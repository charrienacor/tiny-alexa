"""Real-time command recognition: microphone -> log-mel -> CRNN -> action.

The ONNX model has two outputs (intent head + slot head). They are decoded
jointly: only the 31 valid (intent, slot) combinations can be predicted,
so the system can never output an impossible combination.

Run from the repo root:
    python src/infer.py --wav data/MYVOICE/ALARM/ALARM_4_AM_v1_r1.wav
    python src/infer.py --myvoice         # classify your own recordings
    python src/infer.py --no-wakeword     # record + classify once, no wake word
    python src/infer.py                   # wake word -> one command
    python src/infer.py --loop            # wake word -> commands, forever
    python src/infer.py --model models/crnn_int8.onnx   # use the INT8 model
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from features import (SAMPLE_RATE, N_MELS, MAX_FRAMES,
                      load_wav, log_mel, trim_silence, load_intents_slots)
from wakeword import load_wake_word_model, wait_for_wake_word

ROOT = Path(__file__).resolve().parents[1]
MYVOICE_DIR = ROOT / "data" / "MYVOICE"
MYVOICE_CSV = MYVOICE_DIR / "myvoice_manifest.csv"

# Human-readable action per intent. "{slot}" is replaced by the slot value.
INTENT_MESSAGES = {
    "PLAY_MUSIC": "Playing music",
    "PAUSE": "Paused playback",
    "STOP": "Stopped playback",
    "NEXT": "Skipping to the next song",
    "VOLUME_UP": "Turning the volume up",
    "VOLUME_DOWN": "Turning the volume down",
    "LIGHT_ON": "Turning the lights on",
    "LIGHT_OFF": "Turning the lights off",
    "BRIGHTNESS": "Brightness set to {slot}",
    "COLOR": "Light color set to {slot}",
    "TEMPERATURE": "Temperature set to {slot}",
    "TIMER": "Timer started for {slot}",
    "ALARM": "Alarm set for {slot}",
    "CALL": "Starting a call",
    "MESSAGE": "Sending a message",
    "WEATHER": "Weather information is not available offline",
    "TIME": "TIME",  # replaced with the actual clock time below
    "CREATE_REMINDER": "Reminder created: {slot}",
    "LIST_REMINDERS": "You have no reminders yet",
}

# ---- unknown-command rejection ----
# If the joint probability of the best label falls below this, the
# command is treated as out-of-vocabulary. Calibrate with --myvoice.
UNKNOWN_THRESHOLD = 0.50
UNKNOWN_MESSAGE = ("Sorry, that's beyond what I can do. I understand alarms, "
                   "timers, lights, brightness, colors, music and more — try "
                   "something like 'set an alarm for 6 AM'.")

def handle_command(session, wav, labels, label_map):
    """Classify and execute, with unknown-command rejection.

    Returns (label, prob, message, rejected).
    """
    label, prob = predict(session, wav, labels)
    if prob < UNKNOWN_THRESHOLD:
        print(">>> " + UNKNOWN_MESSAGE)
        return label, prob, UNKNOWN_MESSAGE, True
    return label, prob, execute_command(label, label_map), False


_COMBO = None      # (intent index per label, slot index per label)
_LABEL_MAP = None  # label -> (intent, slot value or NO_SLOT)


def load_labels():
    """Ordered label list from data/labels.json."""
    with open(ROOT / "data" / "labels.json", encoding="utf-8") as f:
        return json.load(f)


def load_label_map():
    """label -> (intent, slot value), taken from data/manifest.csv.

    Also builds the label -> (intent idx, slot idx) combo table used for
    joint decoding. Loads once, then cached.
    """
    global _COMBO, _LABEL_MAP
    if _LABEL_MAP is None:
        labels = load_labels()
        intents, slots, label_map = load_intents_slots(ROOT / "data")
        intent_to_idx = {name: i for i, name in enumerate(intents)}
        slot_to_idx = {name: i for i, name in enumerate(slots)}
        _COMBO = (np.array([intent_to_idx[label_map[l][0]] for l in labels]),
                  np.array([slot_to_idx[label_map[l][1]] for l in labels]))
        _LABEL_MAP = label_map
    return _LABEL_MAP


def _log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def create_session(model_path=ROOT / "models" / "crnn.onnx"):
    """Load the exported ONNX model for fast CPU inference."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3   # leave one core for the microphone stream
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    # warmup: the first run is always slower (kernel selection, caches)
    session.run(None, {"logmel": np.zeros((1, 1, N_MELS, MAX_FRAMES), dtype=np.float32)})
    return session


def predict(session, wav, labels):
    """Waveform -> (predicted label, probability) via joint decoding."""
    load_label_map()                                    # builds _COMBO once
    mel = log_mel(trim_silence(wav))[np.newaxis]        # (1, 1, 64, 480)
    intent_logits, slot_logits = session.run(None, {"logmel": mel})
    log_pi = _log_softmax(intent_logits[0])             # (n_intents,)
    log_ps = _log_softmax(slot_logits[0])               # (n_slots,)
    combo_i, combo_s = _COMBO
    # joint score of every valid (intent, slot) combination = the 31 labels
    scores = log_pi[combo_i] + log_ps[combo_s]          # (31,)
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    idx = int(probs.argmax())
    return labels[idx], float(probs[idx])


def record_command(max_sec=4.0, silence_sec=0.8):
    """Record one spoken command from the microphone.

    Waits for speech to start, then records until the speaker has been
    silent for `silence_sec` (or `max_sec` is reached). This endpointing
    is what makes the system respond as soon as you finish speaking.
    Returns float32 samples at 16 kHz, or None if no speech was heard.
    """
    block = int(SAMPLE_RATE * 0.1)                       # 100 ms per block
    max_blocks = int(max_sec * SAMPLE_RATE / block)
    silence_blocks = max(1, int(silence_sec / 0.1))

    print("listening for a command...")
    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=block) as stream:
        # 1) measure the room noise for 0.5 s to set the speech threshold
        noise = []
        for _ in range(5):
            data, _ = stream.read(block)
            noise.append(data[:, 0])
        noise_rms = float(np.sqrt(np.mean(np.concatenate(noise) ** 2)))
        threshold = max(0.004, 4.0 * noise_rms)

        # 2) wait up to 6 s for speech to start
        for _ in range(60):
            data, _ = stream.read(block)
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > threshold:
                frames.append(data[:, 0])
                break
        if not frames:
            print("no speech detected")
            return None

        # 3) keep recording until the speaker stops talking
        silent = 0
        while len(frames) < max_blocks:
            data, _ = stream.read(block)
            frames.append(data[:, 0])
            rms = float(np.sqrt(np.mean(data ** 2)))
            silent = silent + 1 if rms < threshold else 0
            if silent >= silence_blocks:
                break

    wav = np.concatenate(frames).astype(np.float32)
    print(f"captured {len(wav) / SAMPLE_RATE:.2f} s of speech")
    return wav


def execute_command(label, label_map):
    """Turn a predicted label into an action and return a readable message."""
    intent, slot_value = label_map.get(label, (label, ""))
    message = INTENT_MESSAGES.get(intent, intent.replace("_", " ").lower())
    if "{slot}" in message:
        message = message.replace("{slot}", slot_value)
    if intent == "TIME":
        message = "It is " + datetime.now().strftime("%I:%M %p").lstrip("0")
    print(">>> " + message)
    return message


def classify_myvoice(session, labels):
    """Classify every recording listed in data/MYVOICE/myvoice_manifest.csv.

    The manifest needs at least two columns: "path" (relative to
    data/MYVOICE/) and "label" (the expected label, e.g. ALARM_6_00AM).
    Every file goes through the exact same pipeline as the microphone
    (trim -> log-mel -> ONNX -> joint decode), so this measures the
    deployed system on your own voice.
    """
    if not MYVOICE_CSV.exists():
        print(f"not found: {MYVOICE_CSV}")
        print("create it with columns: path,label")
        return
    with open(MYVOICE_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("path") or "").strip()]
    print(f"my voice: {len(rows)} recordings")

    n_labeled, n_correct, n_missing = 0, 0, 0
    for row in rows:
        wav_path = MYVOICE_DIR / row["path"]
        if not wav_path.exists():          # path may already start with MYVOICE/
            wav_path = ROOT / "data" / row["path"]
        if not wav_path.exists():
            print(f"  MISSING {row['path']}")
            n_missing += 1
            continue
        label, prob = predict(session, load_wav(wav_path), labels)
        expected = (row.get("label") or "").strip()
        if expected:
            n_labeled += 1
            if label == expected:
                n_correct += 1
                print(f"  ok    {row['path']} -> {label} ({prob * 100:.0f}%)")
            else:
                print(f"  WRONG {row['path']} -> {label} ({prob * 100:.0f}%)   "
                      f"expected {expected}")
        else:
            print(f"        {row['path']} -> {label} ({prob * 100:.0f}%)")

    if n_missing:
        print(f"{n_missing} file(s) not found")
    if n_labeled:
        print(f"my voice accuracy: {n_correct}/{n_labeled} "
              f"({n_correct / n_labeled * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Tiny voice command recognizer")
    parser.add_argument("--wav", help="classify a WAV file instead of using the microphone")
    parser.add_argument("--myvoice", action="store_true",
                        help="classify the WAVs listed in data/MYVOICE/myvoice_manifest.csv")
    parser.add_argument("--model", default=str(ROOT / "models" / "crnn.onnx"),
                        help="ONNX model to use (e.g. models/crnn_int8.onnx)")
    parser.add_argument("--loop", action="store_true", help="keep listening after each command")
    parser.add_argument("--no-wakeword", action="store_true", help="skip wake-word detection")
    args = parser.parse_args()

    session = create_session(args.model)
    labels = load_labels()
    label_map = load_label_map()

    if args.wav:  # offline test on a file
        label, prob = predict(session, load_wav(args.wav), labels)
        print(f"{args.wav} -> {label} ({prob * 100:.1f}% confidence)")
        return

    if args.myvoice:  # batch test on your own recordings
        classify_myvoice(session, labels)
        return

    wake_model = None if args.no_wakeword else load_wake_word_model()
    try:
        while True:
            if wake_model is not None:
                wait_for_wake_word(wake_model)
            wav = record_command()
            if wav is None:
                if args.loop:
                    continue
                return
            t0 = time.perf_counter()
            label, prob, message, rejected = handle_command(
                session, wav, labels, label_map)
            elapsed = (time.perf_counter() - t0) * 1000
            if rejected:
                print(f"unknown command (confidence only {prob * 100:.0f}%, "
                      f"{elapsed:.0f} ms)")
            else:
                print(f"predicted {label} ({prob * 100:.1f}% confidence, "
                      f"{elapsed:.0f} ms)")
            execute_command(label, label_map)
            if not args.loop:
                return
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
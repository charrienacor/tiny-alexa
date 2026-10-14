"""Real-time voice-command pipeline for the Raspberry Pi 4.

mic (16 kHz) -> openWakeWord 0.4.0 "Alexa" -> record 4.8 s -> TCN.onnx
-> "INTENT  slot=value". Everything runs locally: no ASR, no LLM, no cloud.

Prerequisite: models/TCN.onnx (train.py + export_onnx.py on the GPU box,
then copy to the Pi).

Usage:
    python src/infer.py                 # live microphone
    python src/infer.py --file a.wav    # test with a WAV file
    python src/infer.py --int8          # use TCN_int8.onnx if present

Requires: pip install numpy soundfile onnxruntime sounddevice openwakeword==0.4.0
          sudo apt install libportaudio2
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# Silence onnxruntime's CUDAExecutionProvider warning (requested by
# openWakeWord, absent on the Pi; harmless).
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import openwakeword
from openwakeword.model import Model

from features import SAMPLE_RATE, load_intents_slots, log_mel, trim_silence

ROOT = Path(__file__).resolve().parents[1]

# ── openWakeWord 0.4.0 ─────────────────────────────────────────────
# predict() wants int16 PCM in 80 ms chunks (1280 samples @ 16 kHz);
# float input overflows the internal int16 cast and corrupts features.
WAKE_CHUNK = 1280

# Bundled "Alexa" model — ships inside the wheel, no download needed.
ALEXA_MODEL_PATH = openwakeword.models["alexa"]["model_path"]

# Detection policy.
WAKE_THRESHOLD = 0.5
PATIENCE = 2           # 2 x 80 ms of sustained score before firing
WARMUP_CHUNKS = 10     # ~0.8 s after startup/reset before trusting scores
REFRACTORY_CHUNKS = 15 # ~1.2 s after a detection: the score stays high for
                       # the whole spoken word, so without this one "Alexa"
                       # fires several times

# After the wake word: record this much audio, then run the TCN once.
COMMAND_WINDOW_S = 4.8          # matches the 480-frame TCN input (10 ms hop)
COMMAND_WINDOW_SAMPLES = int(COMMAND_WINDOW_S * SAMPLE_RATE)   # 76800

# Below this confidence the command is reported as unclear. 0.4 turns the
# low-confidence "lights out" -> LIGHT_ON mistakes into "didn't catch that"
# without rejecting any correct val/test prediction (0.5 starts to).
COMMAND_THRESHOLD = 0.4


# ── wake word ────────────────────────────────────────────────────────────────

def make_wakeword_model():
    """Load only the bundled Alexa model (all five would waste ~5 MB on the Pi)."""
    assert Path(ALEXA_MODEL_PATH).exists(), f"missing bundled model: {ALEXA_MODEL_PATH}"
    return Model(wakeword_model_paths=[ALEXA_MODEL_PATH])


def wake_key(wakeword):
    """Key of the loaded model in predict()'s return dict ('alexa_v0.1')."""
    return next(iter(wakeword.models))


class WakeWordDetector:
    """Streaming wrapper around openWakeWord's Model.

    Feed int16 chunks of exactly WAKE_CHUNK samples; reports a detection
    after PATIENCE consecutive chunks above the threshold.
    """

    def __init__(self, wakeword, threshold=WAKE_THRESHOLD, patience=PATIENCE):
        self.wakeword = wakeword
        self.key = wake_key(wakeword)
        self.threshold = threshold
        self.patience = patience
        self.streak = 0
        self.reset()

    def reset(self):
        """Clear detection state; call after each command."""
        self.streak = 0
        self._chunks_since_reset = 0
        self._refractory_left = 0
        self.wakeword.reset()          # v0.4.0: clears the 30-frame score buffer

    def process(self, chunk_int16):
        """One 80 ms chunk in -> True if the wake word fired."""
        assert chunk_int16.dtype == np.int16 and len(chunk_int16) == WAKE_CHUNK, \
            f"expected int16[1280], got {chunk_int16.dtype}[{len(chunk_int16)}]"
        scores = self.wakeword.predict(chunk_int16)
        score = float(scores[self.key])
        self._chunks_since_reset += 1

        if self._refractory_left > 0:
            self._refractory_left -= 1
            return False
        if self._chunks_since_reset <= WARMUP_CHUNKS:
            # Let the streaming front end settle before trusting scores.
            return False

        if score >= self.threshold:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.patience:
            self.streak = 0
            self._refractory_left = REFRACTORY_CHUNKS
            return True
        return False


# ── command model ────────────────────────────────────────────────────────────

class CommandModel:
    """TCN.onnx inference + joint decode over the 31 valid combinations."""

    def __init__(self, onnx_path, data_dir, use_int8=False):
        path = Path(onnx_path)
        if use_int8:
            int8_path = path.with_name(path.stem + "_int8" + path.suffix)
            if int8_path.exists():
                path = int8_path
            else:
                print(f"note: {int8_path.name} not found, using {path.name}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1    # 1 thread: less contention with the
        opts.inter_op_num_threads = 1    # wake-word loop on a 4-core Pi
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name   # "logmel"
        self.output_names = [o.name for o in self.session.get_outputs()]
        # output order is fixed by export_onnx.py: [intent_logits, slot_logits]
        assert self.output_names == ["intent_logits", "slot_logits"], \
            f"unexpected ONNX outputs: {self.output_names}"

        intents, slots, label_map = load_intents_slots(data_dir)
        self.intents = intents
        self.slots = slots
        intent_to_idx = {n: i for i, n in enumerate(intents)}
        slot_to_idx = {n: i for i, n in enumerate(slots)}
        with open(Path(data_dir) / "labels.json", encoding="utf-8") as f:
            labels = json.load(f)
        # same 31-combination layout as train.py / evaluate.py
        self.combo = np.array(
            [[intent_to_idx[label_map[l][0]], slot_to_idx[label_map[l][1]]]
             for l in labels], dtype=np.int64)
        self.labels = labels

    def predict(self, wav_float):
        """16 kHz mono float32 waveform -> (label, intent, slot_value, prob).

        Returns (None, None, None, prob) when the best combination's
        probability is below COMMAND_THRESHOLD (unclear command).
        """
        mel = log_mel(trim_silence(np.asarray(wav_float, dtype=np.float32)))
        mel = mel[None, ...]    # add batch axis: (1, 64, 480) -> (1, 1, 64, 480)
        intent_logits, slot_logits = self.session.run(
            self.output_names, {self.input_name: mel})
        intent_logits = intent_logits[0]      # (19,) — drop batch dim
        slot_logits = slot_logits[0]
        # joint decode: log p(intent) + log p(slot) over valid pairs only
        scores = _log_softmax(intent_logits)[self.combo[:, 0]] + \
                 _log_softmax(slot_logits)[self.combo[:, 1]]
        k = int(scores.argmax())
        prob = float(np.exp(scores[k] - _logsumexp(scores)))
        if prob < COMMAND_THRESHOLD:
            return None, None, None, prob
        label = self.labels[k]
        intent = self.intents[self.combo[k, 0]]
        slot_value = self.slots[self.combo[k, 1]]
        return label, intent, slot_value, prob


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


# ── audio capture ────────────────────────────────────────────────────────────

def read_wav_int16(path):
    """WAV -> int16 mono 16 kHz (same input domain the mic provides)."""
    wav, sr = sf.read(path, dtype="int16")
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SAMPLE_RATE)
        wav = resample_poly(wav.astype(np.float32), SAMPLE_RATE // g,
                            sr // g).astype(np.int16)
    return wav


def mic_stream():
    """Generator of int16 chunks of exactly WAKE_CHUNK samples (80 ms)."""
    import sounddevice as sd
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=WAKE_CHUNK,
                           dtype="int16", channels=1) as stream:
        while True:
            chunk, _overflowed = stream.read(WAKE_CHUNK)
            yield np.frombuffer(chunk, dtype=np.int16)


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="process a WAV file instead of the microphone")
    parser.add_argument("--int8", action="store_true",
                        help="use models/TCN_int8.onnx if present")
    parser.add_argument("--wake-threshold", type=float, default=WAKE_THRESHOLD)
    parser.add_argument("--command-window", type=float, default=COMMAND_WINDOW_S,
                        help="seconds of audio recorded after the wake word")
    args = parser.parse_args()

    # int8 if requested and present, else fp32
    base = ROOT / "models" / "TCN.onnx"
    onnx_path = base.with_name("TCN_int8.onnx") if (args.int8 and (ROOT / "models" / "TCN_int8.onnx").exists()) else base
    if not onnx_path.exists():
        sys.exit(f"error: {onnx_path.name} not found in models/.\n"
                 f"Train and export first (GPU box):  python src/train.py && python src/export_onnx.py\n"
                 f"then copy the file to the Pi's models/ directory.")

    print(f"loading wake word: {Path(ALEXA_MODEL_PATH).name} (openwakeword 0.4.0)")
    wakeword = make_wakeword_model()
    detector = WakeWordDetector(wakeword, threshold=args.wake_threshold)
    print(f"loading command model: {onnx_path.name}")
    cmd = CommandModel(onnx_path, ROOT / "data")

    if args.file:
        run_file(detector, cmd, read_wav_int16(args.file), args.command_window)
    else:
        run_live(detector, cmd, args.command_window)


def run_file(detector, cmd, wav_int16, command_window_s=COMMAND_WINDOW_S):
    """Offline mode: stream a WAV through the wake word, then classify the
    command window that follows each detection. Same code path as live."""
    window_samples = int(command_window_s * SAMPLE_RATE)
    n = len(wav_int16)
    i = 0
    detections = 0
    while i + WAKE_CHUNK <= n:
        if detector.process(wav_int16[i:i + WAKE_CHUNK]):
            detections += 1
            print(f"\n>>> wake word detected at {i / SAMPLE_RATE:.2f} s")
            i += WAKE_CHUNK
            window_end = min(n, i + window_samples)
            window = wav_int16[i:window_end]
            i = window_end
            if len(window) < SAMPLE_RATE:          # < 1 s of audio: skip
                print("    (command window too short, skipping)")
                continue
            t0 = time.perf_counter()
            wav_f = window.astype(np.float32) / 32768.0
            label, intent, slot_value, prob = cmd.predict(wav_f)
            dt = time.perf_counter() - t0
            if label is None:
                print(f"    unclear command (best prob {prob:.2f})")
            else:
                print(f"    {label}   [{intent}{'  ' + slot_value if slot_value != 'NO_SLOT' else ''}]  "
                      f"prob={prob:.2f}  ({dt * 1000:.0f} ms)")
            detector.reset()
        else:
            i += WAKE_CHUNK
    if detections == 0:
        print("no wake word detected in file")
    else:
        print(f"\n{detections} wake word detection(s)")


def run_live(detector, cmd, command_window_s):
    """Live microphone loop: stream 80 ms chunks into the wake word; on a
    detection, read the command window straight from the mic and classify."""
    window_samples = int(command_window_s * SAMPLE_RATE)
    print(f'say "Alexa", then speak your command. Ctrl+C to quit.')
    stream = mic_stream()
    try:
        while True:
            chunk = next(stream)
            if detector.process(chunk):
                print("\n>>> Alexa!")
                # record the command window: 76800 samples = 60 x 1280
                window = np.empty(window_samples, dtype=np.int16)
                for j in range(0, window_samples, WAKE_CHUNK):
                    window[j:j + WAKE_CHUNK] = next(stream)
                t0 = time.perf_counter()
                label, intent, slot_value, prob = cmd.predict(
                    window.astype(np.float32) / 32768.0)
                dt = time.perf_counter() - t0
                if label is None:
                    print(f"    Sorry, I didn't catch that. (best prob {prob:.2f})")
                else:
                    slot_txt = "" if slot_value == "NO_SLOT" else f"  {slot_value}"
                    print(f"    {intent}{slot_txt}   prob={prob:.2f}  ({dt * 1000:.0f} ms)")
                detector.reset()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

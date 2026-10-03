"""Record your own voice for every label (data preparation tool).

Walks you through all 31 labels x 3 phrase variants = 93 recordings,
showing the exact trained phrase to say each time. Each take is a fixed
window (default 2 s) that starts when you press ENTER. Files are saved
as data/MYVOICE/<LABEL>/<LABEL>_v<variant>_r<take>.wav and a
myvoice_manifest.csv is written that infer.py --myvoice can grade.

Run from the repo root:   python record.py

After each take you get a menu:
    enter / y = keep it and move to the next phrase
    p         = play it back (repeat as often as you like)
    r         = re-record the same phrase
    w         = change the recording window length (e.g. 5 for 5 s)
    s         = skip this phrase
    Ctrl+C    = quit safely (writes the manifest for what you have so far)
"""

import csv
import json
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
WINDOW_SEC = 2.0                     # default fixed recording window
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "MYVOICE"
MANIFEST = OUT_DIR / "myvoice_manifest.csv"
LABELS_JSON = ROOT / "data" / "labels.json"

# slot value per label ("" = no slot)
SLOT_VALUE = {
    "ALARM_6_00AM": "6 AM", "ALARM_8_00AM": "8 AM", "ALARM_9_00PM": "9 PM",
    "BRIGHTNESS_100": "100 percent", "BRIGHTNESS_20": "20 percent",
    "BRIGHTNESS_60": "60 percent",
    "COLOR_BLUE": "blue", "COLOR_GREEN": "green", "COLOR_RED": "red",
    "CREATE_REMINDER_DRINK_WATER": "drink water",
    "CREATE_REMINDER_EXERCISE": "exercise",
    "CREATE_REMINDER_STUDY": "study",
    "TEMPERATURE_18": "18 degrees", "TEMPERATURE_22": "22 degrees",
    "TEMPERATURE_26": "26 degrees",
    "TIMER_10s": "10 seconds", "TIMER_30s": "30 seconds", "TIMER_1m": "1 minute",
}

TEMPLATES = {
    "PLAY_MUSIC": ["Play music", "Start music", "Play some music"],
    "WEATHER": ["Weather", "What's the weather?", "Tell me the weather"],
    "TIME": ["Time", "What time is it?", "Tell me the time"],
    "LIGHT_ON": ["Lights on", "Power on the lights", "Turn on the lights"],
    "LIGHT_OFF": ["Lights out", "Kill the lights", "Shut off the lights"],
    "PAUSE": ["Pause", "Pause audio", "Pause song"],
    "STOP": ["Stop", "Stop playing", "End playback"],
    "NEXT": ["Next song", "Skip song", "Play next song"],
    "VOLUME_UP": ["Volume up", "Increase the volume", "Turn the volume up"],
    "VOLUME_DOWN": ["Volume down", "Lower the volume", "Turn the volume down"],
    "CALL": ["Call", "Make a call", "Make a phone call"],
    "MESSAGE": ["Message", "Send a message", "Send my message"],
    "LIST_REMINDERS": ["Reminders", "Show my reminders", "List my reminders"],
    "TIMER": ["Timer {slot}", "Countdown for {slot}", "Start a timer for {slot}"],
    "ALARM": ["Alarm {slot}", "Wake me up at {slot}", "Set an alarm for {slot}"],
    "TEMPERATURE": ["Temperature {slot}", "Change the temperature to {slot}",
                    "Set the temperature to {slot}"],
    "BRIGHTNESS": ["Brightness {slot}", "Adjust brightness to {slot}",
                   "Brightness level {slot}"],
    "COLOR": ["Change color to {slot}", "Switch color to {slot}",
              "Set color to {slot}"],
    "CREATE_REMINDER": ["Reminder {slot}", "Remind me to {slot}",
                        "Create a reminder to {slot}"],
}

# slot name per intent (for the manifest's slot column)
SLOT_NAMES = {"ALARM": "time", "BRIGHTNESS": "percent", "COLOR": "color",
              "CREATE_REMINDER": "task", "TEMPERATURE": "degrees",
              "TIMER": "duration"}

# recording order: fixed intents first, then slotted — change freely
INTENT_ORDER = [
    "PLAY_MUSIC", "PAUSE", "STOP", "NEXT", 
    "VOLUME_UP", "VOLUME_DOWN", "LIGHT_ON", "LIGHT_OFF", 
    "CALL", "MESSAGE", "LIST_REMINDERS", "WEATHER", 
    "TIME", "ALARM", "TIMER", "TEMPERATURE", 
    "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
]


def label_intent(label):
    """ALARM_8_00AM -> ALARM (longest matching intent prefix)."""
    return max((i for i in TEMPLATES if label.startswith(i)), key=len)


def phrases_for(label):
    """The 3 trained phrases for one label, slots filled in."""
    intent = label_intent(label)
    value = SLOT_VALUE.get(label, "")
    return [t.replace("{slot}", value) for t in TEMPLATES[intent]]


def ordered_labels():
    """All 31 labels in recording order (grouped by intent)."""
    by_intent = {}
    with open(LABELS_JSON, encoding="utf-8") as f:
        for label in json.load(f):
            by_intent.setdefault(label_intent(label), []).append(label)
    labels = []
    for intent in INTENT_ORDER:
        labels += sorted(by_intent.get(intent, []))
    return labels


def record_clip(seconds):
    """Record a fixed-length window.

    Starts immediately when called — speak right after pressing ENTER.
    Leading/trailing silence is fine: trim_silence() removes it during
    classification, and log_mel() pads the rest.
    """
    block = int(SAMPLE_RATE * 0.1)                    # 100 ms blocks
    n_blocks = int(seconds * SAMPLE_RATE / block)
    frames = []
    print(f"   recording ({seconds:.0f} s)", end="", flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=block) as stream:
        for _ in range(n_blocks):
            data, _ = stream.read(block)
            frames.append(data[:, 0])
            print(".", end="", flush=True)
    print(" done")
    return np.concatenate(frames).astype(np.float32)


def play_back(wav):
    print("   playing back...")
    sd.play(wav, SAMPLE_RATE)
    sd.wait()


def confirm_take(wav, window_sec):
    """Ask what to do with a fresh take.

    Returns one of: 'keep', 'redo', 'skip', or a float (new window
    length, meaning: re-record with that window).
    """
    while True:
        choice = input("   [ENTER/y]=keep   p=play again   r=re-record   "
                       "w=window   s=skip: ").strip().lower()
        if choice in ("", "y", "yes"):
            return "keep"
        if choice == "p":
            play_back(wav)
            continue                       # menu again after playback
        if choice == "r":
            return "redo"
        if choice == "s":
            return "skip"
        if choice == "w":
            try:
                new = float(input(f"      new window in seconds "
                                  f"(current {window_sec:.0f}): "))
                if 1.0 <= new <= 10.0:
                    return new             # re-record with this window
                print("      enter a value between 1 and 10")
            except ValueError:
                print("      that's not a number")
            continue


def main():
    labels = ordered_labels()
    window_sec = WINDOW_SEC

    print("-" * 60)
    print("MYVOICE recording session — fixed window")
    print(f"{len(labels)} labels x 3 phrases = {len(labels) * 3} recordings")
    print("Speak 10-20 cm from the mic, normal volume, quiet room.")
    print("Press ENTER, then say the phrase immediately.")
    print(f"Current window: {window_sec:.0f} s (change anytime with 'w').")
    print("Ctrl+C quits safely (writes the manifest so far).")
    print("-" * 60)

    results = []                 # rows for the manifest
    n_redo = 0                   # re-takes, for the summary
    try:
        for label in labels:
            intent = label_intent(label)
            phrases = phrases_for(label)
            for v, phrase in enumerate(phrases, start=1):
                while True:
                    print(f"\n[{label}]  variant v{v}")
                    print(f"  SAY: \"{phrase}\"")
                    input(f"   press ENTER, then speak ({window_sec:.0f} s window)... ")
                    wav = record_clip(window_sec)
                    print(f"   captured {len(wav) / SAMPLE_RATE:.2f} s")

                    decision = confirm_take(wav, window_sec)
                    if decision == "redo":
                        n_redo += 1
                        continue
                    if isinstance(decision, float):   # new window -> redo
                        window_sec = decision
                        n_redo += 1
                        print(f"   window set to {window_sec:.0f} s")
                        continue
                    if decision == "skip":
                        break

                    path = f"{label}/{label}_v{v}_r1.wav"
                    (OUT_DIR / path).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(OUT_DIR / path, wav, SAMPLE_RATE, subtype="PCM_16")
                    results.append({
                        "path": path,
                        "label": label,
                        "intent": intent,
                        "speaker": "me",
                        "split": "test",
                        "phrase_id": f"v{v}",
                        "variant_id": "clean",
                        "transcript": phrase,
                        "slot": SLOT_NAMES.get(intent, ""),
                        "slot_value": SLOT_VALUE.get(label, ""),
                        "duration_sec": f"{len(wav) / SAMPLE_RATE:.3f}",
                    })
                    print(f"   saved -> data/MYVOICE/{path}")
                    break
    except (KeyboardInterrupt, EOFError):
        print("\n\nstopping...")

    # write the manifest
    if results:
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{len(results)} recordings saved")
        print(f"manifest written to {MANIFEST}")
    else:
        print("\nno recordings made — nothing written")

    # summary
    print(f"re-takes: {n_redo}")
    covered = {r["label"] for r in results}
    missing_labels = [l for l in labels if l not in covered]
    if missing_labels:
        print(f"labels NOT covered ({len(missing_labels)}):")
        for l in missing_labels:
            print(f"  {l}")
    else:
        print("all labels covered")

    print("\nnext: python src/infer.py --myvoice")


if __name__ == "__main__":
    main()
"""Record 3 seconds from the microphone and classify it immediately —
no WAV file is saved. Uses the exact deployed pipeline (ONNX, two heads,
joint decoding).

Run from the repo root:
    python classify_mic.py                     # countdown -> record -> predict
    python classify_mic.py ALARM_8_00AM        # also grade: OK or WRONG
"""

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))      # make src/ importable

import infer                               # noqa: E402  (ONNX + labels)

SAMPLE_RATE = 16000
SECONDS = 3.0


def main():
    expected = sys.argv[1] if len(sys.argv) > 1 else ""

    # load the model FIRST so the countdown isn't wasted waiting
    print("loading model...")
    session = infer.create_session()
    labels = infer.load_labels()

    print(f"\nrecording {SECONDS:.0f} s (nothing is saved)")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print("  SPEAK NOW")
    wav = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=1, dtype="float32")
    sd.wait()
    wav = wav[:, 0].astype(np.float32)     # (N, 1) -> (N,) mono 1-D array

    peak = float(np.abs(wav).max())
    if peak < 0.01:
        print(f"warning: nearly silent recording (peak {peak:.3f}) — "
              f"check mic/input volume")
    else:
        print(f"captured {len(wav) / SAMPLE_RATE:.2f} s, peak {peak:.3f}\n")

    # classify with the deployed pipeline (trim -> log-mel -> ONNX -> joint)
    t0 = time.perf_counter()
    label, prob = infer.predict(session, wav, labels)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"predicted: {label}  ({prob * 100:.1f}% confidence, "
          f"{elapsed:.0f} ms)")

    if expected:
        if label == expected:
            print(f"OK — matches expected {expected}")
        else:
            print(f"WRONG — expected {expected}, got {label} "
                  f"({prob * 100:.1f}%)")


if __name__ == "__main__":
    main()
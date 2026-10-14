"""Raspberry Pi 4 inference report: wake word + TCN latency (run on the Pi).

Produces results/pi_report.json, the Pi half of the project dashboard:
  * keyword (wake word) true-accept rate  — clips containing "Alexa"
  * false-accept rate                     — clips without the wake word
  * TCN latency p50/p95/p99 + RTF         — the deployed ONNX model
  * wake-word per-chunk cost              — the streaming loop's budget
  * (optional) ONNX intent accuracy       — proves the exported model
                                            matches the GPU-trained one

Run from the repo root on the Pi (wake_tests/ is used by default):
    python src/bench_pi.py
    python src/bench_pi.py --accuracy 200
    python src/bench_pi.py --tp-dir MY_POS --far-dir MY_NEG

Test clips must be 16 kHz mono WAV. Put one "Alexa ..." utterance per
positive file (extra speech after the wake word is fine — it is what the
device hears anyway). Negative files should be everyday speech, TV/radio,
music, or silence; the more of them, the more trustworthy the FAR.

Live check (optional, needs a microphone):
    python src/bench_pi.py --live --expected-count 5
Say "Alexa" exactly 5 times during the window and stay quiet otherwise.
"""

import argparse
import csv
import json
import platform
import socket
import sys
import time
import warnings
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

# Silence onnxruntime's CUDAExecutionProvider warning (requested by
# openWakeWord, absent on the Pi; harmless).
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import onnxruntime as ort
import openwakeword

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import SAMPLE_RATE, load_intents_slots, load_wav, log_mel, trim_silence
from infer import (COMMAND_THRESHOLD, WAKE_CHUNK, WARMUP_CHUNKS, make_wakeword_model,
                   read_wav_int16, wake_key, WakeWordDetector)

ROOT = Path(__file__).resolve().parents[1]
COMMAND_WINDOW_SAMPLES = int(4.8 * SAMPLE_RATE)   # the 4.8 s command window


# ── machine info ─────────────────────────────────────────────────────────────

def _read_first(pattern_lines, key):
    for line in pattern_lines:
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    return None


def machine_info():
    """Host snapshot: OS, CPU, RAM, arch — goes into pi_report.json."""
    info = {"hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "arch": platform.machine()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
        info["cpu"] = (_read_first(cpuinfo, "Model name")
                       or _read_first(cpuinfo, "Hardware")
                       or _read_first(cpuinfo, "processor"))
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        total_kb = int(_read_first(meminfo, "MemTotal:") or 0)
        info["ram_mb"] = round(total_kb / 1024)
    except (OSError, ValueError):
        pass
    return info


def software_versions():
    return {"onnxruntime": version("onnxruntime"),
            "openwakeword": version("openwakeword"),
            "numpy": version("numpy")}


def pct(arr, q):
    return float(np.percentile(np.asarray(arr, dtype=float), q))


# ── wake word tests ──────────────────────────────────────────────────────────

def run_wake_files(detector, folder, expected):
    """Feed every WAV in folder through the detector, one file at a time.

    Each file is preceded by WARMUP_CHUNKS of silence so the detector is in
    steady state when the audio starts — the live equivalent is a mic that
    has been running for a while before the user speaks.

    Returns (detected_count, per-file results). expected=True marks files
    that contain the wake word (TP set); False marks the FAR set.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"error: {folder} does not exist — record test clips "
                         f"there (16 kHz mono WAV) or pass --tp-dir/--far-dir")
    files = sorted(folder.glob("*.wav"))
    if not files:
        raise SystemExit(f"error: no .wav files in {folder}")
    silence = np.zeros(WAKE_CHUNK, dtype=np.int16)
    per_file = []
    for path in files:
        wav = read_wav_int16(path)
        if len(wav) % WAKE_CHUNK:                    # pad to whole 80 ms chunks
            wav = np.pad(wav, (0, WAKE_CHUNK - len(wav) % WAKE_CHUNK))
        detector.reset()
        for _ in range(WARMUP_CHUNKS):               # pre-warm (see docstring)
            detector.process(silence)
        detected_at = None
        for i in range(0, len(wav), WAKE_CHUNK):
            if detector.process(wav[i:i + WAKE_CHUNK]):
                detected_at = (i + WAKE_CHUNK) / SAMPLE_RATE
                break
        ok = detected_at is not None
        per_file.append({"file": path.name,
                         "expected": expected,
                         "detected": ok,
                         "detection_s": round(detected_at, 2) if ok else None})
        mark = "OK " if ok == expected else "MISS"
        print(f"  {mark} {path.name}"
              + (f"  (fired at {detected_at:.2f} s)" if ok else ""))
    return sum(r["detected"] for r in per_file), per_file


def run_live_wake(detector, seconds, expected_count):
    """Stream the microphone for `seconds`; user says "Alexa" expected_count
    times and stays quiet otherwise. Returns the number of detections."""
    import sounddevice as sd
    print(f"listening for {seconds} s — say 'Alexa' exactly {expected_count} "
          f"time(s), stay quiet otherwise")
    time.sleep(1)
    n = 0
    t_end = time.perf_counter() + seconds
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=WAKE_CHUNK,
                           dtype="int16", channels=1) as stream:
        while time.perf_counter() < t_end:
            chunk, _ = stream.read(WAKE_CHUNK)
            if detector.process(np.frombuffer(chunk, dtype=np.int16)):
                n += 1
                print(f"  ... detection #{n}")
    return n


# ── latency benchmarks ───────────────────────────────────────────────────────

def bench_tcn(session, input_name, output_names, mel, n_runs):
    """Time n_runs inferences of the TCN on one fixed log-mel."""
    mel = mel[None, ...]                             # add batch axis
    for _ in range(5):                               # warm up
        session.run(output_names, {input_name: mel})
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: mel})
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


def bench_wake_chunks(wakeword, n_chunks):
    """Time n_chunks wake-word predictions on silence (steady-state cost)."""
    silence = np.zeros(WAKE_CHUNK, dtype=np.int16)
    for _ in range(20):                               # warm up
        wakeword.predict(silence)
    times_ms = []
    for _ in range(n_chunks):
        t0 = time.perf_counter()
        wakeword.predict(silence)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


def bench_command_total(cmd, wav, n_runs):
    """Full per-command cost: trim + log-mel + ONNX + joint decode."""
    def once(w):
        mel = log_mel(trim_silence(np.asarray(w, dtype=np.float32)))
        decode(cmd, mel)
    for _ in range(3):
        once(wav)
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        once(wav)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


# ── TCN session + decode (same math as infer.CommandModel) ──────────────────

def build_command(onnx_path, data_dir, threads=1):
    """ONNX session + the 31-combination joint-decode table."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    assert output_names == ["intent_logits", "slot_logits"], output_names

    intents, slots, label_map = load_intents_slots(data_dir)
    intent_to_idx = {n: i for i, n in enumerate(intents)}
    slot_to_idx = {n: i for i, n in enumerate(slots)}
    labels = json.loads((Path(data_dir) / "labels.json").read_text(encoding="utf-8"))
    combo = np.array([[intent_to_idx[label_map[l][0]],
                       slot_to_idx[label_map[l][1]]] for l in labels],
                     dtype=np.int64)
    return {
        "session": session, "input_name": input_name,
        "output_names": output_names, "labels": labels,
        "intents": intents, "slots": slots, "combo": combo,
    }


def decode(cmd, mel):
    """(1, 64, 480) log-mel -> (label_index, intent_index, prob)."""
    intent_logits, slot_logits = cmd["session"].run(
        cmd["output_names"], {cmd["input_name"]: mel[None, ...]})
    il, sl = intent_logits[0], slot_logits[0]
    combo = cmd["combo"]
    log_i = il - il.max()
    log_s = sl - sl.max()
    scores = (log_i[combo[:, 0]] - np.log(np.exp(log_i).sum())) + \
             (log_s[combo[:, 1]] - np.log(np.exp(log_s).sum()))
    k = int(scores.argmax())
    prob = float(np.exp(scores[k] - (scores.max()
                                     + np.log(np.exp(scores - scores.max()).sum()))))
    return k, int(combo[k, 0]), prob


def read_test_rows(data_dir):
    """Held-out test rows from manifest.csv (no torch needed on the Pi)."""
    rows = []
    with open(Path(data_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"].strip().lower() == "test":
                rows.append(row)
    return rows


def bench_accuracy(cmd, data_dir, n_utts):
    """Run the ONNX model on n_utts held-out test utterances (spread across
    labels and speakers) and compare against the manifest. Reports label
    accuracy with the production COMMAND_THRESHOLD applied, and unthresholded
    intent accuracy. Proves the exported model matches the GPU-trained one."""
    rows = read_test_rows(data_dir)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    rng = np.random.default_rng(0)
    for lab in by_label:
        rng.shuffle(by_label[lab])    # manifest is path-sorted, so without a
                                      # shuffle the first rows per label all
                                      # come from one speaker
    picked = []
    labels_sorted = sorted(by_label)
    while len(picked) < n_utts:
        progressed = False
        for lab in labels_sorted:
            if by_label[lab] and len(picked) < n_utts:
                picked.append(by_label[lab].pop(0))
                progressed = True
        if not progressed:
            break

    correct_label = correct_intent = 0
    for row in picked:
        # load_wav resamples the 12 kHz recordings to 16 kHz, like training
        wav = load_wav(Path(data_dir) / row["path"])
        mel = log_mel(trim_silence(wav))
        k, intent_idx, prob = decode(cmd, mel)
        if prob >= COMMAND_THRESHOLD and cmd["labels"][k] == row["label"]:
            correct_label += 1
        if intent_idx == cmd["combo"][labels_index(cmd, row["label"]), 0]:
            correct_intent += 1
    return {"n": len(picked),
            "label_accuracy_pct": round(100.0 * correct_label / len(picked), 2),
            "intent_accuracy_pct": round(100.0 * correct_intent / len(picked), 2)}


def labels_index(cmd, label):
    return cmd["labels"].index(label)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp-dir", default=ROOT / "wake_tests" / "positives",
                        help="folder of WAVs containing the wake word "
                             "(default: wake_tests/positives)")
    parser.add_argument("--far-dir", default=ROOT / "wake_tests" / "negatives",
                        help="folder of WAVs WITHOUT the wake word "
                             "(default: wake_tests/negatives)")
    parser.add_argument("--accuracy", type=int, default=0, metavar="N",
                        help="also run the ONNX model on N held-out test "
                             "utterances (needs data/manifest.csv)")
    parser.add_argument("--latency-n", type=int, default=200,
                        help="number of timed TCN inferences (default 200)")
    parser.add_argument("--live", action="store_true",
                        help="interactive microphone check instead of file folders")
    parser.add_argument("--expected-count", type=int, default=5,
                        help="how many times you will say the wake word in --live mode")
    parser.add_argument("--live-seconds", type=int, default=60)
    parser.add_argument("--int8", action="store_true",
                        help="benchmark models/TCN_int8.onnx if present")
    parser.add_argument("--results", default=ROOT / "results")
    args = parser.parse_args()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    onnx_path = ROOT / "models" / "TCN.onnx"
    if args.int8 and (ROOT / "models" / "TCN_int8.onnx").exists():
        onnx_path = ROOT / "models" / "TCN_int8.onnx"
    if not onnx_path.exists():
        sys.exit(f"error: {onnx_path} not found — copy it from the GPU box")

    print("loading wake word model (alexa_v0.1, openwakeword 0.4.0)")
    wakeword = make_wakeword_model()
    detector = WakeWordDetector(wakeword)
    print(f"loading command model: {onnx_path.name}")
    cmd = build_command(onnx_path, ROOT / "data")

    report = {
        "stage": "pi_inference",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": machine_info(),
        "software": software_versions(),
        "model": {"path": onnx_path.name,
                  "size_kb": round(onnx_path.stat().st_size / 1024),
                  "threads": 1},
        "wake_word": {"model": wake_key(wakeword),
                      "threshold": detector.threshold,
                      "patience_chunks": detector.patience},
    }

    # ── wake word accuracy ────────────────────────────────────────────────
    if args.tp_dir:
        print(f"\nwake word — positive set ({args.tp_dir})")
        n_det, per_file = run_wake_files(detector, args.tp_dir, expected=True)
        report["wake_word"]["true_accept"] = {
            "files": len(per_file), "detected": n_det,
            "rate_pct": round(100.0 * n_det / len(per_file), 2),
            "per_file": per_file}
    if args.far_dir:
        print(f"\nwake word — negative set ({args.far_dir})")
        n_det, per_file = run_wake_files(detector, args.far_dir, expected=False)
        report["wake_word"]["false_accept"] = {
            "files": len(per_file), "detected": n_det,
            "rate_pct": round(100.0 * n_det / len(per_file), 2),
            "per_file": per_file}
    if args.live:
        print()
        n_det = run_live_wake(detector, args.live_seconds, args.expected_count)
        report["wake_word"]["live"] = {
            "expected": args.expected_count, "detected": n_det,
            "seconds": args.live_seconds}

    # ── latency ───────────────────────────────────────────────────────────
    print(f"\nbenchmarking TCN latency ({args.latency_n} runs, 1 thread)")
    mel = log_mel(np.zeros(COMMAND_WINDOW_SAMPLES, dtype=np.float32))
    tcn_ms = bench_tcn(cmd["session"], cmd["input_name"], cmd["output_names"],
                       mel, args.latency_n)
    wake_ms = bench_wake_chunks(wakeword, max(100, args.latency_n // 2))
    cmd_wav = np.zeros(COMMAND_WINDOW_SAMPLES, dtype=np.float32)
    total_ms = bench_command_total(cmd, cmd_wav, max(20, args.latency_n // 4))

    report["latency_ms"] = {
        "tcn": {"n": len(tcn_ms), "p50": round(pct(tcn_ms, 50), 2),
                "p95": round(pct(tcn_ms, 95), 2), "p99": round(pct(tcn_ms, 99), 2),
                "mean": round(float(np.mean(tcn_ms)), 2)},
        "rtf": round(4.8 / (pct(tcn_ms, 95) / 1000.0), 1),
        "wake_chunk": {"n": len(wake_ms), "p50": round(pct(wake_ms, 50), 2),
                       "p95": round(pct(wake_ms, 95), 2)},
        "command_total": {"n": len(total_ms),
                          "p50": round(pct(total_ms, 50), 2),
                          "p95": round(pct(total_ms, 95), 2)},
    }

    # ── ONNX accuracy on the held-out test set (optional) ─────────────────
    if args.accuracy:
        print(f"\nrunning ONNX model on {args.accuracy} held-out test utterances")
        report["accuracy_onnx"] = bench_accuracy(cmd, ROOT / "data", args.accuracy)

    out = results / "pi_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ── summary (the exact dashboard lines) ───────────────────────────────
    print(f"\nwrote {out}")
    ww = report["wake_word"]
    if "true_accept" in ww:
        ta = ww["true_accept"]
        print(f"  keyword (wake word) accuracy: {ta['rate_pct']:.2f}%  "
              f"({ta['detected']}/{ta['files']})")
    if "false_accept" in ww:
        fa = ww["false_accept"]
        print(f"  false-accept rate:            {fa['rate_pct']:.2f}%  "
              f"({fa['detected']}/{fa['files']} files)")
    lat = report["latency_ms"]
    print(f"  TCN latency p95:            {lat['tcn']['p95']} ms  "
          f"(p50 {lat['tcn']['p50']} ms, RTF {lat['rtf']})")
    if "live" in ww:
        lv = ww["live"]
        print(f"  live check:                 {lv['detected']}/{lv['expected']} detections")
    if "accuracy_onnx" in report:
        acc = report["accuracy_onnx"]
        print(f"  ONNX accuracy (test set):   label {acc['label_accuracy_pct']}%  "
              f"intent {acc['intent_accuracy_pct']}%  (n={acc['n']})")
    print("\nnext: copy results/pi_report.json + results/gpu_report.json to one")
    print("machine and run  python src/dashboard.py  to fill the dashboard table")


if __name__ == "__main__":
    main()

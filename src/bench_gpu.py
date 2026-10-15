"""GPU-side training report (run on the A100 box, after train + evaluate).

Run from the repo root:
    python src/train.py
    python src/evaluate.py
    python src/bench_gpu.py

Merges the artifacts those two scripts wrote — results/train_config.json
(machine + recipe), results/history.csv (per-epoch log) and
results/metrics.txt (held-out test results) — into results/gpu_report.json.
Keep that file: bench_pi.py and dashboard.py combine it with the Pi report
to fill the project dashboard.
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_metrics(text):
    """Pull the headline numbers out of metrics.txt."""
    out = {}
    patterns = {
        "test_utts": r"^test utterances:\s*(\d+)",
        "intent_accuracy": r"^intent accuracy:\s*([\d.]+)",
        "slot_accuracy": r"^slot accuracy:\s*([\d.]+)",
        "label_accuracy": r"label accuracy \(intent \+ slot decoded jointly\):\s*([\d.]+)",
        "clean_accuracy": r"^clean accuracy:\s*([\d.]+)",
        "noisy_accuracy": r"^noisy accuracy:\s*([\d.]+)",
        "training_time": r"^training time:\s*(\d+) s",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    # the readable form too: "324 s (0 hr 05 min 23.9 s / 0.09 hr)"
    m = re.search(r"^training time:\s*(\d+ s \(.*\))\s*$", text, flags=re.MULTILINE)
    if m:
        out["training_time_text"] = m.group(1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=ROOT / "results",
                        help="folder holding train_config.json, history.csv, metrics.txt")
    args = parser.parse_args()
    results = Path(args.results)

    missing = [n for n in ("train_config.json", "history.csv", "metrics.txt")
               if not (results / n).exists()]
    if missing:
        raise SystemExit(f"error: missing {', '.join(missing)} in {results} — "
                         f"run src/train.py and src/evaluate.py first")

    config = json.loads((results / "train_config.json").read_text(encoding="utf-8"))
    with open(results / "history.csv", newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    metrics = parse_metrics((results / "metrics.txt").read_text(encoding="utf-8"))

    def col(name):
        return [float(row[name]) for row in history]

    epochs_run = len(history)
    total_s = sum(col("elapsed_s"))

    report = {
        "stage": "gpu_training",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": {k: config[k] for k in
                    ("hostname", "platform", "python", "torch") if k in config},
        "hardware": {k: config[k] for k in
                     ("gpu", "gpu_count", "gpu_memory_gib", "cuda", "driver")
                     if k in config},
        "recipe": {k: config[k] for k in
                   ("objective", "optimizer", "batch_size", "patience",
                    "epochs_planned", "steps_per_epoch", "model_params",
                    "train_utts", "valid_utts") if k in config},
        "training": {
            "epochs_run": epochs_run,
            "total_steps": epochs_run * config.get("steps_per_epoch", 0),
            "total_seconds": round(total_s, 1),
            "final_val_loss": col("val_loss")[-1],
            "best_val_acc_pct": max(col("val_acc")),
            "last_lr": col("lr")[-1],
            "history": history,
        },
        "test": metrics,
    }

    out = results / "gpu_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    hw = report["hardware"]
    print(f"wrote {out}")
    if hw.get("gpu"):
        print(f"  machine:  {report['machine'].get('hostname')}  "
              f"{hw['gpu']} x{hw.get('gpu_count', 1)}  "
              f"({hw.get('gpu_memory_gib')} GiB)")
    else:
        print(f"  machine:  {report['machine'].get('hostname')}  "
              f"(no NVIDIA GPU — CPU/MPS run)")
    print(f"  training: {epochs_run} epoch(s), {report['training']['total_steps']} steps, "
          f"{report['training']['total_seconds']} s, "
          f"final val loss {report['training']['final_val_loss']:.3f}")
    print(f"  test:     intent {metrics.get('intent_accuracy', float('nan')):.4f}  "
          f"slot {metrics.get('slot_accuracy', float('nan')):.4f}  "
          f"label {metrics.get('label_accuracy', float('nan')):.4f}")
    print("\nnext: copy models/TCN.onnx to the Pi and run src/bench_pi.py there")


if __name__ == "__main__":
    main()

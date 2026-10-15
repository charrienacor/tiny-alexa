"""Merge the GPU and Pi reports into the project dashboard table.

Run on any machine that has both report files (usually the GPU box):
    python src/dashboard.py

Fills the four boxes of the ME2 deck — Model, Dataset, Training on A100,
Validation on RPi 4 — and writes results/dashboard.md, ready to paste into
the project sheet:
  - Model:      read from src/model.py + src/features.py and the exported
                ONNX file(s) in models/
  - Dataset:    counted from data/manifest.csv
  - Training:   results/gpu_report.json (from src/bench_gpu.py on the A100)
  - Validation: results/pi_report.json  (from src/bench_pi.py on the Pi 4)
Rows whose data is still missing show (TBD) with a hint of what to run.
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import HOP, MAX_FRAMES, N_FFT, N_MELS, SAMPLE_RATE, load_intents_slots  # noqa: E402

TBD = "(TBD)"

# Where the recordings come from (see data/README.md) — not in the manifest.
DATASET_SOURCE = ("LibriSpeech + SilencioPH (Filipino-English) speakers, "
                  "transcribed with simple-audio-transcriber; clean + noisy takes")


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt(value, suffix="", nd=2):
    if value is None:
        return TBD
    if isinstance(value, float):
        return f"{value:.{nd}f}{suffix}"
    return f"{value}{suffix}"


def fmt_duration(total):
    """323.9 -> '324 s (0 hr 05 min 23.9 s / 0.09 hr)', as in metrics.txt."""
    t = round(total, 1)                  # so 59.96 s shows as 1 min 00.0 s
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{total:.0f} s ({h} hr {m:02d} min {s:04.1f} s / {total / 3600:.2f} hr)"


def model_constants():
    """Architecture constants from src/model.py, read without importing it
    (model.py needs torch, which the Pi doesn't have)."""
    tree = ast.parse((ROOT / "src" / "model.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            for target in node.targets:
                names = target.elts if isinstance(target, ast.Tuple) else [target]
                values = value if isinstance(target, ast.Tuple) else [value]
                for t, v in zip(names, values):
                    if isinstance(t, ast.Name):
                        consts[t.id] = v
    return consts


def onnx_param_count(path):
    """Weights stored in an ONNX file (BatchNorm is folded into the convs on
    export, so this is a little below the PyTorch count). None without onnx."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return None
    m = onnx.load(str(path))
    return sum(numpy_helper.to_array(t).size for t in m.graph.initializer)


def model_rows(gpu_recipe):
    c = model_constants()
    intents, slots, label_map = load_intents_slots(ROOT / "data")
    n_slot_values = len(slots) - 1                       # minus NO_SLOT
    blocks, layers = c.get("N_BLOCKS"), c.get("LAYERS_PER_BLOCK")
    width, kernel = c.get("C2"), c.get("KERNEL")
    dils = ", ".join(str(2 ** b) for b in range(blocks)) if blocks else "?"
    seconds = (N_FFT + HOP * (MAX_FRAMES - 1)) / SAMPLE_RATE

    params = gpu_recipe.get("model_params")
    files = [f for f in (ROOT / "models" / "TCN.onnx", ROOT / "models" / "TCN_int8.onnx")
             if f.exists()]
    if params is None and files:
        params = onnx_param_count(files[0])
    weights = " · ".join(f"{f.stat().st_size / 1e6:.2f} MB {f.name}" for f in files)
    if params is not None and weights:
        param_line = f"{params / 1e6:.2f} M ({params:,}) · {weights}"
    elif params is not None:
        param_line = f"{params / 1e6:.2f} M ({params:,}) · {TBD} MB — run src/export_onnx.py"
    else:
        param_line = f"{TBD} — run src/train.py and src/export_onnx.py"

    return [
        f"| Front end | Mic · {SAMPLE_RATE // 1000} kHz → log-mel {N_MELS} × {MAX_FRAMES} "
        f"({N_FFT * 1000 // SAMPLE_RATE} ms window, {HOP * 1000 // SAMPLE_RATE} ms hop, "
        f"{seconds:.1f} s) |",
        f"| Encoder | TCN × {blocks} dilated blocks ({layers} convs each, k={kernel}, "
        f"dilations {dils}) · {width}-dim; non-causal, whole-utterance "
        f"(no KV-cache / streaming) |",
        f"| Heads | intent head ({len(intents)}) + slot head ({len(slots)}, incl. NO_SLOT), "
        f"joint decode over {len(label_map)} valid pairs |",
        "| Actuator | src/server.py → tiny-alexa.html (lights, music, timer, alarm, …) |",
        "| Target | Raspberry Pi 4 · onnxruntime (CPU) · openWakeWord \"Alexa\" wake word |",
        f"| Parameters / weights | {param_line} |",
    ]


def dataset_rows():
    with open(ROOT / "data" / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    intents, slots, label_map = load_intents_slots(ROOT / "data")
    hours = sum(float(r.get("duration_sec") or 0) for r in rows) / 3600
    splits = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    split_txt = " / ".join(f"{splits.get(k, 0):,}" for k in ("train", "val", "test"))

    speakers = {r["speaker"] for r in rows}
    try:                                   # dataset.py imports torch
        from dataset import FILIPINO_SPEAKERS
        fil = len(speakers & FILIPINO_SPEAKERS)
        speaker_txt = f"{len(speakers)} ({len(speakers) - fil} foreign + {fil} Filipino-English)"
    except ImportError:
        speaker_txt = str(len(speakers))
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["speaker"])
    speaker_txt += " · " + " / ".join(str(len(by_split.get(k, ()))) for k in ("train", "val", "test")) \
        + " train/val/test"

    clean = sum(1 for r in rows if r["path"].endswith("_clean.wav"))
    return [
        f"| Source | {DATASET_SOURCE} |",
        f"| Hours / utts | {hours:.2f} h / {len(rows):,} ({split_txt} train/val/test; "
        f"{clean:,} clean + {len(rows) - clean:,} noisy) |",
        f"| Speakers | {speaker_txt} |",
        f"| Labels | {len(intents)} intents · {len(slots) - 1} slot values "
        f"({len(label_map)} command labels) |",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-report", default=ROOT / "results" / "gpu_report.json")
    parser.add_argument("--pi-report", default=ROOT / "results" / "pi_report.json")
    parser.add_argument("--out", default=ROOT / "results" / "dashboard.md")
    args = parser.parse_args()

    gpu = load(Path(args.gpu_report))
    pi = load(Path(args.pi_report))

    g = gpu or {}
    p = pi or {}
    ghw, gtr, gtest, grec = g.get("hardware", {}), g.get("training", {}), \
        g.get("test", {}), g.get("recipe", {})
    pmach, pww, plat, pacc = p.get("machine", {}), p.get("wake_word", {}), \
        p.get("latency_ms", {}), p.get("accuracy_onnx", {})

    # ── Training on A100 ──────────────────────────────────────────────────
    gpu_name = ghw.get("gpu")
    if gpu_name:
        cluster = (f"{g.get('machine', {}).get('hostname', '?')} — "
                   f"{gpu_name} x{ghw.get('gpu_count', 1)} "
                   f"({ghw.get('gpu_memory_gib')} GiB, CUDA {ghw.get('cuda')}, "
                   f"driver {ghw.get('driver')})")
    elif g:
        cluster = (f"{g.get('machine', {}).get('hostname', '?')} — no NVIDIA GPU "
                   f"found (trained on CPU/MPS); re-run on the A100 to fill this")
    else:
        cluster = TBD
    objective = grec.get("objective", TBD)
    optimizer = grec.get("optimizer", TBD)
    train_time = gtest.get("training_time_text") or (
        fmt_duration(gtr["total_seconds"]) if gtr.get("total_seconds") is not None else TBD)
    steps = (f"{gtr.get('total_steps')} steps "
             f"({gtr.get('epochs_run')} epoch(s) x {grec.get('steps_per_epoch')}, "
             f"{train_time})") if gtr else TBD
    val_loss = fmt(gtr.get("final_val_loss"), nd=3) if gtr else TBD

    # ── Validation on RPi 4 ───────────────────────────────────────────────
    ta = pww.get("true_accept")
    fa = pww.get("false_accept")
    kw_acc = (f"{fmt(ta.get('rate_pct'))}% ({ta['detected']}/{ta['files']} clips)") \
        if ta else TBD
    far = (f"{fmt(fa.get('rate_pct'))}% ({fa['detected']}/{fa['files']} clips)") \
        if fa else TBD

    # Prefer the end-to-end VCM holdout: the same clips as the keyword
    # accuracy, so both numbers describe one test.
    vcm_ia = p.get("vcm_holdout", {}).get("intent_accuracy")
    if vcm_ia:
        intent_acc = (f"{fmt(vcm_ia.get('rate_pct'))}% "
                      f"({vcm_ia['count']}/{vcm_ia['n']} clips)")
    elif gtest.get("intent_accuracy") is not None:
        intent_acc = (f"{gtest['intent_accuracy'] * 100:.2f}% "
                      f"(full test set, n={int(gtest.get('test_utts', 0))})")
        if pacc:
            intent_acc += (f" — ONNX check {pacc.get('intent_accuracy_pct')}% "
                           f"(n={pacc['n']})")
    elif pacc:
        intent_acc = (f"{fmt(pacc.get('intent_accuracy_pct'))}% "
                      f"(ONNX, n={pacc['n']})")
    else:
        intent_acc = TBD

    tcn = plat.get("tcn", {})
    latency = (f"p95 {tcn.get('p95')} ms / RTF {plat.get('rtf')} "
               f"(p50 {tcn.get('p50')} ms, n={tcn.get('n')})") if tcn else TBD
    runtime = (f"onnxruntime {p.get('software', {}).get('onnxruntime')} — "
               f"{p.get('model', {}).get('threads', 1)} thread(s) "
               f"({p.get('model', {}).get('path')}, "
               f"{p.get('model', {}).get('size_kb')} KB)") if p else TBD

    hw_parts = [str(v) for v in (pmach.get("hostname"), pmach.get("cpu"),
                                 (f"{pmach['ram_mb']} MB RAM"
                                  if pmach.get("ram_mb") else None),
                                 pmach.get("arch")) if v]
    hw_line = " — ".join(hw_parts) if hw_parts else TBD
    ww_line = (f"{pww.get('model')} (openwakeword "
               f"{p.get('software', {}).get('openwakeword')}), threshold "
               f"{pww.get('threshold')}, patience {pww.get('patience_chunks')} chunks") \
        if pww.get("model") else TBD

    lines = [
        "# Project dashboard — auto-filled",
        "",
        f"_generated from: {args.gpu_report} "
        f"{'(present)' if gpu else '(MISSING — run src/bench_gpu.py on the GPU box)'}"
        f", {args.pi_report} "
        f"{'(present)' if pi else '(MISSING — run src/bench_pi.py on the Pi)'}_",
        "",
        "| **Model** | |",
        "|---|---|",
        *model_rows(grec),
        "",
        "| **Dataset** | |",
        "|---|---|",
        *dataset_rows(),
        "",
        "| **Training on A100** | |",
        "|---|---|",
        f"| Cluster (# A100) | {cluster} |",
        f"| Objective | {objective} |",
        f"| Optimizer | {optimizer} |",
        f"| Steps / loss | {steps} / {val_loss} |",
        "",
        "| **Validation on RPi 4** | |",
        "|---|---|",
        f"| Hardware | {hw_line} |",
        f"| Wake word | {ww_line} |",
        f"| Keyword / intent accuracy | {kw_acc} / {intent_acc} |",
        f"| False-accept rate | {far} |",
        f"| Latency p95 / RTF | {latency} |",
        f"| Runtime | {runtime} |",
        "",
    ]

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

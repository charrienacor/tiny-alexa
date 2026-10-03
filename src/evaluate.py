"""Evaluate the trained two-head CRNN on the test split.

Run from the repo root:   python src/evaluate.py

Writes to results/:
    metrics.txt               - training summary + intent/slot/label metrics
    class_metrics_intent.png  - intent per-class precision/recall/F1 bars
    class_metrics_slot.png    - slot per-class precision/recall/F1 bars
    confusion_intent.png      - intent confusion matrix (cells = raw counts)
    confusion_slot.png        - slot confusion matrix (cells = raw counts)
    accuracy_label.png        - training curves from history.csv

Label accuracy = both heads correct after joint decoding (comparable to
the old single-head model's number).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VCMDataset, load_labels
from features import load_intents_slots
from model import CRNN

ROOT = Path(__file__).resolve().parents[1]


def predict_all(model, dataset, device, combo, batch_size=64):
    """Run both heads over the dataset.

    Returns (intent_true, intent_pred, slot_true, slot_pred,
             label_true, label_pred) as class-index arrays.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    it, ip, st, sp, lt, lp = [], [], [], [], [], []
    combo_i, combo_s = combo[:, 0], combo[:, 1]
    with torch.no_grad():
        for mel, intent, slot, label in loader:
            mel = mel.to(device)
            intent_logits, slot_logits = model(mel)
            # joint decode over the 31 valid (intent, slot) combinations
            log_pi = torch.log_softmax(intent_logits, dim=1)
            log_ps = torch.log_softmax(slot_logits, dim=1)
            scores = log_pi[:, combo_i] + log_ps[:, combo_s]      # (B, 31)
            lp.extend(scores.argmax(1).cpu().tolist())
            ip.extend(intent_logits.argmax(1).cpu().tolist())
            sp.extend(slot_logits.argmax(1).cpu().tolist())
            it.extend(intent.tolist())
            st.extend(slot.tolist())
            lt.extend(label.tolist())
    return (np.array(it), np.array(ip), np.array(st),
            np.array(sp), np.array(lt), np.array(lp))


def compute_metrics(y_true, y_pred, n_classes):
    """Confusion matrix + per-class precision/recall/F1 + accuracy."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    pred_sum = cm.sum(axis=0).astype(float)
    true_sum = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_sum, out=np.zeros(n_classes), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros(n_classes), where=true_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=(precision + recall) > 0)
    acc = tp.sum() / cm.sum()
    return cm, precision, recall, f1, acc


def grouped_bar_chart(classes, precision, recall, f1, acc, title, path):
    """Per-class precision/recall/F1 grouped bars with value labels."""
    n = len(classes)
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(10, n * 0.68), 4.8))
    bars = [
        ax.bar(x - w, precision * 100, w, color="#1f77b4", label="precision"),
        ax.bar(x,      recall * 100,    w, color="#ff7f0e", label="recall"),
        ax.bar(x + w,  f1 * 100,        w, color="#2ca02c", label="F1"),
    ]
    for b in bars:
        ax.bar_label(b, fmt="%.0f", fontsize=5 if n > 20 else 6, padding=1)
    ax.axhline(f1.mean() * 100, color="gray", ls="--", lw=1,
               label=f"macro F1 {f1.mean() * 100:.1f}")
    ax.set_ylim(0, 120)                     # headroom for the value labels
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=6 if n > 20 else 8)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_title(
        f"{title}\n"
        f"macro P/R/F1 = {precision.mean() * 100:.1f}/"
        f"{recall.mean() * 100:.1f}/{f1.mean() * 100:.1f}"
        f"   micro P/R/F1 = {acc * 100:.1f}/{acc * 100:.1f}/{acc * 100:.1f}",
        fontsize=11)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_chart(cm, classes, acc, title, path):
    """Confusion matrix PNG with raw counts printed in the cells.

    The cell NUMBER is the raw count. The cell COLOR stays normalized to
    % of true class so errors remain visible for classes with few
    samples — otherwise one big class (e.g. NO_SLOT) would wash out the
    whole color scale. Blank cells are zero.
    """
    n = len(classes)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, row_sum, out=np.zeros(cm.shape, dtype=float),
                    where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.65), max(9, n * 0.5)))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=4 if n > 25 else 6,
                        color="white" if pct[i, j] > 50 else "black")
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=5 if n > 25 else 7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=5 if n > 25 else 7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="color: % of true class",
                 fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def training_curves_chart(history, name, path):
    """Train acc and val acc (solid lines, circle marker per epoch) plus
    val loss (dashed) in one combined plot.

    The y-axis auto-scales with headroom above 100% so a saturated train
    curve (100.0) is not jammed against the plot border.
    """
    train_acc = np.array(history["train_acc"], dtype=float)
    val_acc = np.array(history["val_acc"], dtype=float)
    val_loss = np.array(history["val_loss"], dtype=float)
    if max(train_acc.max(), val_acc.max()) <= 1.0:   # fractions -> percent
        train_acc *= 100
        val_acc *= 100

    epochs = np.arange(1, len(val_acc) + 1)
    best = int(val_acc.argmax())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # train acc — blue line, circle marker per epoch
    l1, = ax.plot(epochs, train_acc, "o-", color="#1f77b4", lw=2, ms=5,
                  label="train acc")
    # val acc — red line, circle marker per epoch
    l2, = ax.plot(epochs, val_acc, "s-", color="#d62728", lw=2, ms=5,
                  label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")

    # y-axis with headroom: bottom drops just below the lowest value,
    # top extends past 100 so the saturated curve has breathing room
    ymin = min(train_acc.min(), val_acc.min())
    ax.set_ylim(max(0.0, ymin - 8), 104)
    ax.grid(True, alpha=0.3)

    # val loss — dashed line, second y-axis (its scale differs from
    # accuracy, one shared axis would flatten it to nothing)
    ax2 = ax.twinx()
    l3, = ax2.plot(epochs, val_loss, "--", color="gray", lw=1.8, label="val loss")
    ax2.set_ylabel("val loss", color="gray")
    ax2.tick_params(axis="y", colors="gray")

    # best-epoch marker and annotation — anchored to the axis top so it
    # never collides with the curves, whatever the y-range is
    ytop = ax.get_ylim()[1]
    ax.axvline(epochs[best], color="gray", ls=":", lw=1)
    ax.plot(epochs[best], val_acc[best], "o", color="#d62728", ms=9,
            mfc="none", mew=1.5)
    ax.text(epochs[best] - 0.5, ytop - 1.0,
            f"best ep {epochs[best]} ({val_acc[best]:.2f}%)",
            ha="right", va="top", fontsize=8, color="dimgray")

    ax.set_title(f"{name} — training curves")
    ax.legend(handles=[l1, l2, l3], loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=ROOT / "data", help="dataset folder")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"          # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps"           # Apple Silicon
    else:
        device = "cpu"

    labels, label_to_idx = load_labels(args.data)
    intents, slots, label_map = load_intents_slots(args.data)
    intent_to_idx = {name: i for i, name in enumerate(intents)}
    slot_to_idx = {name: i for i, name in enumerate(slots)}
    combo = torch.tensor([[intent_to_idx[label_map[l][0]],
                           slot_to_idx[label_map[l][1]]] for l in labels],
                         dtype=torch.long, device=device)

    test_ds = VCMDataset(args.data, "test", label_to_idx)
    print(f"test: {len(test_ds)} utterances, intents: {len(intents)}, "
          f"slots: {len(slots)}")

    model = CRNN(n_intents=len(intents), n_slots=len(slots)).to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "crnn.pt",
                                     map_location=device))

    it, ip, st, sp, lt, lp = predict_all(model, test_ds, device, combo,
                                         args.batch_size)

    # metrics for all three views
    cm_i, prec_i, rec_i, f1_i, acc_i = compute_metrics(it, ip, len(intents))
    cm_s, prec_s, rec_s, f1_s, acc_s = compute_metrics(st, sp, len(slots))
    cm_l, prec_l, rec_l, f1_l, acc_l = compute_metrics(lt, lp, len(labels))

    # clean vs noisy accuracy (the condition is part of the filename)
    conds = np.array(["clean" if path.endswith("_clean.wav") else "noisy"
                      for path, *_ in test_ds.items])
    correct = lt == lp

    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # training history (history.csv, written by train.py) — connects
    # the training results with the test results below
    # ---------------------------------------------------------------
    hist_path = results / "history.csv"
    history = None
    if hist_path.exists():
        with open(hist_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            history = {
                "train_acc": [float(r["train_acc"]) for r in rows],
                "val_acc":   [float(r["val_acc"])   for r in rows],
                "val_loss":  [float(r["val_loss"])  for r in rows],
            }
            # total training time = sum of per-epoch elapsed_s
            # (column is absent in old history.csv files -> skip silently)
            if all(r.get("elapsed_s") for r in rows):
                history["elapsed_s"] = sum(float(r["elapsed_s"]) for r in rows)

    # ---- metrics.txt ----
    lines = [f"test utterances: {len(test_ds)}", ""]
    if history is not None:
        best_ep = int(np.argmax(history["val_acc"])) + 1
        best_va = max(history["val_acc"])
        lines.append(f"best epoch: {best_ep}   val acc: {best_va:.2f}%   (from history.csv)")
        if "elapsed_s" in history:
            total = history["elapsed_s"]
            h, m, s = int(total // 3600), int(total % 3600 // 60), total % 60
            lines.append(f"training time: {total:.0f} s "
                         f"({h} hr {m:02d} min {s:04.1f} s / "
                         f"{total / 3600:.2f} hr)")
        lines.append("")
    lines += [
        f"intent accuracy: {acc_i:.4f}   macro P/R/F1: "
        f"{prec_i.mean():.3f}/{rec_i.mean():.3f}/{f1_i.mean():.3f}",
        f"slot accuracy:   {acc_s:.4f}   macro P/R/F1: "
        f"{prec_s.mean():.3f}/{rec_s.mean():.3f}/{f1_s.mean():.3f}",
        f"label accuracy (intent + slot decoded jointly): {acc_l:.4f}   "
        f"macro F1: {f1_l.mean():.3f}   (same metric as val acc in training)",
        "",
    ]
    for cond in ("clean", "noisy"):
        mask = conds == cond
        if mask.any():
            lines.append(f"{cond} accuracy: {correct[mask].mean():.4f} "
                         f"({int(mask.sum())} utterances)")
    lines.append("")
    for i, name in enumerate(intents):
        lines.append(f"{name}: precision {prec_i[i]:.3f}  "
                     f"recall {rec_i[i]:.3f}  f1 {f1_i[i]:.3f}")
    lines.append("")
    for i, name in enumerate(slots):
        lines.append(f"{name}: precision {prec_s[i]:.3f}  "
                     f"recall {rec_s[i]:.3f}  f1 {f1_s[i]:.3f}")
    (results / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- charts ----
    grouped_bar_chart(intents, prec_i, rec_i, f1_i, acc_i,
                      "intent — per-class precision / recall / F1 (test split)",
                      results / "class_metrics_intent.png")
    grouped_bar_chart(slots, prec_s, rec_s, f1_s, acc_s,
                      "slot — per-class precision / recall / F1 (test split)",
                      results / "class_metrics_slot.png")
    confusion_chart(cm_i, intents, acc_i,
                    f"intent — confusion matrix (test split, acc {acc_i * 100:.2f}%; "
                    "cells = raw counts)",
                    results / "confusion_intent.png")
    confusion_chart(cm_s, slots, acc_s,
                    f"slot — confusion matrix (test split, acc {acc_s * 100:.2f}%; "
                    "cells = raw counts)",
                    results / "confusion_slot.png")

    # ---- training curves from history.csv ----
    if history is not None:
        training_curves_chart(history, "label", results / "accuracy_label.png")
    else:
        print(f"[skip] {hist_path.name} not found - no training-curves plot")

    print(f"intent acc {acc_i:.4f}  slot acc {acc_s:.4f}  label acc {acc_l:.4f}")
    print(f"results written to {results}/")


if __name__ == "__main__":
    main()
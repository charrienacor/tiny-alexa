"""Regenerate the result charts in the tiny-alexa visual theme.

Run from the repo root:   python src/theme_plots.py

Re-runs the same test-set inference as evaluate.py and rewrites, in the
black / hot-pink theme of tiny-alexa.html:

    results/accuracy_label.png     train/val accuracy + val loss curves
    results/confusion_intent.png   intent confusion matrix
    results/confusion_slot.png     slot confusion matrix

and dumps results/deck_data.json (history + confusion matrices + class
names) so presentation.html can animate the same data natively.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch.utils.data import DataLoader

from dataset import VCMDataset, load_labels
from features import load_intents_slots
from model import TCN
from evaluate import predict_all, compute_metrics

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ theme
BG      = "#050506"
PANEL   = "#0b0b0e"
WHITE   = "#f6f6f8"
MUTED   = "#8b8b93"
PINK_100 = "#ffe1ef"
PINK_200 = "#ffc2e0"
PINK_300 = "#ff9ecb"
PINK_400 = "#ff6fb5"
PINK_500 = "#ff3d9e"
PINK_600 = "#f0128a"
PINK_700 = "#c4006e"

CMAP = LinearSegmentedColormap.from_list(
    "tinyalexapink",
    [PANEL, "#2b0720", PINK_700, PINK_500, PINK_300],
    N=256,
)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": WHITE,
    "axes.edgecolor": (1, 1, 1, 0.14),
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": (1, 1, 1, 0.07),
    "font.family": "sans-serif",
})


def _style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", alpha=0.5)


def accuracy_chart(history, best_epoch, path):
    """Train/val accuracy (solid pink family) + val loss (dashed, 2nd axis)."""
    x = np.arange(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    ax.plot(x, history["train_acc"], color=PINK_300, lw=1.4, alpha=0.75,
            label="train acc")
    ax.plot(x, history["val_acc"], color=PINK_500, lw=2.2, label="val acc")
    ax.fill_between(x, history["val_acc"], alpha=0.10, color=PINK_500)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("accuracy %")
    ax.set_xlabel("epoch")
    ax.set_title("label accuracy — train vs validation",
                 color=WHITE, fontsize=13, loc="left", pad=14)

    # best-epoch marker
    bi = int(best_epoch) - 1
    bv = float(history["val_acc"][bi])
    ax.scatter([bi + 1], [bv], s=34, color=PINK_100, zorder=5,
               edgecolor=PINK_500, linewidth=1.5)
    ax.annotate(f"best  ep {bi + 1} · {bv:.2f}%",
                xy=(bi + 1, bv), xytext=(bi + 1 - 12, bv - 16),
                color=PINK_200, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=PINK_400, lw=0.8))

    ax2 = ax.twinx()
    ax2.plot(x, history["val_loss"], color=MUTED, lw=1.2, ls="--",
             alpha=0.8, label="val loss")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=MUTED)
    ax2.set_ylabel("val loss", color=MUTED)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", framealpha=0.35,
              facecolor=PANEL, edgecolor="none", fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_chart(cm, classes, acc, title, path):
    """Confusion matrix, % of true class, on the pink colormap."""
    n = len(classes)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, row_sum, out=np.zeros(cm.shape, dtype=float),
                    where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.65), max(9, n * 0.5)),
                           dpi=200)
    im = ax.imshow(pct, cmap=CMAP, vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=4 if n > 25 else 6,
                        color=BG if pct[i, j] > 55 else WHITE,
                        alpha=0.95)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=5 if n > 25 else 7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=5 if n > 25 else 7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}   ·   accuracy {acc * 100:.2f}%",
                 color=WHITE, fontsize=13, loc="left", pad=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_edgecolor("none")
    cb.set_label("% of true class", color=MUTED, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED, fontsize=7)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    data = ROOT / "data"
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    labels, label_to_idx = load_labels(data)
    intents, slots, label_map = load_intents_slots(data)
    intent_to_idx = {n: i for i, n in enumerate(intents)}
    slot_to_idx = {n: i for i, n in enumerate(slots)}
    combo = torch.tensor(
        [[intent_to_idx[label_map[l][0]], slot_to_idx[label_map[l][1]]]
         for l in labels], dtype=torch.long, device=device)

    test_ds = VCMDataset(data, "test", label_to_idx)
    print(f"test: {len(test_ds)} utterances, intents: {len(intents)}, "
          f"slots: {len(slots)}")

    model = TCN(n_intents=len(intents), n_slots=len(slots)).to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "TCN.pt",
                                     map_location=device))
    it, ip, st, sp, lt, lp = predict_all(model, test_ds, device, combo)

    cm_i, _, _, _, acc_i = compute_metrics(it, ip, len(intents))
    cm_s, _, _, _, acc_s = compute_metrics(st, sp, len(slots))
    print(f"intent acc {acc_i * 100:.2f}%   slot acc {acc_s * 100:.2f}%")

    # training history
    rows = list(csv.DictReader(open(ROOT / "results" / "history.csv",
                                    newline="", encoding="utf-8")))
    history = {
        "train_acc": [float(r["train_acc"]) for r in rows],
        "val_acc":   [float(r["val_acc"])   for r in rows],
        "val_loss":  [float(r["val_loss"])  for r in rows],
    }
    best_epoch = int(max(rows, key=lambda r: float(r["val_acc"]))["epoch"])

    results = ROOT / "results"
    accuracy_chart(history, best_epoch, results / "accuracy_label.png")
    confusion_chart(cm_i, intents, acc_i, "intent confusion",
                    results / "confusion_intent.png")
    confusion_chart(cm_s, slots, acc_s, "slot confusion",
                    results / "confusion_slot.png")

    # native data for presentation.html
    deck = {
        "history": history,
        "best_epoch": best_epoch,
        "intents": intents,
        "slots": slots,
        "cm_intent": cm_i.tolist(),
        "cm_slot": cm_s.tolist(),
        "acc_intent": round(acc_i * 100, 2),
        "acc_slot": round(acc_s * 100, 2),
    }
    (results / "deck_data.json").write_text(json.dumps(deck))
    print("wrote accuracy_label.png, confusion_intent.png, "
          "confusion_slot.png, deck_data.json (themed)")


if __name__ == "__main__":
    main()

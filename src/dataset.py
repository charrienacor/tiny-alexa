"""Datasets built from manifest.csv (two-head targets: intent + slot)."""

import csv
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from features import load_wav, log_mel, trim_silence, load_intents_slots


def load_labels(data_dir):
    """labels.json -> (ordered label list, label -> class index)."""
    with open(Path(data_dir) / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    return labels, {label: i for i, label in enumerate(labels)}


def read_manifest(data_dir, split):
    """Return the manifest rows for one split: 'train', 'valid' or 'test'."""
    wanted = {"train": {"train"}, "valid": {"val"}, "test": {"test"}}[split]
    rows = []
    with open(Path(data_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"].strip().lower() in wanted:
                rows.append(row)
    return rows


class VCMDataset(Dataset):
    """Loads the WAV files of one split.

    Each item is (log-mel tensor, intent index, slot index, label index).
    The label index is the 31-way class (intent + slot combined) and is
    used to measure end-to-end command accuracy after joint decoding.
    """

    def __init__(self, data_dir, split, label_to_idx, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        intents, slots, label_map = load_intents_slots(self.data_dir)
        intent_to_idx = {name: i for i, name in enumerate(intents)}
        slot_to_idx = {name: i for i, name in enumerate(slots)}
        self.items = []
        for row in read_manifest(self.data_dir, split):
            if row["label"] in label_to_idx:
                intent, slot_value = label_map[row["label"]]
                self.items.append((row["path"],
                                   label_to_idx[row["label"]],
                                   intent_to_idx[intent],
                                   slot_to_idx[slot_value]))
        if not self.items:
            raise ValueError(
                f"no '{split}' rows found in manifest.csv "
                f"(train/valid/test splits are read from its 'split' column)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, intent, slot = self.items[i]
        wav = load_wav(self.data_dir / path)
        if self.augment:
            wav = wav * random.uniform(0.5, 1.2)   # volume robustness
        mel = log_mel(trim_silence(wav))           # same front end as the Pi
        return torch.from_numpy(mel), intent, slot, label
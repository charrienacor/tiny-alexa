"""Datasets built from manifest.csv (two-head targets: intent + slot)."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features import SAMPLE_RATE, load_wav, log_mel, trim_silence, load_intents_slots

# Filipino-English speakers (SilencioPH), see data/README.md; everyone else
# is a LibriSpeech ("foreign") speaker. Used for per-group accuracy and for
# oversampling the smaller group during training.
FILIPINO_SPEAKERS = {f"s{i}" for i in [*range(68, 81), 89, 90, 100]}
GROUPS = ("foreign", "filipino")


def speaker_group(speaker):
    return "filipino" if speaker in FILIPINO_SPEAKERS else "foreign"


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


# ── training-time augmentation ───────────────────────────────────────────────

def speed_perturb(wav, lo=0.9, hi=1.1):
    """Resample by a random factor: changes speaking rate and pitch together."""
    factor = random.uniform(lo, hi)
    n_out = int(len(wav) / factor)
    return np.interp(np.arange(n_out) * factor, np.arange(len(wav)),
                     wav).astype(np.float32)


def time_shift(wav, max_s=0.25):
    """Prepend random silence: the Pi's trim is not sample-exact."""
    lead = random.randint(0, int(max_s * SAMPLE_RATE))
    return np.pad(wav, (lead, 0))


def add_noise(wav, rng, snr_lo=5.0, snr_hi=30.0):
    """Mix in white or pink noise at a random SNR (dB)."""
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    if random.random() < 0.5:                       # pink: 1/f power
        spec = np.fft.rfft(noise)
        spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
        noise = np.fft.irfft(spec, n=len(wav)).astype(np.float32)
    snr = random.uniform(snr_lo, snr_hi)
    sig_pow = np.mean(wav ** 2) + 1e-10
    noise *= np.sqrt(sig_pow / (np.mean(noise ** 2) + 1e-10) / 10 ** (snr / 10))
    return wav + noise


def spec_augment(mel, n_freq=2, max_f=8, n_time=2, max_t=20):
    """SpecAugment: zero out random mel bands and time spans of (1, F, T)."""
    _, n_mels, n_frames = mel.shape
    for _ in range(n_freq):
        w = random.randint(0, max_f)
        f0 = random.randint(0, n_mels - w)
        mel[:, f0:f0 + w, :] = 0.0
    for _ in range(n_time):
        w = random.randint(0, max_t)
        t0 = random.randint(0, n_frames - w)
        mel[:, :, t0:t0 + w] = 0.0
    return mel


class VCMDataset(Dataset):
    """Loads the WAV files of one split.

    Each item is (log-mel tensor, intent index, slot index, label index).
    The label index is the 31-way class (intent + slot combined) and is
    used to measure end-to-end command accuracy after joint decoding.
    self.groups holds each item's speaker group ("foreign"/"filipino").
    """

    def __init__(self, data_dir, split, label_to_idx, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        intents, slots, label_map = load_intents_slots(self.data_dir)
        intent_to_idx = {name: i for i, name in enumerate(intents)}
        slot_to_idx = {name: i for i, name in enumerate(slots)}
        self.items = []
        self.groups = []
        for row in read_manifest(self.data_dir, split):
            if row["label"] in label_to_idx:
                intent, slot_value = label_map[row["label"]]
                self.items.append((row["path"],
                                   label_to_idx[row["label"]],
                                   intent_to_idx[intent],
                                   slot_to_idx[slot_value]))
                self.groups.append(speaker_group(row["speaker"]))
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
            # python `random` is reseeded per DataLoader worker by torch;
            # derive the numpy generator from it so workers differ
            rng = np.random.default_rng(random.getrandbits(32))
            wav = speed_perturb(wav)
            # trim first, then shift and add noise over the lead-in, like a
            # noisy Pi recording whose trim stops short of the speech onset
            wav = time_shift(trim_silence(wav))
            if random.random() < 0.5:
                wav = add_noise(wav, rng)
            mel = spec_augment(log_mel(wav))
        else:
            mel = log_mel(trim_silence(wav))       # same front end as the Pi
        return torch.from_numpy(mel), intent, slot, label

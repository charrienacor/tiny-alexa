"""Audio front end: WAV -> log-mel spectrogram.

NumPy only, so training and Pi inference produce identical features.
Every waveform goes through the same steps: trim silence -> remove DC
offset -> pad with true silence to a fixed length -> log-mel.
"""

import numpy as np
import soundfile as sf

import csv
import json
from pathlib import Path


SAMPLE_RATE = 16000   # the dataset is 16 kHz mono
N_FFT = 400           # 25 ms analysis window
HOP = 160             # 10 ms hop -> 100 frames per second
N_MELS = 64           # mel bands (front-end width for training and Pi inference)
MAX_FRAMES = 480      # ~4.8 s; every utterance is padded/trimmed to this
LOG_FLOOR = 1e-6      # added to mel power before log

_WINDOW = np.hanning(N_FFT)
_MEL_FB = None        # mel filterbank matrix, built on first use


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank():
    """Triangular mel filters as a (N_MELS, N_FFT//2+1) matrix."""
    global _MEL_FB
    if _MEL_FB is None:
        n_freqs = N_FFT // 2 + 1
        mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2.0), N_MELS + 2)
        bins = np.floor(_mel_to_hz(mel_points) / SAMPLE_RATE * N_FFT).astype(int)
        fb = np.zeros((N_MELS, n_freqs))
        for m in range(1, N_MELS + 1):
            left, center, right = bins[m - 1], bins[m], bins[m + 1]
            if center > left:   # rising edge of the triangle
                fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:  # falling edge of the triangle
                fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
        _MEL_FB = fb
    return _MEL_FB


def load_wav(path):
    """Load a WAV file as mono float32 in [-1, 1] at 16 kHz.

    Files at other sample rates (e.g. 12 kHz) are resampled here. scipy
    is imported lazily: Pi audio is already 16 kHz, so it is never needed
    there.
    """
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav


def trim_silence(wav, threshold_db=-40.0, pad_ms=100):
    """Cut leading/trailing silence from a recording.

    The threshold is relative to the clip's own peak, so trimming behaves
    the same on quiet and loud recordings. Applied everywhere so training
    and inference see the same thing.
    """
    wav = np.asarray(wav, dtype=np.float32)
    frame = int(SAMPLE_RATE * 0.025)               # 25 ms analysis frames
    n = len(wav) // frame
    if n == 0:
        return wav
    rms = np.sqrt((wav[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
    thresh = (10.0 ** (threshold_db / 20.0)) * rms.max()
    speech = np.where(rms > thresh)[0]
    if len(speech) == 0:                           # basically a silent file
        return wav
    pad = int(SAMPLE_RATE * pad_ms / 1000)
    start = max(0, speech[0] * frame - pad)
    end = min(len(wav), (speech[-1] + 1) * frame + pad)
    return wav[start:end]


def log_mel(wav):
    """Waveform -> log-mel spectrogram of shape (1, N_MELS, MAX_FRAMES).

    Remove DC offset -> pad the waveform with true silence to a fixed
    length -> Hann window -> FFT -> power spectrum -> mel filterbank ->
    log -> per-utterance normalization. Padding the waveform (not the
    spectrogram) means padded frames are real silence, so live recordings
    match training data.

    Normalization uses only the frames that cover real audio: each mel
    band has its mean subtracted (cancels gain and the microphone's
    frequency response) and the whole spectrogram is divided by one
    std. Padded frames are then set to each band's quietest real value,
    so padding looks like "this speaker's silence" regardless of level.

    Because padded frames are overwritten anyway, only the frames that
    overlap real audio go through the FFT (~3x faster for a typical
    1-2 s command; the output is bit-identical to transforming all
    MAX_FRAMES frames).
    """
    arr = np.asarray(wav)
    if np.issubdtype(arr.dtype, np.integer):        # int16 mic data -> floats
        wav = arr.astype(np.float32) / 32768.0
    else:
        wav = arr.astype(np.float32)
    wav = wav - wav.mean()                          # remove DC offset
    # frames that overlap real audio (the rest are padding); recordings
    # longer than MAX_FRAMES are trimmed to the first MAX_FRAMES frames
    n_real = min(MAX_FRAMES, max(1, -(-len(wav) // HOP)))

    # pad with true silence so the last real frame is complete
    need_len = N_FFT + HOP * (n_real - 1)
    if len(wav) < need_len:
        wav = np.pad(wav, (0, need_len - len(wav)))

    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_real)[:, None]
    frames = wav[idx] * _WINDOW[None, :]                       # (n_real, N_FFT)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).T       # (N_FFT//2+1, n_real)
    mel = _mel_filterbank() @ power                            # (N_MELS, n_real)
    real = np.log(mel + LOG_FLOOR)

    logmel = np.empty((N_MELS, MAX_FRAMES))
    logmel[:, :n_real] = (real - real.mean(axis=1, keepdims=True)) / (real.std() + 1e-5)
    logmel[:, n_real:] = logmel[:, :n_real].min(axis=1, keepdims=True)

    return logmel[np.newaxis, ...].astype(np.float32)          # (1, N_MELS, MAX_FRAMES)

def load_intents_slots(data_dir):
    """Read manifest.csv and derive the two-head class mappings.

    Returns (intents, slots, label_map): 19 intent names, 18 slot values
    + "NO_SLOT" (last), and label -> (intent, slot value or "NO_SLOT").
    """
    data_dir = Path(data_dir)
    with open(data_dir / "labels.json", encoding="utf-8") as f:
        labels = set(json.load(f))
    label_map = {}
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row["label"]
            if lab in labels and lab not in label_map:
                value = (row.get("slot_value") or "").strip()
                if value in ("", "-", "none"):
                    value = "NO_SLOT"
                label_map[lab] = (row["intent"], value)
    intents = sorted({i for i, _ in label_map.values()})
    slots = sorted({s for _, s in label_map.values() if s != "NO_SLOT"}) + ["NO_SLOT"]
    return intents, slots, label_map
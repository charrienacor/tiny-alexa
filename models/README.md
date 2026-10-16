# Model weights

| File | Format | Size | Used by |
|---|---|---|---|
| `TCN.onnx` | ONNX | ~0.92 MB | The Pi 4 (and `./run.sh`) — 1-thread ONNX Runtime, ~6.8 ms/command |
| `TCN.pt` | PyTorch | ~0.95 MB | Re-evaluation, fine-tuning, or re-export |

**Architecture:** two-head dilated TCN, 0.23 M parameters — one head for the
19 intents, one for the slot values. Trained from scratch on this project's
command dataset; no pretrained checkpoints were used.

## Licence

These weights are released under the **MIT licence** — the `LICENSE` file at
the repository root covers them along with the rest of the code.

## Provenance

- **Trained on:** 17,851 synthetic utterances (100 synthetic voices, 19
  intents) from the companion dataset,
  [`github.com/markandrian30/AI231/MEX2/Data`](https://github.com/markandrian30/AI231/tree/main/MEX2/Data).
- **Reproduce:** `python src/train.py` (CUDA, Apple MPS or CPU) → `models/TCN.pt`;
  `python src/export_onnx.py` → `models/TCN.onnx`. Seed 0.
- **Data caveat:** the training corpus is *derived* audio — it was synthesised
  from source recordings that are themselves under restrictive licences
  (e.g. non-commercial). The weights and code here are MIT, but the underlying
  data is **not** redistributed with this repo and is subject to its own terms
  — see the dataset repository for access conditions before commercial use.

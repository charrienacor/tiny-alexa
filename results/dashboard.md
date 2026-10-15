# Project dashboard — auto-filled

_generated from: /Users/charrieannenacor/Desktop/tiny-alexa/results/gpu_report.json (present), /Users/charrieannenacor/Desktop/tiny-alexa/results/pi_report.json (present)_

| **Model** | |
|---|---|
| Front end | Mic · 16 kHz → log-mel 64 × 480 (25 ms window, 10 ms hop, 4.8 s) |
| Encoder | TCN × 5 dilated blocks (2 convs each, k=5, dilations 1, 2, 4, 8, 16) · 64-dim; non-causal, whole-utterance (no KV-cache / streaming) |
| Heads | intent head (19) + slot head (19, incl. NO_SLOT), joint decode over 31 valid pairs |
| Actuator | src/server.py → tiny-alexa.html (lights, music, timer, alarm, …) |
| Target | Raspberry Pi 4 · onnxruntime (CPU) · openWakeWord "Alexa" wake word |
| Parameters / weights | 0.23 M (229,958) · 0.92 MB TCN.onnx |

| **Dataset** | |
|---|---|
| Source | LibriSpeech + SilencioPH (Filipino-English) speakers, transcribed with simple-audio-transcriber; clean + noisy takes |
| Hours / utts | 8.64 h / 17,851 (14,256 / 1,806 / 1,789 train/val/test; 8,927 clean + 8,924 noisy) |
| Speakers | 100 (84 foreign + 16 Filipino-English) · 80 / 10 / 10 train/val/test |
| Labels | 19 intents · 18 slot values (31 command labels) |

| **Training on A100** | |
|---|---|
| Cluster (# A100) | Mac.lan — no NVIDIA GPU found (trained on CPU/MPS); re-run on the A100 to fill this |
| Objective | cross-entropy (intent head) + cross-entropy (slot head), label smoothing 0.1; joint decode over the 31 valid (intent, slot) combinations |
| Optimizer | AdamW (lr=0.001, weight_decay=0.01) + 2-epoch linear warmup + per-step cosine decay to 1e-5 over 40 epochs |
| Steps / loss | 8697 steps (39 epoch(s) x 223, 557 s (0 hr 09 min 16.9 s / 0.15 hr)) / 1.207 |

| **Validation on RPi 4** | |
|---|---|
| Hardware | ai231 — 0 — aarch64 |
| Wake word | alexa_v0.1 (openwakeword 0.4.0), threshold 0.5, patience 2 chunks |
| Keyword / intent accuracy | 100.00% (202/202 clips) / 86.14% (174/202 clips) |
| False-accept rate | 0.00% (0/202 clips) |
| Latency p95 / RTF | p95 6.89 ms / RTF 696.7 (p50 6.78 ms, n=200) |
| Runtime | onnxruntime 1.31.0 — 1 thread(s) (TCN.onnx, 898 KB) |

# tiny-alexa

An offline voice assistant for the Raspberry Pi 4. It runs the wake word
("Alexa", openWakeWord 0.4.0), then a small two-head TCN that classifies
the spoken command into an intent plus a slot value (e.g. `ALARM` + `6 AM`).
There is no ASR, no LLM and no cloud.

```
mic 16 kHz → openWakeWord "Alexa" → record ≤ 4.8 s → log-mel 64×480
          → TCN.onnx → intent head (19) + slot head (19)
          → joint decode over the 31 valid (intent, slot) commands
```

- **Model:** 0.23 M parameters (0.92 MB fp32 ONNX), 5 dilated conv blocks, receptive field 1005 frames.
- **Data:** 17,851 utterances from 100 speakers (84 LibriSpeech + 16 Filipino-English). See [data/README.md](data/README.md).
- **Test results** ([results/metrics.txt](results/metrics.txt)): intent 99.66 %, slot 100 %, command (label) 99.66 %.

## Try it now (no mic, no server, no Pi)

```
open tiny-alexa.html?demo
```

Plays a scripted conversation — wake → listening → command → animation — cycling
through every scene. Add `&debug` to show the model latency chip.

## Layout

| Path | What |
|---|---|
| `src/features.py` | NumPy front end (trim → log-mel), shared by training and the Pi |
| `src/model.py` | Two-head TCN |
| `src/dataset.py` | Manifest-driven dataset + augmentation |
| `src/train.py` | Training → `models/TCN.pt`, `results/history.csv`, `results/train_config.json` |
| `src/evaluate.py` | Test-split metrics and charts → `results/` |
| `src/export_onnx.py` | `TCN.pt` → `models/TCN.onnx` (optionally `TCN_int8.onnx`) |
| `src/bench_gpu.py` | Training report → `results/gpu_report.json` |
| `src/infer.py` | Terminal pipeline on the Pi (microphone or `--file`) |
| `src/server.py` + `tiny-alexa.html` | Browser UI: the browser streams mic audio to the Pi over a WebSocket |
| `src/bench_pi.py` | Pi report: wake-word TAR/FAR, latency, ONNX accuracy → `results/pi_report.json` |
| `src/dashboard.py` | Merges both reports → `results/dashboard.md` |

## Training machine (Python 3.13)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/train.py          # CUDA, Apple MPS or CPU
python src/evaluate.py
python src/export_onnx.py    # add --int8 for a quantized copy
python src/bench_gpu.py
```

## Raspberry Pi 4

Copy `src/`, `data/labels.json`, `data/manifest.csv`, `models/TCN.onnx` and
`tiny-alexa.html` to the Pi. The Pi doesn't need torch. The dataset audio
(~950 MB) is not in the repo — only the manifests and class tables are.

**Browser UI**

```bash
pip install numpy soundfile onnxruntime openwakeword==0.4.0 "aiohttp>=3.9"
python3 src/server.py        # then open http://<pi-ip>:8321
```

Browsers only allow the microphone on `https://` or `http://localhost`.
The weather chip uses the browser's GPS location when allowed, named via the
server's `/geo` endpoint (OpenStreetMap Nominatim). Otherwise it uses the
city from the public IP address.
Optional: put music files in `data/music/` for the music player
(`pip install mutagen` for exact track durations).

**Terminal**

```bash
sudo apt install libportaudio2
pip install numpy soundfile onnxruntime sounddevice openwakeword==0.4.0
python src/infer.py                 # live microphone
python src/infer.py --file a.wav    # offline test
```

**Benchmark + dashboard**

```bash
python src/bench_pi.py              # uses data/wake_tests/ by default
python src/bench_pi.py --accuracy 200
python src/dashboard.py             # on a machine that has both reports
```

`--tp-dir` / `--far-dir` override the default `data/wake_tests/positives` and
`data/wake_tests/negatives` folders (16 kHz mono WAV — record your own to improve
the FAR; a few positives saying "Alexa" and a dozen negatives of everyday
speech, TV, music, or silence). `--accuracy` reads the dataset WAVs and needs
`scipy`, because about a third of them are 12 kHz.

## Supported commands

Lights on/off, brightness (20/60/100 %), color (red/blue/green), temperature
(18/22/26 °C), timer (10 s/30 s/1 min), alarm (6 AM/8 AM/9 PM), reminders
(drink water/study/exercise, list), music (play/pause/stop/next, volume up/down),
time, weather, call and message.

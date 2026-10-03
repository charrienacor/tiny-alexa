# tiny alexa

A real-time, **fully on-device voice assistant** for the **Raspberry Pi 4 (4 GB)** — and any
browser on your LAN. Say **"Hey Alexa"**, then a command. The app listens, understands, and
answers with an animated **pink-on-black** UI.

- **No cloud, no ASR, no API keys.** Wake word (openWakeWord) and command classification
  (a custom 4 MB CRNN) run on the Pi. The *only* network call the whole system makes is the
  weather — fetched **directly by the browser** from the free, keyless open-meteo API, never
  by the Pi.
- **19 intents, 31 command labels** — music, lights, brightness, color, temperature, timer,
  alarm, call, message, reminders, weather, time, volume.
- **Every command has a hand-built animation** — a counting timer ring, a thermostat, a
  pendant lamp that flickers on, a ringing twin-bell alarm clock, a phone screen that dials
  and connects, a chat bubble that types and sends, natural objects for colors. All pure
  CSS + Web Audio in one self-contained HTML file: no CDN, no frameworks, no external assets.
- **Trained on 17,851 utterances from 100 speakers** (clean + noisy) → **99.55 % test
  accuracy**, 99.94 % slot accuracy.
- Command → response in **< 50 ms** on an M-series Mac; expected **well under 200 ms** on
  the Pi 4.

![tiny alexa — Hey Alexa wake word](screenshots/TO_WAKE_ALEXA.jpg)

*The full UI in action — every scene below is captured from the live app. Full gallery:
[screenshots](#screenshots).*

> **Status:** developed and verified on macOS (M-series). Deployment to the Raspberry Pi 4
> is fully prepared (see [`rpi/`](rpi/)) but **hardware testing is still pending** — see
> [Status & pending verification](#status--pending-verification).

---

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [The 31 commands](#the-31-commands)
4. [The UI](#the-ui)
5. [The animated scenes](#the-animated-scenes)
6. [Screenshots](#screenshots)
7. [The model (CRNN)](#the-model-crnn)
8. [The dataset](#the-dataset)
9. [Training & results](#training--results)
10. [Latency](#latency)
11. [Deploying to the Raspberry Pi 4](#deploying-to-the-raspberry-pi-4)
12. [Development on Mac / PC](#development-on-mac--pc)
13. [Project layout](#project-layout)
14. [Tools & utilities](#tools--utilities)
15. [Weather & music](#weather--music)
16. [Engineering notes](#engineering-notes)
17. [Status & pending verification](#status--pending-verification)

---

## Quick start

**Try it in 10 seconds, no mic, no server, no Pi:**

```
open rpi/tiny-alexa.html?demo
```

The page plays a scripted conversation — wake → listening → confirm → animation → idle —
cycling through every scene. Add `&debug` to show the CRNN latency chip.

**Full local run (Mac/PC):**

```bash
cd rpi
python3 -m pip install aiohttp onnxruntime openwakeword resampy soundfile mutagen
python3 server.py                      # → http://localhost:8321
```

Open `http://localhost:8321`, grant the microphone permission, tap **start**, and say
**"Hey Alexa, …"**.

**On the Pi:** see [Deploying to the Raspberry Pi 4](#deploying-to-the-raspberry-pi-4).

---

## Architecture

```
browser (rpi/tiny-alexa.html)                    Raspberry Pi 4
┌───────────────────────────────────────────┐    WebSocket    ┌──────────────────────────────────┐
│ mic → AudioWorklet → 16 kHz int16 PCM ────┼───────────────▶│ openWakeWord ("hey alexa")       │
│ AnalyserNode → live waveform              │                │ adaptive-noise endpointing       │
│ state machine + all animations            │◀───────────────│ CRNN (crnn.onnx) → 31 labels     │
│ Web Audio: every tone, synthesized        │   JSON events  │ ID3 scan of music/               │
└───────────────────────────────────────────┘                └──────────────────────────────────┘
        │  (weather only)
        └────────────────────────────▶ open-meteo.com / ipinfo.io   (free, no key, CORS)
```

Division of labor — the Pi does **only** DSP + inference; the browser does **everything visible**:

| Where | What |
|---|---|
| **Browser** | Mic capture (AudioWorklet, echo cancellation on), resample to 16 kHz, stream 80 ms int16 chunks; live waveform (`AnalyserNode`); the whole state machine; every animation and every sound effect (Web Audio synthesis — there are **zero audio files** in the UI); weather fetch; reminders persistence (`localStorage`). |
| **Pi / server.py** | Wake-word detection on each chunk; adaptive endpointing (measures the room-noise floor, stops at 0.8 s silence or 4 s cap); silence-trim → log-mel → CRNN inference; joint intent+slot decoding; confidence gating; ID3 tag + album-art scan of `music/`; serves the page, `/music`, `/health`, `/ws`. |

Server endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` · `GET /tiny-alexa.html` | The UI (single self-contained file) |
| `GET /health` | Liveness + loaded models |
| `GET /music` | Playlist JSON (title/artist/album/duration/art from ID3) |
| `GET /music/{name}` | The MP3 + extracted cover art |
| `WS /ws` | Binary in: 16 kHz int16 PCM · JSON out: `ready` / `awake` / `result` / `rejected` / `idle` / `error` |

---

## The 31 commands

19 intents; 6 of them carry a **slot** (value), giving 31 valid (intent, slot) labels.
The model never outputs an impossible pair — see [joint decoding](#the-model-crnn).

| Intent | Example things to say | Slot values | What happens |
|---|---|---|---|
| `PLAY_MUSIC` | "Play music", "Start music" | — | Local MP3 playlist starts; cover art + title on screen; wave reacts to the music |
| `PAUSE` | "Pause", "Pause song" | — | Playback pauses (toggle) |
| `STOP` | "Stop", "End playback" | — | Stops whatever is active — walks the full [STOP chain](#the-stop-chain) |
| `NEXT` | "Next song", "Skip song" | — | Advances the playlist (auto-advances on track end too) |
| `VOLUME_UP` / `VOLUME_DOWN` | "Volume up", "Lower the volume" | — | ±15 % with a toast |
| `LIGHT_ON` / `LIGHT_OFF` | "Lights on", "Kill the lights" | — | Confirm + lamp scene |
| `BRIGHTNESS` | "Brightness 20 percent", "Adjust brightness to 60" | `20 percent` · `60 percent` · `100 percent` | **Lamp scene** — the light tweens to exactly that level |
| `COLOR` | "Set color to red", "Change color to blue" | `red` · `green` · `blue` | **Natural-object scene** — apple / sprout / raindrop |
| `TEMPERATURE` | "Temperature 18 degrees", "Set the temperature to 26" | `18` · `22` · `26` degrees | **Thermostat scene** — cool / eco / heat |
| `TIMER` | "Timer 1 minute", "Countdown for 30 seconds" | `10 seconds` · `30 seconds` · `1 minute` | **Timer ring** — counts down live, chimes at zero |
| `ALARM` | "Alarm 6 AM", "Wake me up at 9 PM" | `6 AM` · `8 AM` · `9 PM` | **Twin-bell alarm clock** — hands sweep to the time, it rings, snoozes |
| `CALL` | "Call", "Make a phone call" | — | **Phone screen** — dials (real ring tone), connects, live call timer |
| `MESSAGE` | "Message", "Send a message" | — | **Chat bubble** — typing dots → typewriter text → paper plane + ✓✓ |
| `CREATE_REMINDER` | "Remind me to drink water", "Reminder study" | `drink water` · `exercise` · `study` | Persisted to `localStorage`, bell chip swings, listed later |
| `LIST_REMINDERS` | "Show my reminders", "Reminders" | — | Slide-up panel with the saved list |
| `WEATHER` | "What's the weather?" | — | Live open-meteo answer rendered in the UI (browser-side fetch) |
| `TIME` | "What time is it?" | — | Clock readout |

Anything with confidence below **0.50** is rejected with
*"Sorry, that's beyond what I can do. Try 'set an alarm for 6 AM' or 'turn on the lights'."*

---

## The UI

Single file, `rpi/tiny-alexa.html` (~150 KB, no dependencies): black background, pink
(`#ff3d9e`) and white. Four states, driven by server events:

```
idle ──"hey alexa"──▶ awake ──you speak──▶ listening ──result──▶ result ──hold──▶ idle
 ▲                                                             │
 └──────────────────────────── auto-return ◀───────────────────┘
```

- **idle** — a waveform that "breathes" (slow sine amplitude), the wordmark, and a hint line.
- **awake** — a randomized acknowledgement (*"Uh-huh" / "Yes" / "Yeah?"*).
- **listening** — the waveform locks onto your live mic (and later, the music).
- **result** — a randomized confirm (*"Got it" / "Okay" / "Sure" / "On it"…*) plus the
  resolved command line, an intent·slot tag, and — for the rich intents — the full scene.
  Every scene **owns its hold time** and fades back to idle on its own.

Always-on chrome: a **LIVE/OFF** indicator, a **latency chip** (with `?debug`), a
**weather pill** (top-right: icon + temp + place; tap to refresh, hover for details), a
**reminder bell chip**, and a **music card** (cover, seek bar, prev/play/next, volume).

Query params: `?demo` (scripted tour, no mic/server) · `?debug` (latency chip + console).

---

## The animated scenes

Seven hand-built scenes. Each is pure CSS + a small JS module inside the HTML file; every
sound is **synthesized live with the Web Audio API** (ring tones, chimes, key clicks, the
twin-bell strike) — the UI ships **zero audio assets**.

| Scene | Trigger | What you see |
|---|---|---|
| **Timer** | `TIMER` | A progress ring counts down in real time (ticks at 1 s resolution), goes **urgent** in the last seconds, chimes and flashes "time's up" at zero, then fades. |
| **Aircon / thermostat** | `TEMPERATURE` | A round thermostat: the dial sweeps to the target, the mode flips **cool / eco / heat** (18° → cool, 22° → eco, 26° → heat), fan blades spin, the glow takes on the mode's hue. |
| **Color** | `COLOR` | A **natural object** for each color — 🍎 a glossy bobbing **apple** (red), 🌱 a **sprout growing from soil** with unfurling leaves (green), 💧 a **raindrop** over expanding ripples (blue). While it's up, the whole page's ambient glow pulses in that color. |
| **Lamp / brightness** | `BRIGHTNESS`, `LIGHT_ON/OFF` | A **pendant lamp**: ignition flicker like a filament catching, the readout **tweens** from the previous level (0→20→60, easeOutCubic), halo + bulb + floor light-pool scale with the level (**gamma-corrected**, so 20 % genuinely looks dim), dust motes drift through the beam, mode label *off/dim/bright/full*. |
| **Call** | `CALL` | A phone screen: avatar with **sonar rings**, randomized contact, blinking `calling…`, a **real US ring tone** (350 + 440 Hz, 2 s on / 4 s off). Then a pickup chirp, `connected`, a **live call timer**, and a dancing 7-bar equalizer. Hang up via the red button, "stop", or auto at 20 s — with a descending hang-up tone. |
| **Alarm** | `ALARM` | A **twin-bell clock**: hands **sweep from 12:00 to the set time**, then the clock shakes, bells rock, the hammer strikes, sound waves arc off, and a **metallic twin-bell tone** (three detuned G6 partials every 700 ms) rings. **Snooze** button: it dozes (`z z z`) and rings again after ~2.6 s. Stopping settles the shake with a final clink. |
| **Message** | `MESSAGE` | A chat bubble springs in: bouncing **typing dots** → **typewriter text** (~40 ms/char, blinking caret, human cadence) → a **paper plane flies away** + **double ✓✓** pop, with a rising send swoop. Closes via ✕, "stop", or auto. |

### The scene contract

Every scene follows the same contract, which is what keeps seven concurrent-capable scenes
from ever fighting over the screen:

- `startX()` — shows the scene, sets `body.<x>-active`, marks the UI busy.
- `tickX()` — runs **inside the single existing `requestAnimationFrame` loop** (there is
  never a second animation loop; nothing extra lands on the Pi's CPU or the GPU driver).
- **One-shot fade guard** — a `finishing` flag (or a flip to a terminal phase) guarantees
  the fade-out timer is armed exactly once and allowed to run.
- `finishX(holdMs)` — clears everything, restores `state = "idle"` and the hint line.
- `dismissX()` — silent, instant teardown. Called from **every other intent's branch** so a
  new command always supersedes the current scene cleanly.

### The STOP chain

"Stop" resolves the most recent thing that's alive, in this order:

```
music → timer → aircon → color → lamp → call (hangUp) → alarm (alarmOff) → message (closeOut) → generic
```

---

## Screenshots

Every scene below was captured from the **live app** (macOS, full-resolution browser,
`?debug` off) — nothing mocked. All 19 originals live in [`screenshots/`](screenshots/).

### The flow

| "Hey Alexa" — wake | Listening | Music playing |
|---|---|---|
| ![wake](screenshots/TO_WAKE_ALEXA.jpg) | *(waveform locked to the mic)* | ![play music](screenshots/PLAY_MUSIC.jpg) |

| Timer 30 s | Alarm 8:00 AM | Stop |
|---|---|---|
| ![timer](screenshots/TIMER_30s.jpg) | ![alarm](screenshots/ALARM_8_00_AM.jpg) | ![stop](screenshots/STOP.jpg) |

### Lights & climate

| Light on | Brightness 100 | Color — green | Temperature 22° |
|---|---|---|---|
| ![light on](screenshots/LIGHT_ON.jpg) | ![brightness](screenshots/BRIGHTNESS_100.jpg) | ![color green](screenshots/COLOR_GREEN.jpg) | ![temperature](screenshots/TEMPERATURE_22.jpg) |

### Calls, messages, info

| Call | Message | Weather | Time |
|---|---|---|---|
| ![call](screenshots/CALL.jpg) | ![message](screenshots/MESSAGE.jpg) | ![weather](screenshots/WEATHER.jpg) | ![time](screenshots/TIME.jpg) |

### Volume & reminders

| Volume down | Volume up | Create reminder | List reminders |
|---|---|---|---|
| ![volume down](screenshots/VOLUME_DOWN.jpg) | ![volume up](screenshots/VOLUME_UP.jpg) | ![create reminder](screenshots/CREATE_REMINDER_EXERCISE.jpg) | ![list reminders](screenshots/LIST_REMINDERS.jpg) |

Two extra takes (same scenes, different moments):
[`MESSAGE_ALT`](screenshots/MESSAGE_ALT.jpg) — the bubble mid-typewriter ·
[`VOLUME_DOWN_ALT`](screenshots/VOLUME_DOWN_ALT.jpg) — volume at a lower level.

---

## The model (CRNN)

A compact two-head CRNN classifies each command clip. Defined in [`src/model.py`](src/model.py),
exported to ONNX in [`rpi/crnn.onnx`](rpi/crnn.onnx) (**3.96 MB**).

```
log-mel (1 × 64 × 480)      16 kHz, 10 ms hop, 64 mel bands, ~4.8 s padded
  → Conv 3×3/BN/ReLU/MaxPool   1 → 32 → 64 → 128
  → reshape (60 steps × 1024)
  → BiGRU (128 × 2 directions)
  → mean over time → Dropout(0.3)
  → intent head: Linear 256 → 19
  → slot head:   Linear 256 → 19   (18 slot values + NO_SLOT)
```

- **Multi-task by construction** — both heads share the encoder, so intent and slot
  regularize the same features.
- **Joint decoding** — at inference the two softmax vectors are combined and the argmax is
  taken **only over the 31 valid (intent, slot) pairs** (the label space in
  `data/labels.json`). An impossible combination — e.g. `TIME` with slot `red` — can never
  be output, and `NO_SLOT` is correctly predicted for the 13 slot-less intents.
- **Pi-friendly** — BatchNorm is folded into the convolutions at export (zero extra
  latency); the whole forward pass is a few milliseconds. An INT8 export is supported via
  `--model crnn_int8.onnx` but has not been produced yet (see [pending](#status--pending-verification)).

Feature front end (shared by training, evaluation, and the server — one implementation in
[`src/features.py`](src/features.py)): load → resample to 16 kHz → **silence-trim** →
STFT → triangular mel filters → log → pad/trim to 480 frames.

---

## The dataset

Built and documented in [`data/`](data/README.md). The WAV files themselves are too large for a git repo (~950 MB), so the repo ships the full dataset **documentation, class tables, and manifests** — `data/README.md`, `data/labels.json`, `data/slots.json`, `data/manifest.csv`, and `data/MYVOICE/myvoice_manifest.csv` — which fully describe every file:


| Item | Value |
|---|---:|
| Speakers | **100** — 84 foreign (LibriSpeech) + 16 Filipino-English (SilencioPH) |
| Intents | 19 (13 without slots, 6 with slots) |
| Phrase variants | 3 per intent (slotted intents: 3 templates × 3 slot values) |
| Acoustic conditions | 2 per utterance — **clean** and **light background noise** |
| Original WAV files | 18,600 |
| Filtered out (→ `data/FLAGGED/`) | 749 |
| **Active utterances** | **17,851** (≈8,927 clean + ≈8,924 noisy) |
| Splits | train **14,256** · val **1,806** · test **1,789** |

Every file is `data/<LABEL>/<LABEL>_s<speaker>_v<variant>_<clean|noisy>.wav` at 16 kHz mono,
tracked in `data/manifest.csv` (path, label, intent, speaker, split, transcript, slot,
duration). `data/labels.json` is the authoritative 31-label order the model was trained
against; `data/MYVOICE/` holds an optional personal-voice grading set (93 recordings —
31 labels × 3 variants — recorded with `src/record.py`).

---

## Training & results

Trained with PyTorch on an M-series Mac: Adam (lr 1e-3 → 1e-5 cosine), batch 64, 50-epoch
budget with early patience, train-only augmentation (noise + speed/pitch). Best model at
**epoch 28** (val 97.56 %). Full evaluation on the held-out test split
(`results/metrics.txt`, charts in `results/`):

| Metric | Value |
|---|---:|
| Test utterances | 1,789 |
| **Label accuracy (joint intent + slot)** | **99.55 %** (macro F1 0.995) |
| Intent accuracy | 99.55 % (macro P/R/F1 0.994 / 0.993 / 0.993) |
| **Slot accuracy** | **99.94 %** (macro P/R/F1 0.999 / 0.999 / 0.999) |
| Clean subset | 99.33 % (895) |
| Noisy subset | 99.78 % (894) |
| Training time | 50 min 46 s (33 epochs) |

Per-intent F1 (test): perfect (1.000) on `BRIGHTNESS`, `CALL`, `LIGHT_ON`, `MESSAGE`,
`TEMPERATURE`, `TIMER`, `WEATHER`; lowest are `TIME` 0.961, `VOLUME_UP` 0.983,
`VOLUME_DOWN` 0.990, `PAUSE`/`NEXT`/`PLAY_MUSIC`/`STOP`/`LIGHT_OFF`/`LIST_REMINDERS`
0.991 — i.e. **every intent ≥ 0.96**. Per-slot F1: `18 degrees` and `22 degrees` at 0.992; all other slot values 1.000;
`NO_SLOT` 1.000.
(Confusion matrices and per-class bars: `results/confusion_intent.png`,
`confusion_slot.png`, `class_metrics_*.png`, `accuracy_label.png`.)

---

## Latency

| Stage | Measured (M-series Mac) | Pi 4 (expected) |
|---|---|---|
| Wake word, per 80 ms chunk | ~5–15 ms | ~15–25 ms |
| CRNN forward (after endpointing) | **3–9 ms** | **~10–25 ms** |
| **Command → response** | **< 50 ms** | **well under 200 ms** |

The dominant perceived delay is the **0.8 s silence endpointing** — how long you pause
before the system knows you're done — not inference. Endpointing is adaptive: the server
measures the room-noise floor in the pause right after the wake word and derives the
silence threshold from it (so the wake word itself can't inflate it), capped at 4 s.

---

## Deploying to the Raspberry Pi 4

> Prepared, but **not yet executed on real hardware** — see
> [Status & pending verification](#status--pending-verification).

The deployable artifact is the **`rpi/` folder** — copy it whole to the Pi:

```
rpi/
├── tiny-alexa.html      # the entire UI (self-contained)
├── server.py            # wake word + endpointing + CRNN + music + HTTP/WS
├── crnn.onnx            # the 3.96 MB command model
├── src/                 # features.py, infer.py, wakeword.py (shared front end)
├── data/                # labels.json, slots.json, manifest.csv (class tables)
└── music/               # your MP3s (ID3 tags + embedded art read at startup)
```

```bash
# on the Pi
sudo apt update && sudo apt install -y python3-pip libportaudio2
python3 -m pip install aiohttp onnxruntime openwakeword resampy soundfile mutagen
python3 server.py                 # → http://<pi-ip>:8321
```

First run downloads the small openWakeWord "alexa" model (~3 MB) into the package cache.
Then open `http://<pi-ip>:8321` in any modern browser (Chrome/Edge/Safari — phone or
tablet work fine; the UI is responsive), grant the mic, tap **start**, and talk.

Useful flags:

```bash
python3 server.py --port 9000
python3 server.py --model crnn_int8.onnx   # quantized model, if you export one
python3 server.py --threshold 0.5          # wake sensitivity (lower = easier to trigger)
python3 server.py --debug                  # log every wake/command + latency
```

Notes for the 4 GB board: the voice path is tiny (ONNX Runtime + a 4 MB model); the 4
Cortex-A72 cores are plenty for the CRNN. If you ever add more clients, `--threads 3`
keeps inference off the audio thread.

---

## Development on Mac / PC

```bash
# training / evaluation environment (repo root)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # torch, numpy, scipy, soundfile, matplotlib, onnx, onnxruntime

python3 src/train.py                   # train → models/crnn.pt + results/
python3 src/export_onnx.py             # → models/crnn.onnx (verify with onnxruntime)
python3 src/evaluate.py                # full metrics + charts → results/
python3 src/infer.py --wav some.wav    # classify a file
python3 src/infer.py --loop            # live mic, wake word on
python3 src/infer.py --myvoice         # grade the personal-voice set (data/MYVOICE)
python3 classify_mic.py                # 3 s mic clip → instant prediction (deployed pipeline)
```

Rebuilding the model: `src/train.py` (PyTorch) → `src/export_onnx.py` (ONNX, BatchNorm
folded) → copy `crnn.onnx` into `rpi/`. The class tables (`data/labels.json`,
`data/manifest.csv`) are the single source of truth shared by training, evaluation, and
the server.

---

## Project layout

```
tiny-alexa/
├── README.md               ← this file
├── requirements.txt        # training env + Pi env (commented)
├── classify_mic.py         # quick 3-second mic classification (deployed pipeline)
├── src/                    # model + data tooling (PyTorch side)
│   ├── model.py            #   CRNN definition (2 heads)
│   ├── features.py         #   log-mel front end + class mappings (SHARED with server)
│   ├── dataset.py          #   manifest-driven datasets (intent + slot + label targets)
│   ├── train.py            #   training loop (Adam + cosine, early stop, history.csv)
│   ├── evaluate.py         #   test metrics + PNG charts
│   ├── export_onnx.py      #   torch → ONNX (+ runtime verification)
│   ├── infer.py            #   reference CLI inference (mic or WAV, wake word optional)
│   ├── record.py           #   guided recorder for your own voice (data/MYVOICE)
│   └── wakeword.py         #   reference openWakeWord wrapper
├── data/                   # dataset docs + class tables (the ≈950 MB WAVs stay local)
│   ├── README.md           #   full dataset documentation (speakers, splits, filtering)
│   ├── labels.json         #   the 31 labels, in model order
│   ├── slots.json          #   slot vocabulary
│   ├── manifest.csv        #   path/label/intent/speaker/split/transcript/slot/duration
│   └── MYVOICE/
│       └── myvoice_manifest.csv  #   personal-voice grading set (93 recordings)
├── models/                 # crnn.pt (3.97 MB) · crnn.onnx (3.96 MB)
├── results/                # metrics.txt · history.csv · confusion/class PNGs
├── screenshots/            # 19 UI captures (JPEG) — referenced throughout this README
└── rpi/                    # ★ the deployable folder — copy this to the Pi
    ├── tiny-alexa.html     #   the whole UI: state machine, 7 scenes, music, weather
    ├── server.py           #   aiohttp WS+HTTP server (wake word, endpointing, CRNN, music)
    ├── crnn.onnx           #   the model
    ├── src/                #   features.py · infer.py · wakeword.py
    ├── data/               #   labels.json · slots.json · manifest.csv
    └── music/              #   Juna (Clairo) · blink (Clara Benin) + extracted covers
```

---

## Tools & utilities

| Tool | Use |
|---|---|
| `?demo` | Scripted conversation cycling **every** scene — the fastest way to review the UI. |
| `?debug` | Shows the CRNN latency chip and enables console logging. |
| `--debug` (server) | Logs every wake word, command, decision, and latency. |
| `classify_mic.py [LABEL]` | Records 3 s from the mic and classifies with the *exact deployed pipeline*; pass a label to grade OK/WRONG. |
| `record.py` | Guided 93-take recorder for your own voice; writes `data/MYVOICE/myvoice_manifest.csv`. |
| `infer.py --myvoice` | Grades that personal set against the model. |
| `/health` | Quick liveness + model check for the Pi. |

---

## Weather & music

**Weather** is the only online feature, and it costs the Pi nothing: the browser calls
[open-meteo](https://open-meteo.com) (free, open-source, no API key, CORS-enabled) with
your browser Geolocation (fallback: [ipinfo.io](https://ipinfo.io), also keyless). A pink
pill sits top-right at all times (icon + temp + place; tap to refresh, hover for humidity,
wind, high/low). "What's the weather?" renders e.g. *"Partly cloudy, 28°C — feels like
33°, high 33° low 25°"* and holds it ~5 s. Offline, the pill shows **offline** and the
voice path degrades gracefully — it never waits on the network.

**Music** is a local playlist: drop `.mp3` files into `rpi/music/` and that's the whole
setup. `server.py` reads each file's ID3 tags (title/artist/album/duration) at startup and
extracts the embedded album art to `<name>_art.jpg`. The browser fetches `/music`, builds
the player card (cover, seek, prev/play/next, volume), and — while a track plays — the
main waveform **reacts to the actual music** via the `AnalyserNode`. Auto-advance on track
end. Current playlist: **Juna** (Clairo, *Charm*) and **blink** (Clara Benin,
*befriending my tears*).

---

## Engineering notes

Patterns and fixes that came out of building and stress-testing the UI:

- **One animation loop, always.** Every scene's `tickX()` rides the single existing
  `requestAnimationFrame` draw loop. No scene ever spawns its own loop — important on a
  Pi driving a browser.
- **One-shot fade guards.** The original aircon bug: the hold-expiry check ran *every
  frame* and re-armed the fade timer every frame, so the fade was cancelled 60×/s and the
  scene never left the screen. Fix: a `finishing` flag makes the fade arm exactly once.
  Same class of bug (tick guard excluding a non-terminal phase so the deadline never
  fires) was caught in the message scene by simulation before shipping.
- **Regex capture groups.** `String.match(/\d+/)` puts the number in index **0**; the
  old code read `match[1]` (always `undefined`) — so "brightness 20/60/100" all animated
  to 60 % and "temperature 18/26" all to 22°. Fixed at both dispatch sites.
- **Browser autoplay policy.** `player.play()` fired from a WebSocket handler (a
  non-gesture context) is rejected after every page refresh → "tap the page once".
  Fixed with the standard unlock pattern: the first `pointerdown`/`keydown` primes the
  `<audio>` element with a silent 0-frame WAV inside the gesture and resumes the
  `AudioContext`; the unlock sticks for the element's lifetime, so every later play
  succeeds.
- **Supersede discipline.** Starting any scene calls `dismissX()` on whichever scene is
  currently up (wired into every intent branch), so rapid-fire commands never stack.
- **Verification stack** (used for every UI change): extract the page's `<script>` →
  JXA syntax check; a mock-DOM harness (fake `classList`, time-keyed timers, 16 ms frame
  stepping) drives the **real module code** through 15+ lifecycle scenarios (natural
  lifecycle, STOP mid-phase, double-STOP, supersede mid-scene, restarts, mid-fade close —
  each must end in a byte-clean idle state); `curl` against the live server to confirm the
  served page.

---

## Status & pending verification

| Area | Status |
|---|---|
| Dataset (100 speakers, 17,851 utterances, clean + noisy) | ✅ built, filtered, documented (docs + manifests in `data/`; WAVs kept local) |
| CRNN training (99.55 % test, 99.94 % slot) | ✅ done, charts in `results/` |
| ONNX export + runtime verification | ✅ `crnn.onnx` (3.96 MB) |
| Server: wake word, adaptive endpointing, inference, music, WS protocol | ✅ running and tested on macOS (port 8321) |
| UI: state machine, all 7 animated scenes, STOP chain, supersede | ✅ implemented; 15+ scenario simulations pass; served-page checks pass |
| Weather (open-meteo, browser-side) | ✅ implemented, graceful offline fallback |
| Music pipeline (ID3 + art extraction + reactive waveform) | ✅ verified end-to-end (2-track playlist) |
| **Real Raspberry Pi 4 (4 GB) hardware run** | ⏳ **pending** — deployment steps are prepared (`rpi/`), not yet executed |
| Pi: mic input, wake-word reliability, sustained latency | ⏳ pending (expect: CRNN ~10–25 ms, command→response < 200 ms) |
| Pi: RAM/CPU profile under load | ⏳ pending |
| INT8-quantized model (`crnn_int8.onnx`) | ⏳ not yet exported (flag already supported) |
| Pi autostart (systemd service) | ⏳ not yet configured |

When the Pi run happens, the numbers in the [Latency](#latency) table marked *expected*
should be replaced with measurements, and this table updated.

---

*tiny alexa — built for the Raspberry Pi 4, runs anywhere a browser does.*

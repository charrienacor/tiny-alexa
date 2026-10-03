# tiny alexa

A real-time voice assistant for the **Raspberry Pi 4 (4 GB)**.
Say **"Hey Alexa"**, then a command — the app listens, understands, and
responds with an animated pink-on-black UI. Voice runs 100% on-device;
the only thing that ever touches the network is the **weather**, and it's
fetched **directly by the browser** (open-meteo, no API key) — never by the
Pi.

```
browser (tiny-alexa.html)                 Raspberry Pi 4
┌───────────────────────────┐   WebSocket   ┌──────────────────────────────┐
│ mic → 16 kHz int16 PCM ───┼──────────────▶│ openWakeWord  ("hey alexa")  │
│ AnalyserNode → waveform   │               │ endpointing (0.8 s silence)  │
│ state machine + anim      │◀──────────────┤ CRNN (crnn.onnx) → 31 labels│
└───────────────────────────┘   JSON events └──────────────────────────────┘
```

Everything voice-related runs on-device. The only network call is the
weather, made **by the browser** (not the Pi) to the free, open-source
[open-meteo](https://open-meteo.com) API — no API key, no server dependency.
If the browser is offline, the weather chip shows "offline" and the voice
pipeline is unaffected.

## Files

| File | What it is |
|---|---|
| `tiny-alexa.html` | The whole UI — self-contained, no CDN, no external assets. |
| `server.py` | Local WebSocket server: wake word + endpointing + CRNN inference. |
| `crnn.onnx` | The command model (2 heads: 19 intents × 19 slots → 31 valid labels). |
| `src/features.py` | Audio front end (log-mel) + class mappings — shared with training. |
| `src/infer.py` | Reference CLI inference (the Python equivalent of `server.py`). |
| `src/wakeword.py` | Reference openWakeWord wrapper. |
| `data/labels.json` | The 31 labels, in model order. |
| `data/manifest.csv` | label → intent / slot mapping. |
| `music/` | Local MP3 playlist (ID3 tags + embedded album art are read automatically). |
| `requirements.txt` | Python dependencies for `server.py` (install with `pip install -r requirements.txt`). |

## Run on the Raspberry Pi 4

Raspberry Pi OS (Bookworm) blocks `pip install` into the system Python
(`externally-managed-environment`), so use a virtual environment:

```bash
sudo apt update && sudo apt install -y python3-venv libportaudio2
cd ~/tiny-alexa/rpi            # wherever you copied this folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 server.py              # → http://<pi-ip>:8321
```

(Re-run `source .venv/bin/activate` in each new terminal, or call
`.venv/bin/python server.py` directly.)

First run downloads the small openWakeWord models (~3 MB) into the
`openwakeword` package. Open `http://<pi-ip>:8321` in Chrome/Edge/Safari,
grant the microphone permission, and say **"Hey Alexa"**.

Useful flags:

```bash
python3 server.py --port 9000
python3 server.py --model crnn_int8.onnx   # quantized model = faster on Pi
python3 server.py --threshold 0.5          # wake-word sensitivity (0.3 easier)
python3 server.py --debug                  # log every wake/command + latency
```

## Run on a Mac / PC (for development)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 server.py
```

Then open **http://localhost:8321**.

> **Design preview without a mic or server:** open
> `tiny-alexa.html?demo` — it plays a scripted conversation so you can see
> the full animation flow (wake → listening → confirm → idle) instantly.
> Add `&debug` to show the CRNN latency chip.

## Troubleshooting

- **`error: externally-managed-environment`** — Raspberry Pi OS (Bookworm)
  protects the system Python. Don't bypass it with `--break-system-packages`;
  use the venv flow above.
- **`cannot import name 'download_models' from 'openwakeword'`** — you have
  openwakeword < 0.6 installed (older wheels bundle the models differently and
  have no downloader). Fix:
  ```bash
  pip install -U "openwakeword==0.6.0"
  ```
  `server.py` tolerates both: on < 0.6 it skips the explicit download and
  relies on the bundled model files.

## Latency (measured)

| Stage | Mac (M-series) | Pi 4 (expected) |
|---|---|---|
| Wake word per 80 ms chunk | ~5 ms | ~15–25 ms |
| CRNN (after you stop speaking) | **3–9 ms** | **~10–25 ms** |
| **Command → response** | **< 50 ms** | **well under 200 ms** |

The 200 ms budget is dominated by the 0.8 s silence endpointing (how long you
pause before the app knows you're done) — the actual inference is a few
milliseconds. On the Pi, `--model crnn_int8.onnx` (if you export one) and
`--threads 3` keep the CRNN comfortably fast on the 4 Cortex-A72 cores.

## How it works

1. **Capture** — `tiny-alexa.html` grabs the mic with an AudioWorklet,
   resamples to 16 kHz, and streams 80 ms int16 chunks over a WebSocket.
   The same stream feeds an `AnalyserNode` that draws the live waveform.
2. **Wake word** — `server.py` runs openWakeWord's "alexa" model on each
   chunk. On detection it emits `awake` (the UI shows "Uh-huh / Yes / Yeah?").
3. **Endpointing** — the server measures the room-noise floor in the pause
   after the wake word, then records until you've been silent for 0.8 s
   (or 4 s max) — so it responds the instant you finish.
4. **Classify** — the recorded clip goes through the exact training front end
   (trim silence → log-mel 64×480) into `crnn.onnx`. The two heads are
   **jointly decoded** over the 31 valid (intent, slot) pairs, so an
   impossible combination can never be output.
5. **Respond** — the UI shows a randomized confirm ("Got it / Okay / Sure")
   plus the resolved command (e.g. *Brightness set to 60 percent*), then the
   waveform relaxes back to its idle breathing state.

Commands with confidence below 0.50 are rejected with "Sorry, that's beyond
what I can do…".

## Weather (online, open-source)

"Hey Alexa, what's the weather" now answers with **live data** from
[open-meteo](https://open-meteo.com) — a free, open-source weather API that
needs **no API key** and is CORS-enabled, so the browser calls it directly:

```
tiny-alexa.html ──fetch──▶ api.open-meteo.com   (current + today's high/low)
```

- **Location** — browser Geolocation if you allow it (chip says "your
  location"); otherwise it falls back to [ipinfo.io](https://ipinfo.io)
  (also free, no key) and shows your city, e.g. "Quezon City".
- **UI** — a small pink pill sits in the top-right corner at all times
  (icon + temperature + place). Tap it to refresh. Hover for humidity,
  wind, and high/low.
- **Voice** — the WEATHER intent renders the live answer, e.g.
  *"Partly cloudy, 28°C — feels like 33°, high 33° low 25°"*, and the
  response stays on screen a bit longer (5 s) so you can read it.
- **Offline** — if the fetch fails (no internet, DNS blocked), the chip
  shows "offline" and the voice reply degrades to "Checking the weather…"
  + a retry. Nothing crashes; the voice path never waits on the network.
- **Cost to the Pi** — zero. The Pi never sees the weather request; it only
  receives the `WEATHER` intent from the CRNN.

No new dependencies on the Pi: weather uses only browser `fetch`.

## Music (local MP3s)

Drop `.mp3` files into `music/` — that's the whole setup. `server.py` reads
each file's ID3 tags (title, artist, album, track number) and extracts the
embedded album art at startup, then serves:

- `GET /music` — the track list (JSON) the player card renders from
- `GET /music/<file>` — the MP3s and cover images (cached for 1 h)

Playback happens in the **browser's `<audio>` element**, so the Pi only serves
small static files — zero audio decoding on the CPU. The card shows the album
cover, title, artist, a seekable progress bar, and prev / play-pause / next /
volume buttons; the sound wave reacts to the music in real time (the audio is
routed through the shared AudioContext analyser).

| Voice command | Action |
|---|---|
| "play music" | start the playlist (or resume if paused) |
| "pause" | pause / resume toggle |
| "stop" | stop playback, hide the card |
| "next" | next track (auto-advances when a track ends) |
| "volume up / down" | nudge player volume ±15% |

`mutagen` is optional but recommended (exact durations in the track list);
without it the player still works and fills in durations from the audio
element.

## The 31 commands

19 intents; 6 of them carry a slot value:

- **Music:** play music · pause · stop · next
- **Volume:** volume up · volume down
- **Lights:** lights on · lights off · brightness 20/60/100 · color red/green/blue
- **Thermostat:** temperature 18/22/26 degrees
- **Time:** alarm 6 AM / 8 AM / 9 PM · timer 10 s / 30 s / 1 minute · what time is it
- **Reminders:** remind me to drink water / exercise / study · list reminders
- **Other:** call · send a message · what's the weather

## Aircon (animated thermostat)

Saying **"temperature 18 / 22 / 26 degrees"** brings up an animated wall
unit instead of a plain text line:

- the LED display **tweens** from the previous setting to the target
  (22 → 18 counts down; 18 → 26 counts up)
- the **fan spins faster the colder** the target (0.35 s/rev at 18°,
  2.15 s/rev at 22° eco)
- while cooling, **cold-air streaks fall** from the louver and a mist
  glows under the unit; while heating (26°), **warm streaks rise**
- the mode label reads `cool` / `eco` / `heat`; the status LED pulses
- the unit holds ~3.4 s, then fades back to idle — same contract as the
  timer ring. Saying **"stop"** dismisses it early.

Pure CSS + one JS module in `tiny-alexa.html`; no new assets, no network,
nothing added to the Pi's voice path (the CRNN still just emits the
`TEMPERATURE` intent + slot).

## Lamp (animated light)

Saying **"brightness 20 / 60 / 100 percent"** or **"lights on / off"**
brings up an animated pendant lamp instead of a plain text line:

- the lamp hangs from the top of the screen and **sways gently**; the
  bulb **ignites with a filament flicker** (a stepped opacity ramp)
- the **percentage tweens** from the previous level to the target
  (0 → 60 counts up; 100 → 20 counts down)
- the **glow scales with the level** — a big soft halo, a lit bulb core,
  and a light pool on the floor all track a `--lum` value (gamma-corrected
  so 20% looks dim, not half-lit)
- while lit, **dust motes drift** down through the beam
- the mode label reads `off` / `dim` / `bright` / `full`
- the lamp holds ~3.4 s, then fades back to idle — same contract as the
  aircon and timer ring. Saying **"stop"** dismisses it early.

Pure CSS + one JS module in `tiny-alexa.html`; no new assets, no network,
nothing added to the Pi's voice path (the CRNN still just emits the
`BRIGHTNESS` / `LIGHT_ON` / `LIGHT_OFF` intent + slot).

## Phone (animated call)

Saying **"call"** or **"make a call"** brings up a phone dialing screen
instead of a plain text line:

- **Dialing phase** (~2–3.4 s): a pink phone icon in a glass avatar with
  **expanding sonar rings**, a randomized contact name (Alexa / Home /
  Office / Mom / Dad / Work), a blinking `calling…` status, and a
  **real ring tone** — classic US cadence, dual 350 + 440 Hz, 2 s on /
  4 s off, synthesized with the Web Audio API (no audio files)
- **Connected phase**: rings stop, a **live call timer** counts up
  (0:00, 0:01, …), the status reads `connected`, and a **7-bar
  equalizer** dances under the name
- **Hang up**: tap the red button, say **"stop"**, or let it run — the
  call auto-hangs up after 20 s so the screen always returns to idle.
  The screen flashes briefly, a descending hang-up tone plays, and a
  randomized bye phrase appears (*Bye / Talk soon / All done*)
- saying any other command (timer, temperature, lights, music, …)
  silently supersedes the call mid-conversation

Same contract as the timer ring, aircon, and lamp: the screen owns its
hold time and fades back to idle on its own. Pure CSS + one JS module in
`tiny-alexa.html`; no new assets, no network, nothing added to the Pi's
voice path (the CRNN still just emits the `CALL` intent).

## Alarm (animated clock)

Saying **"alarm at 6 AM"** (or any `ALARM` command) brings up a
**twin-bell alarm clock** instead of a plain text line:

- **Set phase** (~1.5 s): the clock appears quietly, its hands **sweep
  from 12:00 to the set time** (CSS-transition eased), and the set time
  is shown on the face + subline (`alarm set for 6:00 AM`)
- **Ringing phase**: the whole clock **shakes**, both **bells rock** and
  the **hammer strikes** between them, **sound waves** arc off the bells,
  the glow pulses, and a **real twin-bell tone** rings — three detuned
  metallic partials (G6 + harmonics) struck every 700 ms, synthesized
  with the Web Audio API (no audio files)
- **Snooze**: tap the `snooze` button (or it's the natural "I'll sleep
  in" path) — the ringing stops, the clock **dozes** with a floating
  `z z z`, then **wakes and rings again** after ~2.6 s
- **Stop**: tap the `✕` button, say **"stop"** (6th in the STOP chain:
  music → timer → AC → lamp → call → **alarm**), or let it run — the
  alarm auto-stops after ~5 s so the screen always returns to idle. On
  stop the shaking **settles out** like a real clock, a final "clink"
  plays, and a good-morning phrase appears (*All quiet / Good morning /
  See you tomorrow*)
- saying any other command (timer, temperature, lights, music, …)
  silently supersedes the alarm mid-conversation

Same contract as the timer ring, aircon, lamp, and phone: the clock owns
its hold time and fades back to idle on its own. Pure CSS + one JS
module in `tiny-alexa.html`; no new assets, no network, nothing added to
the Pi's voice path (the CRNN still just emits the `ALARM` intent with
the time slot).

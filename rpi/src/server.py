#!/usr/bin/env python3
"""
tiny alexa — local real-time voice server (Raspberry Pi 4 ready)

Runs the wake word ("hey alexa", openWakeWord) and the CRNN command model
(crnn.onnx) entirely on-device. The browser (tiny-alexa.html) streams 16 kHz
int16 microphone PCM over a WebSocket; this server detects the wake word,
records the command with endpointing, runs the CRNN, and sends state events
back. Everything is local — no cloud, no internet after the first model load.

Run (from the rpi folder):
    python3 src/server.py               # http://localhost:8321
    python3 src/server.py --port 9000
    python3 src/server.py --model model/crnn_int8.onnx   # quantized model (faster on Pi)
    python3 src/server.py --debug       # print every event + latency

On the Raspberry Pi 4 (4 GB):
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python3 src/server.py
then open http://<pi-ip>:8321 in a browser (Chrome / Edge / Safari).

Latency budget (measured on M-series; Pi 4 is ~3-4x slower on the CPU parts):
    wake word  ~5-15 ms / 80 ms chunk
    CRNN       ~3 ms  (M-series)  ->  ~10-20 ms (Pi 4, 4 cores)
    => command -> response well under 200 ms.
"""
import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path

try:
    import mutagen
except ImportError:          # optional: gives exact durations in /music
    mutagen = None

import numpy as np
import onnxruntime as ort
from aiohttp import web

# Layout: this file lives in rpi/src/, so the project root is one level up.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
from features import (SAMPLE_RATE, N_MELS, MAX_FRAMES,
                      log_mel, trim_silence, load_intents_slots)  # noqa: E402

DATA = HERE / "data"
UNKNOWN_THRESHOLD = 0.50
UNKNOWN_MESSAGE = ("Sorry, that's beyond what I can do. Try 'set an alarm "
                   "for 6 AM' or 'turn on the lights'.")

# ---------------------------------------------------------------------------
# Class tables (built once, identical to src/infer.py)
# ---------------------------------------------------------------------------
with open(DATA / "labels.json", encoding="utf-8") as f:
    LABELS = json.load(f)                                   # 31 ordered labels
INTENTS, SLOTS, LABEL_MAP = load_intents_slots(DATA)
_INT_TO_IDX = {n: i for i, n in enumerate(INTENTS)}
_SLO_TO_IDX = {n: i for i, n in enumerate(SLOTS)}
COMBO_I = np.array([_INT_TO_IDX[LABEL_MAP[l][0]] for l in LABELS])
COMBO_S = np.array([_SLO_TO_IDX[LABEL_MAP[l][1]] for l in LABELS])
LABEL_INTENT = [LABEL_MAP[l][0] for l in LABELS]
LABEL_SLOT = [LABEL_MAP[l][1] for l in LABELS]


def _log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def decode(intent_logits, slot_logits):
    """Joint decode: only the 31 valid (intent, slot) pairs can win."""
    lp = _log_softmax(intent_logits[0])
    ls = _log_softmax(slot_logits[0])
    scores = lp[COMBO_I] + ls[COMBO_S]
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx]), LABEL_INTENT[idx], LABEL_SLOT[idx]


# ---------------------------------------------------------------------------
# Global models (loaded once)
# ---------------------------------------------------------------------------
CRNN = None
WW = None
ARGS = None


def load_models(model_path):
    global CRNN, WW
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ARGS.threads
    opts.inter_op_num_threads = 1
    CRNN = ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    # warmup (first run is slow: kernel selection + caches)
    CRNN.run(None, {"logmel": np.zeros((1, 1, N_MELS, MAX_FRAMES), dtype=np.float32)})

    # openWakeWord "alexa" model — supports both API generations:
    #   0.4.x: Model(wakeword_model_paths=[...]) — model ships inside the wheel
    #   0.6+ : Model(wakeword_models=["alexa"])  — downloads on first run
    import inspect
    import openwakeword
    from openwakeword.model import Model
    if "wakeword_model_paths" in inspect.signature(Model.__init__).parameters:
        WW = Model(wakeword_model_paths=[openwakeword.models["alexa"]["model_path"]])
    else:
        try:
            from openwakeword.utils import download_models
            download_models(["alexa"])   # 0.6+ ships no models; ~a few MB first run
        except ImportError:
            pass
        WW = Model(wakeword_models=["alexa"], inference_framework="onnx")
    print(f"[tiny-alexa] CRNN loaded: {model_path.name} "
          f"(threads={ARGS.threads})")
    print("[tiny-alexa] openWakeWord 'alexa' loaded")


def classify(wav):
    """Waveform -> (label, prob, intent, slot, latency_ms)."""
    t0 = time.perf_counter()
    mel = log_mel(trim_silence(wav))[np.newaxis]              # (1,1,64,480)
    il, sl = CRNN.run(None, {"logmel": mel})
    dt = (time.perf_counter() - t0) * 1000.0
    label, prob, intent, slot = decode(il, sl)
    return label, prob, intent, slot, dt


# ---------------------------------------------------------------------------
# Local music library  (music/*.mp3, ID3 tags + embedded album art)
# Served to the browser; the AUDIO ITSELF is played by the browser's
# <audio> element, so the Pi's CPU only serves small static files.
# ---------------------------------------------------------------------------
MUSIC_DIR = HERE / "music"

def _id3(path):
    """Minimal ID3v2 reader: TIT2/TPE1/TALB/TRCK text + APIC album art.
    Returns (meta_dict, art_bytes_or_None). No third-party dependency."""
    meta, art = {}, None
    try:
        with open(path, "rb") as f:
            head = f.read(10)
            if head[:3] != b"ID3":
                return meta, art
            size = (head[6] << 21) | (head[7] << 14) | (head[8] << 7) | head[9]
            blob = f.read(size)
        i, n = 0, len(blob)
        while i + 10 <= n:
            fid = blob[i:i + 4]
            if not fid or fid[0] == 0:
                break
            flen = (blob[i + 4] << 24) | (blob[i + 5] << 16) | (blob[i + 6] << 8) | blob[i + 7]
            if flen <= 0:
                break
            payload = blob[i + 10:i + 10 + flen]
            if fid in (b"TIT2", b"TPE1", b"TALB", b"TRCK"):
                enc = payload[0]
                try:
                    meta[fid.decode()] = payload[1:].decode(
                        "utf-16" if enc == 1 else "latin-1", "ignore").strip("\x00 ").strip()
                except Exception:
                    pass
            elif fid == b"APIC" and art is None:
                enc = payload[0]
                m0 = payload.find(b"\x00", 1)
                if m0 < 0:
                    break
                m1 = payload.find(b"\x00", m0 + 2)
                if m1 < 0:
                    break
                art = payload[m1 + 1:]
            i += 10 + flen
    except OSError:
        pass
    return meta, art

def _duration(path):
    if mutagen is None:
        return None
    try:
        return round(mutagen.File(path, easy=False).info.length, 1)
    except Exception:
        return None

def build_tracks():
    tracks = []
    if MUSIC_DIR.is_dir():
        for p in sorted(MUSIC_DIR.glob("*.mp3")):
            meta, art = _id3(p)
            if art:
                cover = p.stem + "_art.jpg"
                (MUSIC_DIR / cover).write_bytes(art)
            else:
                cover = None
            stem = p.stem
            title = meta.get("TIT2") or stem
            artist = meta.get("TPE1") or ""
            tracks.append({
                "title": title,
                "artist": artist,
                "album": meta.get("TALB", ""),
                "track": int(meta["TRCK"]) if meta.get("TRCK", "").isdigit() else 99,
                "file": p.name,
                "cover": f"music/{cover}" if cover else None,
                "duration": _duration(p),
            })
    tracks.sort(key=lambda t: (t["track"], t["title"]))
    return tracks

TRACKS = build_tracks()


def music_info():
    return {"tracks": TRACKS}


# ---------------------------------------------------------------------------
# Per-connection worker (runs in a thread so blocking model calls never
# stall the event loop)
# ---------------------------------------------------------------------------
def worker(q, evq, loop, debug):
    st = {"state": "idle", "noise_rms": 0.004, "rec": None}

    def emit(ev):
        loop.call_soon_threadsafe(evq.put_nowait, ev)

    while True:
        item = q.get()
        if item is None:
            break
        chunk = item                                          # int16, 1280
        f = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f * f)))

        if st["state"] == "idle":
            scores = WW.predict(chunk)
            if isinstance(scores, dict):
                # 0.4.x keys by model-file stem ("alexa_v0.1"); 0.6+ by name ("alexa").
                # We load exactly one wake word, so fall back to the top value.
                s = scores.get("alexa", max(scores.values(), default=0.0))
            else:
                s = float(scores)
            if s >= ARGS.threshold:
                try:
                    WW.reset()
                except AttributeError:
                    pass
                st["state"] = "recording"
                st["rec"] = {"phase": "noise", "noise": [], "buf": [],
                             "silent": 0, "t0": time.time(), "threshold": 0.01}
                emit({"type": "awake"})
                if debug:
                    print(f"  [wake] score={s:.2f}")
            else:
                st["noise_rms"] = 0.9 * st["noise_rms"] + 0.1 * rms
            continue

        rec = st["rec"]
        th = rec["threshold"]
        if rec["phase"] == "noise":
            # Measure the room-noise floor in the natural pause right after the
            # wake word (mirrors the 0.5 s noise probe in src/infer.py). Doing it
            # here — instead of reusing the pre-wake estimate — keeps the speech
            # threshold from being inflated by the wake word itself.
            rec["noise"].append(f)
            if rms > 0.02:                        # user started the command early
                nrms = (float(np.sqrt(np.mean(np.concatenate(rec["noise"]) ** 2)))
                        if rec["noise"] else 0.004)
                rec["threshold"] = max(0.006, 4.0 * nrms)
                rec["phase"] = "recording"
                rec["buf"].append(f)
            elif len(rec["noise"]) >= 5:          # ~0.4 s of silence measured
                nrms = float(np.sqrt(np.mean(np.concatenate(rec["noise"]) ** 2)))
                rec["threshold"] = max(0.006, 4.0 * nrms)
                rec["phase"] = "wait_speech"
                if debug:
                    print(f"  [noise] rms={nrms:.4f} threshold={rec['threshold']:.4f}")
            continue
        if rec["phase"] == "wait_speech":
            if rms > th:
                rec["phase"] = "recording"
                rec["buf"].append(f)
            elif time.time() - rec["t0"] > 6.0:
                st["state"] = "idle"
                emit({"type": "idle"})
            continue
        if rec["phase"] == "recording":
            rec["buf"].append(f)
            rec["silent"] = rec["silent"] + 1 if rms < th else 0
            if rec["silent"] >= 10 or len(rec["buf"]) >= 50:   # 0.8 s silence or 4 s cap
                wav = np.concatenate(rec["buf"]).astype(np.float32)
                label, prob, intent, slot, dt = classify(wav)
                st["state"] = "idle"
                if debug:
                    print(f"  [cmd] {label} p={prob:.2f} crnn={dt:.0f}ms "
                          f"({len(wav)/SAMPLE_RATE:.2f}s)")
                if prob < UNKNOWN_THRESHOLD:
                    emit({"type": "rejected", "message": UNKNOWN_MESSAGE})
                else:
                    emit({"type": "result", "label": label, "intent": intent,
                          "slot": slot, "prob": round(prob, 3),
                          "latency_ms": round(dt, 1)})
    emit({"type": "_end"})


# ---------------------------------------------------------------------------
# HTTP / WebSocket
# ---------------------------------------------------------------------------
def make_app():
    app = web.Application(client_max_size=2 ** 20)

    async def index(request):
        return web.FileResponse(HERE / "tiny-alexa.html")

    async def health(request):
        return web.json_response({"ok": True, "models": ["alexa", "crnn"],
                                  "labels": len(LABELS)})

    async def music_json(request):
        return web.json_response(music_info())

    async def music_file(request):
        name = Path(request.match_info["name"]).name        # no path traversal
        fp = MUSIC_DIR / name
        if not fp.is_file():
            raise web.HTTPNotFound()
        resp = web.FileResponse(fp)
        resp.headers["Cache-Control"] = "max-age=3600"
        return resp

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        loop = asyncio.get_event_loop()
        q = queue.Queue()            # thread-safe: browser thread -> worker thread
        evq = asyncio.Queue()        # worker thread -> event loop (via call_soon_threadsafe)

        t = threading.Thread(target=worker,
                             args=(q, evq, loop, bool(ARGS.debug)), daemon=True)
        t.start()

        async def sender():
            while True:
                ev = await evq.get()
                if ev is None or ev.get("type") == "_end":
                    break
                try:
                    await ws.send_str(json.dumps(ev))
                except Exception:
                    break

        sender_task = asyncio.ensure_future(sender())
        sample_rate = SAMPLE_RATE

        async def resample(chunk, src_rate):
            if src_rate == SAMPLE_RATE:
                return chunk
            import resampy
            out = resampy.resample(chunk.astype(np.float32), src_rate, SAMPLE_RATE)
            return (np.clip(out, -1, 1) * 32767).astype(np.int16)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        sample_rate = int(msg.data)
                    except ValueError:
                        pass
                elif msg.type == web.WSMsgType.BINARY:
                    raw = msg.data
                    if len(raw) % 2:
                        raw += b"\x00"
                    chunk = np.frombuffer(raw, dtype=np.int16)
                    if sample_rate != SAMPLE_RATE:
                        chunk = await resample(chunk, sample_rate)
                    q.put(chunk)
                elif msg.type in (web.WSMsgType.CLOSE,
                                  web.WSMsgType.CLOSING,
                                  web.WSMsgType.CLOSED):
                    break
        finally:
            q.put(None)
            t.join(timeout=1.0)
            await evq.put(None)
            sender_task.cancel()
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/tiny-alexa.html", index)
    app.router.add_get("/health", health)
    app.router.add_get("/music", music_json)
    app.router.add_get("/music/{name}", music_file)
    app.router.add_get("/ws", ws_handler)
    return app


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--model", default=str(HERE / "model" / "crnn.onnx"))
    ap.add_argument("--threads", type=int, default=3,
                    help="CRNN intra-op threads (Pi 4 has 4 cores; 3 leaves one for audio)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="wake-word detection threshold")
    ap.add_argument("--debug", action="store_true")
    ARGS = ap.parse_args()

    print("[tiny-alexa] loading models ...")
    t0 = time.perf_counter()
    load_models(Path(ARGS.model))
    print(f"[tiny-alexa] ready in {(time.perf_counter()-t0)*1000:.0f} ms")

    app = make_app()
    if TRACKS:
        print(f"[tiny-alexa] music: {len(TRACKS)} local track(s) in music/")
    else:
        print("[tiny-alexa] music: no tracks found in music/ (drop .mp3 files there)")
    print(f"[tiny-alexa] serving http://{ARGS.host}:{ARGS.port}  (open this in a browser)")
    web.run_app(app, host=ARGS.host, port=ARGS.port, print=None)


if __name__ == "__main__":
    main()

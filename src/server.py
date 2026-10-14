#!/usr/bin/env python3
"""
tiny alexa — local real-time voice server (Raspberry Pi 4 ready)

Runs the wake word ("Alexa", openWakeWord 0.4.0) and the two-head TCN
command model (models/TCN.onnx) entirely on-device. The browser
(tiny-alexa.html) streams 16 kHz int16 microphone PCM over a WebSocket;
this server detects the wake word, records the command with endpointing,
runs the TCN, and sends state events back. Everything is local — no cloud.

The wake-word detector and command model are the same classes src/infer.py
uses, so the browser and the terminal pipeline behave identically.

Run (from the project root):
    python3 src/server.py               # http://localhost:8321
    python3 src/server.py --port 9000
    python3 src/server.py --int8        # models/TCN_int8.onnx if present
    python3 src/server.py --debug       # print every event + latency

On the Raspberry Pi 4:
    pip install numpy soundfile onnxruntime aiohttp openwakeword==0.4.0
    python3 src/server.py
then open http://<pi-ip>:8321 in a browser (Chrome / Edge / Safari).
Note: browsers only allow the microphone on https:// or http://localhost.

Weather: the browser fetches it from open-meteo; GET /geo?lat=..&lon=..
turns its GPS fix into a place name (OpenStreetMap Nominatim, cached), so
the Pi needs internet access for the weather chip to show the real location.

WebSocket protocol (/ws):
    browser -> server   text "16000" (sample rate), then binary int16 PCM
    server  -> browser  {"type": "ready" | "awake" | "result" | "rejected"
                                 | "idle" | "error", ...}
"""
import argparse
import asyncio
import collections
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

import aiohttp
import numpy as np
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import SAMPLE_RATE  # noqa: E402
from infer import (COMMAND_THRESHOLD, WAKE_CHUNK, CommandModel,  # noqa: E402
                   WakeWordDetector, make_wakeword_model)

DATA = ROOT / "data"
PAGE = ROOT / "tiny-alexa.html"
MUSIC_DIR = DATA / "music"

UNKNOWN_MESSAGE = ("Sorry, that's beyond what I can do. Try 'set an alarm "
                   "for 6 AM' or 'turn on the lights'.")

# Endpointing after the wake word, in 80 ms chunks.
WAKE_TAIL_CHUNKS = 3       # skip ~0.24 s: the detector fires before "Alexa" ends
# Room-noise floor = a low percentile of the chunk loudness over the ~3 s
# BEFORE the wake word. The quiet gaps between words dominate a low
# percentile, so neither "Alexa" nor the command inflates it. (Measuring
# after the wake word instead swallowed the start of a command said right
# away: one-word commands like "call" and "stop" lost their first sound.)
NOISE_HISTORY_CHUNKS = 40
NOISE_PERCENTILE = 20
MIN_SPEECH_RMS = 0.006     # floor for the speech threshold
SPEECH_OVER_NOISE = 4.0    # speech threshold = this x room-noise RMS
# Pre-roll: audio kept from just before the speech threshold was crossed,
# so the quiet onset of a word ("k" in "call", "s" in "stop") isn't cut off.
# The command model trims leading silence itself, exactly as in training.
PREROLL_CHUNKS = 3
# Cap on the speech threshold: the pre-wake floor can include music that the
# page ducks to 25% as soon as the wake word fires, and an uncapped 4x that
# level would be louder than a quiet command (simulated: 10% -> 98% correct).
MAX_SPEECH_THRESHOLD = 0.04
END_SILENCE_CHUNKS = 7     # 0.56 s of silence ends the command
# Early exit: once this much silence follows the speech, classify on every
# chunk and commit as soon as the model is this sure; otherwise keep waiting
# for END_SILENCE_CHUNKS. On the test set this answers ~74% of commands
# 0.16 s after speech ends (vs 0.8 s) with no loss of accuracy.
EARLY_EXIT_SILENCE_CHUNKS = 2
EARLY_EXIT_PROB = 0.95
MAX_COMMAND_CHUNKS = 60    # 4.8 s cap = the TCN's 480-frame input window
NO_SPEECH_TIMEOUT_S = 6.0  # give up if nothing is said after the wake word

# Reverse geocoding for the weather chip: GPS lat/lon -> "Quezon City".
# Done here, not in the browser, because Nominatim's usage policy requires an
# identifying User-Agent (browsers can't set one), at most 1 request/s, and
# caching. Results are cached per ~1 km cell for the life of the server.
GEO_URL = "https://nominatim.openstreetmap.org/reverse"
GEO_HEADERS = {"User-Agent": "tiny-alexa-rpi4/1.0 (offline voice assistant demo)"}
GEO_MIN_INTERVAL_S = 1.0
GEO_PLACE_KEYS = ("city", "town", "village", "municipality", "suburb",
                  "city_district", "county", "state_district", "state")

ARGS = None
CMD = None                # CommandModel: stateless, shared by all connections
CMD_NAME = "TCN"
CMD_LOCK = threading.Lock()


def classify(wav):
    """Float waveform -> (label, intent, slot, prob, latency_ms)."""
    t0 = time.perf_counter()
    with CMD_LOCK:
        label, intent, slot, prob = CMD.predict(wav)
    return label, intent, slot, prob, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Local music library  (data/music/*.mp3, ID3 tags + embedded album art)
# Served to the browser; the AUDIO ITSELF is played by the browser's
# <audio> element, so the Pi's CPU only serves small static files.
# ---------------------------------------------------------------------------
_ID3_ENCODINGS = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}


def _syncsafe(b):
    return (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]


def _id3(path):
    """Minimal ID3v2.3/2.4 reader: TIT2/TPE1/TALB/TRCK text + APIC album art.
    Returns (meta_dict, art_bytes_or_None). No third-party dependency."""
    meta, art = {}, None
    try:
        with open(path, "rb") as f:
            head = f.read(10)
            if len(head) < 10 or head[:3] != b"ID3" or head[3] not in (3, 4):
                return meta, art
            blob = f.read(_syncsafe(head[6:10]))
    except OSError:
        return meta, art
    version, flags = head[3], head[5]
    i = 0
    if flags & 0x40 and len(blob) >= 4:          # skip the extended header
        i = _syncsafe(blob[:4]) if version == 4 else 4 + int.from_bytes(blob[:4], "big")
    n = len(blob)
    while i + 10 <= n:
        fid = blob[i:i + 4]
        if fid[0] == 0:                          # padding
            break
        size_bytes = blob[i + 4:i + 8]
        # v2.4 frame sizes are syncsafe; v2.3 are plain 32-bit
        flen = _syncsafe(size_bytes) if version == 4 else int.from_bytes(size_bytes, "big")
        if flen <= 0 or i + 10 + flen > n:
            break
        payload = blob[i + 10:i + 10 + flen]
        i += 10 + flen
        if fid in (b"TIT2", b"TPE1", b"TALB", b"TRCK"):
            enc = _ID3_ENCODINGS.get(payload[0], "latin-1")
            meta[fid.decode()] = payload[1:].decode(enc, "ignore").strip("\x00 ").strip()
        elif fid == b"APIC" and art is None:
            # enc(1) mime\0 picture-type(1) description\0(\0) data
            enc = payload[0]
            m0 = payload.find(b"\x00", 1)
            if m0 < 0:
                continue
            term = b"\x00\x00" if enc in (1, 2) else b"\x00"
            d = m0 + 2
            while True:
                d = payload.find(term, d)
                if d < 0 or enc not in (1, 2) or (d - (m0 + 2)) % 2 == 0:
                    break                        # utf-16 terminator must be aligned
                d += 1
            if d >= 0:
                art = payload[d + len(term):] or None
    return meta, art


def _duration(path):
    if mutagen is None:
        return None
    try:
        return round(mutagen.File(path).info.length, 1)
    except Exception:
        return None


# Formats Chromium / Safari play natively. Matched case-insensitively
# ("SONG.MP3" counts); the explicit types avoid relying on the OS mime table.
AUDIO_TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
               ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
               ".wav": "audio/wav", ".flac": "audio/flac"}


def _mutagen_tags(path):
    """Title/artist/album/track from non-ID3 files (m4a, ogg, flac) when
    mutagen is installed; {} otherwise."""
    if mutagen is None:
        return {}
    try:
        tags = mutagen.File(path, easy=True).tags or {}
    except Exception:
        return {}
    first = lambda k: (tags.get(k) or [""])[0]
    return {"TIT2": first("title"), "TPE1": first("artist"),
            "TALB": first("album"), "TRCK": first("tracknumber")}


def build_tracks():
    """Scan MUSIC_DIR -> (track list, {file name: (mime, album art bytes)}).

    Album art stays in memory and is served from /cover/<file>, so the
    music folder is only ever read: a folder owned by another user (e.g.
    copied with sudo onto the Pi) can't stop the server from starting.
    """
    tracks, covers = [], {}
    if not MUSIC_DIR.is_dir():
        return tracks, covers
    files = [p for p in MUSIC_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in AUDIO_TYPES]
    for p in sorted(files):
        meta, art = _id3(p)
        if not meta:
            meta = _mutagen_tags(p)
        if art:
            covers[p.name] = ("image/png" if art[:8] == b"\x89PNG\r\n\x1a\n"
                              else "image/jpeg", art)
        trck = meta.get("TRCK", "").split("/")[0]  # "3/12" -> "3"
        tracks.append({
            "title": meta.get("TIT2") or p.stem,
            "artist": meta.get("TPE1", ""),
            "album": meta.get("TALB", ""),
            "track": int(trck) if trck.isdigit() else 99,
            "file": p.name,
            "cover": f"cover/{p.name}" if art else None,
            "duration": _duration(p),
        })
    tracks.sort(key=lambda t: (t["track"], t["title"]))
    return tracks, covers


# ---------------------------------------------------------------------------
# Per-connection worker (runs in a thread so blocking model calls never
# stall the event loop). Each connection gets its own wake-word model:
# openWakeWord keeps streaming state, so sharing one across tabs would mix
# their audio together.
# ---------------------------------------------------------------------------
def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def worker(q, emit, debug):
    try:
        detector = WakeWordDetector(make_wakeword_model(), threshold=ARGS.threshold)
    except Exception as e:
        emit({"type": "error", "message": f"wake word failed to load: {e}"})
        emit(None)
        return
    emit({"type": "ready", "models": ["alexa", CMD_NAME]})

    rec = None                 # None while waiting for the wake word
    levels = collections.deque(maxlen=NOISE_HISTORY_CHUNKS)   # pre-wake chunk RMS
    while True:
        chunk = q.get()
        if chunk is None:
            break

        # Feed the wake word every chunk, even while recording a command:
        # openWakeWord's reset() only clears scores, not its audio features,
        # so a starved detector would still "hear" the last "Alexa" later.
        woke = detector.process(chunk)
        f = chunk.astype(np.float32) / 32768.0
        rms = _rms(f)
        if rec is None:
            if woke:
                nrms = float(np.percentile(levels, NOISE_PERCENTILE)) if levels else 0.0
                rec = {"phase": "tail", "tail": 0, "buf": [], "silent": 0,
                       "preroll": collections.deque(maxlen=PREROLL_CHUNKS),
                       "t0": time.monotonic(),
                       "threshold": min(MAX_SPEECH_THRESHOLD,
                                     max(MIN_SPEECH_RMS, SPEECH_OVER_NOISE * nrms))}
                emit({"type": "awake"})
                if debug:
                    print(f"  [wake] noise rms={nrms:.4f} threshold={rec['threshold']:.4f}")
            else:
                levels.append(rms)
            continue

        if rec["phase"] == "tail":
            # Discard the end of the wake word so it isn't mistaken for the
            # start of the command (or for room noise).
            rec["tail"] += 1
            if rec["tail"] >= WAKE_TAIL_CHUNKS:
                rec["phase"] = "wait_speech"
            continue

        if rec["phase"] == "wait_speech":
            if rms > rec["threshold"]:
                rec["phase"] = "recording"
                rec["buf"] = [*rec["preroll"], f]       # keep the quiet onset
            elif time.monotonic() - rec["t0"] > NO_SPEECH_TIMEOUT_S:
                rec = None
                detector.reset()
                emit({"type": "idle"})
            else:
                rec["preroll"].append(f)
            continue

        # phase == "recording"
        rec["buf"].append(f)
        rec["silent"] = rec["silent"] + 1 if rms < rec["threshold"] else 0
        final = (rec["silent"] >= END_SILENCE_CHUNKS
                 or len(rec["buf"]) >= MAX_COMMAND_CHUNKS)
        if not final and rec["silent"] < EARLY_EXIT_SILENCE_CHUNKS:
            continue
        wav = np.concatenate(rec["buf"])
        try:
            label, intent, slot, prob, dt = classify(wav)
        except Exception as e:
            rec = None
            emit({"type": "error", "message": f"command model failed: {e}"})
            detector.reset()
            continue
        if not final and (label is None or prob < EARLY_EXIT_PROB):
            continue             # not sure yet: the user may only be pausing
        waited_ms = rec["silent"] * 80
        rec = None
        if debug:
            print(f"  [cmd] {label or 'unclear'} p={prob:.2f} tcn={dt:.0f}ms "
                  f"({len(wav) / SAMPLE_RATE:.2f}s, committed after "
                  f"{waited_ms} ms of silence{'' if final else ', early'})")
        if label is None:
            emit({"type": "rejected", "message": UNKNOWN_MESSAGE,
                  "prob": round(prob, 3)})
        else:
            emit({"type": "result", "label": label, "intent": intent,
                  "slot": slot, "prob": round(prob, 3),
                  "latency_ms": round(dt, 1)})
        detector.reset()         # drop audio heard while the command was spoken
    emit(None)


# ---------------------------------------------------------------------------
# HTTP / WebSocket
# ---------------------------------------------------------------------------
async def index(request):
    resp = web.FileResponse(PAGE)
    resp.headers["Cache-Control"] = "no-cache"
    return resp


async def health(request):
    return web.json_response({"ok": True, "models": ["alexa", CMD_NAME],
                              "labels": len(CMD.labels),
                              "clients": len(request.app["sockets"])})


async def music_json(request):
    return web.json_response({"tracks": request.app["tracks"]})


async def music_file(request):
    name = Path(request.match_info["name"]).name        # no path traversal
    fp = MUSIC_DIR / name
    if not fp.is_file() or fp.suffix.lower() not in AUDIO_TYPES:
        raise web.HTTPNotFound()
    return web.FileResponse(fp, headers={
        "Content-Type": AUDIO_TYPES[fp.suffix.lower()],
        "Cache-Control": "max-age=3600"})


async def cover_file(request):
    art = request.app["covers"].get(request.match_info["name"])
    if art is None:
        raise web.HTTPNotFound()
    mime, data = art
    return web.Response(body=data, content_type=mime,
                        headers={"Cache-Control": "max-age=3600"})


async def geo(request):
    """/geo?lat=..&lon=.. -> {"place", "region", "country"} via Nominatim."""
    try:
        lat, lon = float(request.query["lat"]), float(request.query["lon"])
    except (KeyError, ValueError):
        raise web.HTTPBadRequest(text="lat and lon are required")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise web.HTTPBadRequest(text="lat/lon out of range")

    app = request.app
    key = (round(lat, 2), round(lon, 2))           # ~1 km cell
    if key not in app["geo_cache"]:
        async with app["geo_lock"]:                # one request at a time, >= 1 s apart
            if key not in app["geo_cache"]:
                wait = app["geo_last"] + GEO_MIN_INTERVAL_S - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                params = {"lat": f"{lat:.5f}", "lon": f"{lon:.5f}", "format": "jsonv2",
                          "zoom": 14, "addressdetails": 1}
                headers = {**GEO_HEADERS, "Accept-Language":
                           request.headers.get("Accept-Language", "en")}
                try:
                    async with app["http"].get(GEO_URL, params=params, headers=headers,
                                               timeout=aiohttp.ClientTimeout(total=6)) as r:
                        r.raise_for_status()
                        j = await r.json(content_type=None)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    return web.json_response({"error": f"reverse geocoding failed: {e}"},
                                             status=502)
                finally:
                    app["geo_last"] = time.monotonic()
                addr = j.get("address") or {}
                place = next((addr[k] for k in GEO_PLACE_KEYS if addr.get(k)), None)
                region = addr.get("state") or addr.get("region") or addr.get("county")
                app["geo_cache"][key] = {
                    "place": place,
                    "region": region if region != place else None,
                    "country": addr.get("country"),
                }
                if ARGS.debug:
                    print(f"  [geo] {key} -> {app['geo_cache'][key]}")
    return web.json_response(app["geo_cache"][key])


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    q = queue.Queue()            # event loop -> worker thread
    evq = asyncio.Queue()        # worker thread -> event loop

    def emit(ev):
        loop.call_soon_threadsafe(evq.put_nowait, ev)

    t = threading.Thread(target=worker, args=(q, emit, ARGS.debug), daemon=True)
    t.start()
    request.app["sockets"].add(ws)
    peer = request.remote
    print(f"[tiny-alexa] client connected: {peer}")

    async def sender():
        while (ev := await evq.get()) is not None:
            if ws.closed:
                break
            await ws.send_str(json.dumps(ev))

    sender_task = asyncio.create_task(sender())
    # The browser sends 80 ms chunks, but re-chunk anyway: the wake word
    # needs exactly WAKE_CHUNK samples per call.
    pending = np.zeros(0, dtype=np.int16)
    odd = b""

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    rate = int(msg.data)
                except ValueError:
                    continue
                if rate != SAMPLE_RATE:
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": f"expected {SAMPLE_RATE} Hz audio, got {rate} Hz"}))
            elif msg.type == web.WSMsgType.BINARY:
                raw = odd + msg.data
                odd = raw[len(raw) & ~1:]               # keep a split int16 byte
                pending = np.concatenate(
                    [pending, np.frombuffer(raw[:len(raw) & ~1], dtype=np.int16)])
                n = len(pending) // WAKE_CHUNK * WAKE_CHUNK
                for i in range(0, n, WAKE_CHUNK):
                    q.put(pending[i:i + WAKE_CHUNK])
                pending = pending[n:]
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        q.put(None)
        await asyncio.to_thread(t.join, 2.0)
        emit(None)
        sender_task.cancel()
        request.app["sockets"].discard(ws)
        print(f"[tiny-alexa] client disconnected: {peer}")
    return ws


def make_app(tracks, covers):
    app = web.Application(client_max_size=2 ** 20)
    app["tracks"], app["covers"] = tracks, covers
    app["sockets"] = set()
    app["geo_cache"], app["geo_last"] = {}, 0.0

    async def http_session(app):
        app["http"] = aiohttp.ClientSession()      # for /geo; needs an event loop
        app["geo_lock"] = asyncio.Lock()
        yield
        await app["http"].close()

    async def close_sockets(app):
        for ws in list(app["sockets"]):
            await ws.close(code=1001, message=b"server shutdown")
    app.cleanup_ctx.append(http_session)
    app.on_shutdown.append(close_sockets)
    app.router.add_get("/", index)
    app.router.add_get(f"/{PAGE.name}", index)
    app.router.add_get("/health", health)
    app.router.add_get("/music", music_json)
    app.router.add_get("/music/{name}", music_file)
    app.router.add_get("/cover/{name}", cover_file)
    app.router.add_get("/geo", geo)
    app.router.add_get("/ws", ws_handler)
    return app


def main():
    global ARGS, CMD, CMD_NAME
    ap = argparse.ArgumentParser(description="tiny alexa browser server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--model", default=str(ROOT / "models" / "TCN.onnx"))
    ap.add_argument("--int8", action="store_true",
                    help="use the _int8 variant of --model if present (faster on the Pi)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="wake-word detection threshold")
    ap.add_argument("--debug", action="store_true")
    ARGS = ap.parse_args()

    model = Path(ARGS.model)
    if not model.exists():
        sys.exit(f"error: {model} not found.\n"
                 f"Train and export first:  python src/train.py && python src/export_onnx.py")
    if not PAGE.exists():
        sys.exit(f"error: {PAGE.name} not found in {ROOT}")

    print("[tiny-alexa] loading models ...")
    t0 = time.perf_counter()
    int8 = model.with_name(model.stem + "_int8" + model.suffix)
    CMD_NAME = int8.stem if ARGS.int8 and int8.exists() else model.stem
    CMD = CommandModel(model, DATA, use_int8=ARGS.int8)
    classify(np.zeros(SAMPLE_RATE, dtype=np.float32))     # warm-up: first run is slow
    make_wakeword_model()                                 # fail fast if it can't load
    print(f"[tiny-alexa] {CMD_NAME} + openWakeWord 'alexa' ready in "
          f"{(time.perf_counter() - t0) * 1000:.0f} ms "
          f"(command threshold {COMMAND_THRESHOLD})")

    tracks, covers = build_tracks()
    if tracks:
        print(f"[tiny-alexa] music: {len(tracks)} local track(s) in data/music/")
    else:
        print("[tiny-alexa] music: no tracks found in data/music/ "
              f"(drop {', '.join(AUDIO_TYPES)} files there)")
    print(f"[tiny-alexa] serving http://{ARGS.host}:{ARGS.port}  (open this in a browser)")
    web.run_app(make_app(tracks, covers), host=ARGS.host, port=ARGS.port, print=None)


if __name__ == "__main__":
    main()

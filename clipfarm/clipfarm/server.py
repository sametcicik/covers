"""Local web app: FastAPI + SSE progress + static UI (studio/dist)."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import queue
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import clip as cliplib
from . import download, moments, pipeline, profile
from .config import ASSETS, STREAMERS_DIR, STUDIO_DIR, settings
from .platforms import channel_info, parse_link
from .util import read_json, write_json

log = logging.getLogger("clipfarm")
app = FastAPI(title="ClipFarm")
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# the Remotion renderer (headless Chrome on its own origin) loads fonts/SFX from this server
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD"])
settings.ensure()

# ------------------------------------------------------------------ job runner (one heavy job at a time)

_events: dict[str, list[dict]] = {}
_listeners: dict[str, list[queue.Queue]] = {}
_work_q: queue.Queue = queue.Queue()
_lock = threading.Lock()


def _emit(job_key: str, ev: dict) -> None:
    ev = {"t": time.time(), **ev}
    with _lock:
        _events.setdefault(job_key, []).append(ev)
        _events[job_key] = _events[job_key][-400:]
        for q in _listeners.get(job_key, []):
            q.put(ev)


def _worker() -> None:
    while True:
        kind, key, args = _work_q.get()
        try:
            if kind == "job":
                cb = lambda step, frac, msg: _emit(key, {"type": "progress", "step": step, "frac": frac, "msg": msg})  # noqa: E731
                job = pipeline.run_job(**args, progress_cb=cb, job_id=key)
                _emit(key, {"type": "done", "job": job["id"]})
            elif kind == "rerender":
                cb = lambda step, frac, msg: _emit(key, {"type": "progress", "step": step, "frac": frac, "msg": msg, "clip": args["clip_id"]})  # noqa: E731
                res = pipeline.rerender(args["job_id"], args["clip_id"], args["settings"], progress_cb=cb)
                _emit(key, {"type": "clip", "clip": args["clip_id"], "result": res})
        except Exception as e:  # noqa: BLE001
            log.exception("worker failed")
            _emit(key, {"type": "error", "msg": str(e)[:600]})


threading.Thread(target=_worker, daemon=True).start()


# ------------------------------------------------------------------ API


class JobReq(BaseModel):
    url: str
    clips: int = 10
    start: float | None = None
    end: float | None = None
    max_minutes: float | None = None
    settings: dict = {}


class RerenderReq(BaseModel):
    settings: dict = {}


@app.get("/api/health")
def health():
    from .hardware import detect

    return {"ok": True, "hardware": detect().as_dict(), "llm": settings.llm_enabled}


@app.get("/api/resolve")
def resolve(url: str):
    try:
        src = parse_link(url)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    slug = src.channel
    title = None
    duration = None
    if not slug:
        try:
            m = download.resolve(url)
            slug, title, duration = m["channel"], m["title"], m["duration"]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Link çözümlenemedi: {e}") from e
    prof = profile.load(src.platform, slug)
    info = channel_info(src.platform, slug) if not prof else {}
    pal = (prof or {}).get("palette")
    if not pal and info:
        prof = profile.ensure(src.platform, slug, info.get("display_name"))
        pal = prof.get("palette")
    return {"platform": src.platform, "kind": src.kind, "channel": slug,
            "display_name": (prof or {}).get("display_name") or info.get("display_name") or slug,
            "avatar": f"/streamers/{slug}/art/avatar.img" if ((prof or {}).get("art") or {}).get("avatar") else info.get("avatar"),
            "palette": pal, "known": bool(prof), "title": title, "duration": duration}


@app.post("/api/jobs")
def create_job(req: JobReq):
    m = download.resolve(req.url)
    from .util import slugify

    jid = f"{m['platform']}_{slugify(m['channel'] or 'x', 20)}_{slugify(str(m['id']), 14)}"
    _emit(jid, {"type": "queued"})
    _work_q.put(("job", jid, {"url": req.url, "n_clips": req.clips, "start": req.start, "end": req.end,
                             "max_minutes": req.max_minutes, "clip_settings": req.settings}))
    return {"job_id": jid}


def _job_view(jid: str) -> dict:
    job = pipeline.load_job(jid)
    if not job:
        raise HTTPException(404, "job yok")
    jd = pipeline.job_dir(jid)
    clips = []
    for c in job.get("clips", []):
        cd = jd / "clips" / c["id"]
        info = read_json(cd / "info.json") or {}
        clips.append({**c, "info": info,
                      "cover": f"/work/{jid}/clips/{c['id']}/cover.jpg" if (cd / "cover.jpg").exists() else None,
                      "video": f"/work/{jid}/clips/{c['id']}/final.mp4" if (cd / "final.mp4").exists() else None,
                      "mtime": (cd / "final.mp4").stat().st_mtime if (cd / "final.mp4").exists() else 0})
    job["clips"] = clips
    m = read_json(jd / "meta.json") or {}
    prof = profile.load(m.get("platform", job.get("platform")), job.get("channel")) or {}
    job["streamer"] = {"name": prof.get("display_name"), "palette": prof.get("palette"), "platform": job.get("platform"),
                       "avatar": f"/streamers/{prof.get('slug')}/art/avatar.img" if (prof.get("art") or {}).get("avatar") else None}
    return job


@app.get("/api/jobs")
def list_jobs():
    out = []
    for p in sorted(settings.work_dir.glob("*/job.json"), key=lambda p: -p.stat().st_mtime):
        j = read_json(p)
        if j:
            out.append({k: j.get(k) for k in ("id", "platform", "channel", "title", "status", "n_clips")} |
                       {"ready": sum(1 for c in j.get("clips", []) if c.get("status") == "ready")})
    return out


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _job_view(jid)


@app.get("/api/jobs/{jid}/events")
async def events(jid: str, request: Request):
    q: queue.Queue = queue.Queue()
    with _lock:
        _listeners.setdefault(jid, []).append(q)
        backlog = list(_events.get(jid, []))[-50:]

    async def gen():
        try:
            for ev in backlog:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = q.get_nowait()
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.4)
                    yield ": ping\n\n"
        finally:
            with _lock:
                _listeners.get(jid, []).remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/jobs/{jid}/clips/{cid}/props")
def preview_props(jid: str, cid: str, req: RerenderReq):
    """Props for the in-browser Remotion Player (same composition as the export)."""
    jd = pipeline.job_dir(jid)
    cd = jd / "clips" / cid
    if not (cd / "moment.json").exists():
        raise HTTPException(404, "klip yok")
    st = {**(read_json(cd / "settings.json") or {}), **req.settings}
    m = read_json(jd / "meta.json")
    prof = profile.ensure(m["platform"], m["channel"])
    tr = read_json(jd / "analysis" / "transcript.json")
    sig_d = read_json(jd / "analysis" / "signals.json")
    import numpy as np

    sig = moments.Signals(sig_d["t0"], np.array(sig_d["heat"]), sig_d["audio"], sig_d["chat"])
    prev = read_json(jd / "source" / "preview.json")
    hq = read_json(cd / "hq.json")
    lay = prof.get("facecam") or {"mode": "game"}
    job = pipeline.load_job(jid)
    tmp = cd / "preview_build"
    tmp.mkdir(exist_ok=True)
    for f in ("track.json",):
        if (cd / f).exists() and not (tmp / f).exists():
            (tmp / f).write_bytes((cd / f).read_bytes())
    built = cliplib.build(job, read_json(cd / "moment.json"), tmp, hq, prof, tr["words"], sig, lay, prev, st)
    p = built["props"]
    p["video"]["src"] = f"/work/{jid}/clips/{cid}/hq.mp4"
    p["fontBaseUrl"] = "/assets/fonts"
    p["sfxBaseUrl"] = "/assets/sfx"
    p["muteVideo"] = False
    p["playSfx"] = True
    return {"props": p, "settings": st}


@app.post("/api/jobs/{jid}/clips/{cid}/rerender")
def rerender(jid: str, cid: str, req: RerenderReq):
    _work_q.put(("rerender", jid, {"job_id": jid, "clip_id": cid, "settings": req.settings}))
    _emit(jid, {"type": "queued", "clip": cid})
    return {"queued": True}


@app.get("/api/jobs/{jid}/clips/{cid}/download")
def download_clip(jid: str, cid: str):
    cd = pipeline.job_dir(jid) / "clips" / cid
    f = cd / "final.mp4"
    if not f.exists():
        raise HTTPException(404, "render yok")
    info = read_json(cd / "info.json") or {}
    from .util import slugify

    name = f"{jid}_{cid}_{slugify((info.get('titles') or ['clip'])[0], 40)}.mp4"
    return FileResponse(f, media_type="video/mp4", filename=name)


@app.get("/api/jobs/{jid}/download.zip")
def download_all(jid: str):
    job = _job_view(jid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i, c in enumerate(sorted([c for c in job["clips"] if c.get("status") == "ready"], key=lambda c: -c["score"]), 1):
            cd = pipeline.job_dir(jid) / "clips" / c["id"]
            base = f"{i:02d}_{c['id']}"
            for src, ext in (("final.mp4", "mp4"), ("cover.jpg", "jpg"), ("info.json", "json")):
                if (cd / src).exists():
                    z.write(cd / src, f"{base}.{ext}")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{jid}.zip"'})


@app.get("/api/profiles/{slug}")
def get_profile(slug: str):
    p = read_json(STREAMERS_DIR / slug / "profile.json")
    if not p:
        raise HTTPException(404, "profil yok")
    return p


@app.put("/api/profiles/{slug}")
def put_profile(slug: str, body: dict):
    p = read_json(STREAMERS_DIR / slug / "profile.json")
    if not p:
        raise HTTPException(404, "profil yok")
    for k in ("palette", "dictionary", "campaign", "label", "captions", "sfx", "hook", "language", "facecam", "fonts"):
        if k in body:
            p[k] = body[k]
    if "palette" in body:
        p["palette"]["locked"] = True
        profile.render_palette_png(p["palette"], STREAMERS_DIR / slug / "palette.png", p.get("display_name", slug))
    write_json(STREAMERS_DIR / slug / "profile.json", p)
    return p


# ------------------------------------------------------------------ static

app.mount("/work", StaticFiles(directory=str(settings.work_dir)), name="work")
app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")
STREAMERS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/streamers", StaticFiles(directory=str(STREAMERS_DIR)), name="streamers")
_dist = STUDIO_DIR / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")
else:
    @app.get("/")
    def _no_ui():
        return JSONResponse({"error": "UI build yok: cd studio && npm run build:ui"})


def serve() -> None:
    import os

    import uvicorn

    os.environ["CLIPFARM_MEDIA_BASE"] = f"http://127.0.0.1:{settings.port}"
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")

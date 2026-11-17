"""Render a clip with Remotion (video, muted) + ffmpeg audio mix, then mux to the final mp4."""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from . import audio
from .config import ASSETS, STUDIO_DIR, settings
from .util import log, read_json, run, write_json

_SERVER = {"port": None, "thread": None, "server": None}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def media_base() -> str:
    """Base URL that serves /work and /assets (the web app server, or a private one for the CLI)."""
    if os.environ.get("CLIPFARM_MEDIA_BASE"):
        return os.environ["CLIPFARM_MEDIA_BASE"].rstrip("/")
    if _SERVER["port"] is None:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles

        settings.ensure()
        from starlette.middleware import Middleware
        from starlette.middleware.cors import CORSMiddleware

        app = Starlette(routes=[Mount("/work", StaticFiles(directory=str(settings.work_dir))),
                                Mount("/assets", StaticFiles(directory=str(ASSETS)))],
                        middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD"])])
        port = _free_port()
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        _SERVER.update(port=port, thread=t, server=server)
    return f"http://127.0.0.1:{_SERVER['port']}"


def work_url(path: Path) -> str:
    rel = Path(path).resolve().relative_to(settings.work_dir.resolve())
    return f"{media_base()}/work/{rel.as_posix()}"


def with_urls(props: dict, hq_path: Path) -> dict:
    p = dict(props)
    p["video"] = {**p["video"], "src": work_url(hq_path)}
    p["fontBaseUrl"] = f"{media_base()}/assets/fonts"
    p["sfxBaseUrl"] = f"{media_base()}/assets/sfx"
    return p


def _node() -> str:
    return os.environ.get("NODE_BIN", "node")


def remotion(props_path: Path, out: Path, progress=None, frames: list[int] | None = None, stills_dir: Path | None = None) -> None:
    cmd = [_node(), str(STUDIO_DIR / "render.mjs"), str(props_path), str(out)]
    if frames:
        cmd += [f"--frames={','.join(map(str, frames))}", f"--stills-dir={stills_dir}"]
    if settings.remotion_concurrency:
        cmd += [f"--concurrency={settings.remotion_concurrency}"]
    env = {**os.environ, "VIDEO_BITRATE": settings.video_bitrate, "REMOTION_LICENSE_KEY": settings.remotion_license_key}
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        env.setdefault("NODE_USE_ENV_PROXY", "1")  # Node >= 22.21 honours HTTP(S)_PROXY with this flag
    log.info("remotion render -> %s", out)
    proc = subprocess.Popen(cmd, cwd=str(STUDIO_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    err_lines: list[str] = []

    def _err():
        for line in proc.stderr:  # type: ignore[union-attr]
            err_lines.append(line)

    th = threading.Thread(target=_err, daemon=True)
    th.start()
    for line in proc.stdout:  # type: ignore[union-attr]
        if line.startswith("PROGRESS") and progress:
            try:
                progress(float(line.split()[1]))
            except ValueError:
                pass
    proc.wait()
    th.join(timeout=2)
    if proc.returncode != 0:
        raise RuntimeError("remotion render failed:\n" + "".join(err_lines[-40:]))


def render_clip(clip_dir: Path, hq_path: Path, progress=None, denoise: bool = True) -> dict:
    props = read_json(clip_dir / "props.json")
    plan = read_json(clip_dir / "plan.json")
    rp = with_urls(props, hq_path)
    write_json(clip_dir / "props.render.json", rp)
    video = clip_dir / "video.mp4"
    remotion(clip_dir / "props.render.json", video, (lambda f: progress(f * 0.9)) if progress else None)
    mix = audio.build_mix(hq_path, plan["segments"], int(plan["fps"]), plan["sfx"], clip_dir / "audio", denoise)
    final = clip_dir / "final.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", mix["path"], "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", final])
    if progress:
        progress(1.0)
    return {"final": str(final), "video": str(video), "mix": mix}


def extract_frames(video: Path, times: list[float], out_dir: Path, width: int = 540) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        p = out_dir / f"qa_{i:02d}_{t:06.2f}.jpg"
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", f"scale={width}:-2",
             "-q:v", "3", p])
        paths.append(p)
    return paths


def cover(video: Path, t: float, out: Path) -> Path:
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-q:v", "2", out])
    return out

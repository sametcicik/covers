"""Small shared helpers: json io, subprocess, ffprobe, progress reporting, slugging."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("clipfarm")

TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def tr_upper(s: str) -> str:
    return s.translate(TR_UPPER).upper()


def tr_lower(s: str) -> str:
    return s.translate(TR_LOWER).lower()


def slugify(s: str, maxlen: int = 48) -> str:
    s = s.translate(str.maketrans("ğüşıöçĞÜŞİÖÇ", "gusiocGUSIOC"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:maxlen].strip("-") or "x"


def read_json(p: Path | str, default: Any = None) -> Any:
    p = Path(p)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def write_json(p: Path | str, data: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=_json_default)
    tmp.replace(p)


def run(cmd: list[str], check: bool = True, capture: bool = True, timeout: float | None = None,
        cwd: str | Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(map(str, cmd)))
    r = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True, timeout=timeout, cwd=cwd, env=env)
    if check and r.returncode != 0:
        tail = (r.stderr or "")[-3000:]
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd[:6]))} ...\n{tail}")
    return r


def ffprobe(path: Path | str) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    return json.loads(r.stdout)


def video_info(path: Path | str) -> dict:
    j = ffprobe(path)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    info = {"duration": float(j["format"].get("duration", 0) or 0), "has_audio": a is not None}
    if v:
        num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 30.0
        if fps <= 0 or fps > 240:
            num, den = v.get("r_frame_rate", "30/1").split("/")
            fps = float(num) / float(den or 1)
        info.update(width=int(v["width"]), height=int(v["height"]), fps=fps,
                    codec=v.get("codec_name"), profile=v.get("profile"),
                    bit_rate=int(v.get("bit_rate") or 0))
    if a:
        info.update(audio_codec=a.get("codec_name"), audio_rate=int(a.get("sample_rate") or 0),
                    audio_bit_rate=int(a.get("bit_rate") or 0))
    return info


def snap_fps(fps: float) -> int:
    """Normalize odd stream fps (59.94, 29.97, 60.02) to an integer render fps."""
    for target in (24, 25, 30, 48, 50, 60):
        if abs(fps - target) < 1.2:
            return target
    return int(round(fps)) or 30


ProgressFn = Callable[[str, float, str], None]


class Progress:
    """Step-based progress reporter. Callbacks receive (step, fraction 0..1, message)."""

    STEPS = [
        ("resolve", "Yayın çözümleniyor"),
        ("profile", "Yayıncı profili"),
        ("download_preview", "Önizleme + ses indiriliyor"),
        ("chat", "Chat replay"),
        ("signals", "Ses / chat / görüntü sinyalleri"),
        ("transcribe", "Transkript"),
        ("moments", "En iyi anlar"),
        ("download_hq", "Seçilen aralıklar yüksek kalitede"),
        ("edit", "Kurgu, hook, kadraj"),
        ("render", "Render"),
        ("qa", "Denetim"),
        ("done", "Bitti"),
    ]

    def __init__(self, cb: ProgressFn | None = None):
        self.cb = cb
        self.t0 = time.time()
        self._last: tuple[str, int] = ("", -1)

    def __call__(self, step: str, frac: float = 0.0, msg: str = "") -> None:
        bucket = int(frac * 10)
        if msg or (step, bucket) != self._last:
            log.info("[%s %3d%%] %s", step, int(frac * 100), msg)
            self._last = (step, bucket)
        if self.cb:
            try:
                self.cb(step, frac, msg)
            except Exception:  # never let UI reporting break the pipeline
                log.exception("progress callback failed")

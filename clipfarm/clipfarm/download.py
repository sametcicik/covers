"""Two-stage, bandwidth-aware downloading.

Stage 1: low-res muxed preview (default 360p, carries the audio track) for the analysis window
         + 16 kHz mono wav for transcription. PREVIEW_HEIGHT=0 means audio only.
Stage 2: only the selected clip ranges at the best quality (<= MAX_HEIGHT, prefers 60 fps),
         with SECTION_PADDING seconds on both sides.

Resolution and format discovery always go through yt-dlp. Section fetching is done by a native
HLS segment fetcher (exact segment range, shared token-bucket rate limit, parallel, retried);
fallback order: yt-dlp --download-sections -> streamlink -> TwitchDownloaderCLI.
"""
from __future__ import annotations

import math
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

from .config import settings
from .platforms import UA, Source, parse_link, twitch_channel
from .util import log, run, snap_fps, video_info

# ------------------------------------------------------------------ rate limiting


def parse_rate(s: str) -> float:
    """'4M' / '500K' / '2.5MB' -> bytes per second (0 = unlimited)."""
    if not s:
        return 0.0
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMG]?)(i?B?)?\s*", s, re.I)
    if not m:
        return 0.0
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2).upper()]
    return float(m.group(1)) * mult


class TokenBucket:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, n: int) -> None:
        if self.rate <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n or self.tokens >= self.rate * 0.99:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 0.5))


# ------------------------------------------------------------------ yt-dlp resolution


def _ydl_opts(extra: dict | None = None) -> dict:
    o = {"quiet": True, "no_warnings": True, "noprogress": True, "socket_timeout": 30, "retries": 5}
    if settings.cookies_file and Path(settings.cookies_file).exists():
        o["cookiefile"] = settings.cookies_file
    if extra:
        o.update(extra)
    return o


def _extract(url: str, flat: bool = False) -> dict:
    import yt_dlp

    opts = _ydl_opts({"extract_flat": "in_playlist" if flat else False})
    if flat:
        opts["playlistend"] = 3
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=False))


def resolve(url: str) -> dict:
    """Resolve any supported link into VOD/clip metadata with format list."""
    src = parse_link(url)
    target = src.url
    note = ""
    if src.kind == "channel":
        if src.platform == "twitch":
            ch = twitch_channel(src.channel)
            arch = ch.get("latest_archive")
            if not arch:
                raise RuntimeError(f"{src.channel}: herkese açık VOD bulunamadı")
            target = f"https://www.twitch.tv/videos/{arch['id']}"
            note = "live-in-progress" if arch.get("status") == "RECORDING" else "latest-vod"
        else:
            lst = _extract(f"https://kick.com/{src.channel}/videos", flat=True)
            entries = [e for e in (lst.get("entries") or []) if e]
            if not entries:
                raise RuntimeError(f"{src.channel}: Kick VOD bulunamadı")
            target = entries[0]["url"]
            note = "live-in-progress" if not entries[0].get("duration") else "latest-vod"
        src = parse_link(target.split("#")[0]) if "kick.com" in target or "twitch.tv" in target else src
    info = _extract(target)
    fmts = []
    for f in info.get("formats") or []:
        if f.get("vcodec") in (None, "none") and f.get("acodec") in (None, "none"):
            continue
        if (f.get("ext") == "mhtml") or "storyboard" in (f.get("format_note") or ""):
            continue
        fmts.append({
            "format_id": f.get("format_id"), "height": f.get("height") or 0, "fps": f.get("fps") or 0,
            "url": f.get("url"), "protocol": f.get("protocol"), "tbr": f.get("tbr") or 0,
            "vcodec": f.get("vcodec"), "acodec": f.get("acodec"),
            "audio_only": f.get("vcodec") in (None, "none"),
        })
    channel = (info.get("uploader_id") or info.get("channel") or src.channel or "").lower()
    if src.platform == "twitch" and not channel and src.video_id:
        from .platforms import twitch_video_owner

        channel = (twitch_video_owner(src.video_id) or "").lower()
    duration = float(info.get("duration") or 0)
    return {
        "url": target,
        "input_url": url,
        "platform": src.platform,
        "kind": src.kind if src.kind != "channel" else "vod",
        "note": note,
        "id": str(info.get("id")),
        "title": info.get("title") or "",
        "channel": channel,
        "channel_display": info.get("uploader") or info.get("channel") or channel,
        "channel_id": info.get("channel_id"),
        "timestamp": info.get("timestamp") or info.get("release_timestamp"),
        "duration": duration,
        "is_live": bool(info.get("is_live")) or note == "live-in-progress",
        "formats": fmts,
    }


def pick_format(meta: dict, purpose: str) -> dict:
    fmts = [f for f in meta["formats"] if f.get("url")]
    if not fmts:
        raise RuntimeError("Hiç indirilebilir format yok (sub-only ise COOKIES_FILE ayarlayın)")
    video = [f for f in fmts if not f["audio_only"]]
    if purpose == "audio":
        aud = [f for f in fmts if f["audio_only"]]
        if aud:
            return min(aud, key=lambda f: f["tbr"] or 1e9)
        return min(video, key=lambda f: (f["height"] or 1e9, f["tbr"] or 1e9))
    if purpose == "preview":
        ph = settings.preview_height
        if ph <= 0:
            return pick_format(meta, "audio")
        ok = [f for f in video if f["height"] and f["height"] <= ph and f.get("acodec") not in (None, "none")] or \
             [f for f in video if f["height"] and f["height"] <= ph]
        if ok:
            return max(ok, key=lambda f: (f["height"], -(f["tbr"] or 0)))
        return min(video, key=lambda f: f["height"] or 1e9)
    # hq
    ok = [f for f in video if (f["height"] or 0) <= settings.max_height] or video
    return max(ok, key=lambda f: ((f["height"] or 0), min(f["fps"] or 30, settings.max_fps), f["tbr"] or 0))


# ------------------------------------------------------------------ HLS section fetcher


@dataclass
class Segment:
    uri: str
    start: float
    dur: float
    init: str | None = None


def _get(client: httpx.Client, url: str) -> str:
    r = client.get(url)
    r.raise_for_status()
    return r.text


def media_segments(client: httpx.Client, url: str) -> list[Segment]:
    text = _get(client, url)
    if "#EXT-X-STREAM-INF" in text:  # master playlist: take highest bandwidth variant
        best, bw = None, -1
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("#EXT-X-STREAM-INF"):
                m = re.search(r"BANDWIDTH=(\d+)", ln)
                b = int(m.group(1)) if m else 0
                if b > bw and i + 1 < len(lines):
                    best, bw = urljoin(url, lines[i + 1].strip()), b
        if not best:
            raise RuntimeError("empty master playlist")
        return media_segments(client, best)
    segs: list[Segment] = []
    t = 0.0
    dur = None
    init = None
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("#EXTINF:"):
            dur = float(ln[8:].split(",")[0])
        elif ln.startswith("#EXT-X-MAP:"):
            m = re.search(r'URI="([^"]+)"', ln)
            if m:
                init = urljoin(url, m.group(1))
        elif ln and not ln.startswith("#") and dur is not None:
            segs.append(Segment(urljoin(url, ln), t, dur, init))
            t += dur
            dur = None
    return segs


def fetch_hls_section(m3u8: str, start: float, end: float, out: Path, rate: float = 0.0,
                      threads: int = 4, progress=None) -> float:
    """Download only the HLS segments covering [start, end]. Returns the absolute start time of `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    bucket = TokenBucket(rate)
    with httpx.Client(headers={"User-Agent": UA}, timeout=60, follow_redirects=True) as client:
        segs = media_segments(client, m3u8)
        if not segs:
            raise RuntimeError("no segments")
        sel = [s for s in segs if s.start + s.dur > start and s.start < end]
        if not sel:
            raise RuntimeError(f"section {start:.0f}-{end:.0f} outside playlist (len {segs[-1].start + segs[-1].dur:.0f}s)")
        tmpdir = out.parent / (out.stem + "_segs")
        tmpdir.mkdir(exist_ok=True)
        done = [0]

        def dl(idx_seg: tuple[int, Segment]) -> Path:
            i, s = idx_seg
            p = tmpdir / f"{i:06d}.bin"
            if p.exists() and p.stat().st_size > 0:
                done[0] += 1
                return p
            last = None
            for attempt in range(5):
                try:
                    with client.stream("GET", s.uri) as r:
                        if r.status_code == 403 and "-unmuted" in s.uri:
                            s.uri = s.uri.replace("-unmuted", "-muted")
                            raise httpx.HTTPError("muted segment")
                        r.raise_for_status()
                        with open(p.with_suffix(".part"), "wb") as f:
                            for chunk in r.iter_bytes(65536):
                                bucket.consume(len(chunk))
                                f.write(chunk)
                    p.with_suffix(".part").replace(p)
                    done[0] += 1
                    if progress:
                        progress(done[0] / len(sel))
                    return p
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(1.5 * (attempt + 1))
            raise RuntimeError(f"segment failed: {s.uri[:80]}: {last}")

        with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            parts = list(ex.map(dl, list(enumerate(sel))))
        raw = out.with_suffix(".raw")
        with open(raw, "wb") as f:
            if sel[0].init:
                f.write(client.get(sel[0].init).content)
            for p in parts:
                with open(p, "rb") as g:
                    shutil.copyfileobj(g, f)
    run(["ffmpeg", "-y", "-v", "error", "-fflags", "+genpts+discardcorrupt", "-i", raw,
         "-map", "0:v?", "-map", "0:a?", "-c", "copy", "-bsf:a", "aac_adtstoasc", out])
    raw.unlink(missing_ok=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return sel[0].start


# ------------------------------------------------------------------ fallbacks


def tool(name: str) -> str | None:
    """Find a CLI tool: venv bin first (yt-dlp, streamlink), then ./bin, then PATH."""
    import sys

    for d in (Path(sys.executable).parent, Path(__file__).resolve().parent.parent / "bin"):
        p = d / name
        if p.exists():
            return str(p)
    return shutil.which(name)


def _hms(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def ytdlp_section(url: str, fmt_id: str, start: float, end: float, out: Path) -> float:
    cmd = ["yt-dlp", "-f", fmt_id, "--download-sections", f"*{_hms(start)}-{_hms(end)}",
           "-o", str(out), "--no-part", "--force-overwrites", "-N", str(settings.concurrent_fragments)]
    if settings.rate_limit:
        cmd += ["--limit-rate", settings.rate_limit]
    if settings.cookies_file:
        cmd += ["--cookies", settings.cookies_file]
    cmd[0] = tool("yt-dlp") or "yt-dlp"
    run(cmd + [url], timeout=7200)
    return start


def streamlink_section(url: str, start: float, end: float, out: Path) -> float:
    exe = tool("streamlink") or "streamlink"
    raw = out.with_suffix(".ts")
    run([exe, "--hls-start-offset", _hms(start), "--hls-duration", _hms(end - start), "-f", "-o", raw, url, "best"],
        timeout=7200)
    run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-c", "copy", out])
    raw.unlink(missing_ok=True)
    return start


def twitchdownloader_section(meta: dict, start: float, end: float, out: Path) -> float:
    exe = tool("TwitchDownloaderCLI")
    if not exe or meta["platform"] != "twitch":
        raise RuntimeError("TwitchDownloaderCLI not available")
    run([exe, "videodownload", "--id", meta["id"].lstrip("v"), "-b", str(int(start)), "-e", str(int(math.ceil(end))),
         "-q", "1080p60", "-o", out, "--collision", "Overwrite"], timeout=7200)
    return float(int(start))


def download_section(meta: dict, purpose: str, start: float, end: float, out: Path, progress=None) -> dict:
    fmt = pick_format(meta, purpose)
    rate = parse_rate(settings.rate_limit)
    errors = []
    if fmt.get("protocol", "").startswith("m3u8"):
        try:
            t0 = fetch_hls_section(fmt["url"], start, end, out, rate, settings.concurrent_fragments, progress)
            return {"path": str(out), "start": t0, "format": fmt["format_id"], "method": "hls"}
        except Exception as e:  # noqa: BLE001
            errors.append(f"hls: {e}")
            log.warning("HLS fetch failed, falling back to yt-dlp: %s", e)
    for name, fn in (
        ("yt-dlp", lambda: ytdlp_section(meta["url"], fmt["format_id"], start, end, out)),
        ("streamlink", lambda: streamlink_section(meta["url"], start, end, out)),
        ("twitchdownloader", lambda: twitchdownloader_section(meta, start, end, out)),
    ):
        try:
            t0 = fn()
            if out.exists() and out.stat().st_size > 0:
                return {"path": str(out), "start": t0, "format": fmt["format_id"], "method": name}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            log.warning("%s failed: %s", name, str(e)[:300])
    raise RuntimeError("Tüm indirme yöntemleri başarısız:\n" + "\n".join(errors))


# ------------------------------------------------------------------ stage helpers


def stage1(meta: dict, start: float, end: float, src_dir: Path, progress=None) -> dict:
    """Preview video (+ its audio) for the analysis window, and a 16 kHz mono wav."""
    src_dir.mkdir(parents=True, exist_ok=True)
    prev = src_dir / "preview.mp4"
    info_p = src_dir / "preview.json"
    from .util import read_json, write_json

    res = read_json(info_p)
    if not (res and prev.exists() and abs(res.get("req_start", -1) - start) < 1 and abs(res.get("req_end", -1) - end) < 1):
        res = download_section(meta, "preview", start, end, prev, progress)
        res.update(req_start=start, req_end=end)
        write_json(info_p, res)
    wav = src_dir / "audio16k.wav"
    if not wav.exists():
        run(["ffmpeg", "-y", "-v", "error", "-i", prev, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
    vi = video_info(prev)
    res.update(audio=str(wav), duration=vi["duration"], has_video="width" in vi,
               width=vi.get("width"), height=vi.get("height"))
    write_json(info_p, res)
    return res


def stage2(meta: dict, start: float, end: float, out_dir: Path, progress=None) -> dict:
    """HQ section with padding, normalized to CFR / short GOP for frame-accurate rendering."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = settings.section_padding
    s, e = max(0.0, start - pad), end + pad
    if meta.get("duration"):
        e = min(e, meta["duration"])
    final = out_dir / "hq.mp4"
    info_p = out_dir / "hq.json"
    from .util import read_json, write_json

    cached = read_json(info_p)
    if cached and final.exists() and cached.get("req_start") == s and cached.get("req_end") == e:
        return cached
    raw = out_dir / "hq_raw.mp4"
    res = download_section(meta, "hq", s, e, raw, progress)
    vi = video_info(raw)
    fps = min(snap_fps(vi.get("fps", 30)), settings.max_fps)
    offset = max(0.0, s - res["start"])
    dur = e - s
    from .hardware import detect

    hw = detect()
    venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "17", "-g", str(fps // 2)] if hw.nvenc_ok else \
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-g", str(max(1, fps // 2)), "-bf", "0"]
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}", "-i", raw, "-t", f"{dur:.3f}",
         "-vf", f"fps={fps}", *venc, "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", final], timeout=7200)
    raw.unlink(missing_ok=True)
    vi = video_info(final)
    out = {"path": str(final), "start": s, "end": e, "req_start": s, "req_end": e, "fps": fps,
           "width": vi.get("width"), "height": vi.get("height"), "format": res["format"], "method": res["method"]}
    write_json(info_p, out)
    return out


def download_image(url: str, out: Path) -> Path | None:
    try:
        r = httpx.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return out
    except Exception:
        return None


def _which_tools() -> dict:
    return {t: tool(t) for t in ("yt-dlp", "streamlink", "TwitchDownloaderCLI", "ffmpeg", "node")}


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(resolve(sys.argv[1]))

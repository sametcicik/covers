"""Streamer profiles: streamers/<slug>/profile.json, channel art, 5-colour palette + palette.png."""
from __future__ import annotations

import colorsys
import datetime as dt
from pathlib import Path

import numpy as np

from .config import FONTS_DIR, STREAMERS_DIR
from .download import download_image
from .platforms import channel_info
from .util import log, read_json, write_json

DEFAULT_PROFILE = {
    "language": "auto",
    "facecam": None,
    "palette": None,
    "dictionary": {"terms": [], "replace": {}},
    "fonts": {"set": "default", "exclude": []},
    "captions": {"style": "dynamic", "position": "auto"},
    "label": {"enabled": True},
    "hook": {"enabled": True},
    "sfx": {"density": 2},
    "music": {"enabled": False},
    "campaign": {
        "max_duration": 59,
        "target_duration": [20, 45],
        "required_hashtags": [],
        "required_mentions": [],
        "censor_profanity": False,
        "cta": "",
    },
}


def profile_dir(platform: str, slug: str) -> Path:
    return STREAMERS_DIR / f"{slug}" if platform == "kick" else STREAMERS_DIR / f"{slug}"


def load(platform: str, slug: str) -> dict | None:
    return read_json(profile_dir(platform, slug) / "profile.json")


def save(profile: dict) -> None:
    d = profile_dir(profile["platform"], profile["slug"])
    profile["updated"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    write_json(d / "profile.json", profile)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure(platform: str, slug: str, display_name: str | None = None, refresh_art: bool = False) -> dict:
    """Load or auto-create the profile; fetch channel art and build the palette on first sight."""
    existing = load(platform, slug) or {}
    prof = _deep_merge(DEFAULT_PROFILE, existing)
    prof.update(platform=platform, slug=slug)
    d = profile_dir(platform, slug)
    d.mkdir(parents=True, exist_ok=True)
    first = not existing
    if first or refresh_art or not prof.get("palette"):
        info = channel_info(platform, slug)
        prof["display_name"] = info.get("display_name") or display_name or slug
        prof.setdefault("created", dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
        art = []
        for key in ("avatar", "banner", "offline"):
            if info.get(key):
                p = download_image(info[key], d / "art" / f"{key}.img")
                if p:
                    art.append((key, p))
        for i, url in enumerate((info.get("panels") or [])[:6]):
            p = download_image(url, d / "art" / f"panel_{i}.img")
            if p:
                art.append((f"panel_{i}", p))
        prof["art"] = {k: str(p.relative_to(d)) for k, p in art}
        if not (prof.get("palette") or {}).get("locked"):
            prof["palette"] = build_palette([p for _, p in art], info.get("brand_color"))
        terms = prof["dictionary"].setdefault("terms", [])
        for t in (prof["display_name"], slug):
            if t and t not in terms:
                terms.insert(0, t)
        render_palette_png(prof["palette"], d / "palette.png", prof["display_name"])
        save(prof)
        log.info("profile %s/%s %s", platform, slug, "created" if first else "refreshed")
    elif display_name and not prof.get("display_name"):
        prof["display_name"] = display_name
    if not (d / "palette.png").exists() and prof.get("palette"):
        render_palette_png(prof["palette"], d / "palette.png", prof.get("display_name", slug))
    return prof


# ------------------------------------------------------------------ colour science


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(max(0, min(255, round(c)))) for c in rgb)


def _lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h: str) -> float:
    r, g, b = hex_to_rgb(h)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _hls(h: str):
    r, g, b = (c / 255 for c in hex_to_rgb(h))
    return colorsys.rgb_to_hls(r, g, b)


def _from_hls(hh, l, s) -> str:
    return rgb_to_hex([c * 255 for c in colorsys.hls_to_rgb(hh % 1.0, max(0, min(1, l)), max(0, min(1, s)))])


def _ensure_contrast(fg: str, bg: str, target: float, lighten: bool = True) -> str:
    h, l, s = _hls(fg)
    for _ in range(40):
        if contrast(fg, bg) >= target:
            return fg
        l = l + 0.025 if lighten else l - 0.025
        fg = _from_hls(h, l, s)
    return fg


def _vivid(h: str, min_s: float = 0.65, l_range=(0.5, 0.66)) -> str:
    hh, l, s = _hls(h)
    return _from_hls(hh, min(max(l, l_range[0]), l_range[1]), max(s, min_s))


def dominant_colors(images: list[Path], k: int = 8) -> list[dict]:
    import cv2

    px = []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            try:
                from PIL import Image

                img = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                continue
        img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
        px.append(img.reshape(-1, 3))
    if not px:
        return []
    data = np.concatenate(px).astype(np.float32)
    k = min(k, len(data))
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 4,
                                    cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    out = []
    for c, n in zip(centers, counts):
        hexc = rgb_to_hex((c[2], c[1], c[0]))
        hh, l, s = _hls(hexc)
        out.append({"hex": hexc, "share": float(n / counts.sum()), "h": hh, "l": l, "s": s,
                    "vivid": s * (1 - abs(l - 0.5) * 2)})
    return sorted(out, key=lambda x: -x["share"])


def build_palette(images: list[Path], brand_hex: str | None = None) -> dict:
    cols = dominant_colors(images)
    vivid = sorted([c for c in cols if c["vivid"] > 0.18], key=lambda c: -(c["share"] ** 0.5 * c["vivid"]))
    if brand_hex:
        primary = _vivid(brand_hex)
        src = "brand+art"
    elif vivid:
        primary = _vivid(vivid[0]["hex"])
        src = "art"
    else:
        primary = "#7C5CFF"
        src = "default"
    ph, _, _ = _hls(primary)

    def hue_dist(a, b):
        d = abs(a - b) % 1.0
        return min(d, 1 - d)

    acc = [c for c in vivid if hue_dist(c["h"], ph) > 0.12]
    accent = _vivid(acc[0]["hex"], 0.75, (0.55, 0.68)) if acc else _from_hls(ph + 0.42, 0.6, 0.85)
    ah, _, _ = _hls(accent)
    acc2 = [c for c in vivid if hue_dist(c["h"], ph) > 0.08 and hue_dist(c["h"], ah) > 0.08]
    accent2 = _vivid(acc2[0]["hex"], 0.7, (0.55, 0.7)) if acc2 else _from_hls(ph - 0.12, 0.66, 0.9)
    stroke = _from_hls(ph, 0.06, 0.35)
    text = _from_hls(ph, 0.975, 0.4)
    # contrast: caption colours sit on a dark stroke -> need >= 4.5:1
    accent = _ensure_contrast(accent, stroke, 6.0)
    accent2 = _ensure_contrast(accent2, stroke, 6.0)
    primary_ui = _ensure_contrast(primary, "#0B0B0F", 4.5)
    return {
        "primary": primary_ui, "accent": accent, "accent2": accent2, "text": text, "stroke": stroke,
        "source": src, "locked": False,
        "contrast": {k: round(contrast(v, stroke), 2) for k, v in
                     (("primary", primary_ui), ("accent", accent), ("accent2", accent2), ("text", text))},
        "extracted": [c["hex"] for c in cols[:8]],
    }


def render_palette_png(pal: dict, out: Path, name: str = "") -> None:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1200, 360
    img = Image.new("RGB", (W, H), "#0B0B0F")
    dr = ImageDraw.Draw(img)
    try:
        f_big = ImageFont.truetype(str(FONTS_DIR / "main_archivo_black.ttf"), 30)
        f_small = ImageFont.truetype(str(FONTS_DIR / "mono_jetbrains_xbold.ttf"), 20)
    except Exception:
        f_big = f_small = ImageFont.load_default()
    dr.text((32, 22), f"{name}  palette", font=f_big, fill="#F2F2F5")
    keys = ["primary", "accent", "accent2", "text", "stroke"]
    sw = (W - 64 - 4 * 16) // 5
    for i, k in enumerate(keys):
        x = 32 + i * (sw + 16)
        dr.rounded_rectangle([x, 80, x + sw, 280], 18, fill=pal[k], outline="#2A2A33", width=2)
        dr.text((x, 292), k, font=f_small, fill="#9A9AA6")
        dr.text((x, 318), pal[k], font=f_small, fill="#F2F2F5")
        if k not in ("stroke",):
            c = pal.get("contrast", {}).get(k)
            if c:
                dr.text((x + 14, 244), f"{c}:1", font=f_small, fill=pal["stroke"] if luminance(pal[k]) > 0.3 else "#FFFFFF")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)

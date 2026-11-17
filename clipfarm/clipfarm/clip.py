"""Assemble one clip: EDL + layout + captions + effects -> Remotion props (same props drive preview and export)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import captions as cap
from . import edit, layout
from .moments import END_PUNCT, Signals
from .util import read_json, write_json

DEFAULT_SETTINGS = {
    "caption_style": "dynamic",
    "captions": True,
    "hook": True,
    "sfx_density": 2,
    "effects": True,
    "label": True,
    "palette": None,  # override dict
    "upper": None,
}


def clip_seed(job_id: str, clip_id: str) -> int:
    return int(hashlib.sha1(f"{job_id}/{clip_id}".encode()).hexdigest()[:8], 16)


def caption_y(lay: dict, mode: str, sizes: dict, face_out: list[float] | None, block_h: float = 260) -> int:
    """Caption block centre: never over the face, inside the platform-safe band (y 15%..72%)."""
    H = layout.OUT_H
    lo, hi = int(H * 0.15 + block_h / 2), int(H * 0.72 - block_h / 2)
    if mode == "split":
        y = sizes["top_h"] + sizes["divider"] + 250  # upper area of the game panel, just below the facecam
    elif face_out:
        fy1 = face_out[1] + face_out[3]
        y = max(int(fy1 + 90 + block_h / 2), int(H * 0.6))
    else:
        y = int(H * 0.6)
    return int(min(max(y, lo), hi))


def out_words(words: list[dict], fsegs: list[dict], fps: int, emph_scores: list[float]) -> list[dict]:
    res = []
    idx = {id(w): i for i, w in enumerate(words)}
    for si, s in enumerate(fsegs):
        a = s["absStart"]
        b = a + (s["outEnd"] - s["outStart"]) / fps
        seg_words = [w for w in words if w["s"] >= a - 0.02 and w["s"] < b - 0.02]
        for k, w in enumerate(seg_words):
            i = idx[id(w)]
            st = (s["outStart"] / fps + (w["s"] - a))
            en = (s["outStart"] / fps + (min(w["e"], b) - a))
            nxt_gap = (words[i + 1]["s"] - w["e"]) if i + 1 < len(words) else 9
            res.append({"w": w["w"], "startMs": int(round(max(0, st) * 1000)), "endMs": int(round(max(st + 0.05, en) * 1000)),
                        "seg": si, "emph": emph_scores[i], "p": w.get("p"),
                        "sent_end": bool(END_PUNCT.search(w["w"])) or nxt_gap > 0.6 or k == len(seg_words) - 1,
                        "abs": w["s"]})
    return res


def build(job: dict, moment: dict, clip_dir: Path, hq: dict, profile: dict, words: list[dict], sig: Signals,
          lay: dict, preview: dict, settings: dict) -> dict:
    """Returns props (also written to clip_dir/props.json) and the edit plan."""
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    fps = int(hq["fps"])
    seed = clip_seed(job["id"], clip_dir.name)
    camp = profile.get("campaign") or {}
    max_total = float(min(59.0, camp.get("max_duration", 59)))
    segs = edit.plan_segments(moment, words, sig, bool(st["hook"]), max_total)
    fsegs = edit.to_frames(segs, hq["start"], fps)
    total = fsegs[-1]["outEnd"]

    # captions
    from .transcribe import clean_words

    clip_words = clean_words([w for w in words if moment["start"] - 3 <= w["s"] <= moment["end"] + 3],
                             profile.get("language_detected") or profile.get("language"))
    scores = cap.emphasis_scores(clip_words, moment.get("emphasis") or [],
                                 (profile.get("dictionary") or {}).get("terms", []), tuple(moment["peak"]))
    ow = out_words(clip_words, fsegs, fps, scores)
    off = int(st.get("caption_offset_ms") or 0)
    if off:
        for w in ow:
            w["startMs"] = max(0, w["startMs"] + off)
            w["endMs"] = max(w["startMs"] + 40, w["endMs"] + off)
    palette = {**(profile.get("palette") or {}), **(st.get("palette") or {})}
    pages = cap.build_pages(ow, st["caption_style"], palette, seed, st.get("upper"),
                            bool(camp.get("censor_profanity")))
    # emphasis words -> frames for effect planning
    emph = []
    for p in pages["pages"]:
        for ln in p["lines"]:
            for w in ln["words"]:
                if w["emph"]:
                    emph.append({"frames": [int(round(w["startMs"] / 1000 * fps))], "dur": (w["endMs"] - w["startMs"]) / 1000})
    fx = edit.plan_effects(moment, fsegs, words, sig, fps, emph, int(st["sfx_density"]) if st["effects"] else 0, seed)

    # layout
    src_w, src_h = hq["width"], hq["height"]
    mode = lay["mode"]
    top_ratio = layout.top_ratio_for(lay, src_h)
    sizes = layout.panel_sizes(top_ratio)
    prev_path = Path(preview["path"])
    file_start = hq["start"] - preview["start"]
    file_end = hq["end"] - preview["start"]
    track_key = [mode, round(file_start, 2), round(file_end, 2), top_ratio, lay.get("box")]
    cached = read_json(clip_dir / "track.json") or read_json(clip_dir.parent / "track.json")
    if cached and cached.get("key") == track_key:
        tr = cached["track"]
    elif preview.get("has_video"):
        tr = layout.track(prev_path, file_start, file_end, lay, mode, top_ratio=top_ratio)
        write_json(clip_dir / "track.json", {"key": track_key, "track": tr})
    else:
        aspect = layout.OUT_W / (sizes["bottom_h"] if mode == "split" else layout.OUT_H)
        w = min(1.0, src_h * aspect / src_w)
        tr = {"w": w, "h": 1.0, "keys": [{"t": 0, "x": (1 - w) / 2, "y": 0}]}
    facecam = layout.facecam_crop(lay, src_w, src_h, top_ratio) if mode == "split" and lay.get("box") else None
    face_out = None
    if mode == "cam" and lay.get("face"):
        fx_, fy_, fw_, fh_ = lay["face"]
        face_out = [0, fy_ * layout.OUT_H, 0, fh_ * layout.OUT_H]
    y = caption_y(lay, mode if facecam or mode != "split" else "game", sizes, face_out)
    y = int(min(layout.OUT_H * 0.7, max(layout.OUT_H * 0.2, y + int(st.get("caption_y_shift") or 0))))

    props = {
        "fps": fps, "width": layout.OUT_W, "height": layout.OUT_H, "durationInFrames": total,
        "video": {"src": "", "width": src_w, "height": src_h, "fps": fps},
        "muteVideo": True,
        "segments": [{k: s[k] for k in ("kind", "srcStart", "srcEnd", "outStart", "outEnd")} for s in fsegs],
        "layout": {"mode": mode if (mode != "split" or facecam) else "game", "topH": sizes["top_h"],
                   "divider": sizes["divider"], "bottomH": sizes["bottom_h"], "facecam": facecam,
                   "track": {"w": tr["w"], "h": tr["h"], "keys": tr["keys"]}},
        "effects": {"zooms": fx["zooms"], "base": fx["base"], "shakes": fx["shakes"], "flashes": fx["flashes"],
                    "enabled": bool(st["effects"])},
        "captions": {"enabled": bool(st["captions"]), "style": st["caption_style"], "y": y, "pages": pages["pages"],
                     "tokens": cap.caption_tokens(ow), "fonts": pages["fonts"], "neutralFont": pages["neutralFont"]},
        "fontBaseUrl": "", "sfxBaseUrl": "",
        "sfx": [{"cat": e["cat"], "file": e["file"], "start": e["start"], "gainDb": -12.0} for e in fx["sfx"]],
        "playSfx": False,
        "palette": {k: palette[k] for k in ("primary", "accent", "accent2", "text", "stroke")},
        "label": {"enabled": bool(st["label"]) and bool((profile.get("label") or {}).get("enabled", True)),
                  "platform": job["platform"], "name": profile.get("display_name") or job["channel"]},
        "seed": seed % 100000,
    }
    plan = {"segments": fsegs, "sfx": fx["sfx"], "settings": st, "moment": moment, "out_words": ow,
            "words_abs": clip_words, "duration": total / fps, "fps": fps}
    write_json(clip_dir / "props.json", props)
    write_json(clip_dir / "plan.json", plan)
    return {"props": props, "plan": plan}


def load(clip_dir: Path) -> tuple[dict, dict]:
    return read_json(clip_dir / "props.json"), read_json(clip_dir / "plan.json")


def caption_rects(props: dict, t_ms: float) -> list[list[float]]:
    """Approximate on-screen rectangle of the caption page visible at t (for QA)."""
    c = props["captions"]
    if not c["enabled"]:
        return []
    page = next((p for p in c["pages"] if p["startMs"] <= t_ms < p["endMs"]), None)
    if not page:
        return []
    h = sum(max(w["size"] for w in ln["words"]) * 1.1 for ln in page["lines"])
    w = max(ln["width"] for ln in page["lines"]) + 40
    cx = layout.OUT_W / 2
    return [[cx - w / 2, c["y"] - h / 2, w, h]]


def mean_conf(words: list[dict]) -> float:
    ps = [w.get("p") or 0 for w in words]
    return float(np.mean(ps)) if ps else 0.0

"""Self-review after every render: automatic checks + optional LLM vision review of 8-10 frames."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import audio, layout, llm
from .captions import _metrics, fonts
from .clip import caption_rects
from .moments import END_PUNCT
from .render import extract_frames
from .util import log, run, video_info

MOJIBAKE = re.compile(r"Ã.|Ä.|Å.|�")


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / max(1e-6, bw * bh)


def check_turkish_glyphs(props: dict) -> list[str]:
    bad = []
    for p in props["captions"]["pages"]:
        for ln in p["lines"]:
            for w in ln["words"]:
                if MOJIBAKE.search(w["text"]):
                    bad.append(f"mojibake: {w['text']}")
                if w["font"] not in fonts():
                    continue
                cmap, _, _ = _metrics(w["font"])
                for ch in w["text"]:
                    if not ch.isspace() and ord(ch) not in cmap:
                        bad.append(f"missing glyph '{ch}' in {w['font']}")
    return sorted(set(bad))


def sync_check(final: Path, plan: dict, props: dict) -> dict:
    """Re-transcribe the rendered audio and compare word onsets against caption timings."""
    from .transcribe import get_model

    tmp = final.parent / "audio" / "qa16k.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", final, "-ac", "1", "-ar", "16000", tmp])
    import soundfile as sf

    a, _ = sf.read(str(tmp), dtype="float32")
    segs, _ = get_model().transcribe(a, language=None, word_timestamps=True, vad_filter=True, beam_size=1,
                                     condition_on_previous_text=False)
    heard = [(w.word.strip(), w.start * 1000) for s in segs for w in (s.words or []) if w.word.strip()]
    cap = [(t["text"].strip(), t["startMs"]) for t in props["captions"]["tokens"]]

    def norm(x):
        return re.sub(r"[^\w]", "", x.lower())

    import difflib

    a_words = [norm(w) for w, _ in heard]
    b_words = [norm(w) for w, _ in cap]
    sm = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    diffs = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            ha, cb = heard[blk.a + k], cap[blk.b + k]
            if a_words[blk.a + k] and abs(cb[1] - ha[1]) < 700:
                diffs.append(cb[1] - ha[1])
    if len(diffs) < 4:
        return {"matched": len(diffs), "median_abs_ms": None, "offset_ms": None}
    d = np.array(diffs)
    return {"matched": len(diffs), "of": len(cap), "median_abs_ms": round(float(np.median(np.abs(d))), 1),
            "offset_ms": round(float(np.median(d)), 1), "p90_abs_ms": round(float(np.percentile(np.abs(d), 90)), 1)}


def sentence_check(plan: dict) -> list[str]:
    ws = plan["words_abs"]
    m = plan["moment"]
    issues = []
    inside = [w for w in ws if m["start"] <= w["s"] < m["end"]]
    if not inside:
        return ["no words in clip"]
    first, last = inside[0], inside[-1]
    i0 = ws.index(first)
    if i0 > 0:
        prev = ws[i0 - 1]
        if not END_PUNCT.search(prev["w"]) and first["s"] - prev["e"] < 0.3:
            issues.append(f"starts mid-sentence at '{first['w']}'")
    i1 = ws.index(last)
    if not END_PUNCT.search(last["w"]) and i1 + 1 < len(ws) and ws[i1 + 1]["s"] - last["e"] < 0.3:
        issues.append(f"ends mid-sentence at '{last['w']}'")
    for p in plan.get("pages_check", []):
        pass
    return issues


def black_frames(final: Path) -> list[list[float]]:
    r = run(["ffmpeg", "-hide_banner", "-i", final, "-vf", "blackdetect=d=0.08:pix_th=0.06", "-an", "-f", "null", "-"])
    return [[float(a), float(b)] for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]


VISION_SYS = """You are a strict QA reviewer for vertical short-form clips (TikTok/Reels/Shorts, 1080x1920). \
You get frames from one clip. Check: captions readable and not covering faces, captions not cut off, no broken \
Turkish characters (ğ ş ı İ ö ü ç), crop not cutting heads, no black/frozen frames, layout looks premium (not a cheap \
template), nothing in the bottom 20% / right edge UI zones. Return JSON only: \
{"pass": true|false, "score": 0-100, "issues": ["..."], "fix": ["caption_up"|"caption_down"|"recrop"|"none"]}"""


def vision_review(frames: list[Path]) -> dict | None:
    if not llm.enabled():
        return None
    try:
        return llm.chat_json(VISION_SYS, "Review these frames in order.", images=frames,
                             model=None, max_tokens=800)
    except Exception as e:  # noqa: BLE001
        log.warning("vision review failed: %s", e)
        return None


def run_qa(clip_dir: Path, props: dict, plan: dict, max_dur: float = 59.5) -> dict:
    final = clip_dir / "final.mp4"
    res: dict = {"checks": {}, "issues": [], "fixes": []}
    vi = video_info(final)
    dur = vi["duration"]
    res["checks"]["duration"] = round(dur, 2)
    if dur >= 60 or dur > max_dur + 0.5:
        res["issues"].append(f"duration {dur:.1f}s >= 60s")
        res["fixes"].append("shorten")
    res["checks"]["video"] = {k: vi.get(k) for k in ("width", "height", "fps", "codec", "profile", "bit_rate")}
    if (vi.get("width"), vi.get("height")) != (1080, 1920):
        res["issues"].append("resolution not 1080x1920")
    # loudness
    lm = audio.measure(final)
    res["checks"]["loudness"] = lm
    if lm["lufs"] is None or abs(lm["lufs"] + 14) > 1.0:
        res["issues"].append(f"loudness {lm['lufs']} LUFS (target -14)")
        res["fixes"].append("remix")
    if lm["true_peak"] is not None and lm["true_peak"] > -0.9:
        res["issues"].append(f"true peak {lm['true_peak']} dBTP > -1")
        res["fixes"].append("remix")
    # black frames
    bl = black_frames(final)
    res["checks"]["black"] = bl
    if bl:
        res["issues"].append(f"black frames at {bl[:3]}")
    # glyphs
    g = check_turkish_glyphs(props)
    res["checks"]["glyphs"] = g
    if g:
        res["issues"].extend(g[:5])
        res["fixes"].append("glyph")
    # sentence boundaries
    sb = sentence_check(plan)
    res["checks"]["sentence"] = sb
    res["issues"].extend(sb)
    # caption sync
    try:
        sync = sync_check(final, plan, props)
    except Exception as e:  # noqa: BLE001
        sync = {"error": str(e)}
    res["checks"]["sync"] = sync
    if sync.get("median_abs_ms") is not None and sync["median_abs_ms"] > 80:
        res["issues"].append(f"caption sync deviation {sync['median_abs_ms']} ms > 80")
    elif sync.get("median_abs_ms") is None and len(props["captions"]["tokens"]) >= 12 and "error" not in sync:
        res["issues"].append("caption sync could not be verified (too few matching words)")
    # frames: faces vs captions, safe area
    import cv2

    n = 10
    times = list(np.linspace(0.3, max(0.4, dur - 0.4), n))
    frames = extract_frames(final, times, clip_dir / "qa_frames", width=540)
    face_hits = []
    for t, fp in zip(times, frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        sc = layout.OUT_W / img.shape[1]
        faces = [[v * sc for v in f["box"]] for f in layout.detect_faces(img) if f["score"] > 0.8]
        if props["layout"]["mode"] == "split":  # the streamer is in the facecam panel; game characters don't count
            faces = [f for f in faces if f[1] + f[3] / 2 < props["layout"]["topH"]]
        for r in caption_rects(props, t * 1000):
            for f in faces:
                if _overlap(r, f) > 0.15:
                    face_hits.append(round(t, 2))
            x, y, w, h = r
            if y + h > layout.OUT_H * 0.76 or y < layout.OUT_H * 0.12 or x < 40 or x + w > layout.OUT_W - 40:
                res["issues"].append(f"caption outside safe area at {t:.1f}s")
                res["fixes"].append("caption_safe")
    res["checks"]["face_overlap_times"] = face_hits
    if face_hits:
        res["issues"].append(f"captions cover a face at {face_hits[:4]}")
        res["fixes"].append("caption_move")
    res["frames"] = [str(p) for p in frames]
    v = vision_review(frames[:9])
    res["vision"] = v
    if v and not v.get("pass", True):
        res["issues"].extend(f"vision: {x}" for x in v.get("issues", [])[:5])
        res["fixes"].extend(v.get("fix") or [])
    res["passed"] = not res["issues"]
    res["fixes"] = sorted(set(res["fixes"]))
    return res

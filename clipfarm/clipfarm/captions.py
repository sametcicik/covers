"""Caption pages for the Remotion caption engine.

- 1-4 words per page, never across a sentence boundary or the cold-open cut.
- Neutral words in the clip's main font; emphasis words (LLM-marked or heuristic: names, numbers, shock,
  punchline) in a contrasting accent font, bigger, in the accent colour.
- Seeded rhythm: emphasis fonts cycle through a seeded order, accent colours alternate, so two consecutive
  pages never share the same combination, yet the clip has a steady pattern instead of random chaos.
- Font sizes are fitted with the real font metrics (fontTools) so text never exceeds the safe width.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache

from .config import FONTS_DIR
from .moments import END_PUNCT, PROFANITY, SHOCK
from .util import read_json, tr_lower, tr_upper

SAFE_W = 840  # px of 1080 (keeps clear of the right-hand action icons)
BASE_PX = 92
EMPH_RATIO = 1.42
MIN_PX = 52

STYLE_PRESETS = {
    # neutral font role(s), accent rotation, uppercase neutral, words per page
    "dynamic": {"neutral": ["main_archivo_black"], "accents": ["serif_instrument_italic", "marker_caveat_brush",
                "condensed_anton", "mono_jetbrains_xbold", "display_unbounded_black"], "upper": False, "max_words": 4},
    "bold": {"neutral": ["main_montserrat_black"], "accents": ["condensed_anton", "serif_instrument_italic",
             "display_unbounded_black"], "upper": True, "max_words": 3},
    "clean": {"neutral": ["main_archivo_black"], "accents": ["main_archivo_black"], "upper": False, "max_words": 4},
    "classic": {"neutral": ["main_montserrat_black"], "accents": ["main_montserrat_black"], "upper": True, "max_words": 3},
}


@lru_cache(maxsize=1)
def fonts() -> dict[str, dict]:
    d = read_json(FONTS_DIR / "fonts.json", {"fonts": []})
    return {f["id"]: f for f in d.get("fonts", []) if f.get("turkish_ok", True)}


@lru_cache(maxsize=16)
def _metrics(font_id: str):
    from fontTools.ttLib import TTFont

    f = fonts()[font_id]
    tt = TTFont(str(FONTS_DIR / f["file"]))
    cmap = tt.getBestCmap()
    hmtx = tt["hmtx"]
    upm = tt["head"].unitsPerEm
    return cmap, hmtx, upm


def text_width(text: str, font_id: str, px: float) -> float:
    cmap, hmtx, upm = _metrics(font_id)
    f = fonts()[font_id]
    ls = f.get("letter_spacing_em", 0) * px
    w = 0.0
    for ch in text:
        g = cmap.get(ord(ch)) or cmap.get(ord("?"))
        w += hmtx[g][0] / upm * px + ls if g else px * 0.5
    return w


def font_css(font_id: str) -> dict:
    f = fonts()[font_id]
    return {"id": font_id, "family": f["family"], "file": f["file"], "weight": f.get("weight", 400),
            "style": f.get("style", "normal"), "sizeScale": f.get("size_scale", 1.0),
            "letterSpacingEm": f.get("letter_spacing_em", 0), "upper": f.get("uppercase", False)}


# ------------------------------------------------------------------ emphasis


def _clean(w: str) -> str:
    return re.sub(r"[^\wçğıöşüÇĞİÖŞÜ%]", "", w)


FILLERS = {"ve", "bir", "bu", "şu", "o", "da", "de", "ki", "mi", "mı", "ne", "ya", "yani", "abi", "aga", "lan", "kanka",
           "işte", "hani", "şey", "the", "a", "an", "and", "to", "of", "bro", "dude", "like", "yeah", "so", "uh", "um"}


def emphasis_scores(words: list[dict], llm_words: list[str], dictionary: list[str], peak: tuple[float, float]) -> list[float]:
    llm_set = {tr_lower(_clean(w)) for ws in llm_words for w in ws.split()}
    dic = {tr_lower(_clean(t)) for t in dictionary}
    out = []
    for i, w in enumerate(words):
        c = _clean(w["w"])
        lc = tr_lower(c)
        s = 0.0
        if not c:
            out.append(0.0)
            continue
        if lc in llm_set:
            s += 5
        if re.search(r"\d", c):
            s += 3.5
        if lc in dic:
            s += 3
        prev_end = i == 0 or bool(END_PUNCT.search(words[i - 1]["w"]))
        if c[0].isupper() and not prev_end and len(c) > 2:
            s += 2  # proper noun mid-sentence
        if SHOCK.search(w["w"]) or PROFANITY.search(w["w"]):
            s += 2.5
        if w["w"].endswith("!"):
            s += 1.5
        if peak[0] - 0.05 <= w["s"] <= peak[1]:
            s += 1.5 + (1.0 if END_PUNCT.search(w["w"]) or i + 1 == len(words) else 0)
        if len(c) >= 5:
            s += 0.8
        if len(c) >= 8:
            s += 0.7
        if re.search(r"(.)\1\1", lc):  # stretched words: "yasss", "nooo"
            s += 2.0
        if END_PUNCT.search(w["w"]) or w["w"].endswith(","):
            s += 0.6  # phrase-final words carry the stress in Turkish and English
        if i > 0 and tr_lower(_clean(words[i - 1]["w"])) == lc:
            s -= 1.5  # repetitions ("bitti bitti") only once
        if lc in FILLERS:
            s = 0
        out.append(s)
    return out


def censor(word: str) -> str:
    def rep(m):
        t = m.group(0)
        return t[0] + "*" * max(1, len(t) - 2) + t[-1] if len(t) > 2 else t[0] + "*"

    return PROFANITY.sub(rep, word)


# ------------------------------------------------------------------ paging


def build_pages(out_words: list[dict], style: str, palette: dict, seed: int, upper: bool | None = None,
                censor_profanity: bool = False) -> dict:
    """out_words: [{"w","startMs","endMs","seg","emph"(score),"sent_end"}] already in output time."""
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["dynamic"])
    available = fonts()
    neutral = next((f for f in preset["neutral"] if f in available), next(iter(available)))
    accents = [f for f in preset["accents"] if f in available] or [neutral]
    rng = random.Random(seed)
    accent_cycle = rng.sample(accents, len(accents))
    up = preset["upper"] if upper is None else upper
    max_words = preset["max_words"]

    pages_words: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(out_words):
        if cur:
            gap = w["startMs"] - cur[-1]["endMs"]
            chars = sum(len(x["w"]) for x in cur) + len(w["w"])
            dur = w["endMs"] - cur[0]["startMs"]
            brk = (len(cur) >= max_words or cur[-1]["sent_end"] or w["seg"] != cur[-1]["seg"] or gap > 450
                   or chars > 20 or dur > 1700 or (w["emph"] >= 3 and len(cur) >= 2) or (cur[-1]["emph"] >= 3 and len(cur) >= 2))
            if brk:
                pages_words.append(cur)
                cur = []
        cur.append(w)
    if cur:
        pages_words.append(cur)
    # merge flashes (< 260 ms) into the following page when allowed
    merged: list[list[dict]] = []
    for pw in pages_words:
        if merged:
            prev = merged[-1]
            short = prev[-1]["endMs"] - prev[0]["startMs"] < 260
            if short and not prev[-1]["sent_end"] and prev[-1]["seg"] == pw[0]["seg"] and len(prev) + len(pw) <= max_words:
                merged[-1] = prev + pw
                continue
        merged.append(pw)

    pages = []
    acc_i = 0
    emph_budget_gap = 0
    last_combo = None
    for pi, pw in enumerate(merged):
        best = max(range(len(pw)), key=lambda k: pw[k]["emph"])
        has_emph = pw[best]["emph"] >= 2.2 and (emph_budget_gap >= 1 or pw[best]["emph"] >= 5)
        color_key = "accent" if pi % 2 == 0 else "accent2"
        efont = accent_cycle[acc_i % len(accent_cycle)] if has_emph else None
        combo = (neutral, efont, color_key)
        if combo == last_combo:
            color_key = "accent2" if color_key == "accent" else "accent"
            combo = (neutral, efont, color_key)
        if has_emph:
            acc_i += 1
            emph_budget_gap = 0
        else:
            emph_budget_gap += 1
        last_combo = combo
        items = []
        for k, w in enumerate(pw):
            txt = censor(w["w"]) if censor_profanity else w["w"]
            is_e = has_emph and k == best
            fid = efont if is_e else neutral
            f_up = fonts()[fid].get("uppercase", False) or (up and not is_e)
            items.append({"text": tr_upper(txt) if f_up else txt, "startMs": w["startMs"], "endMs": w["endMs"],
                          "emph": is_e, "font": fid})
        lines = _layout_lines(items)
        rot = 0.0
        if has_emph:
            rot = round(rng.choice([-1, 1]) * rng.uniform(2.0, 4.5), 2)
        pages.append({"startMs": pw[0]["startMs"], "endMs": pw[-1]["endMs"], "lines": lines,
                      "color": palette[color_key], "rotate": rot, "hasEmph": has_emph, "combo": list(combo)})
    # display end: hold until next page (if close) else last word + 600 ms
    for a, b in zip(pages, pages[1:]):
        a["endMs"] = b["startMs"] if b["startMs"] - a["endMs"] < 700 else a["endMs"] + 500
    if pages:
        pages[-1]["endMs"] += 600
    used = sorted({neutral, *[p["combo"][1] for p in pages if p["combo"][1]]})
    return {"style": style, "pages": pages, "fonts": [font_css(f) for f in used], "neutralFont": neutral}


def _layout_lines(items: list[dict]) -> list[dict]:
    """Split into 1-2 lines and fit pixel sizes to SAFE_W using real glyph advances."""
    def px_for(it):
        f = fonts()[it["font"]]
        base = BASE_PX * (EMPH_RATIO if it["emph"] else 1.0)
        return base / max(0.6, f.get("size_scale", 1.0)) * (1.0 if f.get("size_scale", 1) >= 0.95 else 0.98)

    for it in items:
        it["size"] = px_for(it)
    # line split: emphasis word alone on its line if page has >= 3 words; else balance by width
    if len(items) <= 1:
        groups = [items]
    else:
        widths = [text_width(it["text"] + " ", it["font"], it["size"]) for it in items]
        total = sum(widths)
        if total <= SAFE_W and len(items) <= 2:
            groups = [items]
        else:
            best_k, best_d = 1, 1e9
            for k in range(1, len(items)):
                d = abs(sum(widths[:k]) - sum(widths[k:]))
                if d < best_d:
                    best_k, best_d = k, d
            emph_idx = next((i for i, it in enumerate(items) if it["emph"]), None)
            if emph_idx is not None and len(items) >= 3:
                best_k = emph_idx if emph_idx > 0 else 1
            groups = [items[:best_k], items[best_k:]]
    lines = []
    for g in groups:
        w = sum(text_width(it["text"], it["font"], it["size"]) for it in g) + 0.28 * BASE_PX * (len(g) - 1)
        scale = min(1.0, SAFE_W / max(w, 1))
        for it in g:
            it["size"] = round(max(MIN_PX, it["size"] * scale), 1)
        lines.append({"words": g, "width": round(w * scale, 1)})
    return lines


def caption_tokens(out_words: list[dict]) -> list[dict]:
    """@remotion/captions `Caption` objects (whitespace-sensitive text)."""
    return [{"text": (" " if i else "") + w["w"], "startMs": w["startMs"], "endMs": w["endMs"],
             "timestampMs": (w["startMs"] + w["endMs"]) // 2, "confidence": w.get("p")} for i, w in enumerate(out_words)]

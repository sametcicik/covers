"""Candidate moment discovery, scoring (LLM + signals, or heuristic), peak finding and boundary snapping."""
from __future__ import annotations

import re

import numpy as np

from . import llm
from .util import log

END_PUNCT = re.compile(r"[.!?…]+[\"')\]]*$")
CONJ_START = {"ve", "ama", "çünkü", "fakat", "yani", "da", "de", "ki", "and", "but", "so", "because", "or", "then"}
SHOCK = re.compile(
    r"\b(yok art[ıi]k|inan(a|ı)m(ı|i)yorum|inan[ıi]lmaz|nas[ıi]l ya|ne(?:yse)?\?|ciddi misin|aman|oha|vay|lan|abi|aga|"
    r"efsane|rezalet|bitti|kaybett|kazand|milyon|bin lira|tl|dolar|"
    r"no way|what the|oh my god|omg|holy|bro|insane|crazy|dude|let'?s go|i can'?t|wtf|are you serious)\b", re.I)
LAUGH_TXT = re.compile(r"\b(ha(ha)+|ah(ah)+|hehe+|lol|lmao|\[?(laughter|kahkaha)\]?)\b", re.I)
PROFANITY = re.compile(r"\b(amk|aq|siktir|lan|orospu|piç|fuck|shit|damn|bitch)\w*", re.I)
NUMBER = re.compile(r"\d")

CATS_TR = {
    "funny": "komik", "shock": "şok", "argument": "tartışma", "confession": "itiraf", "rage": "rage",
    "win": "büyük kazanç", "loss": "büyük kayıp", "quote": "alıntılık", "reaction": "tepki",
}


def sentences(words: list[dict], max_words: int = 28) -> list[dict]:
    out: list[dict] = []
    cur: list[int] = []
    for i, w in enumerate(words):
        cur.append(i)
        gap = (words[i + 1]["s"] - w["e"]) if i + 1 < len(words) else 9.0
        end = bool(END_PUNCT.search(w["w"])) or gap > 0.75
        if not end and len(cur) >= max_words and (w["w"].endswith(",") or gap > 0.3):
            end = True
        if end:
            out.append({"i0": cur[0], "i1": cur[-1], "s": words[cur[0]]["s"], "e": words[cur[-1]]["e"],
                        "text": " ".join(words[j]["w"] for j in cur)})
            cur = []
    return out


class Signals:
    """Helper over 1 Hz heat + 10 Hz audio dB, indexed by absolute VOD seconds."""

    def __init__(self, t0: float, heat: np.ndarray, audio: dict, chat: dict):
        self.t0 = t0
        self.heat = heat
        self.db10 = np.array(audio.get("db10") or [], dtype=float)
        self.loud_z = np.array(audio.get("loud_z") or [], dtype=float)
        self.chat_z = np.array(chat.get("rate_z") or [0.0] * len(heat), dtype=float)
        self.react_z = np.array(chat.get("react_z") or [0.0] * len(heat), dtype=float)

    def _sl(self, arr, a, b, hz=1):
        i0 = int(max(0, (a - self.t0) * hz))
        i1 = int(max(i0 + 1, (b - self.t0) * hz))
        return arr[i0:i1] if len(arr) else np.zeros(1)

    def heat_max(self, a, b):
        s = self._sl(self.heat, a, b)
        return float(s.max()) if len(s) else 0.0

    def heat_top(self, a, b, k=5):
        s = np.sort(self._sl(self.heat, a, b))[::-1][:k]
        return float(s.mean()) if len(s) else 0.0

    def argmax_heat(self, a, b):
        s = self._sl(self.heat, a, b)
        return a + float(np.argmax(s)) if len(s) else a

    def db(self, a, b):
        s = self._sl(self.db10, a, b, 10)
        return s if len(s) else np.array([-60.0])

    def peak_stats(self, a, b) -> dict:
        return {
            "loud": float(self._sl(self.loud_z, a, b).max(initial=0)),
            "chat": float(self._sl(self.chat_z, a, b).max(initial=0)),
            "react": float(self._sl(self.react_z, a, b).max(initial=0)),
            "heat": self.heat_max(a, b),
        }


# ------------------------------------------------------------------ heuristic scoring


def _lex(text: str) -> float:
    s = 0.0
    s += 2.0 * len(SHOCK.findall(text))
    s += 2.5 * len(LAUGH_TXT.findall(text))
    s += 1.0 * text.count("!")
    s += 0.8 * text.count("?")
    s += 1.0 * len(PROFANITY.findall(text))
    s += 0.6 * len(NUMBER.findall(text)) ** 0.5
    return s


def _span_score(sents, i, j, sig: Signals, lang: str) -> tuple[float, dict]:
    a, b = sents[i]["s"], sents[j]["e"]
    dur = b - a
    text = " ".join(x["text"] for x in sents[i:j + 1])
    nwords = len(text.split())
    hmax = sig.heat_max(a, b)
    htop = sig.heat_top(a, b)
    peak_t = sig.argmax_heat(a, b)
    pos = (peak_t - a) / max(dur, 1e-3)
    lex = _lex(text)
    st = sig.peak_stats(a, b)
    score = 46 * hmax + 16 * htop
    score += min(18.0, lex * 1.6)
    score += 10 if 0.3 <= pos <= 0.92 else (3 if pos > 0.92 else 0)
    wps = nwords / max(dur, 1)
    score += 5 if 1.4 <= wps <= 4.5 else 0
    score += 6 * max(0.0, 1 - abs(dur - 32) / 20)
    first = sents[i]["text"].split()[0].lower().strip(",.") if sents[i]["text"] else ""
    if first in CONJ_START:
        score -= 6
    if nwords < 12:
        score -= 10
    return score, {"lex": lex, "pos": pos, **st}


def _reason(st: dict, text: str, lang: str) -> str:
    parts = []
    tr = lang == "tr"
    if st.get("chat", 0) > 2:
        parts.append(f"chat hızı normalin ~{1 + st['chat'] / 2:.1f} katına çıktı" if tr else f"chat velocity spiked (~{1 + st['chat'] / 2:.1f}x)")
    if st.get("react", 0) > 2:
        parts.append("emote/kahkaha spam'i" if tr else "emote/laugh spam in chat")
    if st.get("loud", 0) > 2:
        parts.append("ses enerjisinde ani sıçrama (bağırma/tepki)" if tr else "sudden audio energy jump (shout/reaction)")
    if SHOCK.search(text):
        parts.append("şok/tepki ifadesi" if tr else "shock/reaction line")
    if LAUGH_TXT.search(text):
        parts.append("kahkaha" if tr else "laughter")
    if st.get("pos", 0.5) >= 0.3:
        parts.append("setup sonra payoff yapısı" if tr else "setup then payoff structure")
    return ("; ".join(parts) or ("yüksek genel enerji" if tr else "high overall energy")).capitalize()


def _category(text: str, st: dict) -> str:
    if LAUGH_TXT.search(text) or st.get("react", 0) > 3:
        return "funny"
    if PROFANITY.search(text) and st.get("loud", 0) > 2.5:
        return "rage"
    if re.search(r"kazand|won|win|kazanç|para|milyon|dolar", text, re.I):
        return "win"
    if re.search(r"kaybett|lost|lose|öldü|died|bitti", text, re.I):
        return "loss"
    if SHOCK.search(text):
        return "shock"
    return "reaction"


def heuristic_moments(windows: list[tuple[float, float]], words: list[dict], sig: Signals, lang: str,
                      min_dur: float, max_dur: float) -> list[dict]:
    out = []
    for w0, w1 in windows:
        ws = [w for w in words if w0 <= w["s"] < w1]
        if len(ws) < 10:
            continue
        sents = sentences(ws)
        best = None
        for i in range(len(sents)):
            for j in range(i, len(sents)):
                dur = sents[j]["e"] - sents[i]["s"]
                if dur > max_dur:
                    break
                if dur < min_dur:
                    continue
                sc, st = _span_score(sents, i, j, sig, lang)
                if best is None or sc > best[0]:
                    best = (sc, i, j, st)
        if not best:
            continue
        sc, i, j, st = best
        text = " ".join(x["text"] for x in sents[i:j + 1])
        out.append({
            "start": sents[i]["s"], "end": sents[j]["e"], "score": float(np.clip(100 * (1 - np.exp(-max(0.0, sc) / 55)), 0, 99)),
            "category": _category(text, st), "why": _reason(st, text, lang), "text": text,
            "signals": {k: round(v, 3) for k, v in st.items()}, "source": "heuristic",
        })
    return out


# ------------------------------------------------------------------ LLM scoring

SYSTEM = """You are a senior short-form video editor who finds the most viral moments in livestream transcripts \
(Twitch/Kick). You select clips that work WITHOUT context: they need a clear setup and a payoff. \
Look for: funny moments, shock, arguments, confessions, rage, big wins/losses, quotable lines, strong reactions. \
Each window has numbered sentences with timestamps (seconds) and stream signals \
(loud = shouting/energy z-score, chat = chat-speed z-score, react = emote/laugh spam z-score).
Rules: a clip starts at a sentence start and ends at a sentence end; never mid-sentence. Target 20-45 s, hard max {max_dur} s. \
Pick the peak: the single strongest punchline or reaction (1.5-4 s). \
Mark emphasis words for captions: names, numbers, shock words, the punchline (exact words as written, max 1 per 6 words). \
Write titles in the transcript language, no emojis, no clickbait lies, max 60 chars. Skip windows with nothing worth posting.
Return JSON only."""

USER_TMPL = """Streamer: {streamer}. Language: {lang}.
Return: {{"moments":[{{"window":<id>,"start_idx":<int>,"end_idx":<int>,"peak_idx":<int sentence idx>,\
"peak_quote":"<exact words of the peak>","score":<0-100>,"category":"funny|shock|argument|confession|rage|win|loss|quote|reaction",\
"why":"<1 sentence: why this works, in {lang}>","emphasis":["..."],"titles":["...","...","..."],\
"description":"<1-2 sentences, {lang}>","hashtags":["#..."]}}]}}

{windows}"""


def llm_moments(windows: list[tuple[float, float]], words: list[dict], sig: Signals, lang: str, streamer: str,
                max_dur: float, batch: int = 5) -> list[dict]:
    out = []
    blocks = []
    for wid, (w0, w1) in enumerate(windows):
        ws = [w for w in words if w0 <= w["s"] < w1]
        if len(ws) < 10:
            continue
        sents = sentences(ws)
        lines = []
        for k, s in enumerate(sents):
            st = sig.peak_stats(s["s"], s["e"] + 1)
            lines.append(f"[{k}] {s['s'] - w0:6.1f}-{s['e'] - w0:6.1f}s loud={st['loud']:.1f} chat={st['chat']:.1f} "
                         f"react={st['react']:.1f} | {s['text']}")
        blocks.append((wid, w0, sents, f"### window {wid}\n" + "\n".join(lines)))
    for b in range(0, len(blocks), batch):
        chunk = blocks[b:b + batch]
        try:
            res = llm.chat_json(SYSTEM.format(max_dur=int(max_dur)),
                                USER_TMPL.format(streamer=streamer, lang=lang, windows="\n\n".join(c[3] for c in chunk)))
        except Exception as e:  # noqa: BLE001
            log.warning("LLM moment scoring failed: %s", e)
            continue
        by_id = {c[0]: c for c in chunk}
        for m in (res or {}).get("moments", []):
            c = by_id.get(m.get("window"))
            if not c:
                continue
            _, w0, sents, _ = c
            i, j = int(m.get("start_idx", 0)), int(m.get("end_idx", 0))
            if not (0 <= i <= j < len(sents)):
                continue
            pk = m.get("peak_idx")
            peak_sent = sents[int(pk)] if isinstance(pk, int) and 0 <= pk < len(sents) else None
            st = sig.peak_stats(sents[i]["s"], sents[j]["e"])
            sig_score = 100 * (0.7 * st["heat"] + 0.3 * sig.heat_top(sents[i]["s"], sents[j]["e"]))
            score = 0.62 * float(m.get("score", 50)) + 0.38 * sig_score
            out.append({
                "start": sents[i]["s"], "end": sents[j]["e"], "score": float(np.clip(score, 0, 100)),
                "llm_score": m.get("score"), "category": m.get("category", "reaction"), "why": m.get("why", ""),
                "text": " ".join(x["text"] for x in sents[i:j + 1]),
                "peak_hint": [peak_sent["s"], peak_sent["e"]] if peak_sent else None,
                "peak_quote": m.get("peak_quote"), "emphasis": m.get("emphasis") or [],
                "titles": m.get("titles") or [], "description": m.get("description") or "",
                "hashtags": m.get("hashtags") or [], "signals": {k: round(v, 3) for k, v in st.items()},
                "source": "llm",
            })
    return out


# ------------------------------------------------------------------ refinement


def find_peak(m: dict, words: list[dict], sig: Signals) -> tuple[float, float]:
    """Strongest 1.5-4 s inside the clip, snapped to word boundaries."""
    a, b = m["start"], m["end"]
    if m.get("peak_hint"):
        ha, hb = m["peak_hint"]
        a2, b2 = max(a, ha - 1.0), min(b, hb + 1.5)
        if b2 - a2 >= 1.5:
            a, b = a2, b2
    db = sig.db(a, b)
    n = len(db)
    best, bt = -1e9, a
    # 2.5 s sliding window over 10 Hz loudness + heat
    win = 25
    for k in range(0, max(1, n - win), 2):
        t = a + k / 10
        sc = float(np.percentile(db[k:k + win], 85)) / 6 + 6 * sig.heat_max(t, t + 2.5)
        if m.get("peak_quote") and m.get("peak_hint"):
            sc += 0  # hint already restricts range
        if sc > best:
            best, bt = sc, t
    ws = [w for w in words if m["start"] <= w["s"] < m["end"]]
    if not ws:
        return bt, min(bt + 2.5, m["end"])
    # snap: start at the nearest word start <= bt+0.3, end at a word end making 1.5-4 s
    starts = [w for w in ws if w["s"] <= bt + 0.3]
    w_start = starts[-1] if starts else ws[0]
    # prefer starting at a sentence/phrase start within 1.2 s before
    idx = ws.index(w_start)
    for back in range(1, 5):
        if idx - back < 0:
            break
        prev = ws[idx - back]
        if ws[idx - back + 1]["s"] - prev["e"] > 0.25 or END_PUNCT.search(prev["w"]):
            if w_start["s"] - ws[idx - back + 1]["s"] <= 1.2:
                w_start = ws[idx - back + 1]
            break
    ps = w_start["s"]
    pe = ps + 1.5
    for w in ws[ws.index(w_start):]:
        if w["e"] - ps > 4.0:
            break
        pe = w["e"]
        if w["e"] - ps >= 1.5 and (END_PUNCT.search(w["w"]) or w["w"].endswith(",")):
            break
    if pe - ps < 1.5:
        pe = min(ps + 1.8, m["end"])
    return round(ps, 3), round(pe, 3)


def snap_bounds(m: dict, words: list[dict], sig: Signals, max_dur: float) -> dict:
    ws = [w for w in words if m["start"] - 0.05 <= w["s"] <= m["end"] + 0.05]
    if not ws:
        return m
    i0 = words.index(ws[0])
    i1 = words.index(ws[-1])
    gap_before = ws[0]["s"] - words[i0 - 1]["e"] if i0 > 0 else 1.0
    gap_after = words[i1 + 1]["s"] - ws[-1]["e"] if i1 + 1 < len(words) else 2.5
    start = ws[0]["s"] - min(0.25, max(0.04, gap_before * 0.6))
    end = ws[-1]["e"] + min(0.45, max(0.08, gap_after * 0.6))
    # let laughter / reaction ring out after the last word (stop before the next sentence)
    if gap_after > 0.6:
        db = sig.db(ws[-1]["e"], ws[-1]["e"] + 2.5)
        speech_level = float(np.percentile(sig.db(m["start"], m["end"]), 60))
        ring = 0
        for v in db:
            if v < speech_level - 6:
                break
            ring += 1
        end = max(end, min(ws[-1]["e"] + ring / 10 + 0.15, ws[-1]["e"] + gap_after - 0.08))
    if end - start > max_dur:
        end = start + max_dur
        cut = [w for w in ws if w["e"] <= end and END_PUNCT.search(w["w"])]
        if cut:
            end = cut[-1]["e"] + 0.2
    m = dict(m)
    m["start"], m["end"] = round(start, 3), round(end, 3)
    return m


def dedupe(moments: list[dict], overlap: float = 0.3) -> list[dict]:
    keep: list[dict] = []
    for m in sorted(moments, key=lambda x: -x["score"]):
        ok = True
        for k in keep:
            inter = min(m["end"], k["end"]) - max(m["start"], k["start"])
            if inter > overlap * min(m["end"] - m["start"], k["end"] - k["start"]):
                ok = False
                break
        if ok:
            keep.append(m)
    return keep


def windows_from_peaks(peak_idx: list[int], t0: float, lo: float, hi: float, before: float = 42,
                       after: float = 22) -> list[tuple[float, float]]:
    out = []
    for p in peak_idx:
        t = t0 + p
        out.append((max(lo, t - before), min(hi, t + after)))
    return out


def find_moments(words: list[dict], windows: list[tuple[float, float]], sig: Signals, lang: str, streamer: str,
                 n: int, min_dur: float = 18, max_dur: float = 50) -> list[dict]:
    cands: list[dict] = []
    if llm.enabled():
        cands = llm_moments(windows, words, sig, lang, streamer, max_dur)
    if not cands:
        cands = heuristic_moments(windows, words, sig, lang, min_dur, max_dur)
    cands = [snap_bounds(m, words, sig, max_dur) for m in cands]
    cands = dedupe(cands)
    for m in cands:
        m["peak"] = list(find_peak(m, words, sig))
    cands.sort(key=lambda m: -m["score"])
    log.info("moments: %d candidates", len(cands))
    return cands[: max(n * 2, n + 3)]

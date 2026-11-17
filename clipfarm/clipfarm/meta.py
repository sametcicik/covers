"""Titles, description and hashtags per clip (LLM when available, deterministic fallback otherwise)."""
from __future__ import annotations

import re

from . import llm
from .util import log, tr_upper

CAT_TR = {"funny": "komik an", "shock": "şok anı", "argument": "tartışma", "confession": "itiraf", "rage": "sinir krizi",
          "win": "büyük kazanç", "loss": "büyük kayıp", "quote": "efsane söz", "reaction": "tepki"}
CAT_EN = {"funny": "funny moment", "shock": "shock moment", "argument": "argument", "confession": "confession",
          "rage": "rage moment", "win": "huge win", "loss": "huge loss", "quote": "legendary line", "reaction": "reaction"}

META_SYS = """You write metadata for vertical short clips of livestreams (TikTok/Reels/Shorts). \
Language = transcript language. No emojis. Titles <= 60 chars, punchy, honest, varied (one quote-style, one curiosity, \
one plain). Description 1-2 sentences. 5-8 hashtags (lowercase, no spaces; include streamer and game/topic). \
Return JSON only: {"titles":["","",""],"description":"","hashtags":["#..."]}"""


def _quote(text: str, n: int = 52) -> str:
    from .captions import SLURS
    from .moments import PROFANITY

    sents = []
    for x in re.split(r"(?<=[.!?,])\s+", text.strip()):
        ws = x.split()
        if len(ws) > 12:  # unpunctuated speech: cut into phrases
            sents += [" ".join(ws[k:k + 8]) for k in range(0, len(ws), 8)]
        elif ws:
            sents.append(x.strip().rstrip(","))

    def rank(s: str):
        nw = len(s.split())
        bad = PROFANITY.search(s) or SLURS.search(s)
        return (0 if bad else 1, 1 if 3 <= nw <= 10 else 0, s.count("!") + s.count("?"),
                -abs(len(s) - 38))

    best = max(sents, key=rank) if sents else text
    best = best.strip().strip('"')
    best = best[:1].upper() + best[1:] if best else best
    if len(best) > n:
        best = best[: n - 1].rsplit(" ", 1)[0] + "…"
    return best


def build(moment: dict, job: dict, profile: dict, lang: str) -> dict:
    name = profile.get("display_name") or job["channel"]
    camp = profile.get("campaign") or {}
    tags_req = camp.get("required_hashtags") or []
    titles, desc, tags = moment.get("titles") or [], moment.get("description") or "", moment.get("hashtags") or []
    if (len(titles) < 3 or not desc) and llm.enabled():
        try:
            j = llm.chat_json(META_SYS, f"Streamer: {name}\nPlatform: {job['platform']}\nStream title: {job.get('title')}\n"
                              f"Category: {moment.get('category')}\nTranscript: {moment.get('text', '')[:2500]}")
            titles = titles or j.get("titles") or []
            desc = desc or j.get("description") or ""
            tags = tags or j.get("hashtags") or []
        except Exception as e:  # noqa: BLE001
            log.warning("meta LLM failed: %s", e)
    tr = lang == "tr"
    cat = moment.get("category", "reaction")
    if len(titles) < 3:
        q = _quote(moment.get("text", ""))
        fallback = ([f"\"{q}\"", f"{name} {CAT_TR.get(cat, 'tepki')}: bunu beklemiyordu", f"{name} yayında {CAT_TR.get(cat, 'an')}"]
                    if tr else [f"\"{q}\"", f"{name} did NOT expect this", f"{name}'s {CAT_EN.get(cat, 'moment')} on stream"])
        titles = (titles + fallback)[:3]
    if not desc:
        desc = (f"{name} yayınından kısa bir kesit. {moment.get('why', '')}".strip() if tr
                else f"A moment from {name}'s stream. {moment.get('why', '')}".strip())
    base_tags = [f"#{re.sub(r'[^a-z0-9ğüşıöç]', '', name.lower())}", f"#{job['platform']}",
                 "#yayın" if tr else "#stream", "#keşfet" if tr else "#fyp", "#shorts"]
    title_words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{4,}", job.get("title") or "")
    if title_words:
        base_tags.append("#" + title_words[0].lower())
    all_tags = []
    for t in list(tags) + base_tags + tags_req:
        t = "#" + t.lstrip("#").replace(" ", "").lower()
        if len(t) > 2 and t not in all_tags:
            all_tags.append(t)
    if camp.get("cta"):
        desc = f"{desc} {camp['cta']}".strip()
    for m in camp.get("required_mentions") or []:
        if m not in desc:
            desc += f" {m}"
    from .captions import censor

    titles = [censor(t, bool(camp.get("censor_profanity"))) for t in titles]
    desc = censor(desc, bool(camp.get("censor_profanity")))
    return {"titles": [t[:80] for t in titles[:3]], "description": desc, "hashtags": all_tags[:10],
            "caption_preview": tr_upper(titles[0])[:40] if titles else ""}

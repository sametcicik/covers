"""SFX library access: manifest, variation rotation (never the same file twice in a row)."""
from __future__ import annotations

import random

from .config import SFX_DIR
from .util import read_json

_MANIFEST = None


def manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = read_json(SFX_DIR / "manifest.json", {"items": []})
    return _MANIFEST


def by_category() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for it in manifest().get("items", []):
        out.setdefault(it["category"], []).append(it)
    return out


def assign(events: list[dict], rng: random.Random) -> list[dict]:
    """Attach concrete files. Each category cycles through a seeded shuffle; no file repeats back-to-back."""
    cats = by_category()
    queues: dict[str, list[dict]] = {}
    last_file = None
    out = []
    for e in sorted(events, key=lambda e: e["at"]):
        items = cats.get(e["cat"]) or []
        if not items:
            continue
        q = queues.setdefault(e["cat"], [])
        if not q:
            q.extend(rng.sample(items, len(items)))
        pick = q.pop(0)
        if pick["file"] == last_file and q:
            q.append(pick)
            pick = q.pop(0)
        last_file = pick["file"]
        hit = float(pick.get("hit", 0.0))
        dur = float(pick.get("duration", 1.0))
        start = e["at"] - hit  # risers: hit = their crest, so the crest lands on the punchline
        out.append({**e, "id": pick["id"], "file": pick["file"], "start": round(max(0.0, start), 3),
                    "duration": dur, "lufs": pick.get("loudness_lufs")})
    return out

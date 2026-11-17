"""Edit decision list: cold-open hook, dead-air trimming, zoom/shake/flash events, SFX placement."""
from __future__ import annotations

import random

import numpy as np

from . import sfx as sfxlib
from .moments import END_PUNCT, Signals
from .util import log

DEAD_AIR = 0.4  # gaps above this are candidates for trimming
KEEP_GAP = 0.2  # what remains of a trimmed gap


def _speech_level(sig: Signals, a: float, b: float) -> float:
    return float(np.percentile(sig.db(a, b), 70))


MIN_TOTAL = 18.0  # don't trim a clip below this (target 20-45 s)


def plan_segments(m: dict, words: list[dict], sig: Signals, hook: bool, max_total: float) -> list[dict]:
    """Return segments in absolute VOD seconds: [{"s","e","kind"}]. Cold open first if needed.

    Trimming relaxes step by step when it would push the clip under MIN_TOTAL."""
    for level in (0, 1, 2):
        segs = _plan_segments(m, words, sig, hook, max_total, level)
        if sum(x["e"] - x["s"] for x in segs) >= MIN_TOTAL or m["end"] - m["start"] < MIN_TOTAL:
            break
    return segs


def _plan_segments(m: dict, words: list[dict], sig: Signals, hook: bool, max_total: float, relax: int) -> list[dict]:
    cs, ce = m["start"], m["end"]
    ps, pe = m["peak"]
    ws = [w for w in words if cs <= w["s"] < ce]
    lvl = _speech_level(sig, cs, ce)
    cuts: list[tuple[float, float]] = []  # removed intervals
    for a, b in zip(ws, ws[1:]):
        g = b["s"] - a["e"]
        if g <= DEAD_AIR:
            continue
        gdb = sig.db(a["e"] + 0.05, b["s"] - 0.05)
        if len(gdb) and np.percentile(gdb, 80) > lvl - 7:
            # laughter / reaction / game event in the gap: keep it, but cap long stretches without speech
            if g > 4.5 and not (a["e"] <= ps <= b["s"]):  # always: no long speech-free stretches
                cuts.append((a["e"] + 2.5, b["s"] - 1.0))
            continue
        if relax >= 2 and g < 1.2:
            continue
        keep = KEEP_GAP if relax == 0 else 0.35
        before_punch = abs(b["s"] - ps) < 0.35 or (ps - 3.0 <= b["s"] <= ps + 0.3)
        if before_punch or a["w"].endswith("?"):
            keep = min(g, 0.7)  # comic timing: keep the beat before the punchline / after a question
        if g - keep < 0.15:
            continue
        cuts.append((a["e"] + keep / 2, b["s"] - keep / 2))
    main: list[dict] = []
    t = cs
    for a, b in cuts:
        if a - t > 0.25:
            main.append({"s": round(t, 3), "e": round(a, 3), "kind": "main"})
        t = b
    if ce - t > 0.1:
        main.append({"s": round(t, 3), "e": round(ce, 3), "kind": "main"})
    segs = list(main)
    use_hook = hook and (ps - cs) > 3.0
    if use_hook:
        cold = {"s": round(ps - 0.1, 3), "e": round(pe + 0.15, 3), "kind": "cold"}
        segs = [cold] + main
    total = sum(x["e"] - x["s"] for x in segs)
    # enforce hard max: drop leading main material at sentence boundaries, never the peak
    while total > max_total and len(main) > 1:
        over = total - max_total
        first = main[0]
        if first["e"] - first["s"] <= over + 0.3 and first["e"] < ps:
            main.pop(0)
        else:
            cut_to = first["s"] + over + 0.05
            starts = [w["s"] for w in ws if w["s"] >= cut_to and w["s"] < ps and
                      (ws.index(w) == 0 or END_PUNCT.search(ws[ws.index(w) - 1]["w"]))]
            if starts and starts[0] < first["e"]:
                first["s"] = round(starts[0] - 0.1, 3)
            else:
                break
        segs = ([segs[0]] if use_hook else []) + main
        total = sum(x["e"] - x["s"] for x in segs)
    if total > max_total:
        last = segs[-1]
        last["e"] = round(last["e"] - (total - max_total), 3)
    return segs


def to_frames(segs: list[dict], hq_start: float, fps: int) -> list[dict]:
    out = []
    f = 0
    for s in segs:
        n = max(1, int(round((s["e"] - s["s"]) * fps)))
        out.append({"kind": s["kind"], "srcStart": round(s["s"] - hq_start, 4), "srcEnd": round(s["s"] - hq_start + n / fps, 4),
                    "absStart": s["s"], "outStart": f, "outEnd": f + n})
        f += n
    return out


def abs_to_out(t: float, fsegs: list[dict], fps: int, prefer: str = "main") -> list[int]:
    """Map an absolute VOD time to output frame(s) (a word inside the cold open appears twice)."""
    res = []
    for s in fsegs:
        a = s["absStart"]
        b = a + (s["outEnd"] - s["outStart"]) / fps
        if a - 1e-3 <= t < b:
            res.append(s["outStart"] + int(round((t - a) * fps)))
    return res


def plan_effects(m: dict, fsegs: list[dict], words: list[dict], sig: Signals, fps: int, emphasis_words: list[dict],
                 sfx_density: int, seed: int) -> dict:
    """Zooms, shakes, flashes and SFX in output frames."""
    rng = random.Random(seed)
    total = fsegs[-1]["outEnd"]
    zooms: list[dict] = []
    shakes: list[dict] = []
    flashes: list[dict] = []
    sfx: list[dict] = []
    has_cold = fsegs[0]["kind"] == "cold"

    # 1. opening punch-in within the first second
    zooms.append({"f": 0, "attack": max(3, int(0.12 * fps)), "hold": int(0.9 * fps), "release": int(0.5 * fps),
                  "amount": 0.12, "kind": "open"})
    # 2. cold-open transition: flash + whoosh + impact
    if has_cold:
        cut = fsegs[1]["outStart"]
        flashes.append({"f": cut - 2, "dur": 5})
        zooms.append({"f": cut, "attack": 2, "hold": int(0.15 * fps), "release": int(0.35 * fps), "amount": 0.08,
                      "kind": "transition"})
        sfx.append({"cat": "whoosh", "at": cut / fps, "prio": 0, "gain": -2})
        sfx.append({"cat": "impact", "at": cut / fps, "prio": 0, "gain": -4})
    # 3. jump-cut masking: alternate base scale per main segment
    base = []
    alt = False
    for s in fsegs:
        if s["kind"] == "main":
            base.append({"from": s["outStart"], "to": s["outEnd"], "scale": 1.06 if alt else 1.0})
            alt = not alt
        else:
            base.append({"from": s["outStart"], "to": s["outEnd"], "scale": 1.04})
    # 4. peak (in main): punch-in + light shake; riser into it if there is room
    ps, pe = m["peak"]
    pf = [f for f in abs_to_out(ps, fsegs, fps) if not (has_cold and f < fsegs[1]["outStart"])]
    if pf:
        p0 = pf[0]
        pdur = int((pe - ps) * fps)
        zooms.append({"f": p0, "attack": 3, "hold": max(4, pdur), "release": int(0.4 * fps), "amount": 0.13, "kind": "peak"})
        shakes.append({"f": p0 + 1, "dur": int(0.45 * fps), "amp": 9})
        if p0 / fps > 3.0:
            sfx.append({"cat": "riser", "at": p0 / fps, "prio": 1, "gain": -8, "end_aligned": True})
        sfx.append({"cat": "impact", "at": p0 / fps, "prio": 1, "gain": -9})
    # also shake the cold open peak slightly
    if has_cold:
        shakes.append({"f": 2, "dur": int(0.35 * fps), "amp": 6})
    # 5. emphasis punch-ins (spaced >= 3 s from other zooms) + pops
    zf = sorted(z["f"] for z in zooms)
    for ew in emphasis_words:
        for f in ew["frames"]:
            if has_cold and f < fsegs[1]["outStart"]:
                continue
            if all(abs(f - z) > 3 * fps for z in zf) and f < total - fps:
                zooms.append({"f": f, "attack": 4, "hold": int(max(0.3, ew["dur"] + 0.3) * fps),
                              "release": int(0.3 * fps), "amount": 0.07, "kind": "emphasis"})
                zf.append(f)
                sfx.append({"cat": "swish", "at": f / fps, "prio": 3, "gain": -14})
            sfx.append({"cat": "pop", "at": f / fps, "prio": 4, "gain": -16})
            break
    # 6. density control: max N effects per 10 s window, keep higher priority first
    limit = {0: 0, 1: 2, 2: 3, 3: 4}.get(int(sfx_density), 3)
    chosen: list[dict] = []
    for e in sorted(sfx, key=lambda e: (e["prio"], e["at"])):
        if limit == 0:
            break
        if e["prio"] > 0 and sum(1 for c in chosen if abs(c["at"] - e["at"]) < 5.0) >= limit:
            continue
        if any(abs(c["at"] - e["at"]) < 0.35 and c["cat"] == e["cat"] for c in chosen):
            continue
        chosen.append(e)
    # pair transition sounds are both prio 0; re-check the 10 s rule including them
    chosen.sort(key=lambda e: e["at"])
    final = []
    for e in chosen:
        window = [c for c in final if e["at"] - c["at"] < 10.0]
        if len(window) >= max(limit, 2) and e["prio"] > 0:
            continue
        final.append(e)
    placed = sfxlib.assign(final, rng)
    log.info("effects: %d zooms, %d shakes, %d sfx", len(zooms), len(shakes), len(placed))
    return {"zooms": sorted(zooms, key=lambda z: z["f"]), "base": base, "shakes": shakes, "flashes": flashes,
            "sfx": placed}

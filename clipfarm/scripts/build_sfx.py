"""Build assets/sfx: Mixkit (Sound Effects Free License) + original synthesized sounds (CC0).

Usage: python scripts/build_sfx.py [--raw-dir DIR]
Every file: 48 kHz stereo 16-bit wav, silence-trimmed, peak -1 dBFS, with measured hit offset + LUFS.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "sfx"
SR = 48000
UA = "Mozilla/5.0 (X11; Linux x86_64) ClipFarm-asset-builder"
MIXKIT_LICENSE = "Mixkit Sound Effects Free License"
MIXKIT_LICENSE_URL = "https://mixkit.co/license/#sfxFree"

# category -> (mixkit pages, title must match, title must not match, max length s, wanted)
PLAN = {
    "whoosh": (["whoosh", "swoosh", "transition"], r"whoosh|woosh|swoosh|fast|air", r"voice|laugh|man|woman|pop", 2.5, 7),
    "swish": (["swoosh", "transition", "whoosh"], r"swish|swipe|swoosh|quick|short|fast", r"voice|laugh|long", 1.6, 7),
    "impact": (["impact", "hit", "punch"], r"impact|hit|punch|boom|thud|slam", r"voice|laugh|glass|crowd|punches|heartbeat|body|wood", 2.5, 7),
    "bass_drop": (["bass", "drop", "boom"], r"bass|drop|boom|sub|deep", r"voice|laugh|guitar|loop|buzzer|wrong|hum|rumble", 3.0, 6),
    "pop": (["pop", "bubble", "click"], r"pop|bubble|click|plop", r"music|song|voice|crowd|blood|slide|long", 0.8, 7),
    "glitch": (["glitch", "electric", "technology"], r"glitch|digital|static|error|interference|electric", r"voice|loop", 2.0, 6),
    "ding": (["ding", "bell", "notification"], r"ding|bell|notification|chime|correct|ring", r"church|alarm|phone|door|bike|loop|school|christmas|summer|squeak|toy|cowbell", 2.0, 6),
    "record_scratch": (["scratch", "dj", "vinyl", "record"], r"scratch|record|vinyl|rewind", r"cat|wood|paper|loop", 2.5, 5),
    "riser": (["rise", "riser", "cinematic", "tension", "transition"], r"rise|riser|rising|swell|build|uplifter|reverse", r"voice|loop|music|helicopter|choir|stutter", 6.0, 6),
    "suspense": (["suspense", "tension", "cinematic", "horror"], r"suspense|tension|drone|dark|mystery|eerie|thriller", r"voice|scream|loop|music", 6.0, 5),
}


def mixkit_items(page: str) -> list[dict]:
    try:
        r = httpx.get(f"https://mixkit.co/free-sound-effects/{page}/", headers={"User-Agent": UA}, timeout=30,
                      follow_redirects=True)
        if r.status_code != 200:
            return []
    except Exception:
        return []
    t = r.text
    items = []
    for m in re.finditer(r'data-audio-player-preview-url-value="(https://assets\.mixkit\.co/[^"]+\.mp3)".*?'
                         r'item-grid-card__title">\s*(.*?)\s*</h2>', t, re.S):
        url, title = m.group(1), html.unescape(m.group(2)).strip()
        mid = re.search(r"/sfx/(\d+)/", url)
        items.append({"url": url, "title": title, "mixkit_id": mid.group(1) if mid else "",
                      "page": f"https://mixkit.co/free-sound-effects/{page}/"})
    return items


def load_any(path: Path) -> np.ndarray:
    tmp = path.with_suffix(".dec.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le",
                    str(tmp)], check=True)
    a, _ = sf.read(str(tmp), dtype="float32")
    tmp.unlink(missing_ok=True)
    return a if a.ndim == 2 else np.stack([a, a], 1)


def finish(a: np.ndarray, max_len: float) -> np.ndarray | None:
    mono = np.abs(a).max(axis=1)
    if mono.max() < 1e-3:
        return None
    thr = mono.max() * 10 ** (-48 / 20)
    idx = np.where(mono > thr)[0]
    s = max(0, idx[0] - int(0.005 * SR))
    e = min(len(a), idx[-1] + int(0.02 * SR))
    a = a[s:e]
    if len(a) > max_len * SR:
        a = a[: int(max_len * SR)]
        fl = int(0.25 * SR)
        a[-fl:] *= np.linspace(1, 0, fl)[:, None] ** 2
    if len(a) < int(0.06 * SR):
        return None
    fi = min(int(0.003 * SR), len(a) // 4)
    a[:fi] *= np.linspace(0, 1, fi)[:, None]
    fo = min(int(0.015 * SR), len(a) // 4)
    a[-fo:] *= np.linspace(1, 0, fo)[:, None]
    a = a / (np.abs(a).max() + 1e-9) * 10 ** (-1 / 20)
    return a.astype(np.float32)


def hit_offset(a: np.ndarray, cat: str) -> float:
    mono = np.abs(a).mean(axis=1)
    hop = int(0.01 * SR)
    n = len(mono) // hop
    env = mono[: n * hop].reshape(n, hop).mean(axis=1)
    if cat in ("riser", "suspense"):
        return round(float(np.argmax(env)) * 0.01, 3)
    if cat in ("whoosh", "swish"):
        return round(float(np.argmax(env)) * 0.01, 3)
    pk = env.max()
    onset = int(np.argmax(env > 0.5 * pk))
    return round(onset * 0.01, 3)


def loudness(a: np.ndarray) -> float:
    meter = pyln.Meter(SR, block_size=0.1 if len(a) / SR < 0.4 else 0.4)
    try:
        v = meter.integrated_loudness(a)
        return round(float(v), 2) if np.isfinite(v) else -30.0
    except Exception:
        return -30.0


def looks_like_voice(a: np.ndarray) -> bool:
    """Crude reject of spoken-word clips: sustained energy in 100-300 Hz harmonic band with syllabic modulation."""
    mono = a.mean(axis=1)
    if len(mono) < SR * 0.8:
        return False
    hop = int(0.02 * SR)
    n = len(mono) // hop
    env = np.abs(mono[: n * hop]).reshape(n, hop).mean(axis=1)
    spec = np.abs(np.fft.rfft(env - env.mean()))
    f = np.fft.rfftfreq(len(env), 0.02)
    syl = spec[(f > 3) & (f < 7)].sum() / (spec[(f > 0.3)].sum() + 1e-9)
    return syl > 0.45


# ------------------------------------------------------------------ synthesis (original, CC0)


def _noise(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def _bp(x, lo, hi, order=2):
    sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def _sweep_filter(x, f0, f1, q_bw=0.6, steps=64):
    out = np.zeros_like(x)
    n = len(x)
    seg = n // steps
    for i in range(steps):
        fc = f0 * (f1 / f0) ** (i / max(1, steps - 1))
        lo, hi = fc * (1 - q_bw / 2), min(SR / 2 - 100, fc * (1 + q_bw / 2))
        a, b = i * seg, n if i == steps - 1 else (i + 1) * seg
        pad = 2048
        a0 = max(0, a - pad)
        y = _bp(x[a0:b], max(20, lo), hi)
        out[a:b] = y[a - a0:]
    return out


def synth(cat: str, k: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + k * 17 + hash(cat) % 1000)
    if cat == "riser":
        d = 2.0 + 0.6 * k
        n = int(d * SR)
        t = np.arange(n) / SR
        noise = _sweep_filter(_noise(n, rng), 300 + 80 * k, 6000 + 900 * k, 0.5)
        f = 110 * (2 ** (3 * (t / d) ** 1.6)) * (1 + 0.1 * k)
        tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.4 * np.sin(4 * np.pi * np.cumsum(f) / SR)
        env = (t / d) ** 2.2
        x = (0.7 * noise / (np.abs(noise).max() + 1e-9) + 0.35 * tone) * env
        x[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    elif cat == "suspense":
        d = 3.5 + 0.5 * k
        n = int(d * SR)
        t = np.arange(n) / SR
        base = 55 * (1 + 0.06 * k)
        drone = sum(np.sin(2 * np.pi * base * h * t + rng.uniform(0, 6)) / h for h in (1, 2, 3, 5))
        beat = 0.5 + 0.5 * np.sin(2 * np.pi * (1.2 + 0.3 * k) * t) ** 8
        shimmer = _bp(_noise(n, rng), 2500, 7000) * 0.15
        env = np.minimum(1, t / 0.8) * np.minimum(1, (d - t) / 0.6)
        x = (drone * (0.55 + 0.45 * beat) + shimmer) * env
    elif cat == "record_scratch":
        d = 0.45 + 0.12 * k
        n = int(d * SR)
        t = np.arange(n) / SR
        rate = 9 + 3 * k
        speed = np.sin(2 * np.pi * rate * t * (1 - 0.4 * t / d))
        phase = np.cumsum(180 + 900 * np.abs(speed)) / SR
        saw = 2 * (phase % 1) - 1
        grit = _bp(_noise(n, rng), 800, 5000) * np.abs(speed)
        x = (0.6 * saw * np.abs(speed) + 0.5 * grit) * np.minimum(1, (d - t) / 0.05)
    elif cat == "whoosh":
        d = 0.6 + 0.12 * k
        n = int(d * SR)
        t = np.arange(n) / SR
        x = _sweep_filter(_noise(n, rng), 400 + 60 * k, 3500 + 500 * k, 0.9)
        x *= np.sin(np.pi * np.clip(t / d, 0, 1)) ** 3
    elif cat == "pop":
        d = 0.12
        n = int(d * SR)
        t = np.arange(n) / SR
        f = 900 * np.exp(-t * 30) + 300 + 60 * k
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45)
    else:
        raise ValueError(cat)
    x = np.asarray(x, dtype=np.float32)
    st = np.stack([x, np.roll(x, 12)], 1)
    return st


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir or tempfile.mkdtemp(prefix="sfx_raw_"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    items_out = []
    used_ids = set()
    for cat, (pages, must, mustnot, max_len, want) in PLAN.items():
        (OUT / cat).mkdir(exist_ok=True)
        for old in (OUT / cat).glob("*.wav"):
            old.unlink()
        cands = []
        for p in pages:
            for it in mixkit_items(p):
                if it["mixkit_id"] in used_ids or any(c["mixkit_id"] == it["mixkit_id"] for c in cands):
                    continue
                if re.search(must, it["title"], re.I) and not re.search(mustnot, it["title"], re.I):
                    cands.append(it)
        n = 0
        for it in cands:
            if n >= want:
                break
            rp = raw_dir / f"{it['mixkit_id']}.mp3"
            try:
                if not rp.exists():
                    r = httpx.get(it["url"], headers={"User-Agent": UA}, timeout=60)
                    r.raise_for_status()
                    rp.write_bytes(r.content)
                a = finish(load_any(rp), max_len)
            except Exception as e:  # noqa: BLE001
                print("skip", it["title"], e, file=sys.stderr)
                continue
            if a is None or looks_like_voice(a) or len(a) < 0.1 * SR or loudness(a) < -24:
                continue
            n += 1
            used_ids.add(it["mixkit_id"])
            fid = f"{cat}_{n:02d}"
            sf.write(str(OUT / cat / f"{fid}.wav"), a, SR, subtype="PCM_16")
            items_out.append({"id": fid, "category": cat, "file": f"{cat}/{fid}.wav", "duration": round(len(a) / SR, 3),
                              "hit": hit_offset(a, cat), "loudness_lufs": loudness(a), "source": "mixkit",
                              "title": it["title"], "source_url": it["page"], "asset_url": it["url"],
                              "license": MIXKIT_LICENSE, "license_url": MIXKIT_LICENSE_URL})
        k = 0
        while n < 5 or (cat in ("riser", "suspense", "record_scratch") and n < want and k < 6):
            if cat not in ("riser", "suspense", "record_scratch", "whoosh", "pop"):
                break
            a = finish(synth(cat, k), max_len)
            k += 1
            if a is None:
                continue
            n += 1
            fid = f"{cat}_{n:02d}"
            sf.write(str(OUT / cat / f"{fid}.wav"), a, SR, subtype="PCM_16")
            items_out.append({"id": fid, "category": cat, "file": f"{cat}/{fid}.wav", "duration": round(len(a) / SR, 3),
                              "hit": hit_offset(a, cat), "loudness_lufs": loudness(a), "source": "clipfarm-synth",
                              "title": f"ClipFarm synthesized {cat} #{k}", "source_url": "scripts/build_sfx.py",
                              "license": "CC0-1.0 (original, generated by ClipFarm)",
                              "license_url": "https://creativecommons.org/publicdomain/zero/1.0/"})
        print(f"{cat}: {n}")
    (OUT / "manifest.json").write_text(json.dumps({"version": 1, "items": items_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

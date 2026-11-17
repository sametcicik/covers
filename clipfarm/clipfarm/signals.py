"""Per-second signals over the analysis window: audio energy, chat velocity/emotes, visual change.

All series are indexed by absolute VOD second: value[i] belongs to t = t0 + i.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .util import log

CHAT_DELAY = 6.0  # chat reacts ~4-8 s after the moment on stream

LAUGH = re.compile(r"(kekw|lul|omegalul|lmao+|lmfao|icant|😂|🤣|ha(ha)+|ah(ah)+|sj(sj)+|ks(ks)+|asd(f|asd)?|xd+|ajsj|pfff+|kfkf|skdj)", re.I)
HYPE = re.compile(r"(pog|poggers|pogchamp|w+$|^w\s|lets go+|lesgo+|gg+|ez+|clip|klip|kl[iı]p|monka|omg|wtf|what+|nooo+|yok art[ıi]k|aga+|abi+|efsane|helal|\?{3,}|!{3,}|reis|ya+h+|vay)", re.I)


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0:
        return x
    from scipy.ndimage import median_filter

    return median_filter(x, size=max(3, win), mode="nearest")


def _robust_z(x: np.ndarray, win: int) -> np.ndarray:
    med = _rolling_median(x, win)
    mad = _rolling_median(np.abs(x - med), win) * 1.4826
    floor = max(1e-3, 0.35 * float(np.std(x)) if len(x) else 1e-3)
    return np.clip((x - med) / np.maximum(mad, floor), -10, 25)


def audio_signals(wav: Path, offset: float) -> dict:
    """10 Hz RMS dB + 1 Hz robust-z loudness, energy jumps, 'shout' (loud + bright)."""
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    hop = sr // 10
    n = len(audio) // hop
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-10)
    db10 = 20 * np.log10(rms + 1e-9)
    # brightness: share of energy above 1.5 kHz (shouting / screaming / laughter are bright)
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1 / sr)
    hi = spec[:, freqs > 1500].sum(axis=1) / (spec.sum(axis=1) + 1e-9)
    secs = n // 10
    db = db10[: secs * 10].reshape(secs, 10)
    db1 = np.percentile(db, 90, axis=1)
    hi1 = hi[: secs * 10].reshape(secs, 10).mean(axis=1)
    loud_z = _robust_z(db1, 180)
    prev = np.convolve(db1, np.ones(4) / 4, mode="full")[: len(db1)]
    prev = np.concatenate([[db1[0]], prev[:-1]])
    jump = np.clip(db1 - prev, 0, None)
    shout = np.clip(loud_z, 0, None) * (0.5 + _robust_z(hi1, 180).clip(0, 4) / 4)
    return {
        "t0": offset, "db": db1.round(2).tolist(), "loud_z": loud_z.round(3).tolist(),
        "jump": jump.round(3).tolist(), "shout": shout.round(3).tolist(),
        "db10": db10.round(2).tolist(),
    }


def chat_signals(msgs: list[dict], t0: float, seconds: int) -> dict:
    cnt = np.zeros(seconds)
    emo = np.zeros(seconds)
    laugh = np.zeros(seconds)
    hype = np.zeros(seconds)
    for m in msgs:
        i = int(m["t"] - CHAT_DELAY - t0)
        if 0 <= i < seconds:
            w = m.get("w", 1.0)
            cnt[i] += w
            emo[i] += w * min(3, m.get("e", 0))
            txt = m.get("m", "")
            if LAUGH.search(txt):
                laugh[i] += w
            if HYPE.search(txt):
                hype[i] += w
    k = np.ones(5) / 5
    rate = np.convolve(cnt, k, mode="same")
    dense = len(msgs) > 60 and sum(m.get("w", 1) for m in msgs) / max(1, seconds / 60) >= 3  # >= 3 msg/min
    ratez = _robust_z(rate, 300) if dense else np.zeros(seconds)
    react = np.convolve(emo + 2 * laugh + 1.5 * hype, k, mode="same")
    reactz = _robust_z(react, 300) if dense else np.zeros(seconds)
    return {"t0": t0, "count": len(msgs) if dense else 0, "raw_count": len(msgs), "rate": rate.round(3).tolist(), "rate_z": ratez.round(3).tolist(),
            "react_z": reactz.round(3).tolist(), "laugh": np.convolve(laugh, k, mode="same").round(3).tolist()}


def video_signals(preview: Path, t0: float, seconds: int, facecam: dict | None, fps_sample: float = 2.0) -> dict:
    """Scene change (histogram distance), game-area motion, facecam motion (reaction proxy)."""
    import cv2

    cap = cv2.VideoCapture(str(preview))
    if not cap.isOpened():
        return {"t0": t0, "scene": [0.0] * seconds, "motion": [0.0] * seconds, "face_motion": [0.0] * seconds}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / fps_sample)))
    scene = np.zeros(seconds)
    motion = np.zeros(seconds)
    fmotion = np.zeros(seconds)
    prev = None
    prev_hist = None
    idx = 0
    box = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            sec = int(idx / fps)
            if box is None and facecam and facecam.get("box"):
                x, y, w, h = facecam["box"]
                box = (int(x * 160), int(y * 90), max(2, int(w * 160)), max(2, int(h * 90)))
            if prev is not None and sec < seconds:
                d = np.abs(gray - prev)
                if box:
                    x, y, w, h = box
                    fm = d[y:y + h, x:x + w].mean()
                    d2 = d.copy()
                    d2[y:y + h, x:x + w] = 0
                    mm = d2.sum() / max(1, d2.size - w * h)
                else:
                    fm, mm = 0.0, d.mean()
                motion[sec] = max(motion[sec], mm)
                fmotion[sec] = max(fmotion[sec], fm)
                scene[sec] = max(scene[sec], float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
            prev, prev_hist = gray, hist
        idx += 1
    cap.release()
    return {"t0": t0, "scene": scene.round(4).tolist(), "motion_z": _robust_z(motion, 120).round(3).tolist(),
            "face_motion_z": _robust_z(fmotion, 120).round(3).tolist() if facecam else [0.0] * seconds}


def combine(audio: dict, chat: dict, video: dict, seconds: int) -> np.ndarray:
    """Heat curve 0..1 per second. Weights adapt to which signals are informative."""
    def arr(d, k):
        a = np.array(d.get(k) or [0.0] * seconds, dtype=float)
        if len(a) < seconds:
            a = np.pad(a, (0, seconds - len(a)))
        return a[:seconds]

    a = 0.55 * arr(audio, "loud_z").clip(0, 6) / 6 + 0.25 * arr(audio, "jump").clip(0, 15) / 15 + \
        0.2 * arr(audio, "shout").clip(0, 6) / 6
    has_chat = chat.get("count", 0) > 60
    c = (0.6 * arr(chat, "rate_z").clip(0, 6) + 0.4 * arr(chat, "react_z").clip(0, 6)) / 6 if has_chat else np.zeros(seconds)
    sc = arr(video, "scene")
    v = 0.4 * (sc > 0.45).astype(float) + 0.3 * arr(video, "motion_z").clip(0, 5) / 5 + \
        0.3 * arr(video, "face_motion_z").clip(0, 5) / 5
    if has_chat:
        heat = 0.42 * a + 0.4 * c + 0.18 * v
    else:
        heat = 0.75 * a + 0.25 * v
    heat = np.convolve(heat, np.ones(3) / 3, mode="same")
    mx = np.percentile(heat, 99.5) if len(heat) else 1
    return np.clip(heat / max(mx, 1e-6), 0, 1)


def peaks(heat: np.ndarray, min_gap: int = 50, k: int = 30) -> list[int]:
    order = np.argsort(-heat)
    chosen: list[int] = []
    for i in order:
        if heat[i] <= 0.05:
            break
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break
    log.info("heat peaks: %d", len(chosen))
    return chosen

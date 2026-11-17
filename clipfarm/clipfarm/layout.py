"""Facecam / camera detection and crop tracks for the 1080x1920 frame.

Modes
  split : game/screen + facecam overlay -> facecam top (TOP_RATIO), divider, game bottom (smart crop)
  cam   : camera-only (IRL/just chatting) -> single 9:16 face-tracked crop; multi-person = active speaker
  game  : no face -> single 9:16 action-tracked crop
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import MODELS_DIR
from .util import log

OUT_W, OUT_H = 1080, 1920
TOP_RATIO = 0.375  # facecam panel share of height in split mode
DIVIDER = 8


def panel_sizes(top_ratio: float = TOP_RATIO) -> dict:
    top = int(round(OUT_H * top_ratio))
    return {"top_h": top, "divider": DIVIDER, "bottom_h": OUT_H - top - DIVIDER}


# ------------------------------------------------------------------ face detection

_DET = {}


def _detector(w: int, h: int):
    import cv2

    model = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    key = (w, h)
    if key not in _DET:
        if model.exists():
            d = cv2.FaceDetectorYN.create(str(model), "", (w, h), score_threshold=0.72, nms_threshold=0.3, top_k=20)
            _DET[key] = ("yunet", d)
        else:
            _DET[key] = ("haar", cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
    return _DET[key]


def detect_faces(frame) -> list[dict]:
    """Return faces as dicts with pixel box (x,y,w,h), score and landmarks (mouth y)."""
    import cv2

    h, w = frame.shape[:2]
    kind, det = _detector(w, h)
    out = []
    if kind == "yunet":
        _, faces = det.detect(frame)
        if faces is None and w < 1000:  # small facecams: retry upscaled
            big = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            kind2, det2 = _detector(w * 2, h * 2)
            _, faces = det2.detect(big)
            if faces is not None:
                faces = faces.copy()
                faces[:, :14] /= 2
        for f in faces if faces is not None else []:
            x, y, fw, fh = f[:4]
            out.append({"box": [float(x), float(y), float(fw), float(fh)], "score": float(f[14]),
                        "mouth_y": float((f[11] + f[13]) / 2)})
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, fw, fh in det.detectMultiScale(gray, 1.1, 5, minSize=(18, 18)):
            out.append({"box": [float(x), float(y), float(fw), float(fh)], "score": 0.8, "mouth_y": float(y + fh * 0.8)})
    return out


def _sample_frames(video: Path, t_start: float | None, t_end: float | None, n: int):
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    a = t_start or 0.0
    b = min(t_end or total, total)
    frames = []
    for t in np.linspace(a + 0.5, max(a + 0.6, b - 0.5), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            frames.append((float(t), fr))
    cap.release()
    return frames


# ------------------------------------------------------------------ facecam box


def _edge_box(E: np.ndarray, face: list[float], W: int, H: int) -> list[float] | None:
    """Find the persistent rectangle (overlay frame) around the face from an averaged edge map."""
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2

    def col_score(x, y0, y1):
        x0, x1 = max(0, x - 1), min(W, x + 2)
        return float(E[max(0, y0):min(H, y1), x0:x1].max(axis=1).mean()) if x1 > x0 and y1 > y0 else 0.0

    def row_score(y, x0, x1):
        y0, y1 = max(0, y - 1), min(H, y + 2)
        return float(E[y0:y1, max(0, x0):min(W, x1)].max(axis=0).mean()) if y1 > y0 and x1 > x0 else 0.0

    ys0, ys1 = int(fy - 0.6 * fh), int(fy + 1.8 * fh)
    xs0, xs1 = int(fx - 0.8 * fw), int(fx + 1.8 * fw)
    best = {}
    for side, rng, fn, args in (
        ("l", range(max(1, int(cx - 5 * fw)), int(fx - 0.15 * fw)), col_score, (ys0, ys1)),
        ("r", range(int(fx + fw * 1.15), min(W - 1, int(cx + 5 * fw))), col_score, (ys0, ys1)),
        ("t", range(max(1, int(cy - 4 * fh)), int(fy - 0.2 * fh)), row_score, (xs0, xs1)),
        ("b", range(int(fy + fh * 1.3), min(H - 1, int(cy + 5 * fh))), row_score, (xs0, xs1)),
    ):
        scores = [(fn(v, *args), v) for v in rng]
        if scores:
            s, v = max(scores)
            best[side] = (s, v)
    th = 0.32
    l = best.get("l", (0, 0))
    r = best.get("r", (0, W))
    t = best.get("t", (0, 0))
    b = best.get("b", (0, H))
    x0 = l[1] if l[0] > th else (0 if fx < 4.5 * fw else None)
    x1 = r[1] if r[0] > th else (W if W - (fx + fw) < 4.5 * fw else None)
    y0 = t[1] if t[0] > th else (0 if fy < 3.5 * fh else None)
    y1 = b[1] if b[0] > th else (H if H - (fy + fh) < 4 * fh else None)
    if None in (x0, x1, y0, y1):
        return None
    bw, bh = x1 - x0, y1 - y0
    if bw < fw * 1.3 or bh < fh * 1.3 or bw * bh > 0.5 * W * H:
        return None
    if not (0.6 <= bw / bh <= 2.6):
        return None
    return [float(x0), float(y0), float(bw), float(bh)]


def detect_layout(video: Path, t_start: float | None = None, t_end: float | None = None, n: int = 48) -> dict:
    """Analyse sampled frames of `video` (file seconds). Returns normalized boxes."""
    import cv2

    frames = _sample_frames(video, t_start, t_end, n)
    if not frames:
        return {"mode": "game", "box": None, "face": None, "presence": 0.0}
    H, W = frames[0][1].shape[:2]
    per = []
    edges = np.zeros((H, W), np.float32)
    for _, fr in frames:
        per.append(detect_faces(fr))
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        edges += (cv2.Canny(g, 60, 150) > 0).astype(np.float32)
    edges /= len(frames)
    faces = [f for fl in per for f in fl]
    presence = sum(1 for fl in per if fl) / len(per)
    if presence < 0.25 or not faces:
        return {"mode": "game", "box": None, "face": None, "presence": round(presence, 3), "src": [W, H]}
    # cluster faces by center (grid 1/8 of frame)
    centers = np.array([[f["box"][0] + f["box"][2] / 2, f["box"][1] + f["box"][3] / 2] for f in faces])
    cells = (centers[:, 0] // (W / 8)).astype(int) * 10 + (centers[:, 1] // (H / 6)).astype(int)
    uniq, counts = np.unique(cells, return_counts=True)
    clusters = []
    for u, c in sorted(zip(uniq, counts), key=lambda z: -z[1]):
        if c / len(per) < 0.2:
            continue
        sel = [f for f, cc in zip(faces, cells) if cc == u]
        med = np.median(np.array([f["box"] for f in sel]), axis=0).tolist()
        clusters.append({"box": med, "presence": c / len(per)})
    if not clusters:
        clusters = [{"box": np.median(np.array([f["box"] for f in faces]), axis=0).tolist(), "presence": presence}]
    main = clusters[0]
    fx, fy, fw, fh = main["box"]
    face_rel = fh / H
    norm = lambda b: [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)]  # noqa: E731
    if face_rel > 0.17 or len(clusters) > 1 and all(c["box"][3] / H > 0.12 for c in clusters[:2]):
        return {"mode": "cam", "box": None, "face": norm(main["box"]), "presence": round(presence, 3),
                "people": len(clusters), "faces": [norm(c["box"]) for c in clusters[:4]], "src": [W, H]}
    box = _edge_box(edges, main["box"], W, H)
    method = "edges"
    if box is None:
        method = "expand"
        bw, bh = fw * 3.4, fh * 3.0
        x0 = min(max(0, fx + fw / 2 - bw / 2), W - bw)
        y0 = min(max(0, fy + fh / 2 - bh * 0.45), H - bh)
        box = [x0, y0, bw, bh]
    log.info("facecam box via %s: %s", method, box)
    return {"mode": "split", "box": norm(box), "face": norm(main["box"]), "presence": round(presence, 3),
            "box_method": method, "src": [W, H]}


# ------------------------------------------------------------------ crops


def top_ratio_for(layout: dict, src_h: int) -> float:
    """Smaller facecams (few source pixels) get a slightly shorter panel to limit upscaling softness."""
    if layout.get("mode") != "split" or not layout.get("box"):
        return TOP_RATIO
    return 0.35 if layout["box"][3] * src_h < 300 else 0.385


def facecam_crop(layout: dict, src_w: int, src_h: int, top_ratio: float = TOP_RATIO) -> dict:
    """Crop (normalized) inside the facecam box that covers the top panel aspect, face-centred."""
    p = panel_sizes(top_ratio)
    aspect = OUT_W / p["top_h"]
    bx, by, bw, bh = layout["box"]
    fx, fy, fw, fh = layout["face"]
    bwp, bhp = bw * src_w, bh * src_h
    if bwp / bhp > aspect:
        cw, ch = bhp * aspect, bhp
    else:
        cw, ch = bwp, bwp / aspect
    fcx, fcy = (fx + fw / 2) * src_w, (fy + fh * 0.45) * src_h
    x0 = min(max(bx * src_w, fcx - cw / 2), bx * src_w + bwp - cw)
    y0 = min(max(by * src_h, fcy - ch * 0.45), by * src_h + bhp - ch)
    return {"x": x0 / src_w, "y": y0 / src_h, "w": cw / src_w, "h": ch / src_h}


def _smooth_track(raw: np.ndarray, dt: float, dead: float = 0.035, max_speed: float = 0.22) -> np.ndarray:
    """Dead-zone + speed-limited follower + gentle EMA: camera-operator-like motion, no jitter."""
    from scipy.ndimage import median_filter

    if len(raw) == 0:
        return raw
    x = median_filter(raw, size=min(7, len(raw) | 1), mode="nearest")
    out = np.zeros_like(x)
    cur = x[0]
    for i, tgt in enumerate(x):
        d = tgt - cur
        if abs(d) > dead:
            step = np.clip(d - np.sign(d) * dead * 0.5, -max_speed * dt, max_speed * dt)
            cur += step
        out[i] = cur
    k = np.ones(5) / 5
    pad = np.pad(out, 2, mode="edge")
    return np.convolve(pad, k, mode="valid")


def track(video: Path, file_start: float, file_end: float, layout: dict, mode: str, fps_sample: float = 3.0,
          top_ratio: float = TOP_RATIO) -> dict:
    """Crop track over [file_start, file_end] of the preview (seconds in file time).

    Returns {"mode", "w", "h", "keys": [{"t": rel_sec, "x": left, "y": top}]} in normalized source coords.
    split -> the game panel crop (bottom); cam/game -> the full 9:16 crop.
    """
    import cv2

    p = panel_sizes(top_ratio)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if mode == "split":
        aspect = OUT_W / p["bottom_h"]
    else:
        aspect = OUT_W / OUT_H
    ch = 1.0
    cw = min(1.0, (H * aspect) / W)
    times = np.arange(file_start, file_end, 1 / fps_sample)
    cx_raw = []
    cy_raw = []
    prev = None
    box = layout.get("box")
    speaker_hist: list[int] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            cx_raw.append(cx_raw[-1] if cx_raw else 0.5)
            cy_raw.append(0.5)
            continue
        if mode == "cam":
            faces = detect_faces(fr)
            if faces:
                if len(faces) > 1 and prev is not None:
                    # active speaker: most mouth-region motion
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    best, bi = -1.0, 0
                    for i, f in enumerate(faces):
                        x, y, fw, fh = map(int, f["box"])
                        y0 = max(0, int(f["mouth_y"] - fh * 0.2))
                        roi = np.abs(g[y0:y0 + int(fh * 0.4), max(0, x):x + fw] - prev[y0:y0 + int(fh * 0.4), max(0, x):x + fw])
                        s = float(roi.mean()) if roi.size else 0.0
                        if s > best:
                            best, bi = s, i
                    speaker_hist.append(bi)
                    # hysteresis: switch only if same speaker wins ~1.5 s
                    recent = speaker_hist[-int(fps_sample * 1.5):]
                    choice = max(set(recent), key=recent.count)
                    f = faces[min(choice, len(faces) - 1)]
                else:
                    f = max(faces, key=lambda z: z["box"][2] * z["box"][3])
                x, y, fw, fh = f["box"]
                cx_raw.append((x + fw / 2) / W)
                cy_raw.append((y + fh / 2) / H)
            else:
                cx_raw.append(cx_raw[-1] if cx_raw else 0.5)
                cy_raw.append(cy_raw[-1] if cy_raw else 0.4)
            prev = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                d = np.abs(g - prev)
                if box:
                    bx, by, bw, bh = box
                    d[int(by * 90):int((by + bh) * 90) + 1, int(bx * 160):int((bx + bw) * 160) + 1] = 0
                # suppress HUD-like static borders: ignore 4% edges
                d[:, :6] = 0
                d[:, -6:] = 0
                col = d.sum(axis=0)
                if col.sum() > 50:
                    xs = np.arange(160) / 160
                    m = float((col * xs).sum() / col.sum())
                    cx_raw.append(0.55 * 0.5 + 0.45 * m)  # strong centre prior (crosshair/action is central)
                else:
                    cx_raw.append(cx_raw[-1] if cx_raw else 0.5)
            else:
                cx_raw.append(0.5)
            prev = g
            cy_raw.append(0.5)
    cap.release()
    dt = 1 / fps_sample
    cx = _smooth_track(np.array(cx_raw), dt, dead=0.03 if mode == "cam" else 0.06,
                       max_speed=0.25 if mode == "cam" else 0.12)
    keys = []
    for t, c in zip(times, cx):
        left = float(np.clip(c - cw / 2, 0, 1 - cw))
        if mode == "split" and box:
            left = _avoid_box(left, cw, box)
        keys.append({"t": round(float(t - file_start), 3), "x": round(left, 4), "y": 0.0})
    return {"mode": mode, "w": round(cw, 4), "h": ch, "keys": keys}


def _avoid_box(left: float, cw: float, box: list[float]) -> float:
    """Shift the game crop so the facecam overlay isn't duplicated in the game panel (if a small shift does it)."""
    bx, by, bw, bh = box
    if bh > 0.6:  # tall overlays can't be avoided
        return left
    if left < bx + bw and left + cw > bx:
        right_opt = bx + bw
        left_opt = bx - cw
        cands = [c for c in (right_opt, left_opt) if 0 <= c <= 1 - cw]
        if cands:
            best = min(cands, key=lambda c: abs(c - left))
            if abs(best - left) < 0.18:
                return best
    return left

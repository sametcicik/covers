"""Audio for a clip: speech edit + cleanup, SFX bed (ducked), loudness normalization (-14 LUFS, TP -1 dB)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .config import SFX_DIR
from .util import run

SR = 48000
TARGET_LUFS = -14.0
TARGET_TP = -1.0


def speech_track(hq: Path, segments: list[dict], fps: int, out: Path, denoise: bool = True) -> Path:
    """Cut the edit from the hq file (frame-exact durations), declick cuts, clean and compress speech."""
    parts = []
    for i, s in enumerate(segments):
        a = s["srcStart"]
        d = (s["outEnd"] - s["outStart"]) / fps
        parts.append(f"[0:a]atrim=start={a:.4f}:duration={d:.4f},asetpts=PTS-STARTPTS,"
                     f"aformat=sample_rates={SR}:channel_layouts=stereo,"
                     f"afade=t=in:st=0:d=0.006,afade=t=out:st={max(0.0, d - 0.008):.4f}:d=0.008[s{i}]")
    chain = "".join(f"[s{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=0:a=1[cat]"
    clean = "highpass=f=70,"
    if denoise:
        clean += "afftdn=nr=8:nf=-32:tn=1,"
    clean += "acompressor=threshold=-21dB:ratio=2.8:attack=6:release=140:makeup=2:knee=4"
    fc = ";".join(parts) + ";" + chain + f";[cat]{clean}[out]"
    run(["ffmpeg", "-y", "-v", "error", "-i", hq, "-filter_complex", fc, "-map", "[out]", "-ar", str(SR), "-ac", "2",
         "-c:a", "pcm_s16le", out])
    return out


def _lufs(x: np.ndarray) -> float:
    meter = pyln.Meter(SR)
    try:
        v = meter.integrated_loudness(x)
        return float(v) if np.isfinite(v) else -40.0
    except ValueError:
        return -40.0


def sfx_bed(events: list[dict], duration: float, speech_lufs: float, out: Path) -> Path | None:
    if not events:
        return None
    n = int(duration * SR) + SR
    bed = np.zeros((n, 2), np.float32)
    for e in events:
        try:
            x, sr = sf.read(str(SFX_DIR / e["file"]), dtype="float32")
        except Exception:
            continue
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        below = float(np.clip(10 + (-(e.get("gain", -8) + 2)) * 0.4, 10, 14))
        stacked = sum(1 for o in events if abs(o["at"] - e["at"]) < 0.2)  # layered hits (whoosh+impact)
        below += 10 * np.log10(max(1, stacked))
        src_l = e.get("lufs") if e.get("lufs") is not None else _lufs(x)
        g = 10 ** ((speech_lufs - below - src_l) / 20)
        i0 = int(e["start"] * SR)
        seg = x[: max(0, min(len(x), n - i0))] * g
        bed[i0:i0 + len(seg)] += seg
        e["applied_db_below_speech"] = below
    bed = bed[: int(duration * SR)]
    sf.write(str(out), bed, SR, subtype="PCM_24")
    return out


def loudnorm(inp: Path, out: Path) -> dict:
    r = run(["ffmpeg", "-hide_banner", "-i", inp, "-af",
             f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP - 0.3}:LRA=11:print_format=json", "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    meas = json.loads(m.group(0)) if m else {}
    af = (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP - 0.3}:LRA=11:linear=true"
          + (f":measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}"
             f":measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}" if meas else "")
          + f",alimiter=limit={10 ** ((TARGET_TP - 0.2) / 20):.4f}:attack=2:release=50:level=false,aresample={SR}")
    run(["ffmpeg", "-y", "-v", "error", "-i", inp, "-af", af, "-ar", str(SR), "-c:a", "pcm_s16le", out])
    return meas


def build_mix(hq: Path, segments: list[dict], fps: int, sfx_events: list[dict], work: Path, denoise: bool = True) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    sp = speech_track(hq, segments, fps, work / "speech.wav", denoise)
    x, _ = sf.read(str(sp), dtype="float32")
    duration = len(x) / SR
    sl = _lufs(x)
    bed = sfx_bed(sfx_events, duration, sl, work / "sfx.wav")
    mixed = work / "mix_raw.wav"
    if bed:
        fc = ("[0:a]asplit=2[sp][key];"
              "[1:a][key]sidechaincompress=threshold=0.03:ratio=5:attack=4:release=220:makeup=1[ducked];"
              "[sp][ducked]amix=inputs=2:normalize=0:duration=first[m]")
        run(["ffmpeg", "-y", "-v", "error", "-i", sp, "-i", bed, "-filter_complex", fc, "-map", "[m]",
             "-c:a", "pcm_s24le", mixed])
    else:
        mixed = sp
    final = work / "mix.wav"
    meas = loudnorm(mixed, final)
    return {"path": str(final), "speech_lufs": round(sl, 2), "duration": duration, "loudnorm_in": meas}


def measure(path: Path) -> dict:
    """Integrated loudness + true peak of the final file (ffmpeg ebur128)."""
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true", "-f", "null", "-"])
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", tail)
    tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail)
    return {"lufs": float(i.group(1)) if i else None, "true_peak": float(tp.group(1)) if tp else None}

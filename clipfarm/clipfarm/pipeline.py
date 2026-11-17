"""End-to-end job orchestration (resumable: every step caches its output in work/<job>/)."""
from __future__ import annotations

import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from . import chat as chatlib
from . import clip as cliplib
from . import download, layout, meta, moments, profile, qa, render, signals
from .config import settings
from .hardware import detect
from .transcribe import merge_transcripts, transcribe
from .util import Progress, log, read_json, slugify, write_json

MAX_FIX_ROUNDS = 2


def job_dir(job_id: str) -> Path:
    return settings.work_dir / job_id


def load_job(job_id: str) -> dict | None:
    return read_json(job_dir(job_id) / "job.json")


def save_job(job: dict) -> None:
    write_json(job_dir(job["id"]) / "job.json", job)


def analysis_window(meta_: dict, start: float | None, end: float | None, max_minutes: float | None) -> tuple[float, float]:
    dur = meta_.get("duration") or 0
    a = max(0.0, start or 0.0)
    b = end if end else (dur or a + 3600)
    if dur:
        b = min(b, dur)
    if max_minutes:
        b = min(b, a + max_minutes * 60)
    if b - a < 20:
        raise RuntimeError("analiz aralığı çok kısa")
    return a, b


def run_job(url: str, n_clips: int = 10, start: float | None = None, end: float | None = None,
            max_minutes: float | None = None, progress_cb=None, clip_settings: dict | None = None,
            job_id: str | None = None) -> dict:
    settings.ensure()
    prog = Progress(progress_cb)
    t_start = time.time()
    prog("resolve", 0.1, url)
    m = download.resolve(url)
    jid = job_id or f"{m['platform']}_{slugify(m['channel'] or 'x', 20)}_{slugify(str(m['id']), 14)}"
    jd = job_dir(jid)
    jd.mkdir(parents=True, exist_ok=True)
    write_json(jd / "meta.json", m)
    w0, w1 = analysis_window(m, start, end, max_minutes)
    job = load_job(jid) or {}
    job.update({"id": jid, "url": url, "platform": m["platform"], "channel": m["channel"], "title": m["title"],
                "vod_id": m["id"], "window": [w0, w1], "n_clips": n_clips, "status": "running",
                "hardware": detect().as_dict(), "started": job.get("started") or time.time()})
    save_job(job)
    prog("resolve", 1.0, f"{m['platform']} / {m['channel']} / {m['title'][:60]}")

    try:
        # ---- profile
        prog("profile", 0.2, m["channel"])
        prof = profile.ensure(m["platform"], m["channel"], m.get("channel_display"))
        job["streamer"] = {"name": prof.get("display_name"), "palette": prof.get("palette"),
                           "avatar": f"/streamers/{prof['slug']}/art/avatar.img" if (prof.get("art") or {}).get("avatar") else None}
        save_job(job)
        prog("profile", 1.0, prof.get("display_name", ""))

        # ---- stage 1
        prog("download_preview", 0.0, f"{(w1 - w0) / 60:.0f} dk analiz aralığı")
        prev = download.stage1(m, w0, w1, jd / "source", progress=lambda f: prog("download_preview", f, ""))
        prog("download_preview", 1.0, f"{prev['method']} / {prev['format']}")
        t0 = prev["start"]
        seconds = int(prev["duration"])

        # ---- chat
        chat_p = jd / "analysis" / "chat.json"
        chat = read_json(chat_p)
        if chat is None:
            prog("chat", 0.0, "")
            chat = chatlib.fetch_chat(m, t0, t0 + seconds, progress=lambda f: prog("chat", f, ""))
            write_json(chat_p, chat)
        prog("chat", 1.0, f"{len(chat)} mesaj")

        # ---- layout + signals
        prog("signals", 0.05, "kadraj tespiti")
        lay = (prof.get("facecam") or {}) if (prof.get("facecam") or {}).get("locked") else None
        if not lay:
            lay = layout.detect_layout(Path(prev["path"])) if prev.get("has_video") else {"mode": "game", "box": None}
            prof["facecam"] = {**lay, "locked": False}
            profile.save(prof)
        sig_p = jd / "analysis" / "signals.json"
        sig_d = read_json(sig_p)
        if sig_d is None:
            prog("signals", 0.2, "ses")
            a_sig = signals.audio_signals(Path(prev["audio"]), t0)
            prog("signals", 0.45, "chat")
            c_sig = signals.chat_signals(chat, t0, seconds)
            prog("signals", 0.6, "görüntü")
            v_sig = signals.video_signals(Path(prev["path"]), t0, seconds, lay if lay.get("mode") == "split" else None) \
                if prev.get("has_video") else {}
            heat = signals.combine(a_sig, c_sig, v_sig, seconds)
            sig_d = {"audio": a_sig, "chat": c_sig, "video": v_sig, "heat": heat.round(4).tolist(), "t0": t0}
            write_json(sig_p, sig_d)
        heat = np.array(sig_d["heat"])
        sig = moments.Signals(t0, heat, sig_d["audio"], sig_d["chat"])
        prog("signals", 1.0, f"mod: {lay.get('mode')}")

        # ---- transcription
        hw = detect()
        scope = settings.transcribe_scope
        if scope == "auto":
            scope = "full" if hw.whisper_device == "cuda" or seconds <= 15 * 60 else "candidates"
        k = int(min(24, max(6, math.ceil(n_clips * 2.2) + 2)))
        pk = signals.peaks(heat, min_gap=55, k=k)
        windows = moments.windows_from_peaks(pk, t0, t0, t0 + seconds)
        tr_p = jd / "analysis" / "transcript.json"
        tr = read_json(tr_p)
        if scope == "full":
            ranges = [(0.0, float(seconds))]
        else:
            rel = sorted((a - t0, b - t0) for a, b in windows)
            merged: list[list[float]] = []
            for a, b in rel:
                if merged and a <= merged[-1][1] + 5:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            ranges = [(a, b) for a, b in merged]
        done = [tuple(x) for x in (tr or {}).get("ranges", [])]
        todo = [(a, b) for a, b in ranges if not any(abs(a + t0 - x[0]) < 1 and abs(b + t0 - x[1]) < 1 for x in done)]
        if todo:
            prog("transcribe", 0.0, f"{hw.whisper_model} / {hw.whisper_device} / {sum(b - a for a, b in todo) / 60:.1f} dk")
            new = transcribe(Path(prev["audio"]), t0, todo, prof, None,
                             progress=lambda f: prog("transcribe", f, ""))
            tr = merge_transcripts(tr, new)
            write_json(tr_p, tr)
        lang = tr.get("language") or "tr"
        words = tr["words"]
        if prof.get("language") in (None, "", "auto"):
            prof["language_detected"] = lang
            profile.save(prof)
        prog("transcribe", 1.0, f"{len(words)} kelime, dil: {lang}")

        # ---- moments
        prog("moments", 0.2, "")
        camp = prof.get("campaign") or {}
        max_d = float(min(59, camp.get("max_duration", 59)))
        if scope == "full" and len(windows) < k:
            pass
        if scope == "full":
            # GPU/full transcript: add sliding windows so transcript-only moments are found too
            step = 150
            extra = [(t0 + a, min(t0 + seconds, t0 + a + 240)) for a in range(0, max(1, seconds - 60), step)]
            win_all = windows + extra
        else:
            win_all = windows
        if m["kind"] == "clip" or seconds < 70:
            # a platform clip is already a moment: keep it whole (sentence-snapped), find its peak
            ws_all = [w for w in words if t0 <= w["s"] <= t0 + seconds]
            if ws_all:
                mo = {"start": ws_all[0]["s"], "end": ws_all[-1]["e"], "score": 75.0, "category": "reaction",
                      "why": "Platformda zaten klip olarak işaretlenmiş an" if lang == "tr" else "Already clipped on the platform",
                      "text": " ".join(w["w"] for w in ws_all), "source": "platform-clip", "signals": {}}
                mo = moments.snap_bounds(mo, words, sig, max_d)
                mo["peak"] = list(moments.find_peak(mo, words, sig))
                cands = [mo]
            else:
                cands = []
        else:
            cands = moments.find_moments(words, win_all, sig, lang, prof.get("display_name") or m["channel"], n_clips,
                                         min_dur=18, max_dur=max_d - 6)
        write_json(jd / "analysis" / "moments.json", cands)
        prog("moments", 1.0, f"{len(cands)} aday")

        # ---- per clip
        job["clips"] = job.get("clips") or []
        accepted = [c for c in job["clips"] if c.get("status") == "ready"]
        tried = {c["moment_key"] for c in job["clips"]}
        ci = len(job["clips"])
        for mo in cands:
            if len(accepted) >= n_clips:
                break
            key = f"{mo['start']:.1f}"
            if key in tried:
                continue
            ci += 1
            cid = f"c{ci:02d}"
            entry = {"id": cid, "moment_key": key, "score": round(mo["score"], 1), "status": "processing",
                     "why": mo["why"], "category": mo.get("category")}
            job["clips"].append(entry)
            save_job(job)
            try:
                res = process_clip(job, m, prof, mo, cid, words, sig, lay, prev, lang, clip_settings or {}, prog,
                                   len(accepted), n_clips)
                entry.update(res)
            except Exception as e:  # noqa: BLE001
                log.error("clip %s failed: %s", cid, traceback.format_exc())
                entry.update(status="error", error=str(e)[:500])
            save_job(job)
            if entry["status"] == "ready":
                accepted.append(entry)
        job["status"] = "done"
        job["elapsed"] = round(time.time() - t_start, 1)
        export_outputs(job)
        save_job(job)
        prog("done", 1.0, f"{len(accepted)} klip hazır")
        return job
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)[:2000]
        save_job(job)
        raise


def process_clip(job: dict, m: dict, prof: dict, mo: dict, cid: str, words: list[dict], sig, lay: dict, prev: dict,
                 lang: str, user_settings: dict, prog, done_count: int, total: int) -> dict:
    jd = job_dir(job["id"])
    cd = jd / "clips" / cid
    cd.mkdir(parents=True, exist_ok=True)
    write_json(cd / "moment.json", mo)
    label = f"klip {done_count + 1}/{total}"
    prog("download_hq", 0.0, label)
    pre = 1.0
    hq = download.stage2(m, mo["start"] - pre, mo["end"] + 1.5, cd, progress=lambda f: prog("download_hq", f, label))
    prog("download_hq", 1.0, f"{label}: {hq['height']}p{hq['fps']}")
    st = {**cliplib.DEFAULT_SETTINGS, **(prof.get("clip_defaults") or {}), **user_settings,
          "sfx_density": user_settings.get("sfx_density", (prof.get("sfx") or {}).get("density", 2)),
          "hook": user_settings.get("hook", (prof.get("hook") or {}).get("enabled", True))}
    st.setdefault("caption_style", (prof.get("captions") or {}).get("style", "dynamic"))
    return build_render_qa(job, prof, mo, cd, hq, words, sig, lay, prev, lang, st, prog, label)


def build_render_qa(job, prof, mo, cd: Path, hq: dict, words, sig, lay, prev, lang, st: dict, prog, label: str) -> dict:
    history = []
    for rnd in range(MAX_FIX_ROUNDS + 1):
        prog("edit", 0.3, label)
        built = cliplib.build(job, mo, cd, hq, prof, words, sig, lay, prev, st)
        write_json(cd / "settings.json", st)
        prog("edit", 1.0, f"{label}: {built['plan']['duration']:.1f}s")
        prog("render", 0.0, label)
        render.render_clip(cd, Path(hq["path"]), progress=lambda f: prog("render", f, label),
                           denoise=st.get("denoise", True))
        prog("qa", 0.2, label)
        res = qa.run_qa(cd, built["props"], built["plan"])
        history.append({"round": rnd, "issues": res["issues"], "fixes": res["fixes"]})
        write_json(cd / "qa.json", {**res, "history": history})
        if res["passed"]:
            break
        if rnd == MAX_FIX_ROUNDS:
            break
        changed = False
        if "shorten" in res["fixes"]:
            mo = dict(mo)
            mo["end"] = max(mo["peak"][1] + 1.0, mo["end"] - 4)
            changed = True
        if "caption_move" in res["fixes"] or "caption_safe" in res["fixes"] or "caption_down" in res["fixes"]:
            st["caption_y_shift"] = st.get("caption_y_shift", 0) + (160 if lay.get("mode") == "split" else 140)
            changed = True
        if "caption_up" in res["fixes"]:
            st["caption_y_shift"] = st.get("caption_y_shift", 0) - 120
            changed = True
        if "glyph" in res["fixes"]:
            st["caption_style"] = "clean"
            changed = True
        sync = res["checks"].get("sync") or {}
        if sync.get("offset_ms") is not None and abs(sync["offset_ms"]) > 60:
            st["caption_offset_ms"] = st.get("caption_offset_ms", 0) - sync["offset_ms"]
            changed = True
        if "remix" in res["fixes"]:
            st["denoise"] = False
            changed = True
        if not changed:
            break
    passed = res["passed"]
    out = {"status": "ready" if passed else "rejected", "qa": {"passed": passed, "issues": res["issues"],
                                                                  "rounds": len(history), "checks": res["checks"]},
           "duration": round(res["checks"]["duration"], 2)}
    # cover + metadata
    props, plan = cliplib.load(cd)
    fps = plan["fps"]
    peak_out = 0.45 if plan["segments"][0]["kind"] == "cold" else max(0.4, (mo["peak"][0] - mo["start"]))
    render.cover(cd / "final.mp4", peak_out, cd / "cover.jpg")
    md = meta.build(mo, job, prof, lang)
    info = {"clip": cd.name, "score": round(mo["score"], 1), "why": mo["why"], "category": mo.get("category"),
            "source": {"platform": job["platform"], "channel": job["channel"], "vod": job["vod_id"],
                       "start": mo["start"], "end": mo["end"], "peak": mo["peak"]},
            "hook": plan["segments"][0]["kind"] == "cold", "duration": out["duration"], "fps": fps,
            **md, "qa": out["qa"], "settings": st, "text": mo.get("text", "")}
    write_json(cd / "info.json", info)
    out.update(titles=md["titles"], meta=md)
    prog("qa", 1.0, f"{label}: {'geçti' if passed else 'reddedildi'}")
    return out


def export_outputs(job: dict) -> list[str]:
    """Rebuild output/<channel>/<job>/ from all ready clips, ranked by score: NN_<title>.mp4/.jpg/.json."""
    dest = settings.output_dir / job["channel"] / job["id"]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    jd = job_dir(job["id"])
    ready = sorted([c for c in job.get("clips", []) if c.get("status") == "ready"], key=lambda c: -c.get("score", 0))
    out = []
    for i, c in enumerate(ready, 1):
        cd = jd / "clips" / c["id"]
        info = read_json(cd / "info.json") or {}
        base = f"{i:02d}_{slugify((info.get('titles') or [c['id']])[0], 40)}"
        shutil.copy(cd / "final.mp4", dest / f"{base}.mp4")
        if (cd / "cover.jpg").exists():
            shutil.copy(cd / "cover.jpg", dest / f"{base}.jpg")
        write_json(dest / f"{base}.json", {**info, "rank": i})
        c["output"] = str(dest / f"{base}.mp4")
        out.append(c["output"])
    return out


def rerender(job_id: str, clip_id: str, new_settings: dict, progress_cb=None) -> dict:
    """Re-render a clip with changed settings (no re-download, no re-analysis)."""
    job = load_job(job_id)
    jd = job_dir(job_id)
    cd = jd / "clips" / clip_id
    m = read_json(jd / "meta.json")
    prof = profile.ensure(m["platform"], m["channel"])
    mo = read_json(cd / "moment.json")
    tr = read_json(jd / "analysis" / "transcript.json")
    sig_d = read_json(jd / "analysis" / "signals.json")
    sig = moments.Signals(sig_d["t0"], np.array(sig_d["heat"]), sig_d["audio"], sig_d["chat"])
    prev = read_json(jd / "source" / "preview.json")
    hq = read_json(cd / "hq.json")
    lay = prof.get("facecam") or layout.detect_layout(Path(prev["path"]))
    st = {**(read_json(cd / "settings.json") or {}), **new_settings}
    prog = Progress(progress_cb)
    res = build_render_qa(job, prof, mo, cd, hq, tr["words"], sig, lay, prev, tr.get("language", "tr"), st, prog,
                          f"{clip_id}")
    job = load_job(job_id)  # re-read: another re-render may have finished meanwhile
    for c in job.get("clips", []):
        if c["id"] == clip_id:
            c.update(res)
    export_outputs(job)
    save_job(job)
    return res

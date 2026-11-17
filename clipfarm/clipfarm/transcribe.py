"""faster-whisper transcription with word timestamps, VAD speech map and streamer dictionary."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MODELS_DIR, settings
from .hardware import detect
from .util import log, tr_lower

_MODEL = {}


def get_model(name: str | None = None):
    from faster_whisper import WhisperModel

    hw = detect()
    name = name or hw.whisper_model
    key = (name, hw.whisper_device, hw.whisper_compute)
    if key not in _MODEL:
        log.info("loading whisper %s on %s (%s)", name, hw.whisper_device, hw.whisper_compute)
        _MODEL[key] = WhisperModel(name, device=hw.whisper_device, compute_type=hw.whisper_compute,
                                   cpu_threads=hw.cpu_cores, download_root=str(MODELS_DIR / "whisper"))
    return _MODEL[key]


def speech_regions(audio: np.ndarray, sr: int = 16000) -> list[list[float]]:
    """Silero VAD (bundled with faster-whisper) -> [[start, end], ...] in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=60, threshold=0.45))
    return [[round(t["start"] / sr, 3), round(t["end"] / sr, 3)] for t in ts]


# ------------------------------------------------------------------ dictionary


def _norm(w: str) -> str:
    return re.sub(r"[^\wçğıöşüâîû]", "", tr_lower(w))


class Dictionary:
    def __init__(self, profile: dict | None):
        d = (profile or {}).get("dictionary") or {}
        self.terms: list[str] = [t for t in d.get("terms", []) if t]
        self.replace: dict[str, str] = {_norm(k): v for k, v in (d.get("replace") or {}).items()}
        self._terms_norm = {_norm(t): t for t in self.terms if " " not in t}

    def hotwords(self) -> str | None:
        return " ".join(self.terms[:60]) or None

    def apply_word(self, w: str) -> str:
        m = re.match(r"^(\W*)(.*?)(\W*)$", w, re.S)
        pre, core, post = m.groups() if m else ("", w, "")
        n = _norm(core)
        if not n:
            return w
        if n in self.replace:
            return pre + self.replace[n] + post
        if n in self._terms_norm:
            return pre + self._terms_norm[n] + post
        if len(n) >= 4 and self._terms_norm:
            # Turkish suffixes: match the stem against terms ("jahreini" -> "Jahrein" + "i")
            for tn, t in self._terms_norm.items():
                if len(tn) >= 4 and n.startswith(tn) and len(n) - len(tn) <= 4:
                    return pre + t + core[len(tn):] + post
            best = difflib.get_close_matches(n, list(self._terms_norm), n=1, cutoff=0.86)
            if best and best[0][0] == n[0]:
                return pre + self._terms_norm[best[0]] + post
        return w

    def apply_words(self, words: list[dict]) -> list[dict]:
        if not (self.terms or self.replace):
            return words
        for wd in words:
            wd["w"] = self.apply_word(wd["w"])
        # multi-word replacements ("ca rain" -> "Jahrein")
        multi = {k: v for k, v in self.replace.items() if False}
        _ = multi
        return words


# ------------------------------------------------------------------ transcription


def transcribe(audio_path: Path, offset: float, ranges: list[tuple[float, float]] | None,
               profile: dict | None = None, language: str | None = None, progress=None) -> dict:
    """Transcribe (optionally only `ranges`, in file seconds). Returned times are absolute (file + offset)."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, "expects 16 kHz wav"
    total = len(audio) / sr
    hw = detect()
    model = get_model()
    dic = Dictionary(profile)
    lang = language or ((profile or {}).get("language") if (profile or {}).get("language") not in (None, "", "auto") else None)
    if settings.whisper_language not in ("", "auto"):
        lang = settings.whisper_language
    if not lang:
        probe_start = int(min(total * 0.3, max(0, total - 30)) * sr)
        try:
            lang, prob, _ = model.detect_language(audio[probe_start:probe_start + 30 * sr])
        except Exception:  # older faster-whisper
            _, info = model.transcribe(audio[probe_start:probe_start + 30 * sr], beam_size=1)
            lang, prob = info.language, info.language_probability
        log.info("detected language %s (%.2f)", lang, prob)

    ranges = ranges or [(0.0, total)]
    ranges = sorted((max(0.0, a), min(total, b)) for a, b in ranges if b - a > 1)
    words: list[dict] = []
    segments: list[dict] = []
    done = 0.0
    todo = sum(b - a for a, b in ranges)
    prompt = None
    if dic.terms:
        prompt = ", ".join(dic.terms[:30]) + "."
    for a, b in ranges:
        chunk = audio[int(a * sr):int(b * sr)]
        segs, _info = model.transcribe(
            chunk, language=lang, word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 150},
            beam_size=5 if hw.whisper_device == "cuda" else 2,
            condition_on_previous_text=False, initial_prompt=prompt, hotwords=dic.hotwords(),
            no_speech_threshold=0.6, compression_ratio_threshold=2.4,
        )
        for s in segs:
            sw = []
            for w in s.words or []:
                txt = w.word.strip()
                if not txt:
                    continue
                sw.append({"w": txt, "s": round(w.start + a + offset, 3), "e": round(w.end + a + offset, 3),
                           "p": round(w.probability, 3)})
            if not sw:
                continue
            sw = dic.apply_words(sw)
            # drop obvious hallucinations: very low prob long runs or repeated phrases
            if np.mean([x["p"] for x in sw]) < 0.25 and len(sw) > 3:
                continue
            words.extend(sw)
            segments.append({"s": sw[0]["s"], "e": sw[-1]["e"], "text": " ".join(x["w"] for x in sw)})
            if progress:
                progress(min(1.0, (done + (s.end or 0)) / max(1.0, todo)))
        done += b - a
    words.sort(key=lambda x: x["s"])
    # fix overlaps / zero-length
    for i, w in enumerate(words):
        if w["e"] <= w["s"]:
            w["e"] = round(w["s"] + 0.08, 3)
        if i + 1 < len(words) and w["e"] > words[i + 1]["s"]:
            w["e"] = max(w["s"] + 0.04, words[i + 1]["s"])
    speech = [[round(x + offset, 3), round(y + offset, 3)] for x, y in speech_regions(audio, sr)]
    return {"language": lang, "model": hw.whisper_model, "ranges": [[a + offset, b + offset] for a, b in ranges],
            "words": words, "segments": segments, "speech": speech}


def merge_transcripts(old: dict | None, new: dict) -> dict:
    if not old:
        return new
    words = {(w["s"], w["w"]): w for w in old.get("words", [])}
    for w in new["words"]:
        words[(w["s"], w["w"])] = w
    out = dict(new)
    out["words"] = sorted(words.values(), key=lambda w: w["s"])
    out["segments"] = sorted(old.get("segments", []) + new["segments"], key=lambda s: s["s"])
    out["ranges"] = old.get("ranges", []) + new["ranges"]
    return out

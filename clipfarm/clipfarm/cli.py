"""CLI: clipfarm <link> --clips 10 | clipfarm serve | clipfarm rerender <job> <clip> | clipfarm doctor"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def _ts(s: str | None) -> float | None:
    if s is None:
        return None
    parts = [float(x) for x in s.split(":")]
    v = 0.0
    for p in parts:
        v = v * 60 + p
    return v


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if argv and argv[0] == "serve":
        from .server import serve

        serve()
        return 0
    if argv and argv[0] == "doctor":
        from .hardware import detect
        from .download import _which_tools
        from .config import settings

        print(json.dumps({"hardware": detect().as_dict(), "tools": _which_tools(), "llm": settings.llm_enabled,
                          "rate_limit": settings.rate_limit or "unlimited"}, indent=2))
        return 0
    if argv and argv[0] == "rerender":
        ap = argparse.ArgumentParser(prog="clipfarm rerender")
        ap.add_argument("job")
        ap.add_argument("clip")
        ap.add_argument("--settings", default="{}", help='JSON, e.g. {"caption_style":"bold","hook":false}')
        a = ap.parse_args(argv[1:])
        from .pipeline import rerender

        print(json.dumps(rerender(a.job, a.clip, json.loads(a.settings)), ensure_ascii=False, indent=2))
        return 0
    ap = argparse.ArgumentParser(prog="clipfarm", description="Yayın linkinden yayına hazır dikey klipler.")
    ap.add_argument("url", help="Kick/Twitch VOD, klip veya kanal linki")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--start", help="analiz başlangıcı (sn veya HH:MM:SS)")
    ap.add_argument("--end", help="analiz bitişi (sn veya HH:MM:SS)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--style", default=None, choices=["dynamic", "bold", "clean", "classic"])
    ap.add_argument("--no-hook", action="store_true")
    ap.add_argument("--sfx", type=int, default=None, choices=[0, 1, 2, 3], help="SFX yoğunluğu")
    ap.add_argument("--no-label", action="store_true")
    a = ap.parse_args(argv)
    st = {}
    if a.style:
        st["caption_style"] = a.style
    if a.no_hook:
        st["hook"] = False
    if a.sfx is not None:
        st["sfx_density"] = a.sfx
    if a.no_label:
        st["label"] = False
    from .pipeline import run_job

    job = run_job(a.url, a.clips, _ts(a.start), _ts(a.end), a.max_minutes, clip_settings=st)
    ready = [c for c in job.get("clips", []) if c.get("status") == "ready"]
    print(f"\n{len(ready)} klip hazır:")
    for c in sorted(ready, key=lambda c: -c["score"]):
        print(f"  [{c['score']:5.1f}] {c.get('output')}\n          {c['titles'][0] if c.get('titles') else ''}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

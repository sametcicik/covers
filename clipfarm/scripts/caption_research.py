"""Re-run the caption-style research on the latest Reels with YOUR OWN logged-in cookies.

python scripts/caption_research.py https://www.instagram.com/nocodealex/reels/ --n 10
Requires COOKIES_FILE (Netscape cookies.txt exported from your browser). Writes docs/caption_ref/research/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "caption_ref" / "research"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cookies = os.environ.get("COOKIES_FILE")
    OUT.mkdir(parents=True, exist_ok=True)
    ytdlp = str(ROOT / ".venv" / "bin" / "yt-dlp")
    cmd = [ytdlp, "--playlist-end", str(a.n), "-o", str(OUT / "%(playlist_index)02d_%(id)s.%(ext)s")]
    if cookies:
        cmd += ["--cookies", cookies]
    r = subprocess.run(cmd + [a.url])
    if r.returncode != 0:
        print("download failed (Instagram needs COOKIES_FILE from a logged-in browser session)", file=sys.stderr)
    for v in sorted(OUT.glob("*.mp4")):
        b = v.stem
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-t", "8", "-vf", "fps=2,scale=270:-2,tile=8x2",
                        "-frames:v", "1", str(OUT / f"sheet_{b}.jpg")])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(v), "-t", "0.5", "-vf",
                        "fps=30,scale=360:-2,tile=5x3", "-frames:v", "1", str(OUT / f"anim_{b}.jpg")])
    print(f"contact sheets in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

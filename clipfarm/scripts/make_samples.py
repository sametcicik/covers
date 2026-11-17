"""Copy every ready clip of the given jobs into docs/samples as light previews (720x1280) + cover + frame sheet.

python scripts/make_samples.py <job_id> [<job_id> ...]
The full-quality 1080x1920 files stay in output/ (git-ignored).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "samples"


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def main(jobs: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample clips", "",
             "Light previews (720x1280, about 2 Mbps) of real end-to-end runs. The full exports are 1080x1920 H.264 High at about 14 Mbps",
             "and live in `output/` (not committed).", ""]
    for jid in jobs:
        job = json.loads((ROOT / "work" / jid / "job.json").read_text())
        ready = sorted([c for c in job.get("clips", []) if c.get("status") == "ready"], key=lambda c: -c["score"])
        lines += [f"## {job['platform'].capitalize()}: {job['channel']} ({job['title'][:70]})", "",
                  f"VOD `{job['vod_id']}`, analysed window {int(job['window'][0])}-{int(job['window'][1])} s.", ""]
        for i, c in enumerate(ready, 1):
            cd = ROOT / "work" / jid / "clips" / c["id"]
            info = json.loads((cd / "info.json").read_text())
            base = f"{job['platform']}_{job['channel']}_{i:02d}"
            run(["ffmpeg", "-y", "-v", "error", "-i", cd / "final.mp4", "-vf", "scale=720:1280", "-c:v", "libx264",
                 "-preset", "slow", "-crf", "27", "-maxrate", "2.4M", "-bufsize", "4M", "-c:a", "aac", "-b:a", "128k",
                 "-movflags", "+faststart", OUT / f"{base}.mp4"])
            run(["ffmpeg", "-y", "-v", "error", "-i", cd / "cover.jpg", "-vf", "scale=540:-2", "-q:v", "4", OUT / f"{base}_cover.jpg"])
            dur = info["duration"]
            sel = "+".join(f"eq(n\\,{int(dur * 60 * k / 9) + 2})" for k in range(8))
            run(["ffmpeg", "-y", "-v", "error", "-i", cd / "final.mp4", "-vf", f"select='{sel}',scale=240:-2,tile=8x1",
                 "-frames:v", "1", "-vsync", "0", "-q:v", "4", OUT / f"{base}_sheet.jpg"])
            q = info["qa"]["checks"]
            sync = q.get("sync") or {}
            lines += [f"### {i}. {info['titles'][0]}  (score {info['score']})", "",
                      f"![sheet]({base}_sheet.jpg)", "",
                      f"- Video: [{base}.mp4]({base}.mp4), cover: [{base}_cover.jpg]({base}_cover.jpg)",
                      f"- Why: {info['why']}",
                      f"- Duration {info['duration']} s, hook (cold open): {'yes' if info['hook'] else 'no'}, category: {info['category']}",
                      f"- Alternative titles: {' / '.join(info['titles'][1:])}",
                      f"- Hashtags: {' '.join(info['hashtags'])}",
                      f"- QA: passed in {info['qa']['rounds']} round(s); loudness {q['loudness']['lufs']} LUFS, true peak "
                      f"{q['loudness']['true_peak']} dBTP; caption sync median {sync.get('median_abs_ms')} ms over "
                      f"{sync.get('matched')} words; black frames: {len(q['black'])}; face overlaps: {len(q['face_overlap_times'])}",
                      ""]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Regenerate LICENSES.md from assets/sfx/manifest.json and assets/fonts/fonts.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sfx = json.loads((ROOT / "assets/sfx/manifest.json").read_text())
fonts = json.loads((ROOT / "assets/fonts/fonts.json").read_text())
L = ["# Third-party assets and licenses", "",
     "ClipFarm code is yours. Bundled media assets come from the sources below; every one allows commercial use.", "",
     "## Sound effects (`assets/sfx/`)", "",
     "- **Mixkit Sound Effects Free License** (https://mixkit.co/license/#sfxFree): free for commercial and personal",
     "  projects, no attribution required; you may not redistribute the sounds as a standalone sound library.",
     "  Mixkit's official summary is at https://mixkit.co/llm-info/.",
     "- **CC0-1.0**: sounds synthesized by `scripts/build_sfx.py` (original work, public domain).",
     "- Pixabay and Freesound were evaluated. Pixabay blocks automated access (HTTP 403) and Freesound's API needs a key,",
     "  so neither is used.", "",
     "| File | Category | Source | Original title | License |", "|---|---|---|---|---|"]
for it in sfx["items"]:
    src = f"[mixkit #{it['asset_url'].split('/')[-2]}]({it['source_url']})" if it["source"] == "mixkit" else "ClipFarm synth"
    L.append(f"| `{it['file']}` | {it['category']} | {src} | {it['title']} | {it['license']} |")
L += ["", "## Fonts (`assets/fonts/`)", "",
      "All fonts are SIL Open Font License 1.1 with no Reserved Font Names. Licence texts are in `assets/fonts/licenses/`.",
      "Variable fonts were instanced to a single weight with fontTools, which the OFL allows (no RFN).", "",
      "| File | Role | Family | Source |", "|---|---|---|---|"]
for f in fonts["fonts"]:
    L.append(f"| `{f['file']}` | {f['role']} | {f['family']} | {f.get('source_url', '')} |")
L += ["", "## Models", "",
      "- `assets/models/face_detection_yunet_2023mar.onnx`: OpenCV Zoo YuNet, MIT License.",
      "- Whisper weights (faster-whisper / CTranslate2 conversions, downloaded on first run): MIT License (OpenAI Whisper).", "",
      "## Software", "",
      "- Remotion (https://remotion.dev): free for individuals and companies with up to 3 employees. Larger companies",
      "  need a company license; set `REMOTION_LICENSE_KEY` in `.env`.",
      "- yt-dlp (Unlicense), streamlink (BSD-2), faster-whisper (MIT), FFmpeg (LGPL/GPL build of your system),",
      "  OpenCV (Apache-2.0)."]
(ROOT / "LICENSES.md").write_text("\n".join(L) + "\n")
print("LICENSES.md written")

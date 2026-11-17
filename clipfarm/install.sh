#!/usr/bin/env bash
# One-command install: Python venv + Node deps + UI build + Remotion browser + health check.
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "ffmpeg gerekli (apt install ffmpeg / brew install ffmpeg)"; exit 1; }
command -v node >/dev/null || { echo "Node.js >= 18 gerekli"; exit 1; }
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
.venv/bin/pip install -q -U pip wheel
.venv/bin/pip install -q -e .
.venv/bin/pip install -q -U --pre "yt-dlp[default,curl-cffi]"   # nightly yt-dlp
if command -v nvidia-smi >/dev/null; then
  .venv/bin/pip install -q "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*" || true
fi
[ -f .env ] || cp .env.example .env
[ -f assets/models/face_detection_yunet_2023mar.onnx ] || curl -sSL -o assets/models/face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
(cd studio && npm install --no-audit --no-fund && npx vite build && NODE_USE_ENV_PROXY=1 npx remotion browser ensure)
[ -f assets/sfx/manifest.json ] || .venv/bin/python scripts/build_sfx.py
mkdir -p bin && ln -sf ../.venv/bin/clipfarm bin/clipfarm
# TwitchDownloaderCLI (optional last-resort Twitch fallback)
if [ ! -x bin/TwitchDownloaderCLI ] && [ "$(uname -s)" = "Linux" ]; then
  U=$(curl -s https://api.github.com/repos/lay295/TwitchDownloader/releases/latest | grep -o 'https://[^"]*CLI[^"]*Linux-x64.zip' | head -1 || true)
  if [ -n "$U" ]; then curl -sSL -o /tmp/tdcli.zip "$U" && unzip -o -q /tmp/tdcli.zip -d bin && chmod +x bin/TwitchDownloaderCLI || true; fi
fi
.venv/bin/clipfarm doctor
echo
echo "Kurulum tamam. Arayüz:  ./bin/clipfarm serve   ->  http://127.0.0.1:8765"
echo "CLI:                    ./bin/clipfarm <link> --clips 10"

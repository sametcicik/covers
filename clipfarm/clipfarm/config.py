"""Runtime configuration (.env) and project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

ASSETS = ROOT / "assets"
FONTS_DIR = ASSETS / "fonts"
SFX_DIR = ASSETS / "sfx"
MODELS_DIR = ASSETS / "models"
STREAMERS_DIR = ROOT / "streamers"
STUDIO_DIR = ROOT / "studio"


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    work_dir: Path = field(default_factory=lambda: Path(_env("CLIPFARM_WORK_DIR", str(ROOT / "work"))))
    output_dir: Path = field(default_factory=lambda: Path(_env("CLIPFARM_OUTPUT_DIR", str(ROOT / "output"))))

    # LLM gateway (OpenAI-compatible /chat/completions)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL").rstrip("/"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-5-5"))
    llm_vision_model: str = field(default_factory=lambda: _env("LLM_VISION_MODEL", _env("LLM_MODEL", "claude-sonnet-5-5")))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 120.0))

    # Download
    rate_limit: str = field(default_factory=lambda: _env("DOWNLOAD_RATE_LIMIT", ""))  # e.g. "4M" (bytes/s, yt-dlp syntax)
    concurrent_fragments: int = field(default_factory=lambda: _env_int("DOWNLOAD_CONCURRENT_FRAGMENTS", 4))
    preview_height: int = field(default_factory=lambda: _env_int("PREVIEW_HEIGHT", 360))
    cookies_file: str = field(default_factory=lambda: _env("COOKIES_FILE", ""))
    section_padding: float = field(default_factory=lambda: _env_float("SECTION_PADDING", 5.0))
    max_height: int = field(default_factory=lambda: _env_int("MAX_HEIGHT", 1080))

    # Transcription
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "auto"))
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "auto"))
    transcribe_scope: str = field(default_factory=lambda: _env("TRANSCRIBE_SCOPE", "auto"))  # auto|full|candidates

    # Render
    remotion_concurrency: int = field(default_factory=lambda: _env_int("REMOTION_CONCURRENCY", 0))
    video_bitrate: str = field(default_factory=lambda: _env("VIDEO_BITRATE", "14M"))
    remotion_license_key: str = field(default_factory=lambda: _env("REMOTION_LICENSE_KEY", "free-license"))
    max_fps: int = field(default_factory=lambda: _env_int("MAX_FPS", 60))

    # Server
    host: str = field(default_factory=lambda: _env("CLIPFARM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("CLIPFARM_PORT", 8765))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key)

    def ensure(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        STREAMERS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()

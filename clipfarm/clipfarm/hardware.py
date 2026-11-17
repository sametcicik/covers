"""Hardware detection: RAM, disk, NVIDIA GPU, NVENC. Picks transcription/encode backends."""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass

from .config import settings


@dataclass
class Hardware:
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    gpu_name: str | None
    gpu_vram_gb: float
    cuda_ok: bool
    nvenc_ok: bool
    whisper_model: str
    whisper_device: str
    whisper_compute: str
    video_encoder: str

    def as_dict(self) -> dict:
        return asdict(self)


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        return 8.0


def _gpu() -> tuple[str | None, float]:
    if not shutil.which("nvidia-smi"):
        return None, 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if out:
            name, mem = [x.strip() for x in out[0].split(",")]
            return name, float(mem) / 1024
    except Exception:
        pass
    return None, 0.0


def _cuda_ok() -> bool:
    try:
        import ctranslate2  # type: ignore

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _nvenc_ok() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def detect() -> Hardware:
    cores = os.cpu_count() or 2
    ram = _ram_gb()
    disk = shutil.disk_usage(str(settings.work_dir if settings.work_dir.exists() else settings.work_dir.parent)).free / 1024**3
    gpu_name, vram = _gpu()
    cuda = bool(gpu_name) and _cuda_ok()
    nvenc = bool(gpu_name) and _nvenc_ok()

    if settings.whisper_model != "auto":
        model = settings.whisper_model
    elif cuda and vram >= 9:
        model = "large-v3"
    elif cuda and vram >= 5:
        model = "large-v3"  # int8_float16 fits in ~5 GB
    elif cuda:
        model = "medium"
    elif cores >= 12 and ram >= 16:
        model = "medium"
    else:
        model = "small"

    if cuda:
        device, compute = "cuda", ("float16" if vram >= 9 else "int8_float16")
    else:
        device, compute = "cpu", "int8"

    return Hardware(
        cpu_cores=cores,
        ram_gb=round(ram, 1),
        disk_free_gb=round(disk, 1),
        gpu_name=gpu_name,
        gpu_vram_gb=round(vram, 1),
        cuda_ok=cuda,
        nvenc_ok=nvenc,
        whisper_model=model,
        whisper_device=device,
        whisper_compute=compute,
        video_encoder="h264_nvenc" if nvenc else "libx264",
    )

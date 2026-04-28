"""Capture environment + hardware info for run provenance."""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone

import torch


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def get_hardware_info(device: torch.device | str | None = None) -> dict:
    """Return a flat dict describing the run environment."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["cuda_version"] = torch.version.cuda
    return info

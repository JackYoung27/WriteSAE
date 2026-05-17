"""Shared local-only constants and helpers for Modal experiment scripts.

This module is imported by the top-level driver scripts in `experiments/`
before the Modal image is built. It is NOT shipped to the Modal worker
(it lives outside `/root` and is never `add_local_file`-mounted), so any
function added here must be safe to run on the local machine only.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# Pre-built wheels matching the rest of the image stack (torch 2.8 + cu126,
# Python 3.12, cxx11 ABI). Updating either URL must be done in this one file.
CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    "v1.6.1.post4/"
    "causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
MAMBA_SSM_WHEEL = (
    "https://github.com/state-spaces/mamba/releases/download/"
    "v2.3.1/"
    "mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)


def code_sha() -> str:
    """Return the short git SHA of the working tree, or 'unknown' if it fails."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"

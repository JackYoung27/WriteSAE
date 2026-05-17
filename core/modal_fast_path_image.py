"""Reusable Modal image with CUDA fast paths for GDN (fla) and Mamba-2 (mamba-ssm).

Two strategies are provided:

  STRATEGY A (PRIMARY): CUDA 12.6 + PyTorch 2.8 + pre-built wheels
    Both mamba-ssm and causal-conv1d publish pre-built wheels for cu12+torch2.8.
    These wheels predate the CUDAStream::query() ABI break in PyTorch 2.10, so they
    load without symbol errors.  flash-linear-attention is Triton-based and works
    with any PyTorch >= 2.5.

  STRATEGY B (ALTERNATIVE): CUDA 13.0 + PyTorch 2.10 + build from source
    The pre-built wheels for cu13+torch2.10 are broken: they were compiled against
    c10::cuda::CUDAStream::query(), a symbol removed in PyTorch 2.10.
    The fix (confirmed in github.com/state-spaces/mamba/issues/891) is to build
    both causal-conv1d and mamba-ssm from their GitHub main branches, which have
    the ABI fix.  This takes ~5-10 minutes of compilation during image build.

Usage:
    from modal_fast_path_image import fast_image, fast_image_from_source
    # or:
    from modal_fast_path_image import FAST_IMAGE_PREBUILT, FAST_IMAGE_FROM_SOURCE

Import examples inside Modal functions:
    import causal_conv1d
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


def _current_code_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (FileNotFoundError, OSError):
        sha = ""
    return sha or os.environ.get("MATRIX_SAE_CODE_SHA", "unknown")

CURRENT_CODE_SHA = _current_code_sha()

# Pre-built wheels: PyTorch 2.8 still exposes CUDAStream::query() so cu12+torch2.8+cxx11abiTRUE wheels load cleanly.
CAUSAL_CONV1D_WHEEL_PREBUILT = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    "v1.6.1.post4/"
    "causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
MAMBA_SSM_WHEEL_PREBUILT = (
    "https://github.com/state-spaces/mamba/releases/download/"
    "v2.3.1/"
    "mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

_SMOKE_TEST = (
    "python -c \""
    "import causal_conv1d; "
    "print(f'causal_conv1d {causal_conv1d.__version__}'); "
    "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; "
    "print('mamba_ssm.ops OK'); "
    "from fla.modules import FusedRMSNormGated; "
    "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; "
    "print('fla fast-path OK'); "
    "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}'); "
    "print('ALL_FASTPATH_IMPORTS_OK')"
    "\""
)

# devel base provides nvcc for Triton JIT in flash-linear-attention; cxx11abiTRUE matches devel's default C++11 ABI.
fast_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    # transformers 5.5.x is earliest native Qwen3.5 support; fla 0.4.2 dodges the triton-3.0 'STAGE' autotuner bug.
    .pip_install(
        "transformers==5.5.4", "datasets", "numpy", "tqdm",
        "matplotlib", "wandb", "accelerate", "sentencepiece", "scipy", "scikit-learn",
        "einops", "ninja", "h5py", "flash-linear-attention==0.4.2",
    )
    # --no-deps so the wheels don't drag in a different torch.
    .run_commands(
        f"python -m pip install --no-deps '{CAUSAL_CONV1D_WHEEL_PREBUILT}'",
        f"python -m pip install --no-deps '{MAMBA_SSM_WHEEL_PREBUILT}'",
    )
    .run_commands(_SMOKE_TEST)
    .env({"MATRIX_SAE_CODE_SHA": CURRENT_CODE_SHA})
)

FAST_IMAGE_PREBUILT = fast_image


# Strategy B works around broken cu13+torch2.10 GitHub-release wheels (mamba issue #891) by building from source.
fast_image_from_source = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
            "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
            "CAUSAL_CONV1D_FORCE_CXX11_ABI": "TRUE",
            "MAMBA_FORCE_BUILD": "TRUE",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    # PyPI default resolves to a CPU wheel; pin the cu126 index so source builds compile against CUDA torch.
    .run_commands(
        "python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install(
        "transformers>=5.0", "datasets", "numpy", "tqdm",
        "matplotlib", "wandb", "accelerate", "sentencepiece", "scipy", "scikit-learn",
        "einops", "ninja", "flash-linear-attention",
    )
    .run_commands(
        "python -m pip install --no-build-isolation --no-deps "
        "\"causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git\"",
    )
    # --no-cache-dir avoids reusing a broken cached wheel.
    .run_commands(
        "python -m pip install --no-build-isolation --no-cache-dir --no-deps "
        "\"mamba-ssm @ git+https://github.com/state-spaces/mamba.git\"",
    )
    .run_commands(_SMOKE_TEST)
    .env({"MATRIX_SAE_CODE_SHA": CURRENT_CODE_SHA})
)

FAST_IMAGE_FROM_SOURCE = fast_image_from_source

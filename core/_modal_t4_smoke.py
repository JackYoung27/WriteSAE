"""T4 smoke test for fast_image after adding sm_75 to TORCH_CUDA_ARCH_LIST.

Verifies:
  1. fast_image builds (image rebuild triggered by env var change).
  2. Container runs on T4 (compute capability 7.5).
  3. causal_conv1d, mamba_ssm, fla all import and their Triton kernels
     compile for sm_75 (one real chunk_gated_delta_rule call).

Usage:
    modal run core/_modal_t4_smoke.py
"""
from __future__ import annotations

from pathlib import Path

import modal

from core.modal_fast_path_image import fast_image

# Mount core/ into /root/core so the remote module resolves the same way.
_core_dir = Path(__file__).resolve().parent
_image = fast_image.add_local_dir(str(_core_dir), "/root/core", copy=True)

app = modal.App("matrix-sae-t4-smoke", image=_image)


@app.function(gpu="T4", timeout=900)
def smoke() -> str:
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    assert torch.cuda.is_available(), "no CUDA on T4 container"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU={name} capability=sm_{cap[0]}{cap[1]}")
    assert cap == (7, 5), f"expected sm_75 (T4), got sm_{cap[0]}{cap[1]}"

    import causal_conv1d
    print(f"causal_conv1d {causal_conv1d.__version__}")

    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
    print("mamba_ssm.ops OK")

    # Trigger Triton JIT compilation on sm_75 with a realistic GDN shape.
    b, t, h, dk, dv = 1, 32, 4, 64, 64
    dtype = torch.bfloat16
    q = torch.randn(b, t, h, dk, device="cuda", dtype=dtype)
    k = torch.randn(b, t, h, dk, device="cuda", dtype=dtype)
    v = torch.randn(b, t, h, dv, device="cuda", dtype=dtype)
    g = torch.randn(b, t, h, device="cuda", dtype=torch.float32).clamp(max=-0.01)
    beta = torch.sigmoid(torch.randn(b, t, h, device="cuda", dtype=dtype))

    out, state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert out.shape == (b, t, h, dv), f"unexpected out shape {out.shape}"
    assert state is not None and state.shape == (b, h, dk, dv), \
        f"unexpected state shape {state.shape if state is not None else None}"
    print(f"GDN kernel out={tuple(out.shape)} state={tuple(state.shape)}")
    print("FASTPATH_OK")
    return "FASTPATH_OK"


@app.local_entrypoint()
def main():
    result = smoke.remote()
    print(f"remote returned: {result}")

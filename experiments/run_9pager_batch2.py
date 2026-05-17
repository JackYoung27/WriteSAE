"""9-pager batch 2: FLOP counting, cross-head covariance, JumpReLU downstream.

These run in parallel with E7 (4B downstream on A100).

Usage:
    modal run experiments/run_9pager_batch2.py
"""
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha


CODE_SHA = code_sha()

_ext_dir = Path(__file__).resolve().parent / "extraction"
_core_dir = Path(__file__).resolve().parent.parent / "core"
_analysis_dir = Path(__file__).resolve().parent / "analysis"
_ablation_dir = Path(__file__).resolve().parent / "ablations"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "MAX_JOBS": "4", "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++",
    })
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands("python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers>=5.0", "datasets", "numpy", "tqdm",
        "matplotlib", "accelerate", "sentencepiece", "scipy", "scikit-learn",
        "einops", "ninja", "flash-linear-attention",
    )
    .run_commands(
        f"python -m pip install --no-deps '{CAUSAL_CONV1D_WHEEL}'",
        f"python -m pip install --no-deps '{MAMBA_SSM_WHEEL}'",
    )
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .add_local_file(str(_analysis_dir / "probe_features.py"), "/root/probe_features.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation.py"), "/root/circuit_ablation.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("9pager-batch2")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")

DATA = "/data"
MODELS = "/models"
ABLATION = "/ablation"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
N_FEATURES = 2048
K = 32


# ===================================================================
# FLOP counting per architecture
# ===================================================================

@app.function(
    volumes={DATA: data_vol},
    gpu="A10G", image=image, timeout=600, memory=16384,
)
def exp_flop_count() -> dict[str, Any]:
    """Count FLOPs for encode + decode per architecture at nf=2048."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    print("=== FLOP counting ===")

    from train import train as train_sae
    from sae import FlatSAE, MatrixSAE, BilinearMatrixSAE

    d_k, d_v = 128, 128
    d_in = d_k * d_v
    nf = N_FEATURES
    batch = 64

    results = {}

    for sae_type, sae_cls, kwargs in [
        ("flat", FlatSAE, {"d_in": d_in, "n_features": nf, "k": K}),
        ("rank1", MatrixSAE, {"d_k": d_k, "d_v": d_v, "n_features": nf, "k": K, "rank": 1}),
        ("bilinear", BilinearMatrixSAE, {"d_k": d_k, "d_v": d_v, "n_features": nf, "k": K, "rank": 1}),
    ]:
        sae = sae_cls(**kwargs).cuda().eval()
        n_params = sum(p.numel() for p in sae.parameters())

        if sae_type == "flat":
            x = torch.randn(batch, d_in, device="cuda")
        else:
            x = torch.randn(batch, d_k, d_v, device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = sae(x)

        # Timed encode + decode (1000 iterations)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(100):
                out = sae(x)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        samples_per_sec = (100 * batch) / elapsed

        # Count FLOPs analytically
        if sae_type == "flat":
            # Encode: x @ W_e^T = (batch, d_in) @ (d_in, nf) = batch * d_in * nf MACs
            # TopK: negligible
            # Decode: a @ W_d = (batch, nf) @ (nf, d_in) = batch * nf * d_in MACs
            encode_flops = d_in * nf  # per sample
            decode_flops = nf * d_in
        elif sae_type == "rank1":
            # Encode: same as flat (dense encoder)
            # Decode: sum_i a_i * v_i w_i^T = k * (d_k + d_v) per sample (sparse)
            encode_flops = d_in * nf
            decode_flops = K * (d_k + d_v)  # only k active atoms
        else:  # bilinear
            # Encode: for each atom: v_i^T @ S @ w_i = d_k*d_v + d_k per atom, nf atoms
            # Actually: einsum("irk,bkv,irv->bi") ~ nf * d_k * d_v per sample
            encode_flops = nf * d_k * d_v
            # Decode: same as rank1
            decode_flops = K * (d_k + d_v)

        total_flops = encode_flops + decode_flops

        results[sae_type] = {
            "n_params": n_params,
            "encode_flops_per_sample": encode_flops,
            "decode_flops_per_sample": decode_flops,
            "total_flops_per_sample": total_flops,
            "samples_per_sec": samples_per_sec,
            "ms_per_sample": 1000 / samples_per_sec,
            "flop_ratio_vs_flat": None,  # filled below
        }
        print(f"  {sae_type}: {n_params:,} params, {total_flops:,} FLOPs/sample, "
              f"{samples_per_sec:.0f} samples/s, {1000/samples_per_sec:.3f} ms/sample")

        del sae
        torch.cuda.empty_cache()

    flat_flops = results["flat"]["total_flops_per_sample"]
    for t in results:
        results[t]["flop_ratio_vs_flat"] = results[t]["total_flops_per_sample"] / flat_flops

    print("\n  FLOP ratios vs flat:")
    for t, r in results.items():
        print(f"    {t}: {r['flop_ratio_vs_flat']:.3f}x")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "flop_count"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "flop_count.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# Cross-head covariance of bilinear vs flat activations
# ===================================================================

@app.function(
    volumes={DATA: data_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=3600, memory=32768,
)
def exp_cross_head_covariance(
    layer: int = 9,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure inter-head activation covariance for bilinear vs flat SAEs.

    For each sample, encode through 16 per-head SAEs to get 16 activation
    vectors. Compute the mean pairwise correlation across heads. If bilinear
    activations are more correlated, that explains why collective replacement
    works better than per-head replacement.
    """
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    from scipy.stats import pearsonr

    from sae import build_sae_from_config

    print(f"=== Cross-head covariance, L{layer} ===")

    states_dir = Path(DATA) / "states"
    head_states = []
    for h in range(16):
        path = states_dir / f"layer_{layer}" / f"head_{h}.npy"
        data = np.load(str(path), mmap_mode="r")[:n_samples]
        head_states.append(data)

    results = {"layer": layer, "n_samples": n_samples}

    for sae_type in ["bilinear", "flat"]:
        print(f"\n  {sae_type}:")

        # Load 16 per-head SAEs
        saes = {}
        for h in range(16):
            ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type}_s{seed}"
            best_path = ckpt_dir / "best.pt"
            if not best_path.exists():
                continue
            ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            sd = ckpt.get("model_state_dict", {})
            sae = build_sae_from_config(config=cfg, state_dict=sd)
            sae.load_state_dict(sd)
            sae = sae.cuda().eval()
            saes[h] = sae

        if len(saes) < 16:
            print(f"    Only {len(saes)}/16 SAEs loaded, skipping")
            continue

        # Encode each sample through all 16 heads
        # Collect: for each sample, the TopK activation pattern per head
        # Measure: how correlated are the activation magnitudes across heads?

        all_activations = []  # (n_samples, 16, nf)

        for i in range(0, n_samples, 64):
            j = min(i + 64, n_samples)
            batch_acts = []
            for h in range(16):
                state_batch = torch.from_numpy(
                    head_states[h][i:j].astype(np.float32)
                ).cuda()
                if sae_type == "flat":
                    state_batch = state_batch.reshape(j - i, -1)
                with torch.no_grad():
                    out = saes[h](state_batch)
                # coefficients is (batch, n_features), sparse with zeros from TopK
                batch_acts.append(out.coefficients.cpu().numpy())

            # Stack: (batch, 16, nf)
            stacked = np.stack(batch_acts, axis=1)
            all_activations.append(stacked)

        all_activations = np.concatenate(all_activations, axis=0)  # (n_samples, 16, nf)
        print(f"    Activations shape: {all_activations.shape}")

        # Measure inter-head correlation
        # For each pair of heads, compute mean correlation of activation vectors across samples
        n_heads = 16
        corr_matrix = np.zeros((n_heads, n_heads))

        for h1 in range(n_heads):
            for h2 in range(h1 + 1, n_heads):
                # Flatten: each head's activations across all samples
                a1 = all_activations[:, h1, :].flatten()
                a2 = all_activations[:, h2, :].flatten()
                # Only compute on nonzero entries
                mask = (a1 != 0) | (a2 != 0)
                if mask.sum() > 10:
                    r, _ = pearsonr(a1[mask], a2[mask])
                    corr_matrix[h1, h2] = r
                    corr_matrix[h2, h1] = r

        mean_corr = corr_matrix[np.triu_indices(n_heads, k=1)].mean()
        std_corr = corr_matrix[np.triu_indices(n_heads, k=1)].std()

        # Also measure: fraction of features that fire on multiple heads
        any_active = (all_activations > 0).any(axis=0)  # (16, nf) bool
        multi_head_features = (any_active.sum(axis=0) > 1).sum()
        total_active = (any_active.sum(axis=0) > 0).sum()

        results[sae_type] = {
            "mean_interhead_corr": float(mean_corr),
            "std_interhead_corr": float(std_corr),
            "multi_head_features": int(multi_head_features),
            "total_active_features": int(total_active),
            "multi_head_frac": float(multi_head_features / max(total_active, 1)),
        }
        print(f"    Mean inter-head correlation: {mean_corr:.4f} ± {std_corr:.4f}")
        print(f"    Multi-head features: {multi_head_features}/{total_active} ({multi_head_features/max(total_active,1)*100:.1f}%)")

        # Free memory
        for sae in saes.values():
            del sae
        torch.cuda.empty_cache()

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "cross_head_covariance"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# JumpReLU / Gated SAE comparison (train + downstream)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=7200, memory=32768,
)
def exp_jumprelu_comparison(
    layer: int = 9,
    seeds: list[int] = [0, 1, 42],
) -> dict[str, Any]:
    """Train BilinearSAE and FlatSAE with BatchTopK (proxy for gated/JumpReLU),
    compare MSE. Full JumpReLU needs code changes, but BatchTopK already exists
    in the codebase and tests whether dynamic sparsity changes the ordering.
    """
    import sys
    sys.path.insert(0, "/root")
    import numpy as np

    from train import train as train_sae

    print(f"=== BatchTopK comparison, layer {layer} ===")

    states_dir = f"{DATA}/states"
    results = {"layer": layer, "training": {}}

    configs = [
        ("flat_topk", "flat", False),
        ("flat_batchtopk", "flat", True),
        ("bilinear_topk", "bilinear", False),
        ("bilinear_batchtopk", "bilinear", True),
    ]

    for tag, sae_type, use_btk in configs:
        for seed in seeds:
            key = f"{tag}_s{seed}"
            out_dir = f"{DATA}/checkpoints/sparsity_compare/{key}_L{layer}"

            print(f"  Training {key}...")
            t0 = time.time()
            out = train_sae(
                sae_type=sae_type,
                data_dir=states_dir,
                layer=layer, head=0,
                n_features=N_FEATURES, k=K,
                lr=3e-4, lr_min=3e-5,
                batch_size=256, epochs=20,
                warmup_steps=50, norm_every=100,
                resample_every=250,
                rank=1, seed=seed,
                use_batchtopk=use_btk,
                output_dir=out_dir,
            )
            elapsed = time.time() - t0

            results["training"][key] = {
                "sae_type": sae_type,
                "use_batchtopk": use_btk,
                "seed": seed,
                "best_mse": out.get("best_mse", out.get("best_val_mse")),
                "n_dead": out.get("final_n_dead"),
                "time_s": elapsed,
            }
            print(f"    MSE={results['training'][key]['best_mse']:.6e}, "
                  f"dead={results['training'][key]['n_dead']}, {elapsed:.0f}s")

    # Summary
    print("\n  === Summary ===")
    for tag, _, _ in configs:
        mses = [v["best_mse"] for k, v in results["training"].items() if k.startswith(tag)]
        deads = [v["n_dead"] for k, v in results["training"].items() if k.startswith(tag)]
        if mses:
            print(f"  {tag}: MSE={np.mean(mses):.6e} ± {np.std(mses):.6e}, dead={np.mean(deads):.0f}")

    # Save
    out_dir_path = Path(DATA) / "reviewer_experiments" / "sparsity_compare"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    with open(out_dir_path / f"L{layer}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main():
    t0 = time.time()
    handles = [
        ("flop_count", exp_flop_count.spawn()),
        ("cross_head_cov", exp_cross_head_covariance.spawn(layer=9, n_samples=1000)),
        ("batchtopk_compare", exp_jumprelu_comparison.spawn(layer=9, seeds=[0, 1, 42])),
    ]

    print(f"{len(handles)} jobs launched. Waiting...")

    results = {}
    failures: list[str] = []
    for name, handle in handles:
        try:
            result = handle.get()
            results[name] = result
            print(f"\n✓ {name} complete")

            if name == "flop_count":
                for t, r in result.items():
                    print(f"  {t}: {r['total_flops_per_sample']:,} FLOPs, {r['flop_ratio_vs_flat']:.3f}x flat")
            elif name == "cross_head_cov":
                for t in ["bilinear", "flat"]:
                    if t in result:
                        r = result[t]
                        print(f"  {t}: inter-head corr={r['mean_interhead_corr']:.4f}, multi-head={r['multi_head_frac']*100:.1f}%")
            elif name == "batchtopk_compare":
                import numpy as np
                for tag in ["flat_topk", "flat_batchtopk", "bilinear_topk", "bilinear_batchtopk"]:
                    mses = [v["best_mse"] for k, v in result["training"].items() if k.startswith(tag)]
                    if mses:
                        print(f"  {tag}: MSE={np.mean(mses):.6e}")

        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results[name] = {"error": str(e)}
            failures.append(f"{name}: {e}")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out_path = Path("results/data/9pager_batch2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    if failures:
        raise RuntimeError("One or more batch2 jobs failed: " + "; ".join(failures))

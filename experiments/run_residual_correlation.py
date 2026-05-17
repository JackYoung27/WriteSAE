"""Measure reconstruction residual correlation across heads.

Hypothesis: flat SAE residuals are more correlated across heads than bilinear
residuals. Correlated residuals compound when all 16 heads are replaced
simultaneously; uncorrelated residuals partially cancel.

Metrics:
  1. Mean pairwise cosine similarity of residual vectors across heads
  2. Residual Frobenius norm per head
  3. Variance explained by top-k PCs of residuals (structured vs random)

Data: matrix-sae-data-08b-clean volume, states/layer_9/head_{H}.npy
SAEs:  layer-encoder-swap-v1 volume, layer_9/head_{H}/{type}_s42/best.pt

Usage:
    modal run experiments/run_residual_correlation.py
"""
import json
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha

# Infrastructure (same image/volume setup as run_reviewer_round2.py)


CODE_SHA = code_sha()


_core_dir = Path(__file__).resolve().parent.parent / "core"

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
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("residual-correlation")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")

DATA = "/data"
ABLATION = "/ablation"
N_HEADS = 16
N_FEATURES = 2048
K = 32


def _load_sae(ckpt_dir: str, device: str = "cuda"):
    """Load SAE from checkpoint directory containing best.pt and config.json."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    from sae import build_sae_from_config

    best_path = Path(ckpt_dir) / "best.pt"
    config_path = Path(ckpt_dir) / "config.json"

    cfg = json.loads(config_path.read_text())
    ckpt = torch.load(str(best_path), map_location="cpu", weights_only=True)
    sae = build_sae_from_config(cfg, state_dict=ckpt["model_state_dict"])
    sae.load_state_dict(ckpt["model_state_dict"])
    return sae.to(device).eval()


# ===================================================================
# Main experiment: residual correlation across heads
# ===================================================================

@app.function(
    volumes={DATA: data_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=3600, memory=32768,
)
def measure_residual_correlation(
    layer: int = 9,
    n_samples: int = 1000,
    seed: int = 42,
    n_pcs: int = 10,
) -> dict[str, Any]:
    """Compute residual correlation, norms, and PCA structure for flat vs bilinear."""
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from itertools import combinations

    rng = np.random.default_rng(seed)
    print(f"=== Residual correlation: layer {layer}, N={n_samples} ===")

    # -----------------------------------------------------------------
    # Load states for all 16 heads, sample N indices (shared across heads)
    # -----------------------------------------------------------------
    states_dir = Path(DATA) / "states" / f"layer_{layer}"
    head0_path = states_dir / "head_0.npy"
    total_available = np.load(str(head0_path), mmap_mode="r").shape[0]
    idx = rng.choice(total_available, size=min(n_samples, total_available), replace=False)
    idx.sort()
    N = len(idx)

    print(f"  Total states available: {total_available}, using N={N}")

    head_states = {}  # h -> (N, 128, 128) float32 tensor on cuda
    for h in range(N_HEADS):
        arr = np.load(str(states_dir / f"head_{h}.npy"), mmap_mode="r")
        head_states[h] = torch.from_numpy(arr[idx].astype(np.float32)).cuda()
    print(f"  Loaded {N_HEADS} heads, shape per head: {head_states[0].shape}")

    results = {"layer": layer, "n_samples": N, "seed": seed, "sae_types": {}}

    # -----------------------------------------------------------------
    # For each SAE type, compute residuals and all metrics
    # -----------------------------------------------------------------
    for sae_type in ["flat", "bilinear"]:
        print(f"\n--- {sae_type} ---")

        saes = {}
        missing = []
        for h in range(N_HEADS):
            ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type}_s{seed}"
            if not (ckpt_dir / "best.pt").exists():
                missing.append(h)
                continue
            saes[h] = _load_sae(str(ckpt_dir))

        if missing:
            print(f"  Missing heads: {missing}")
        if len(saes) < 2:
            print(f"  Too few heads ({len(saes)}), skipping {sae_type}")
            continue
        print(f"  Loaded {len(saes)}/{N_HEADS} SAEs")

        # Compute residuals: (N, 16384) per head
        residuals = {}  # h -> (N, 16384) tensor
        norms = {}      # h -> (N,) tensor of per-sample Frobenius norms
        mses = {}       # h -> scalar mean MSE

        for h in sorted(saes.keys()):
            sae = saes[h]
            x = head_states[h]  # (N, 128, 128)

            with torch.no_grad():
                if sae_type == "flat":
                    x_in = x.reshape(N, -1)  # (N, 16384)
                    out = sae(x_in)
                    recon = out.reconstruction  # (N, 16384)
                    residual = x_in - recon  # (N, 16384)
                else:
                    out = sae(x)
                    recon = out.reconstruction  # (N, 128, 128)
                    residual = (x - recon).reshape(N, -1)  # (N, 16384)

            norms[h] = residual.norm(dim=-1)  # (N,)
            mses[h] = (residual ** 2).mean().item()
            residuals[h] = residual  # keep on GPU

            if h % 4 == 0:
                print(f"    head {h}: MSE={mses[h]:.6f}, "
                      f"mean_norm={norms[h].mean().item():.4f}")

        # -----------------------------------------------------------------
        # Metric 1: Pairwise cosine similarity of residuals
        # -----------------------------------------------------------------
        available_heads = sorted(residuals.keys())
        n_heads_available = len(available_heads)
        pair_cosines = []
        pair_labels = []

        # Normalize residuals once for cosine computation
        normed = {}
        for h in available_heads:
            n = residuals[h].norm(dim=-1, keepdim=True).clamp(min=1e-8)
            normed[h] = residuals[h] / n

        for h1, h2 in combinations(available_heads, 2):
            # Per-sample cosine similarity, then average
            cos = (normed[h1] * normed[h2]).sum(dim=-1)  # (N,)
            pair_cosines.append(cos.mean().item())
            pair_labels.append(f"{h1}-{h2}")

        mean_pairwise_cos = float(np.mean(pair_cosines))
        std_pairwise_cos = float(np.std(pair_cosines))
        median_pairwise_cos = float(np.median(pair_cosines))
        max_pairwise_cos = float(np.max(pair_cosines))
        min_pairwise_cos = float(np.min(pair_cosines))

        print(f"  Pairwise cosine: mean={mean_pairwise_cos:.4f}, "
              f"std={std_pairwise_cos:.4f}, "
              f"range=[{min_pairwise_cos:.4f}, {max_pairwise_cos:.4f}]")

        cos_matrix = np.zeros((n_heads_available, n_heads_available))
        np.fill_diagonal(cos_matrix, 1.0)
        pair_idx = 0
        for i, h1 in enumerate(available_heads):
            for j, h2 in enumerate(available_heads):
                if j <= i:
                    continue
                cos_matrix[i, j] = pair_cosines[pair_idx]
                cos_matrix[j, i] = pair_cosines[pair_idx]
                pair_idx += 1

        # -----------------------------------------------------------------
        # Metric 2: Per-head residual statistics
        # -----------------------------------------------------------------
        per_head_stats = {}
        for h in available_heads:
            per_head_stats[h] = {
                "mse": mses[h],
                "mean_frobenius_norm": float(norms[h].mean().item()),
                "std_frobenius_norm": float(norms[h].std().item()),
            }

        # -----------------------------------------------------------------
        # Metric 3: PCA of residuals (across heads, per sample)
        # For each sample, we have 16 residual vectors (one per head).
        # Stack all heads' residuals: (N * n_heads, 16384) and run PCA.
        # Also: per-head PCA to check if individual residuals are low-rank.
        # -----------------------------------------------------------------

        # 3a: Cross-head PCA on stacked residuals
        # Use a subsample if N*n_heads is large
        max_pca_samples = min(N, 500)
        stacked = torch.stack(
            [residuals[h][:max_pca_samples] for h in available_heads], dim=0
        )  # (n_heads, max_pca_samples, 16384)
        stacked = stacked.reshape(-1, stacked.shape[-1])  # (n_heads * max_pca_samples, 16384)

        # Center
        stacked_mean = stacked.mean(dim=0, keepdim=True)
        stacked_c = stacked - stacked_mean

        # Compute top-k singular values via SVD on the smaller dimension
        # stacked_c is (M, 16384) where M = n_heads * max_pca_samples
        # Use torch.linalg.svdvals for just singular values
        print(f"  Computing PCA on stacked residuals: {stacked_c.shape}")
        svs = torch.linalg.svdvals(stacked_c.float())
        total_var = (svs ** 2).sum().item()
        cumvar = (svs[:n_pcs] ** 2).cumsum(dim=0) / total_var
        var_explained = {f"top_{i+1}": float(cumvar[i].item()) for i in range(min(n_pcs, len(cumvar)))}

        print(f"  Stacked PCA variance explained: "
              f"top-1={var_explained.get('top_1', 0):.4f}, "
              f"top-5={var_explained.get('top_5', 0):.4f}, "
              f"top-10={var_explained.get('top_10', 0):.4f}")

        # 3b: Per-head PCA (is each head's residual low-rank?)
        per_head_pca = {}
        for h in available_heads:
            r = residuals[h][:max_pca_samples].float()
            r_c = r - r.mean(dim=0, keepdim=True)
            svs_h = torch.linalg.svdvals(r_c)
            tv = (svs_h ** 2).sum().item()
            if tv > 0:
                cv_h = (svs_h[:n_pcs] ** 2).cumsum(dim=0) / tv
                per_head_pca[h] = {
                    f"top_{i+1}": float(cv_h[i].item())
                    for i in range(min(n_pcs, len(cv_h)))
                }
            else:
                per_head_pca[h] = {f"top_{i+1}": 0.0 for i in range(n_pcs)}

        # -----------------------------------------------------------------
        # Metric 4: Signed sum test
        # If residuals cancel across heads, the norm of their sum is smaller
        # than the sum of their norms. Compute both.
        # -----------------------------------------------------------------
        summed_residual = torch.zeros(N, residuals[available_heads[0]].shape[-1],
                                       device="cuda")
        sum_of_norms = torch.zeros(N, device="cuda")
        for h in available_heads:
            summed_residual += residuals[h]
            sum_of_norms += norms[h]

        norm_of_sum = summed_residual.norm(dim=-1)  # (N,)
        cancellation_ratio = (norm_of_sum / sum_of_norms.clamp(min=1e-8)).mean().item()
        # ratio=1 means no cancellation (perfectly aligned)
        # ratio~1/sqrt(16)=0.25 means fully random/independent
        print(f"  Cancellation ratio (norm_of_sum / sum_of_norms): {cancellation_ratio:.4f}")
        print(f"    (1.0 = no cancellation, 0.25 = independent random)")

        # -----------------------------------------------------------------
        # Store results
        # -----------------------------------------------------------------
        results["sae_types"][sae_type] = {
            "pairwise_cosine": {
                "mean": mean_pairwise_cos,
                "std": std_pairwise_cos,
                "median": median_pairwise_cos,
                "min": min_pairwise_cos,
                "max": max_pairwise_cos,
                "n_pairs": len(pair_cosines),
                "all_pairs": {label: val for label, val in zip(pair_labels, pair_cosines)},
                "matrix": cos_matrix.tolist(),
            },
            "per_head": per_head_stats,
            "stacked_pca_variance_explained": var_explained,
            "per_head_pca_variance_explained": {str(h): v for h, v in per_head_pca.items()},
            "cancellation_ratio": cancellation_ratio,
            "norm_of_sum_mean": float(norm_of_sum.mean().item()),
            "sum_of_norms_mean": float(sum_of_norms.mean().item()),
            "heads_used": available_heads,
        }

        # Free memory before next SAE type
        del residuals, normed, stacked, stacked_c, summed_residual, saes
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------
    if "flat" in results["sae_types"] and "bilinear" in results["sae_types"]:
        flat_r = results["sae_types"]["flat"]
        bil_r = results["sae_types"]["bilinear"]

        summary = {
            "flat_mean_pairwise_cos": flat_r["pairwise_cosine"]["mean"],
            "bilinear_mean_pairwise_cos": bil_r["pairwise_cosine"]["mean"],
            "cos_difference": flat_r["pairwise_cosine"]["mean"] - bil_r["pairwise_cosine"]["mean"],
            "flat_cancellation_ratio": flat_r["cancellation_ratio"],
            "bilinear_cancellation_ratio": bil_r["cancellation_ratio"],
            "flat_stacked_pca_top1": flat_r["stacked_pca_variance_explained"].get("top_1", 0),
            "bilinear_stacked_pca_top1": bil_r["stacked_pca_variance_explained"].get("top_1", 0),
        }

        # Per-head MSE comparison
        flat_mses = [flat_r["per_head"][h]["mse"] for h in flat_r["heads_used"]]
        bil_mses = [bil_r["per_head"][h]["mse"] for h in bil_r["heads_used"]]
        summary["flat_mean_mse"] = float(np.mean(flat_mses))
        summary["bilinear_mean_mse"] = float(np.mean(bil_mses))

        results["summary"] = summary

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Mean pairwise cosine:  flat={summary['flat_mean_pairwise_cos']:.4f}  "
              f"bilinear={summary['bilinear_mean_pairwise_cos']:.4f}  "
              f"diff={summary['cos_difference']:+.4f}")
        print(f"  Cancellation ratio:    flat={summary['flat_cancellation_ratio']:.4f}  "
              f"bilinear={summary['bilinear_cancellation_ratio']:.4f}")
        print(f"  Stacked PCA top-1:     flat={summary['flat_stacked_pca_top1']:.4f}  "
              f"bilinear={summary['bilinear_stacked_pca_top1']:.4f}")
        print(f"  Mean per-head MSE:     flat={summary['flat_mean_mse']:.6f}  "
              f"bilinear={summary['bilinear_mean_mse']:.6f}")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "residual_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"L{layer}_N{N}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()
    print(f"\nSaved to {out_path}")

    return results


# ===================================================================
# Also run rank1 for completeness (same decoder structure as bilinear)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=3600, memory=32768,
)
def measure_residual_correlation_rank1(
    layer: int = 9,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Same as above but for rank1 SAE type.

    Rank1 uses a linear encoder with rank-1 decoder atoms. If the hypothesis
    is correct (bilinear encoder produces less correlated residuals), rank1
    should look closer to flat (both use linear encoders).
    """
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from itertools import combinations

    rng = np.random.default_rng(seed)
    print(f"=== Residual correlation (rank1): layer {layer}, N={n_samples} ===")

    states_dir = Path(DATA) / "states" / f"layer_{layer}"
    total_available = np.load(str(states_dir / "head_0.npy"), mmap_mode="r").shape[0]
    idx = rng.choice(total_available, size=min(n_samples, total_available), replace=False)
    idx.sort()
    N = len(idx)

    head_states = {}
    for h in range(N_HEADS):
        arr = np.load(str(states_dir / f"head_{h}.npy"), mmap_mode="r")
        head_states[h] = torch.from_numpy(arr[idx].astype(np.float32)).cuda()

    sae_type = "rank1"
    saes = {}
    for h in range(N_HEADS):
        ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type}_s{seed}"
        if (ckpt_dir / "best.pt").exists():
            saes[h] = _load_sae(str(ckpt_dir))

    if len(saes) < 2:
        return {"error": f"Too few rank1 SAEs found ({len(saes)})"}
    print(f"  Loaded {len(saes)}/{N_HEADS} rank1 SAEs")

    residuals = {}
    norms = {}
    mses = {}
    for h in sorted(saes.keys()):
        x = head_states[h]
        with torch.no_grad():
            out = saes[h](x)
            recon = out.reconstruction
            residual = (x - recon).reshape(N, -1)
        norms[h] = residual.norm(dim=-1)
        mses[h] = (residual ** 2).mean().item()
        residuals[h] = residual

    available_heads = sorted(residuals.keys())
    normed = {}
    for h in available_heads:
        n = residuals[h].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed[h] = residuals[h] / n

    pair_cosines = []
    for h1, h2 in combinations(available_heads, 2):
        cos = (normed[h1] * normed[h2]).sum(dim=-1).mean().item()
        pair_cosines.append(cos)

    mean_cos = float(np.mean(pair_cosines))

    # Cancellation ratio
    summed = torch.zeros(N, residuals[available_heads[0]].shape[-1], device="cuda")
    sum_norms = torch.zeros(N, device="cuda")
    for h in available_heads:
        summed += residuals[h]
        sum_norms += norms[h]
    cancel = (summed.norm(dim=-1) / sum_norms.clamp(min=1e-8)).mean().item()

    result = {
        "sae_type": "rank1",
        "layer": layer,
        "n_samples": N,
        "mean_pairwise_cos": mean_cos,
        "cancellation_ratio": cancel,
        "mean_mse": float(np.mean(list(mses.values()))),
        "heads_used": available_heads,
    }

    print(f"  rank1: mean_pairwise_cos={mean_cos:.4f}, cancellation={cancel:.4f}")

    out_dir = Path(DATA) / "reviewer_experiments" / "residual_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"rank1_L{layer}_N{N}_s{seed}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    data_vol.commit()

    return result


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main():
    """Launch flat vs bilinear comparison and rank1 control in parallel."""
    t0 = time.time()

    h_main = measure_residual_correlation.spawn(layer=9, n_samples=1000, seed=42)
    h_rank1 = measure_residual_correlation_rank1.spawn(layer=9, n_samples=1000, seed=42)

    print("Launched 2 jobs: flat-vs-bilinear + rank1 control. Waiting...")

    main_result = h_main.get()
    rank1_result = h_rank1.get()

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"ALL RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 60}")

    if "summary" in main_result:
        s = main_result["summary"]
        print(f"  flat    pairwise_cos={s['flat_mean_pairwise_cos']:.4f}  "
              f"cancel={s['flat_cancellation_ratio']:.4f}  "
              f"MSE={s['flat_mean_mse']:.6f}")
        print(f"  bilinear pairwise_cos={s['bilinear_mean_pairwise_cos']:.4f}  "
              f"cancel={s['bilinear_cancellation_ratio']:.4f}  "
              f"MSE={s['bilinear_mean_mse']:.6f}")

    if "mean_pairwise_cos" in rank1_result:
        print(f"  rank1   pairwise_cos={rank1_result['mean_pairwise_cos']:.4f}  "
              f"cancel={rank1_result['cancellation_ratio']:.4f}  "
              f"MSE={rank1_result['mean_mse']:.6f}")

    # Interpretation
    if "summary" in main_result:
        s = main_result["summary"]
        if s["cos_difference"] > 0:
            print(f"\n  Hypothesis SUPPORTED: flat residuals are more correlated "
                  f"(+{s['cos_difference']:.4f} cosine)")
        else:
            print(f"\n  Hypothesis NOT SUPPORTED: bilinear residuals are more "
                  f"correlated ({s['cos_difference']:+.4f} cosine)")

    combined = {"main": main_result, "rank1": rank1_result}
    out_path = Path("results/data/residual_correlation_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved combined results to {out_path}")

"""Final reviewer experiments: spectral sample-size sensitivity + activation-weighted alignment.

Exp 2: Spectral predictor sample-size sensitivity
  - Load extracted states from matrix-sae-data-08b-clean for all GDN layers
  - For layer 9 head 0: compute sigma_1/sigma_2 on subsets of size {100,250,500,1000,2000,5000}
  - Report mean and std of sigma_1/sigma_2 at each subset size
  - Compute the spectral predictor Spearman rho (alpha vs sv_ratio across layers)
    using only 1000 vs 5000 samples per layer

Exp 3: Activation-weighted alignment metric
  - Load model + bilinear SAE for L9 H4
  - Recompute alignment but weight each position's contribution by activation magnitude
    instead of using unweighted top-50
  - Compare against the existing 0.137 mean combined alignment

Usage:
    modal run experiments/run_final_reviewer_fixes.py
    modal run experiments/run_final_reviewer_fixes.py --experiment spectral
    modal run experiments/run_final_reviewer_fixes.py --experiment alignment
"""
import json
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha

# Infrastructure (shared pattern from round 1/2/3)


CODE_SHA = code_sha()


_ext_dir = Path(__file__).resolve().parent / "extraction"
_core_dir = Path(__file__).resolve().parent.parent / "core"
_analysis_dir = Path(__file__).resolve().parent / "analysis"

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
    .add_local_file(str(_core_dir / "__init__.py"), "/root/core/__init__.py", copy=True)
    .add_local_file(str(_core_dir / "types.py"), "/root/core/types.py", copy=True)
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .add_local_file(str(_analysis_dir / "memory_alignment.py"), "/root/memory_alignment.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("final-reviewer-fixes")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")

DATA = "/data"
MODELS = "/models"
ABLATION = "/ablation"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
N_HEADS = 16

# GDN layer indices for Qwen3.5-0.8B (same list used by spectral analysis)
GDN_LAYERS = [0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21]


# ===================================================================
# EXP 2: Spectral predictor sample-size sensitivity (CPU-only)
# ===================================================================

@app.function(
    volumes={DATA: data_vol},
    image=image, timeout=1800, memory=32768,
    # No GPU: pure SVD on numpy arrays
)
def exp_spectral_sample_sensitivity(
    target_layer: int = 9,
    target_head: int = 0,
    subset_sizes: list[int] = [100, 250, 500, 1000, 2000, 5000],
    n_bootstrap: int = 20,
) -> dict[str, Any]:
    """Measure stability of sigma_1/sigma_2 diagnostic at varying sample sizes.

    Part A: For layer 9 head 0, subsample states at each size, compute mean
    sigma_1/sigma_2 across bootstrap resamples. Report mean and std.

    Part B: Recompute the spectral predictor Spearman rho (across all 12 GDN
    layers) using 1000 vs 5000 samples. Shows whether the cross-layer
    correlation is stable with fewer samples.
    """
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    from scipy.stats import spearmanr

    print(f"=== EXP 2: Spectral sample-size sensitivity ===")
    t0 = time.time()

    # ------------------------------------------------------------------
    # Part A: sigma_1/sigma_2 stability at layer 9 head 0
    # ------------------------------------------------------------------
    states_path = f"{DATA}/states/layer_{target_layer}/head_{target_head}.npy"
    print(f"Loading states from {states_path}...")
    states = np.load(states_path)  # (N, 128, 128)
    N_total = states.shape[0]
    print(f"  Loaded {N_total} states, shape {states.shape}")

    rng = np.random.default_rng(42)

    part_a_results: dict[str, Any] = {
        "layer": target_layer,
        "head": target_head,
        "n_total": N_total,
        "n_bootstrap": n_bootstrap,
        "subset_results": {},
    }

    for n_sub in subset_sizes:
        if n_sub > N_total:
            print(f"  Skipping n={n_sub} (only {N_total} available)")
            continue

        ratios = []
        for boot_i in range(n_bootstrap):
            idx = rng.choice(N_total, size=n_sub, replace=False)
            sub_states = states[idx].astype(np.float32)
            # Compute SVD for each state, collect sigma_1/sigma_2
            sv_ratios_batch = []
            for s in sub_states:
                sv = np.linalg.svd(s, compute_uv=False)
                if sv[1] > 1e-10:
                    sv_ratios_batch.append(float(sv[0] / sv[1]))
            mean_ratio = float(np.mean(sv_ratios_batch))
            ratios.append(mean_ratio)

        part_a_results["subset_results"][n_sub] = {
            "mean_sv_ratio": float(np.mean(ratios)),
            "std_sv_ratio": float(np.std(ratios)),
            "min_sv_ratio": float(np.min(ratios)),
            "max_sv_ratio": float(np.max(ratios)),
            "cv": float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else 0.0,
        }
        print(
            f"  n={n_sub:>5}: sigma1/sigma2 = {np.mean(ratios):.4f} +/- {np.std(ratios):.4f} "
            f"(CV={np.std(ratios)/np.mean(ratios)*100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Part B: Cross-layer Spearman rho at 1000 vs 5000 samples
    # ------------------------------------------------------------------
    print("\n--- Part B: Spearman rho at different sample sizes ---")

    # Pre-existing alpha values from the correlation file
    # (empirical alpha averaged across heads for each GDN layer)
    alpha_empirical = {
        0: 0.7610, 1: 0.5511, 2: 0.6964, 5: 0.8852,
        6: 0.8426, 9: 0.9385, 10: 0.9113, 13: 0.8921,
        14: 0.9098, 17: 0.9218, 18: 0.8799, 21: 0.8633,
    }

    sample_sizes_b = [500, 1000, 2000, 5000]
    part_b_results: dict[str, Any] = {"sample_sizes": sample_sizes_b, "results": {}}

    for n_samples in sample_sizes_b:
        sv_ratios_per_layer = {}
        alphas_ordered = []
        svr_ordered = []

        for layer in GDN_LAYERS:
            head_path = f"{DATA}/states/layer_{layer}/head_0.npy"
            try:
                layer_states = np.load(head_path)  # (N, 128, 128)
            except FileNotFoundError:
                print(f"  Layer {layer}: states not found, skipping")
                continue

            n_avail = layer_states.shape[0]
            n_use = min(n_samples, n_avail)
            idx = rng.choice(n_avail, size=n_use, replace=False)
            sub = layer_states[idx].astype(np.float32)

            batch_ratios = []
            for s in sub:
                sv = np.linalg.svd(s, compute_uv=False)
                if sv[1] > 1e-10:
                    batch_ratios.append(float(sv[0] / sv[1]))
            mean_ratio = float(np.mean(batch_ratios))
            sv_ratios_per_layer[layer] = mean_ratio

            alphas_ordered.append(alpha_empirical[layer])
            svr_ordered.append(mean_ratio)

        if len(alphas_ordered) >= 4:
            rho, p = spearmanr(alphas_ordered, svr_ordered)
        else:
            rho, p = float("nan"), float("nan")

        part_b_results["results"][n_samples] = {
            "sv_ratios": sv_ratios_per_layer,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n_layers": len(alphas_ordered),
        }
        print(f"  n={n_samples:>5}: Spearman rho = {rho:.4f} (p={p:.4f}), {len(alphas_ordered)} layers")

    total_time = time.time() - t0
    results = {
        "part_a": part_a_results,
        "part_b": part_b_results,
        "total_time_s": round(total_time, 1),
    }

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp_spectral_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spectral_sample_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()
    print(f"\n  Saved to {out_path} ({total_time:.0f}s)")

    return results


# ===================================================================
# EXP 3: Activation-weighted alignment metric (needs GPU for SAE + model)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=7200, memory=32768,
)
def exp_activation_weighted_alignment(
    layer: int = 9,
    head: int = 4,
    n_seqs: int = 100,
    seq_len: int = 512,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Recompute memory alignment with activation-magnitude weighting.

    Instead of taking unweighted mean over top-50 activating positions,
    weight each position's alignment by its activation coefficient.
    This tests whether high-activation positions show stronger alignment.

    Also computes alignment at multiple top-N thresholds (10, 50, 200, all)
    and with/without activation weighting for direct comparison.
    """
    import os
    import sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"

    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from memory_alignment import (
        load_model, load_sae, GDNWriteCapture, simulate_recurrence,
        load_corpus_batches,
    )

    print(f"=== EXP 3: Activation-weighted alignment, L{layer} H{head} ===")
    t0 = time.time()

    model, tokenizer = load_model(MODEL_08B, "cuda")
    model_vol.commit()

    import glob
    ckpt_path = None
    ablation_path = Path(ABLATION) / f"layer_{layer}" / f"head_{head}" / "bilinear_s42"
    if (ablation_path / "best.pt").exists():
        ckpt_path = str(ablation_path / "best.pt")

    if ckpt_path is None:
        candidates = glob.glob(
            f"{DATA}/checkpoints/*/bilinear_L{layer}_H{head}_nf*_k*_s42/best.pt"
        )
        if candidates:
            ckpt_path = candidates[0]

    if ckpt_path is None:
        return {"error": f"No bilinear SAE checkpoint found for L{layer} H{head}"}

    print(f"  Using checkpoint: {ckpt_path}")
    sae = load_sae(ckpt_path, "cuda")

    V_dec = sae.V_dec.detach().float().cpu()
    W_dec = sae.W_dec.detach().float().cpu()
    n_features = V_dec.shape[0]
    d_k = V_dec.shape[2]
    d_v = W_dec.shape[2]
    v_dec_dir = F.normalize(V_dec[:, 0, :], dim=-1)
    w_dec_dir = F.normalize(W_dec[:, 0, :], dim=-1)
    print(f"  SAE: {n_features} features, d_k={d_k}, d_v={d_v}")

    print(f"  Loading {n_seqs} sequences of length {seq_len}...")
    batches = load_corpus_batches(tokenizer, n_seqs, seq_len, batch_size)
    total_seqs = sum(b.shape[0] for b in batches)
    print(f"  Loaded {total_seqs} sequences in {len(batches)} batches")

    # Install hook
    hook = GDNWriteCapture(model, layer)

    # Collect per-feature data: (activation_magnitude, k_cos, v_cos)
    feature_data: dict[int, list[tuple[float, float, float]]] = {i: [] for i in range(n_features)}

    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        input_ids = batch.to("cuda")
        bs = input_ids.shape[0]

        outputs = model(input_ids=input_ids, use_cache=True)

        k_all = hook.k
        v_all = hook.v
        beta_all = hook.beta
        g_all = hook.g

        for seq_idx in range(bs):
            k_head = k_all[seq_idx, :, head, :].cpu()
            v_head = v_all[seq_idx, :, head, :].cpu()
            if beta_all.ndim == 4:
                beta_head = beta_all[seq_idx, :, head, 0].cpu()
            else:
                beta_head = beta_all[seq_idx, :, head].cpu()
            g_head = g_all[seq_idx, :, head].cpu()

            states, k_normed, write_v = simulate_recurrence(
                k_head, v_head, beta_head, g_head, dtype=torch.float32,
            )

            # Encode through SAE in sub-batches
            sub_bs = 128
            all_coeffs = []
            for s in range(0, states.shape[0], sub_bs):
                e = min(s + sub_bs, states.shape[0])
                chunk = states[s:e].to("cuda")
                coeffs = sae.encode(chunk)
                all_coeffs.append(coeffs.cpu())
            all_coeffs = torch.cat(all_coeffs, dim=0)

            k_dir = k_normed.cpu()
            wv_raw = write_v.cpu()
            wv_norms = wv_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            wv_dir = wv_raw / wv_norms

            k_cos_all = k_dir @ v_dec_dir.t()
            v_cos_all = wv_dir @ w_dec_dir.t()

            active_pos, active_feat = (all_coeffs > 0).nonzero(as_tuple=True)
            if active_pos.numel() > 0:
                active_vals = all_coeffs[active_pos, active_feat].detach().cpu().numpy()
                active_k_cos = k_cos_all[active_pos, active_feat].detach().cpu().numpy()
                active_v_cos = v_cos_all[active_pos, active_feat].detach().cpu().numpy()
                feat_np = active_feat.detach().cpu().numpy()

                for fi in np.unique(feat_np):
                    mask = feat_np == fi
                    entries = list(zip(
                        active_vals[mask].tolist(),
                        active_k_cos[mask].tolist(),
                        active_v_cos[mask].tolist(),
                    ))
                    feature_data[int(fi)].extend(entries)

        del k_all, v_all, beta_all, g_all, outputs
        torch.cuda.empty_cache()

    hook.remove()
    collection_time = time.time() - t0
    print(f"  Data collection: {collection_time:.1f}s")

    # ------------------------------------------------------------------
    # Compute alignment metrics: unweighted (original) vs activation-weighted
    # at multiple top-N thresholds
    # ------------------------------------------------------------------
    top_n_values = [10, 50, 200, -1]  # -1 = all activations

    per_feature_results: list[dict[str, Any]] = []
    alive_count = 0

    for feat_idx in range(n_features):
        data = feature_data[feat_idx]
        if len(data) == 0:
            per_feature_results.append({
                "feature": feat_idx, "alive": False,
                "n_activations": 0,
            })
            continue

        alive_count += 1
        data.sort(key=lambda x: x[0], reverse=True)

        feat_result: dict[str, Any] = {
            "feature": feat_idx,
            "alive": True,
            "n_activations": len(data),
        }

        for top_n in top_n_values:
            subset = data if top_n == -1 else data[:top_n]
            if len(subset) == 0:
                continue

            acts = np.array([d[0] for d in subset])
            k_cosines = np.abs(np.array([d[1] for d in subset]))
            v_cosines = np.abs(np.array([d[2] for d in subset]))

            combined_unweighted = np.sqrt(k_cosines * v_cosines)

            # Activation-weighted: weight each alignment by activation magnitude
            weights = acts / acts.sum() if acts.sum() > 0 else np.ones_like(acts) / len(acts)
            k_weighted = float(np.sum(weights * k_cosines))
            v_weighted = float(np.sum(weights * v_cosines))
            combined_weighted = np.sqrt(k_weighted * v_weighted)

            tag = f"top{top_n}" if top_n > 0 else "all"
            feat_result[f"unweighted_{tag}"] = {
                "mean_k_cos": float(np.mean(k_cosines)),
                "mean_v_cos": float(np.mean(v_cosines)),
                "mean_combined": float(np.mean(combined_unweighted)),
                "n_used": len(subset),
            }
            feat_result[f"weighted_{tag}"] = {
                "mean_k_cos": k_weighted,
                "mean_v_cos": v_weighted,
                "mean_combined": float(combined_weighted),
                "n_used": len(subset),
            }

        per_feature_results.append(feat_result)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    alive = [r for r in per_feature_results if r.get("alive")]

    summary: dict[str, Any] = {
        "n_features": n_features,
        "n_alive": alive_count,
        "n_dead": n_features - alive_count,
    }

    # Null baseline
    null_expected = float(np.sqrt(2 / np.pi) / np.sqrt(128))
    summary["null_baseline_expected_abs_cos"] = null_expected

    for top_n in top_n_values:
        tag = f"top{top_n}" if top_n > 0 else "all"

        for metric in ["unweighted", "weighted"]:
            key = f"{metric}_{tag}"
            valid = [r for r in alive if key in r]
            if not valid:
                continue

            k_vals = [r[key]["mean_k_cos"] for r in valid]
            v_vals = [r[key]["mean_v_cos"] for r in valid]
            c_vals = [r[key]["mean_combined"] for r in valid]

            summary[key] = {
                "mean_k_cos": float(np.mean(k_vals)),
                "mean_v_cos": float(np.mean(v_vals)),
                "mean_combined": float(np.mean(c_vals)),
                "std_combined": float(np.std(c_vals)),
                "median_combined": float(np.median(c_vals)),
                "n_features": len(valid),
            }

    total_time = time.time() - t0
    results = {
        "layer": layer,
        "head": head,
        "n_seqs": total_seqs,
        "seq_len": seq_len,
        "checkpoint": ckpt_path,
        "summary": summary,
        "per_feature": per_feature_results,
        "collection_time_s": round(collection_time, 1),
        "total_time_s": round(total_time, 1),
    }

    print(f"\n{'='*70}")
    print(f"Alignment comparison: unweighted vs activation-weighted")
    print(f"{'='*70}")
    print(f"  Null baseline (random 128-dim): {null_expected:.4f}")
    print(f"  {alive_count} alive features out of {n_features}")
    print(f"\n  {'Method':<25} {'|k_cos|':>8} {'|v_cos|':>8} {'combined':>8}")
    print(f"  {'-'*55}")
    for top_n in top_n_values:
        tag = f"top{top_n}" if top_n > 0 else "all"
        for metric in ["unweighted", "weighted"]:
            key = f"{metric}_{tag}"
            if key in summary:
                s = summary[key]
                label = f"{metric} ({tag})"
                print(f"  {label:<25} {s['mean_k_cos']:>8.4f} {s['mean_v_cos']:>8.4f} {s['mean_combined']:>8.4f}")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp_weighted_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"L{layer}_H{head}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()
    print(f"\n  Saved to {out_path} ({total_time:.0f}s)")

    return results


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main(experiment: str = "all"):
    """Launch experiments. Use --experiment spectral, alignment, or all."""
    t0 = time.time()
    handles = []

    run_spectral = experiment in ("all", "spectral")
    run_alignment = experiment in ("all", "alignment")

    if run_spectral:
        print("Launching Exp 2: Spectral sample-size sensitivity...")
        h = exp_spectral_sample_sensitivity.spawn(
            target_layer=9,
            target_head=0,
            subset_sizes=[100, 250, 500, 1000, 2000, 5000],
            n_bootstrap=20,
        )
        handles.append(("exp_spectral_sensitivity", h))

    if run_alignment:
        print("Launching Exp 3: Activation-weighted alignment...")
        h = exp_activation_weighted_alignment.spawn(
            layer=9, head=4,
            n_seqs=100, seq_len=512, batch_size=4,
        )
        handles.append(("exp_weighted_alignment", h))

    if not handles:
        print(f"Unknown experiment '{experiment}'. Use: all, spectral, alignment")
        return

    print(f"\n{len(handles)} jobs launched in parallel. Waiting...")

    results = {}
    failures: list[str] = []
    for name, handle in handles:
        try:
            result = handle.get()
            results[name] = result
            print(f"\n{'='*60}")
            print(f"=== {name} complete ===")
            print(f"{'='*60}")

            if name == "exp_spectral_sensitivity":
                print("\nPart A: sigma_1/sigma_2 stability (L9 H0)")
                part_a = result["part_a"]
                for n_sub, stats in sorted(part_a["subset_results"].items(), key=lambda x: int(x[0])):
                    print(f"  n={int(n_sub):>5}: {stats['mean_sv_ratio']:.4f} +/- {stats['std_sv_ratio']:.4f} "
                          f"(CV={stats['cv']*100:.2f}%)")

                print("\nPart B: Spearman rho at different sample sizes")
                part_b = result["part_b"]
                for n_samples, stats in sorted(part_b["results"].items(), key=lambda x: int(x[0])):
                    print(f"  n={int(n_samples):>5}: rho={stats['spearman_rho']:.4f} "
                          f"(p={stats['p_value']:.4f})")

            elif name == "exp_weighted_alignment":
                s = result["summary"]
                print(f"\n  {s['n_alive']} alive features, null baseline = {s['null_baseline_expected_abs_cos']:.4f}")
                for key in sorted(s.keys()):
                    if isinstance(s[key], dict) and "mean_combined" in s[key]:
                        print(f"  {key}: combined={s[key]['mean_combined']:.4f}")

        except Exception as e:
            print(f"\n=== {name} FAILED: {e} ===")
            results[name] = {"error": str(e)}
            failures.append(f"{name}: {e}")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    out_path = Path("results/data/final_reviewer_fixes.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved combined results to {out_path}")

    if failures:
        raise RuntimeError("One or more jobs failed: " + "; ".join(failures))

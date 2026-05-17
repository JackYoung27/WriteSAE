"""Two experiments for Stanford reviewer round 3.

Exp Q2: Low-rank SVD baseline for downstream PPL
  - For each sequence, extract recurrent state at layer 9
  - Replace each head's 128x128 state with rank-r SVD approximation
  - Measure downstream PPL for r = 1, 2, 4, 8, 16, 32
  - Compare against SAE reconstructions (+1.33% bilinear, +3.31% flat)

Exp Q4: Encoder-decoder subspace analysis (untied bilinear)
  - Load untied BilinearMatrixSAE checkpoint (L9 H0)
  - Compare V_enc vs V_dec and W_enc vs W_dec directions per feature
  - Check whether alive features diverge more than dead features

Usage:
    modal run experiments/run_reviewer_round3.py
    modal run experiments/run_reviewer_round3.py --experiment q2
    modal run experiments/run_reviewer_round3.py --experiment q4
"""
import json
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha

# Infrastructure (shared with round 1/2)


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
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("reviewer-round3")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")

DATA = "/data"
MODELS = "/models"
ABLATION = "/ablation"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
N_HEADS = 16


# ===================================================================
# EXP Q2: Low-rank SVD baseline for downstream PPL
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=7200, memory=32768,
)
def exp_q2_svd_baseline_ppl(
    layer: int = 9,
    n_sequences: int = 200,
    svd_ranks: list[int] = [1, 2, 4, 8, 16, 32],
) -> dict[str, Any]:
    """Downstream PPL with rank-r SVD reconstruction of all heads at one layer.

    For each rank r, replace every head's 128x128 state with its best rank-r
    approximation U[:,:r] @ diag(S[:r]) @ Vt[:r,:]. Measure suffix PPL.
    This gives a non-sparse, non-learned baseline for comparison against SAEs.
    """
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    from evaluate_downstream import _patch_gdn_initial_states
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices

    print(f"=== EXP Q2: SVD baseline PPL, layer {layer}, ranks {svd_ranks} ===")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    corpus_ids = np.load(f"{DATA}/states/corpus.npy")
    n_seq = min(n_sequences, len(corpus_ids))
    seq_len = corpus_ids.shape[1]
    split_pos = seq_len // 2

    print(f"  {n_seq} sequences x {seq_len} tokens, split at {split_pos}")
    print(f"  {len(gdn_layers)} GDN layers, patching all for state forwarding")
    print(f"  SVD ranks to test: {svd_ranks}")

    results: dict[str, Any] = {
        "layer": layer,
        "n_sequences": n_seq,
        "seq_len": seq_len,
        "split_pos": split_pos,
        "gdn_layers": gdn_layers,
        "svd_ranks": svd_ranks,
    }

    # Pre-allocate storage for per-sequence losses at each rank + baseline
    baseline_losses = []
    baseline_tokens = 0
    rank_losses: dict[int, list[float]] = {r: [] for r in svd_ranks}
    rank_tokens: dict[int, int] = {r: 0 for r in svd_ranks}
    rank_mses: dict[int, list[float]] = {r: [] for r in svd_ranks}

    t0_total = time.time()

    def _run_prefix(input_ids, split_pos):
        """Run prefix and return fresh cache + clean GDN states."""
        prefix = input_ids[:, :split_pos]
        prefix_out = model(input_ids=prefix, use_cache=True)
        cache = prefix_out.past_key_values
        gdn_states = {}
        for idx in gdn_layers:
            lc = cache.layers[idx]
            if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                gdn_states[idx] = lc.recurrent_states.clone()
        return cache, gdn_states

    for seq_i in range(n_seq):
        input_ids = torch.tensor(
            corpus_ids[seq_i:seq_i + 1], dtype=torch.long, device="cuda"
        )
        suffix = input_ids[:, split_pos:]
        n_tok = suffix.shape[1] - 1

        # Baseline: fresh prefix pass, no modification
        cache, gdn_states_clean = _run_prefix(input_ids, split_pos)
        with _patch_gdn_initial_states(model, gdn_layers, gdn_states_clean):
            baseline_out = model(
                input_ids=suffix, past_key_values=cache,
                use_cache=False, labels=suffix,
            )
        baseline_losses.append(baseline_out.loss.item() * n_tok)
        baseline_tokens += n_tok

        # SVD reconstruction at each rank (fresh prefix each time)
        for rank in svd_ranks:
            cache_r, gdn_states_r = _run_prefix(input_ids, split_pos)
            state = gdn_states_r[layer]
            n_heads_actual = state.shape[1]

            mse_accum = 0.0
            for h in range(n_heads_actual):
                head_state = state[0, h].float()  # (d_k, d_v)
                U, S, Vt = torch.linalg.svd(head_state, full_matrices=False)
                r = min(rank, S.shape[0])
                recon = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]
                mse_accum += ((recon - head_state) ** 2).mean().item()
                state[0, h] = recon.to(state.dtype)

            rank_mses[rank].append(mse_accum / n_heads_actual)

            with _patch_gdn_initial_states(model, gdn_layers, gdn_states_r):
                svd_out = model(
                    input_ids=suffix, past_key_values=cache_r,
                    use_cache=False, labels=suffix,
                )
            rank_losses[rank].append(svd_out.loss.item() * n_tok)
            rank_tokens[rank] += n_tok

        if seq_i % 25 == 0:
            elapsed = time.time() - t0_total
            print(f"  seq {seq_i}/{n_seq} ({elapsed:.0f}s)")

    baseline_avg_loss = sum(baseline_losses) / max(baseline_tokens, 1)
    baseline_ppl = float(np.exp(baseline_avg_loss))
    total_time = time.time() - t0_total

    results["baseline"] = {
        "loss": baseline_avg_loss,
        "perplexity": baseline_ppl,
        "n_tokens": baseline_tokens,
    }
    print(f"\n  Baseline PPL: {baseline_ppl:.4f} (loss={baseline_avg_loss:.6f})")

    results["svd_results"] = {}
    for rank in svd_ranks:
        avg_loss = sum(rank_losses[rank]) / max(rank_tokens[rank], 1)
        ppl = float(np.exp(avg_loss))
        delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100
        avg_mse = float(np.mean(rank_mses[rank]))

        results["svd_results"][rank] = {
            "loss": avg_loss,
            "perplexity": ppl,
            "delta_pct": delta_pct,
            "mean_reconstruction_mse": avg_mse,
            "n_tokens": rank_tokens[rank],
        }
        print(f"  SVD rank {rank:>2}: PPL={ppl:.4f} (delta={delta_pct:+.3f}%), MSE={avg_mse:.2e}")

    results["total_time_s"] = round(total_time, 1)
    print(f"\n  Total time: {total_time:.0f}s")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp_q2_svd_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"L{layer}_n{n_seq}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()
    print(f"  Saved to {out_path}")

    return results


# ===================================================================
# EXP Q4: Encoder-decoder subspace analysis (untied bilinear)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, ABLATION: ablation_vol},
    image=image, timeout=600, memory=16384,
    # No GPU needed: pure weight-matrix analysis
)
def exp_q4_encoder_decoder_subspace(
    layer: int = 9,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare encoder vs decoder directions in untied BilinearMatrixSAE.

    For each feature i, compute:
      cos(V_enc[i], V_dec[i])  and  cos(W_enc[i], W_dec[i])
    If encoder and decoder learn nearly identical subspaces, tying should
    work well. If they diverge, that explains the +7.05% tied failure.

    Also compares alive vs dead features to check whether active features
    have more or less encoder-decoder divergence.
    """
    import sys
    sys.path.insert(0, "/root")
    import torch
    import torch.nn.functional as F
    import numpy as np
    import glob

    from sae import build_sae_from_config

    print(f"=== EXP Q4: encoder-decoder subspace analysis, L{layer} ===")

    # Find untied bilinear checkpoint for all 16 heads
    per_head_results: dict[int, dict[str, Any]] = {}

    for head in range(N_HEADS):
        # Try ablation volume first (per-head matched checkpoints)
        ckpt_path = None
        config_path = None

        ablation_path = Path(ABLATION) / f"layer_{layer}" / f"head_{head}" / f"bilinear_s{seed}"
        if (ablation_path / "best.pt").exists():
            ckpt_path = str(ablation_path / "best.pt")
            config_path = str(ablation_path / "config.json")

        # Fall back to data volume
        if ckpt_path is None:
            candidates = glob.glob(
                f"{DATA}/checkpoints/*/bilinear_L{layer}_H{head}_nf*_k*_s{seed}/best.pt"
            )
            if candidates:
                ckpt_path = candidates[0]
                config_path = str(Path(candidates[0]).parent / "config.json")

        if ckpt_path is None:
            print(f"  Head {head}: no untied bilinear checkpoint found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"]

        # Verify this is untied (has both V_enc and V_dec)
        if "V_enc" not in sd or "V_dec" not in sd:
            print(f"  Head {head}: checkpoint is tied (no V_enc), skipping")
            continue

        # Load config to build SAE for dead-feature detection
        cfg = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
        sae = build_sae_from_config(cfg, state_dict=sd)
        sae.load_state_dict(sd)
        sae.eval()

        # Extract direction matrices
        # Handle backward compat: old checkpoints may have (n_features, d) instead of (n_features, rank, d)
        V_enc = sd["V_enc"]
        W_enc = sd["W_enc"]
        V_dec = sd["V_dec"]
        W_dec = sd["W_dec"]
        if V_enc.ndim == 2:
            V_enc = V_enc.unsqueeze(1)
            W_enc = W_enc.unsqueeze(1)
            V_dec = V_dec.unsqueeze(1)
            W_dec = W_dec.unsqueeze(1)

        n_features = V_enc.shape[0]
        rank = V_enc.shape[1]

        # Compute per-feature cosine similarities across rank components
        # For rank-1: straightforward cos(V_enc[i,0], V_dec[i,0])
        # For rank-r: average across rank components
        v_cos_per_feature = torch.zeros(n_features)
        w_cos_per_feature = torch.zeros(n_features)

        for r_idx in range(rank):
            v_enc_r = F.normalize(V_enc[:, r_idx, :], dim=-1)  # (n_features, d_k)
            v_dec_r = F.normalize(V_dec[:, r_idx, :], dim=-1)
            w_enc_r = F.normalize(W_enc[:, r_idx, :], dim=-1)  # (n_features, d_v)
            w_dec_r = F.normalize(W_dec[:, r_idx, :], dim=-1)

            v_cos_per_feature += (v_enc_r * v_dec_r).sum(dim=-1) / rank
            w_cos_per_feature += (w_enc_r * w_dec_r).sum(dim=-1) / rank

        # Identify alive vs dead features from steps_since_active
        steps_since = sae.steps_since_active
        dead_threshold = sae.dead_threshold if hasattr(sae, "dead_threshold") else 100
        alive_mask = steps_since < dead_threshold
        dead_mask = ~alive_mask
        n_alive = int(alive_mask.sum().item())
        n_dead = int(dead_mask.sum().item())

        v_cos = v_cos_per_feature.numpy()
        w_cos = w_cos_per_feature.numpy()
        combined_cos = (np.abs(v_cos) + np.abs(w_cos)) / 2

        head_result: dict[str, Any] = {
            "ckpt_path": ckpt_path,
            "n_features": n_features,
            "rank": rank,
            "n_alive": n_alive,
            "n_dead": n_dead,
            "all_features": {
                "v_cos_mean": float(np.mean(v_cos)),
                "v_cos_std": float(np.std(v_cos)),
                "v_cos_median": float(np.median(v_cos)),
                "v_cos_abs_mean": float(np.mean(np.abs(v_cos))),
                "w_cos_mean": float(np.mean(w_cos)),
                "w_cos_std": float(np.std(w_cos)),
                "w_cos_median": float(np.median(w_cos)),
                "w_cos_abs_mean": float(np.mean(np.abs(w_cos))),
                "combined_cos_mean": float(np.mean(combined_cos)),
                "pct_v_above_0.9": float((np.abs(v_cos) > 0.9).mean() * 100),
                "pct_w_above_0.9": float((np.abs(w_cos) > 0.9).mean() * 100),
                "pct_v_below_0.5": float((np.abs(v_cos) < 0.5).mean() * 100),
                "pct_w_below_0.5": float((np.abs(w_cos) < 0.5).mean() * 100),
            },
        }

        if n_alive > 0:
            alive_v = v_cos[alive_mask.numpy()]
            alive_w = w_cos[alive_mask.numpy()]
            head_result["alive_features"] = {
                "v_cos_mean": float(np.mean(alive_v)),
                "v_cos_std": float(np.std(alive_v)),
                "v_cos_abs_mean": float(np.mean(np.abs(alive_v))),
                "w_cos_mean": float(np.mean(alive_w)),
                "w_cos_std": float(np.std(alive_w)),
                "w_cos_abs_mean": float(np.mean(np.abs(alive_w))),
            }

        if n_dead > 0:
            dead_v = v_cos[dead_mask.numpy()]
            dead_w = w_cos[dead_mask.numpy()]
            head_result["dead_features"] = {
                "v_cos_mean": float(np.mean(dead_v)),
                "v_cos_std": float(np.std(dead_v)),
                "v_cos_abs_mean": float(np.mean(np.abs(dead_v))),
                "w_cos_mean": float(np.mean(dead_w)),
                "w_cos_std": float(np.std(dead_w)),
                "w_cos_abs_mean": float(np.mean(np.abs(dead_w))),
            }

        # Histogram bins for later plotting
        hist_v, bin_edges_v = np.histogram(v_cos, bins=50, range=(-1, 1))
        hist_w, bin_edges_w = np.histogram(w_cos, bins=50, range=(-1, 1))
        head_result["histogram"] = {
            "v_cos_counts": hist_v.tolist(),
            "w_cos_counts": hist_w.tolist(),
            "bin_edges": bin_edges_v.tolist(),
        }

        per_head_results[head] = head_result

        print(
            f"  Head {head}: V cos={np.mean(v_cos):.3f}+/-{np.std(v_cos):.3f}, "
            f"W cos={np.mean(w_cos):.3f}+/-{np.std(w_cos):.3f}, "
            f"alive={n_alive}, dead={n_dead}"
        )

    all_v_means = [r["all_features"]["v_cos_mean"] for r in per_head_results.values()]
    all_w_means = [r["all_features"]["w_cos_mean"] for r in per_head_results.values()]
    all_v_abs = [r["all_features"]["v_cos_abs_mean"] for r in per_head_results.values()]
    all_w_abs = [r["all_features"]["w_cos_abs_mean"] for r in per_head_results.values()]

    # Alive vs dead divergence comparison (across all heads that have both)
    alive_v_means = [r["alive_features"]["v_cos_abs_mean"]
                     for r in per_head_results.values() if "alive_features" in r]
    alive_w_means = [r["alive_features"]["w_cos_abs_mean"]
                     for r in per_head_results.values() if "alive_features" in r]
    dead_v_means = [r["dead_features"]["v_cos_abs_mean"]
                    for r in per_head_results.values() if "dead_features" in r]
    dead_w_means = [r["dead_features"]["w_cos_abs_mean"]
                    for r in per_head_results.values() if "dead_features" in r]

    summary: dict[str, Any] = {
        "n_heads_analyzed": len(per_head_results),
        "aggregate": {
            "v_cos_mean_across_heads": float(np.mean(all_v_means)) if all_v_means else None,
            "w_cos_mean_across_heads": float(np.mean(all_w_means)) if all_w_means else None,
            "v_cos_abs_mean_across_heads": float(np.mean(all_v_abs)) if all_v_abs else None,
            "w_cos_abs_mean_across_heads": float(np.mean(all_w_abs)) if all_w_abs else None,
        },
        "alive_vs_dead": {
            "alive_v_cos_abs_mean": float(np.mean(alive_v_means)) if alive_v_means else None,
            "alive_w_cos_abs_mean": float(np.mean(alive_w_means)) if alive_w_means else None,
            "dead_v_cos_abs_mean": float(np.mean(dead_v_means)) if dead_v_means else None,
            "dead_w_cos_abs_mean": float(np.mean(dead_w_means)) if dead_w_means else None,
        },
    }

    # Statistical test: do alive features have different alignment than dead?
    if alive_v_means and dead_v_means and len(alive_v_means) > 1 and len(dead_v_means) > 1:
        from scipy.stats import mannwhitneyu
        u_v, p_v = mannwhitneyu(alive_v_means, dead_v_means, alternative="two-sided")
        u_w, p_w = mannwhitneyu(alive_w_means, dead_w_means, alternative="two-sided")
        summary["alive_vs_dead"]["mann_whitney_v"] = {"U": float(u_v), "p": float(p_v)}
        summary["alive_vs_dead"]["mann_whitney_w"] = {"U": float(u_w), "p": float(p_w)}
        print(f"\n  Alive vs dead V alignment: U={u_v:.1f}, p={p_v:.4f}")
        print(f"  Alive vs dead W alignment: U={u_w:.1f}, p={p_w:.4f}")

    results = {
        "layer": layer,
        "seed": seed,
        "summary": summary,
        "per_head": {str(k): v for k, v in per_head_results.items()},
    }

    print(f"\n  Aggregate across {len(per_head_results)} heads:")
    print(f"    V cos (abs mean): {summary['aggregate']['v_cos_abs_mean_across_heads']}")
    print(f"    W cos (abs mean): {summary['aggregate']['w_cos_abs_mean_across_heads']}")
    if summary["alive_vs_dead"]["alive_v_cos_abs_mean"] is not None:
        print(f"    Alive V: {summary['alive_vs_dead']['alive_v_cos_abs_mean']:.3f}, "
              f"Dead V: {summary['alive_vs_dead']['dead_v_cos_abs_mean']:.3f}")
        print(f"    Alive W: {summary['alive_vs_dead']['alive_w_cos_abs_mean']:.3f}, "
              f"Dead W: {summary['alive_vs_dead']['dead_w_cos_abs_mean']:.3f}")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp_q4_encoder_decoder_subspace"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"L{layer}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()
    print(f"  Saved to {out_path}")

    return results


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main(experiment: str = "all"):
    """Launch experiments. Use --experiment q2, q4, or all."""
    t0 = time.time()
    handles = []

    run_q2 = experiment in ("all", "q2")
    run_q4 = experiment in ("all", "q4")

    if run_q2:
        print("Launching Exp Q2: SVD baseline PPL...")
        h = exp_q2_svd_baseline_ppl.spawn(
            layer=9, n_sequences=200, svd_ranks=[1, 2, 4, 8, 16, 32],
        )
        handles.append(("exp_q2_svd_baseline", h))

    if run_q4:
        print("Launching Exp Q4: encoder-decoder subspace analysis...")
        h = exp_q4_encoder_decoder_subspace.spawn(layer=9, seed=42)
        handles.append(("exp_q4_subspace", h))

    if not handles:
        print(f"Unknown experiment '{experiment}'. Use: all, q2, q4")
        return

    print(f"\n{len(handles)} jobs launched. Waiting...")

    results = {}
    failures: list[str] = []
    for name, handle in handles:
        try:
            result = handle.get()
            results[name] = result
            print(f"\n=== {name} complete ===")

            if name == "exp_q2_svd_baseline":
                baseline_ppl = result["baseline"]["perplexity"]
                print(f"  Baseline PPL: {baseline_ppl:.4f}")
                for rank, r in sorted(result["svd_results"].items(), key=lambda x: int(x[0])):
                    print(f"  SVD rank {rank:>2}: PPL={r['perplexity']:.4f} "
                          f"(delta={r['delta_pct']:+.3f}%, MSE={r['mean_reconstruction_mse']:.2e})")

            elif name == "exp_q4_subspace":
                s = result["summary"]
                print(f"  {s['n_heads_analyzed']} heads analyzed")
                agg = s["aggregate"]
                print(f"  V cos |mean|: {agg['v_cos_abs_mean_across_heads']:.3f}")
                print(f"  W cos |mean|: {agg['w_cos_abs_mean_across_heads']:.3f}")
                avd = s["alive_vs_dead"]
                if avd["alive_v_cos_abs_mean"] is not None:
                    print(f"  Alive V: {avd['alive_v_cos_abs_mean']:.3f}, "
                          f"Dead V: {avd['dead_v_cos_abs_mean']:.3f}")

        except Exception as e:
            print(f"\n=== {name} FAILED: {e} ===")
            results[name] = {"error": str(e)}
            failures.append(f"{name}: {e}")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    out_path = Path("results/data/reviewer_round3_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved combined results to {out_path}")

    if failures:
        raise RuntimeError("One or more reviewer round 3 jobs failed: " + "; ".join(failures))

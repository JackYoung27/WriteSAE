"""Five parallel experiments to answer remaining Stanford reviewer questions.

Exp A (Task 17): Per-head PPL deltas for paired t-test
Exp B (Task 18): Per-head σ1/σ2 and correlation with rank-1 advantage
Exp C (Task 19): Tied-bilinear downstream PPL
Exp D (Task 20): Write-alignment sensitivity to top-N positions
Exp E (Task 21): Wall-clock timing per architecture

Usage:
    modal run experiments/run_reviewer_round2.py
"""
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha

# Infrastructure (reuse from round 1)


CODE_SHA = code_sha()


_ext_dir = Path(__file__).resolve().parent / "extraction"
_core_dir = Path(__file__).resolve().parent.parent / "core"
_analysis_dir = Path(__file__).resolve().parent / "analysis"
_ablation_dir = Path(__file__).resolve().parent / "ablations"
_results_dir = Path(__file__).resolve().parent.parent / "results" / "data"

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
    .add_local_file(str(_analysis_dir / "analyze.py"), "/root/analyze.py", copy=True)
    .add_local_file(str(_analysis_dir / "memory_alignment.py"), "/root/memory_alignment.py", copy=True)
    .add_local_file(str(_analysis_dir / "probe_features.py"), "/root/probe_features.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation.py"), "/root/circuit_ablation.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation_v2.py"), "/root/circuit_ablation_v2.py", copy=True)
    # Data files baked into image
    .add_local_file(str(_results_dir / "memory_alignment_L9_H4.json"), "/root/memory_alignment_L9_H4.json", copy=True)
    .add_local_file(str(_results_dir / "gdn_head_level_results.json"), "/root/gdn_head_level_results.json", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("reviewer-round2")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")

DATA = "/data"
MODELS = "/models"
ABLATION = "/ablation"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
N_HEADS = 16
N_FEATURES = 2048
K = 32


# ===================================================================
# EXP A: Per-head PPL deltas (Task 17)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol, ABLATION: ablation_vol},
    gpu="A100", image=image, timeout=7200, memory=32768,
)
def exp_a_perhead_ppl_deltas(
    layer: int = 9,
    n_sequences: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Per-head PPL: reconstruct ONE head at a time, get 16 individual deltas."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    from evaluate_downstream import (
        load_sae_from_checkpoint,
        reconstruct_state_head,
        _patch_gdn_initial_states,
    )
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices

    print(f"=== EXP A: per-head PPL deltas, layer {layer} ===")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    corpus_ids = np.load(f"{DATA}/states/corpus.npy")
    n_seq = min(n_sequences, len(corpus_ids))
    seq_len = corpus_ids.shape[1]
    split_pos = seq_len // 2

    results = {"layer": layer, "n_sequences": n_seq, "per_head_deltas": {}}

    for sae_type_tag in ["bilinear", "flat"]:
        print(f"\n  === {sae_type_tag} ===")

        head_saes = {}
        for h in range(N_HEADS):
            ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type_tag}_s{seed}"
            best_path = ckpt_dir / "best.pt"
            config_path = ckpt_dir / "config.json"
            if not best_path.exists():
                import glob
                candidates = glob.glob(f"{DATA}/checkpoints/*/{sae_type_tag}_L{layer}_H{h}_nf{N_FEATURES}_k{K}_s{seed}/best.pt")
                if candidates:
                    best_path = Path(candidates[0])
                    config_path = best_path.parent / "config.json"
                else:
                    continue
            sae, cfg, _ = load_sae_from_checkpoint(
                str(best_path), str(config_path) if config_path.exists() else None, device="cuda",
            )
            sae.eval()
            head_saes[h] = (sae, sae_type_tag)
        if len(head_saes) != N_HEADS:
            print(f"  {sae_type_tag}: only {len(head_saes)}/{N_HEADS} heads, skipping")
            continue
        print(f"  {sae_type_tag}: loaded {len(head_saes)}/{N_HEADS} heads")

        head_deltas = {h: [] for h in range(N_HEADS)}

        for seq_i in range(n_seq):
            input_ids = torch.tensor(corpus_ids[seq_i:seq_i+1], dtype=torch.long, device="cuda")
            prefix = input_ids[:, :split_pos]
            suffix = input_ids[:, split_pos:]

            # Forward prefix to get cached states
            prefix_out = model(input_ids=prefix, use_cache=True)
            cache = prefix_out.past_key_values
            gdn_states_clean = {}
            for idx in gdn_layers:
                lc = cache.layers[idx]
                if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                    gdn_states_clean[idx] = lc.recurrent_states.clone()

            # Baseline loss (no reconstruction)
            with _patch_gdn_initial_states(model, gdn_layers, gdn_states_clean):
                baseline_out = model(input_ids=suffix, past_key_values=cache, use_cache=False, labels=suffix)
            baseline_loss = baseline_out.loss.item()

            # Per-head: reconstruct only head h
            for h in range(N_HEADS):
                gdn_states_h = {k: v.clone() for k, v in gdn_states_clean.items()}
                state = gdn_states_h[layer]
                sae, stype = head_saes[h]
                original = state[0, h].float()
                reconstructed = reconstruct_state_head(sae, original, stype)
                state[0, h] = reconstructed.to(state.dtype)

                with _patch_gdn_initial_states(model, gdn_layers, gdn_states_h):
                    recon_out = model(input_ids=suffix, past_key_values=cache, use_cache=False, labels=suffix)
                head_deltas[h].append(recon_out.loss.item() - baseline_loss)

            if seq_i % 50 == 0:
                print(f"    seq {seq_i}/{n_seq}")

        per_head_stats = {}
        for h in range(N_HEADS):
            deltas = head_deltas[h]
            per_head_stats[h] = {
                "mean_loss_delta": float(np.mean(deltas)),
                "std_loss_delta": float(np.std(deltas)),
                "n": len(deltas),
            }
        results["per_head_deltas"][sae_type_tag] = per_head_stats

        # Free GPU memory before loading next arch
        del head_saes
        torch.cuda.empty_cache()

    # Paired t-test: bilinear vs flat
    if "bilinear" in results["per_head_deltas"] and "flat" in results["per_head_deltas"]:
        from scipy.stats import ttest_rel
        bil_means = [results["per_head_deltas"]["bilinear"][h]["mean_loss_delta"] for h in range(N_HEADS)]
        flat_means = [results["per_head_deltas"]["flat"][h]["mean_loss_delta"] for h in range(N_HEADS)]
        t_stat, p_val = ttest_rel(bil_means, flat_means)
        results["paired_test_bilinear_vs_flat"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "bilinear_mean": float(np.mean(bil_means)),
            "flat_mean": float(np.mean(flat_means)),
        }
        print(f"\n  Paired t-test (bilinear vs flat): t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    bilinear mean delta: {np.mean(bil_means):.6f}")
        print(f"    flat mean delta: {np.mean(flat_means):.6f}")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp_a_perhead_ppl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_s{seed}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# EXP B: Per-head σ1/σ2 (Task 18) - NO GPU NEEDED
# ===================================================================

@app.function(
    volumes={DATA: data_vol},
    image=image, timeout=600, memory=16384,
    # No GPU - pure CPU SVD on memmaped arrays
)
def exp_b_perhead_sigma(
    layers: list[int] = [1, 9, 17],
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Per-head σ1/σ2 for all heads across layers. CPU only."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    from scipy.stats import spearmanr

    print("=== EXP B: per-head σ1/σ2 ===")

    results = {"layers": {}}

    for layer in layers:
        layer_dir = Path(DATA) / "states" / f"layer_{layer}"
        if not layer_dir.exists():
            print(f"  Layer {layer}: states not found, skipping")
            continue

        head_metrics = {}
        for h in range(N_HEADS):
            head_path = layer_dir / f"head_{h}.npy"
            if not head_path.exists():
                continue

            states = np.load(str(head_path), mmap_mode="r")
            idx = np.random.default_rng(42).choice(len(states), size=min(n_samples, len(states)), replace=False)
            sample = torch.from_numpy(states[idx].astype(np.float32))

            svs = torch.linalg.svdvals(sample).numpy()
            s1 = svs[:, 0]
            s2 = svs[:, 1]
            ratio = s1 / (s2 + 1e-10)

            p = svs / (svs.sum(1, keepdims=True) + 1e-12)
            eff_rank = np.exp(-(p * np.log(p + 1e-12)).sum(1))

            head_metrics[h] = {
                "sigma_ratio_mean": float(ratio.mean()),
                "sigma_ratio_std": float(ratio.std()),
                "effective_rank_mean": float(eff_rank.mean()),
                "sigma_1_mean": float(s1.mean()),
                "sigma_2_mean": float(s2.mean()),
            }

        results["layers"][layer] = head_metrics
        ratios = [head_metrics[h]["sigma_ratio_mean"] for h in sorted(head_metrics)]
        print(f"  Layer {layer}: σ1/σ2 range [{min(ratios):.2f}, {max(ratios):.2f}], mean {np.mean(ratios):.2f}")

    # Correlate with rank-1 advantage if data exists
    try:
        with open("/root/gdn_head_level_results.json") as f:
            head_level = json.load(f)
        per_head = {e["head"]: e for e in head_level.get("per_head", [])}

        if 9 in results["layers"] and per_head:
            sigma_ratios = [results["layers"][9][h]["sigma_ratio_mean"] for h in range(N_HEADS) if h in results["layers"][9]]
            rank1_advs = [per_head[h]["rank1_advantage_pct"] for h in range(N_HEADS) if h in per_head]
            if len(sigma_ratios) == len(rank1_advs):
                rho, p = spearmanr(sigma_ratios, rank1_advs)
                results["perhead_correlation_L9"] = {
                    "spearman_rho": float(rho), "p_value": float(p), "n": len(sigma_ratios),
                }
                print(f"\n  Per-head correlation (L9): ρ={rho:.3f}, p={p:.4f}, N={len(sigma_ratios)}")
    except Exception as e:
        print(f"  Correlation failed: {e}")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp_b_perhead_sigma"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "perhead_sigma.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# EXP C: Tied-bilinear downstream PPL (Task 19)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=3600, memory=32768,
)
def exp_c_tied_bilinear_ppl(
    layer: int = 9,
    n_sequences: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Downstream PPL for tied-bilinear from encoder-swap checkpoints."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    from evaluate_downstream import (
        load_sae_from_checkpoint,
        evaluate_downstream_perhead_matched,
    )
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices

    print(f"=== EXP C: tied-bilinear downstream PPL, layer {layer} ===")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    corpus_ids = np.load(f"{DATA}/states/corpus.npy")
    n_seq = min(n_sequences, len(corpus_ids))
    batches = [torch.tensor(corpus_ids[i:i+1], dtype=torch.long, device="cuda") for i in range(n_seq)]

    sae_type_configs = {}
    for sae_type_tag in ["bilinear_tied"]:
        head_saes = {}
        for h in range(N_HEADS):
            ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type_tag}_s{seed}"
            best_path = ckpt_dir / "best.pt"
            config_path = ckpt_dir / "config.json"
            if not best_path.exists():
                print(f"  MISSING: {best_path}")
                continue
            sae, cfg, _ = load_sae_from_checkpoint(
                str(best_path), str(config_path) if config_path.exists() else None, device="cuda",
            )
            sae.eval()
            head_saes[h] = (sae, sae_type_tag)

        if len(head_saes) == N_HEADS:
            tag = f"{sae_type_tag} (per-head matched, {len(head_saes)}/{N_HEADS} heads)"
            sae_type_configs[tag] = head_saes
            print(f"  {sae_type_tag}: loaded {len(head_saes)}/{N_HEADS} heads")

    if not sae_type_configs:
        return {"error": "No tied-bilinear checkpoints found"}

    result = evaluate_downstream_perhead_matched(
        model, tokenizer, batches, layer_idx=layer,
        sae_type_configs=sae_type_configs, n_heads=N_HEADS, split_fraction=0.5,
    )

    out_dir = Path(DATA) / "reviewer_experiments" / "exp_c_tied_bilinear"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_s{seed}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    data_vol.commit()

    baseline_ppl = result.get("baseline", {}).get("perplexity", 0)
    print(f"\n  Baseline PPL: {baseline_ppl:.4f}")
    for tag, res in result.get("sae_results", {}).items():
        print(f"  {tag}: PPL={res.get('perplexity', 0):.4f} (Δ={res.get('delta_pct', 0):+.2f}%)")

    return result


# ===================================================================
# EXP D: Write-alignment sensitivity to top-N (Task 20)
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=3600, memory=32768,
)
def exp_d_alignment_sensitivity(
    layer: int = 9,
    head: int = 4,
    top_n_values: list[int] = [10, 25, 50, 100],
    n_seqs: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Re-run write alignment with different top-N activating positions."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    print(f"=== EXP D: alignment sensitivity, L{layer} H{head} ===")

    from memory_alignment import load_model, load_sae, compute_alignment

    import glob
    candidates = glob.glob(f"{DATA}/checkpoints/*/bilinear_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}/best.pt")
    if not candidates:
        # Try ablation volume - but we don't have it mounted here
        return {"error": f"No bilinear checkpoint found for L{layer} H{head}"}

    ckpt_path = candidates[0]
    print(f"  SAE: {ckpt_path}")

    model, tokenizer = load_model(MODEL_08B, "cuda")
    sae = load_sae(ckpt_path, "cuda")

    results = {"layer": layer, "head": head, "top_n_results": {}}

    for top_n in top_n_values:
        print(f"  Running top_n={top_n}...")
        result = compute_alignment(
            model, tokenizer, sae,
            layer_idx=layer, head_idx=head,
            n_seqs=n_seqs, seq_len=512,
            batch_size=4, top_n=top_n,
            device="cuda",
        )

        # Extract stats
        alive_features = [r for r in result["results"] if r.get("alive", False)]
        combined = [math.sqrt(abs(r.get("mean_abs_k_cos", 0)) * abs(r.get("mean_abs_v_cos", 0)))
                    for r in alive_features]

        n_above_03 = sum(1 for c in combined if c > 0.3)
        results["top_n_results"][top_n] = {
            "n_alive": len(alive_features),
            "mean_combined": float(np.mean(combined)) if combined else 0,
            "median_combined": float(np.median(combined)) if combined else 0,
            "n_above_0.3": n_above_03,
            "pct_above_0.3": n_above_03 / max(len(combined), 1) * 100,
            "max_combined": float(max(combined)) if combined else 0,
        }
        print(f"    mean={np.mean(combined):.3f}, >0.3: {n_above_03}/{len(alive_features)} ({n_above_03/max(len(alive_features),1)*100:.1f}%)")

    out_dir = Path(DATA) / "reviewer_experiments" / "exp_d_alignment_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_H{head}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# EXP E: Wall-clock timing (Task 21)
# ===================================================================

@app.function(
    volumes={DATA: data_vol},
    gpu="A10G", image=image, timeout=600, memory=16384,
)
def exp_e_timing() -> dict[str, Any]:
    """Measure training + inference wall-clock time per architecture."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    print("=== EXP E: wall-clock timing ===")

    from train import train as train_sae

    # Train each type for 5 epochs on a small subset, measure wall-clock
    states_dir = f"{DATA}/states"
    results = {}

    for sae_type in ["flat", "rank1", "bilinear"]:
        print(f"  Training {sae_type} (5 epochs, timing run)...")
        t0 = time.time()
        out = train_sae(
            sae_type=sae_type,
            data_dir=states_dir,
            layer=9, head=0,
            n_features=N_FEATURES, k=K,
            lr=3e-4, lr_min=3e-5,
            batch_size=256, epochs=5,
            warmup_steps=10, norm_every=50,
            resample_every=1000,  # no resampling in 5 epochs
            rank=1, seed=42,
            output_dir=f"/tmp/timing_{sae_type}",
        )
        train_time = time.time() - t0

        # Inference timing: encode + decode 1000 samples
        from sae import build_sae_from_config
        ckpt = torch.load(f"/tmp/timing_{sae_type}/best.pt", map_location="cuda", weights_only=False)
        cfg = ckpt.get("config", {})
        sd = ckpt.get("model_state_dict", {})

        sae = build_sae_from_config(config=cfg, state_dict=sd)
        sae.load_state_dict(sd)
        sae = sae.cuda().eval()

        head_path = f"{states_dir}/layer_9/head_0.npy"
        states_np = np.load(head_path, mmap_mode="r")[:1000]
        if sae_type == "flat":
            x = torch.from_numpy(states_np.astype(np.float32)).reshape(1000, -1).cuda()
        else:
            x = torch.from_numpy(states_np.astype(np.float32)).cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = sae(x[:10])

        # Timed run
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, 1000, 64):
                _ = sae(x[i:i+64])
        torch.cuda.synchronize()
        infer_time = time.time() - t0

        results[sae_type] = {
            "train_5ep_s": train_time,
            "train_per_epoch_s": train_time / 5,
            "infer_1000_samples_s": infer_time,
            "infer_per_sample_ms": infer_time / 1000 * 1000,
            "n_params": sum(p.numel() for p in sae.parameters()),
        }
        print(f"    train: {train_time:.1f}s (5ep), infer: {infer_time*1000:.1f}ms (1000 samples)")
        print(f"    params: {results[sae_type]['n_params']:,}")

    out_dir = Path(DATA) / "reviewer_experiments" / "exp_e_timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "timing.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main(experiment: int = 0):
    """Launch all 5 experiments in parallel."""
    t0 = time.time()
    handles = []

    if experiment in (0, 1):
        print("Launching Exp A: per-head PPL deltas...")
        h = exp_a_perhead_ppl_deltas.spawn(layer=9, n_sequences=200, seed=42)
        handles.append(("exp_a_perhead_ppl", h))

    if experiment in (0, 2):
        print("Launching Exp B: per-head σ1/σ2...")
        h = exp_b_perhead_sigma.spawn(layers=[1, 9, 17], n_samples=1000)
        handles.append(("exp_b_perhead_sigma", h))

    if experiment in (0, 3):
        print("Launching Exp C: tied-bilinear PPL...")
        h = exp_c_tied_bilinear_ppl.spawn(layer=9, n_sequences=500, seed=42)
        handles.append(("exp_c_tied_bilinear", h))

    if experiment in (0, 4):
        print("Launching Exp D: alignment sensitivity...")
        h = exp_d_alignment_sensitivity.spawn(layer=9, head=4, n_seqs=100, seed=42)
        handles.append(("exp_d_alignment_sensitivity", h))

    if experiment in (0, 5):
        print("Launching Exp E: timing...")
        h = exp_e_timing.spawn()
        handles.append(("exp_e_timing", h))

    print(f"\n{len(handles)} jobs launched. Waiting...")

    results = {}
    for name, handle in handles:
        try:
            result = handle.get()
            results[name] = result
            print(f"\n✓ {name} complete")

            if name == "exp_a_perhead_ppl" and "paired_test_bilinear_vs_flat" in result:
                pt = result["paired_test_bilinear_vs_flat"]
                print(f"  Paired t-test: t={pt['t_stat']:.3f}, p={pt['p_value']:.4f}")
            elif name == "exp_b_perhead_sigma" and "perhead_correlation_L9" in result:
                c = result["perhead_correlation_L9"]
                print(f"  Per-head ρ(σ1/σ2, rank1_adv) = {c['spearman_rho']:.3f}, p={c['p_value']:.4f}")
            elif name == "exp_c_tied_bilinear":
                for tag, r in result.get("sae_results", {}).items():
                    print(f"  {tag}: Δ={r.get('delta_pct', 0):+.2f}%")
            elif name == "exp_d_alignment_sensitivity":
                for tn, r in result.get("top_n_results", {}).items():
                    print(f"  top_n={tn}: mean={r['mean_combined']:.3f}, >0.3: {r['pct_above_0.3']:.1f}%")
            elif name == "exp_e_timing":
                for arch, t in result.items():
                    print(f"  {arch}: train={t['train_per_epoch_s']:.1f}s/ep, infer={t['infer_per_sample_ms']:.2f}ms/sample, params={t['n_params']:,}")

        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results[name] = {"error": str(e)}

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out_path = Path("results/data/reviewer_round2_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")

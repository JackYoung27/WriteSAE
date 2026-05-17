"""Three parallel Modal experiments to raise review score from 4.5 to 6.

Experiment 1: Bilinear-flat downstream PPL (W2, Q2, Q3)
  - Load bilinear_flat checkpoints from layer-encoder-swap-v1 volume
  - Run downstream PPL evaluation for all 16 heads across layers 1, 9, 17
  - Compare against flat/bilinear/rank1 baselines

Experiment 2: Write-alignment feature ablation (W2, Q1)
  - Group features by write-alignment percentile
  - Zero each group, measure PPL change
  - Test: do write-aligned atoms drive disproportionate downstream impact?

Experiment 3: 4B multi-scale validation (W1)
  - Extract states from Qwen3.5-4B
  - Spectral audit (SVD)
  - Train flat/rank1/bilinear SAEs (3 seeds)

Usage:
    modal run experiments/run_reviewer_experiments.py
    modal run experiments/run_reviewer_experiments.py --experiment 1  # just exp 1
"""
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha

# Infrastructure (shared with run_modal.py)


CODE_SHA = code_sha()


# Core scripts live in experiments/extraction/ alongside run_modal.py
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
    # Core scripts
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    # Analysis/evaluation scripts
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .add_local_file(str(_analysis_dir / "analyze.py"), "/root/analyze.py", copy=True)
    .add_local_file(str(_analysis_dir / "memory_alignment.py"), "/root/memory_alignment.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation.py"), "/root/circuit_ablation.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation_v2.py"), "/root/circuit_ablation_v2.py", copy=True)
    .add_local_file(str(_analysis_dir / "probe_features.py"), "/root/probe_features.py", copy=True)
    .add_local_file(str(Path(__file__).resolve().parent.parent / "results" / "data" / "memory_alignment_L9_H4.json"), "/root/memory_alignment_L9_H4.json", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("reviewer-experiments")

# Volumes
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")
data_4b_vol = modal.Volume.from_name("matrix-sae-data-4b", create_if_missing=True)

DATA = "/data"
MODELS = "/models"
ABLATION = "/ablation"
DATA_4B = "/data4b"

MODEL_08B = "Qwen/Qwen3.5-0.8B"
MODEL_4B = "Qwen/Qwen3.5-4B"
LAYERS = [1, 9, 17]
N_HEADS = 16
SEEDS = [0, 1, 42]
N_FEATURES = 2048
K = 32


# ===================================================================
# EXPERIMENT 1: Bilinear-flat downstream PPL
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=7200, memory=32768,
)
def exp1_bilinear_flat_downstream(
    layer: int = 9,
    n_sequences: int = 500,
    seq_len: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    """Downstream PPL with per-head bilinear-flat SAEs from encoder-swap volume."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    from evaluate_downstream import (
        load_sae_from_checkpoint,
        evaluate_downstream_perhead_matched,
    )
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices

    print(f"=== EXP 1: bilinear-flat downstream PPL, layer {layer}, seed {seed} ===")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    corpus_path = Path(DATA) / "states" / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus_ids = np.load(str(corpus_path))
    n_seq = min(n_sequences, len(corpus_ids))
    batches = [torch.tensor(corpus_ids[i:i+1], dtype=torch.long, device="cuda")
               for i in range(n_seq)]

    # Load bilinear_flat checkpoints from ablation volume
    sae_type_configs: dict[str, dict[int, tuple]] = {}

    for sae_type_tag in ["bilinear_flat"]:
        head_saes: dict[int, tuple] = {}
        loaded = 0
        for h in range(N_HEADS):
            ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type_tag}_s{seed}"
            best_path = ckpt_dir / "best.pt"
            config_path = ckpt_dir / "config.json"

            if not best_path.exists():
                print(f"  MISSING: {best_path}")
                continue

            sae, cfg, train_mse = load_sae_from_checkpoint(
                str(best_path), str(config_path) if config_path.exists() else None, device="cuda",
            )
            sae.eval()
            head_saes[h] = (sae, sae_type_tag)
            loaded += 1

        print(f"  {sae_type_tag}: loaded {loaded}/{N_HEADS} heads")
        if loaded == N_HEADS:
            tag = f"{sae_type_tag} (per-head matched, {loaded}/{N_HEADS} heads)"
            sae_type_configs[tag] = head_saes

    # Also load flat + bilinear + rank1 from main data volume for comparison
    for sae_type_tag in ["flat", "bilinear", "rank1"]:
        head_saes = {}
        loaded = 0
        for h in range(N_HEADS):
            nf_tag = f"nf{N_FEATURES}"
            tag_str = f"{sae_type_tag}_L{layer}_H{h}_{nf_tag}_k{K}_s{seed}"
            ckpt_dir = Path(DATA) / "checkpoints" / tag_str
            best_path = ckpt_dir / "best.pt"
            config_path = ckpt_dir / "config.json"

            if not best_path.exists():
                # Try alternative paths
                alt_dir = Path(DATA) / f"states/layer_{layer}" / f"head_{h}" / f"{sae_type_tag}_s{seed}"
                alt_best = alt_dir / "best.pt"
                if alt_best.exists():
                    best_path = alt_best
                    config_path = alt_dir / "config.json"
                else:
                    continue

            sae, cfg, train_mse = load_sae_from_checkpoint(
                str(best_path), str(config_path) if config_path.exists() else None, device="cuda",
            )
            sae.eval()
            head_saes[h] = (sae, sae_type_tag)
            loaded += 1

        if loaded == N_HEADS:
            tag = f"{sae_type_tag} (per-head matched, {loaded}/{N_HEADS} heads)"
            sae_type_configs[tag] = head_saes
            print(f"  {sae_type_tag}: loaded {loaded}/{N_HEADS} heads")

    if not sae_type_configs:
        return {"error": "No SAE types loaded", "layer": layer}

    result = evaluate_downstream_perhead_matched(
        model, tokenizer, batches,
        layer_idx=layer,
        sae_type_configs=sae_type_configs,
        n_heads=N_HEADS,
        split_fraction=0.5,
    )

    out_dir = Path(DATA) / "reviewer_experiments" / "exp1_bilinear_flat_ppl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_{layer}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    data_vol.commit()

    baseline_ppl = result.get("baseline", {}).get("perplexity", 0)
    print(f"\n  Baseline PPL: {baseline_ppl:.4f}")
    for tag, res in result.get("sae_results", {}).items():
        delta = res.get("delta_pct", 0)
        ppl = res.get("perplexity", 0)
        print(f"  {tag}: PPL={ppl:.4f} (Δ={delta:+.2f}%)")

    return result


# ===================================================================
# EXPERIMENT 2: Write-alignment feature ablation
# ===================================================================

@app.function(
    volumes={DATA: data_vol, MODELS: model_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=7200, memory=32768,
)
def exp2_alignment_ablation(
    layer: int = 9,
    head: int = 4,
    n_sequences: int = 200,
    seq_len: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    """Per-feature ablation grouped by write-alignment percentile."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np
    import math

    from evaluate_downstream import load_sae_from_checkpoint
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices
    from circuit_ablation import compute_ppl_with_feature_ablation

    print(f"=== EXP 2: alignment ablation, L{layer} H{head} ===")

    align_path = Path(DATA) / "analysis" / f"memory_alignment_L{layer}_H{head}.json"
    if not align_path.exists():
        # Try results/data path
        align_path = Path("/root/memory_alignment_L9_H4.json")
        if not align_path.exists():
            return {"error": f"Alignment data not found at {align_path}"}

    with open(align_path) as f:
        align_data = json.load(f)

    features = []
    for r in align_data["results"]:
        if not r.get("alive", False):
            continue
        k_cos = abs(r.get("mean_abs_k_cos", 0))
        v_cos = abs(r.get("mean_abs_v_cos", 0))
        combined = math.sqrt(k_cos * v_cos)
        features.append({"idx": r["feature"], "combined": combined})

    features.sort(key=lambda x: x["combined"], reverse=True)
    n_alive = len(features)
    print(f"  {n_alive} alive features, top combined={features[0]['combined']:.3f}")

    # Group by percentile
    n_top = max(1, n_alive // 5)  # top 20%
    n_bot = max(1, n_alive // 5)  # bottom 20%

    groups = {
        "top_20pct": [f["idx"] for f in features[:n_top]],
        "mid_60pct": [f["idx"] for f in features[n_top:-n_bot]],
        "bot_20pct": [f["idx"] for f in features[-n_bot:]],
        "random_20pct": [],  # filled below
    }

    # Random sample of same size as top group
    rng = np.random.default_rng(seed)
    all_alive = [f["idx"] for f in features]
    groups["random_20pct"] = rng.choice(all_alive, size=n_top, replace=False).tolist()

    print(f"  Groups: top={len(groups['top_20pct'])}, mid={len(groups['mid_60pct'])}, "
          f"bot={len(groups['bot_20pct'])}, random={len(groups['random_20pct'])}")
    print(f"  Top 20% mean alignment: {np.mean([f['combined'] for f in features[:n_top]]):.3f}")
    print(f"  Bot 20% mean alignment: {np.mean([f['combined'] for f in features[-n_bot:]]):.3f}")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    # Load SAE checkpoint from ablation volume (known path structure)
    ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{head}" / f"bilinear_s{seed}"
    best_path = ckpt_dir / "best.pt"
    config_path = ckpt_dir / "config.json"

    if not best_path.exists():
        # Fallback: try main data volume with experiment tag
        import glob
        candidates = glob.glob(f"{DATA}/checkpoints/*/bilinear_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}/best.pt")
        if candidates:
            best_path = Path(candidates[0])
            config_path = best_path.parent / "config.json"
        else:
            return {"error": f"SAE checkpoint not found at {ckpt_dir} or {DATA}/checkpoints/*/bilinear_L{layer}_H{head}_*"}

    sae, cfg, train_mse = load_sae_from_checkpoint(
        str(best_path), str(config_path) if config_path.exists() else None, device="cuda",
    )
    sae.eval()
    print(f"  SAE loaded: {best_path}")

    corpus_path = Path(DATA) / "states" / "corpus.npy"
    corpus_ids = np.load(str(corpus_path))
    n_seq = min(n_sequences, len(corpus_ids))
    split_pos = seq_len // 2

    results = {"layer": layer, "head": head, "n_sequences": n_seq, "groups": {}}

    # First: baseline (no ablation)
    print("  Running baseline...")
    baseline_losses = []
    for i in range(n_seq):
        ids = torch.tensor(corpus_ids[i:i+1, :seq_len], dtype=torch.long, device="cuda")
        r = compute_ppl_with_feature_ablation(
            model, ids, split_pos, gdn_layers,
            target_layer_idx=layer, sae=sae, sae_type="bilinear",
            ablate_features=[], head_idx=head,
        )
        baseline_losses.append(r["loss"])
    baseline_ppl = math.exp(np.mean(baseline_losses))
    results["baseline_ppl"] = baseline_ppl
    print(f"  Baseline PPL (SAE, no ablation): {baseline_ppl:.4f}")

    for group_name, group_features in groups.items():
        print(f"  Running group: {group_name} ({len(group_features)} features)...")
        losses = []
        for i in range(n_seq):
            ids = torch.tensor(corpus_ids[i:i+1, :seq_len], dtype=torch.long, device="cuda")
            r = compute_ppl_with_feature_ablation(
                model, ids, split_pos, gdn_layers,
                target_layer_idx=layer, sae=sae, sae_type="bilinear",
                ablate_features=group_features, head_idx=head,
            )
            losses.append(r["loss"])

        group_ppl = math.exp(np.mean(losses))
        delta_pct = (group_ppl - baseline_ppl) / baseline_ppl * 100
        results["groups"][group_name] = {
            "n_features": len(group_features),
            "ppl": group_ppl,
            "delta_pct": delta_pct,
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
        }
        print(f"    PPL={group_ppl:.4f} (Δ={delta_pct:+.2f}%)")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "exp2_alignment_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"L{layer}_H{head}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# EXPERIMENT 3: 4B multi-scale validation
# ===================================================================

@app.function(
    volumes={DATA_4B: data_4b_vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=14400, memory=32768,
)
def exp3_4b_extract_and_train(
    layer: int = 9,
    n_samples: int = 5000,
    seq_len: int = 1024,
) -> dict[str, Any]:
    """Extract 4B states for one layer, train flat/rank1/bilinear SAEs (3 seeds)."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    print(f"=== EXP 3: 4B validation, layer {layer} ===")

    from extract_states import (
        load_model_and_tokenizer, get_gdn_layer_indices,
        extract_states, setup_memmaps, probe_state_dims,
    )

    # Step 1: Extract states
    print(f"  Loading {MODEL_4B}...")
    model, tokenizer, config = load_model_and_tokenizer(MODEL_4B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    print(f"  GDN layers: {gdn_layers}")
    if layer not in gdn_layers:
        nearest = min(gdn_layers, key=lambda x: abs(x - layer))
        print(f"  Layer {layer} is not GDN, using nearest: {nearest}")
        layer = nearest

    n_heads, d_k, d_v = probe_state_dims(model, layer, tokenizer, "cuda")
    print(f"  State dims: {n_heads} heads, {d_k}x{d_v}")

    # Tokenize corpus
    from datasets import load_dataset
    ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
    corpus_ids = []
    for item in ds:
        toks = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=seq_len)
        if toks["input_ids"].shape[1] == seq_len:
            corpus_ids.append(toks["input_ids"][0].numpy())
        if len(corpus_ids) >= n_samples:
            break
    corpus_np = np.array(corpus_ids, dtype=np.int64)
    print(f"  Corpus: {corpus_np.shape}")

    states_dir = Path(DATA_4B) / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(states_dir / "corpus.npy"), corpus_np)

    # Extract states
    layer_dir = states_dir / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    memmaps = setup_memmaps(states_dir, [layer], n_heads, d_k, d_v, n_samples)

    batch_size = 8
    batches = [torch.tensor(corpus_np[i:i+batch_size], dtype=torch.long, device="cuda")
               for i in range(0, n_samples, batch_size)]

    t0 = time.time()
    extract_states(model, config, batches, [layer], memmaps, device="cuda")
    extract_time = time.time() - t0
    print(f"  Extraction: {extract_time:.1f}s")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Step 2: Spectral audit
    print("  Running spectral audit...")
    head_data = np.lib.format.open_memmap(
        str(layer_dir / "head_0.npy"), mode="r",
    )
    sample_idx = np.random.default_rng(42).choice(len(head_data), size=min(1000, len(head_data)), replace=False)
    sample = torch.from_numpy(head_data[sample_idx].astype(np.float32))

    svs = torch.linalg.svdvals(sample).numpy()
    sv1 = svs[:, 0].mean()
    sv2 = svs[:, 1].mean()
    sv_ratio = sv1 / sv2

    p = svs / (svs.sum(1, keepdims=True) + 1e-12)
    eff_rank = np.exp(-(p * np.log(p + 1e-12)).sum(1)).mean()

    print(f"  σ1/σ2 = {sv_ratio:.2f}, eff_rank = {eff_rank:.1f}")

    # Step 3: Train SAEs
    from train import train as train_sae_fn

    train_results = {}
    for sae_type in ["flat", "rank1", "bilinear"]:
        for seed in SEEDS:
            ckpt_dir = Path(DATA_4B) / "checkpoints" / f"{sae_type}_L{layer}_s{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Training {sae_type} s{seed}...")
            t0 = time.time()
            result = train_sae_fn(
                sae_type=sae_type,
                data_dir=str(states_dir),
                layer=layer,
                head=0,  # head 0 for spectral comparison
                n_features=N_FEATURES,
                k=K,
                lr=3e-4,
                lr_min=3e-5,
                batch_size=256,
                epochs=20,
                warmup_steps=50,
                norm_every=100,
                resample_every=250,
                rank=1,
                seed=seed,
                output_dir=str(ckpt_dir),
            )
            train_time = time.time() - t0
            key = f"{sae_type}_s{seed}"
            train_results[key] = {
                "sae_type": sae_type,
                "seed": seed,
                "best_mse": result.get("best_mse", result.get("best_val_mse")),
                "n_dead": result.get("final_n_dead"),
                "time_s": train_time,
            }
            print(f"    MSE={train_results[key]['best_mse']:.6e}, "
                  f"dead={train_results[key]['n_dead']}, {train_time:.0f}s")

    # Compile results
    out = {
        "layer": layer,
        "model": MODEL_4B,
        "n_samples": n_samples,
        "n_heads": n_heads,
        "d_k": d_k,
        "d_v": d_v,
        "spectral": {
            "sv_ratio": float(sv_ratio),
            "sv1": float(sv1),
            "sv2": float(sv2),
            "effective_rank": float(eff_rank),
        },
        "training": train_results,
        "extract_time_s": extract_time,
    }

    # Save
    out_dir = Path(DATA_4B) / "reviewer_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"layer_{layer}_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    data_4b_vol.commit()

    return out


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main(experiment: int = 0):
    """Launch all three experiments in parallel.

    Args:
        experiment: 0 = all, 1/2/3 = specific experiment
    """
    t0 = time.time()
    handles: list[tuple[str, Any]] = []

    if experiment in (0, 1):
        print("Launching Exp 1: bilinear-flat downstream PPL...")
        for layer in LAYERS:
            for seed in [42]:  # single seed first for speed
                h = exp1_bilinear_flat_downstream.spawn(
                    layer=layer, n_sequences=500, seed=seed,
                )
                handles.append((f"exp1_L{layer}_s{seed}", h))

    if experiment in (0, 2):
        print("Launching Exp 2: alignment ablation...")
        h = exp2_alignment_ablation.spawn(
            layer=9, head=4, n_sequences=200, seed=42,
        )
        handles.append(("exp2_L9_H4", h))

    if experiment in (0, 3):
        print("Launching Exp 3: 4B multi-scale...")
        for layer in LAYERS:
            h = exp3_4b_extract_and_train.spawn(
                layer=layer, n_samples=5000,
            )
            handles.append((f"exp3_4b_L{layer}", h))

    print(f"\n{len(handles)} jobs launched in parallel. Waiting for results...")

    results = {}
    failures: list[str] = []
    for name, handle in handles:
        try:
            result = handle.get()
            results[name] = result
            print(f"\n✓ {name} complete")

            if name.startswith("exp1"):
                baseline = result.get("baseline", {}).get("perplexity", 0)
                for tag, r in result.get("sae_results", {}).items():
                    print(f"  {tag}: Δ={r.get('delta_pct', 0):+.2f}%")
            elif name.startswith("exp2"):
                for g, r in result.get("groups", {}).items():
                    print(f"  {g}: Δ={r.get('delta_pct', 0):+.2f}%")
            elif name.startswith("exp3"):
                sp = result.get("spectral", {})
                print(f"  σ1/σ2={sp.get('sv_ratio', 0):.2f}, eff_rank={sp.get('effective_rank', 0):.1f}")
                for k, v in result.get("training", {}).items():
                    print(f"  {k}: MSE={v.get('best_mse', 0):.6e}")

        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results[name] = {"error": str(e)}
            failures.append(f"{name}: {e}")

    elapsed = time.time() - t0
    print(f"\nAll experiments complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out_path = Path("results/data/reviewer_experiments_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Combined results saved to {out_path}")

    if failures:
        raise RuntimeError("One or more reviewer experiments failed: " + "; ".join(failures))

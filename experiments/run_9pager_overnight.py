"""Overnight 9-pager experiments: E7 (4B downstream PPL) + E9 (higher-rank decoder).

E7: Full downstream PPL evaluation on Qwen3.5-4B using 4B-trained SAEs
E9: Retrain BilinearSAE at rank={2,4} on L9/L17, measure MSE + downstream

Usage:
    modal run experiments/run_9pager_overnight.py
    modal run experiments/run_9pager_overnight.py --experiment e7
    modal run experiments/run_9pager_overnight.py --experiment e9
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
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("9pager-overnight")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
data_4b_vol = modal.Volume.from_name("matrix-sae-data-4b", create_if_missing=True)
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
DATA_4B = "/data4b"
MODELS = "/models"
MODEL_4B = "Qwen/Qwen3.5-4B"
MODEL_08B = "Qwen/Qwen3.5-0.8B"


# ===================================================================
# E7: 4B downstream PPL
# ===================================================================

@app.function(
    volumes={DATA_4B: data_4b_vol, MODELS: model_vol},
    gpu="A100", image=image, timeout=14400, memory=32768,
)
def exp_e7_4b_downstream(
    layer: int = 9,
    n_sequences: int = 200,
) -> dict[str, Any]:
    """Downstream PPL on 4B using SAEs trained on 4B states."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    from evaluate_downstream import (
        _patch_gdn_initial_states,
        reconstruct_state_head,
        load_sae_from_checkpoint,
    )
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices

    print(f"=== E7: 4B downstream PPL, layer {layer} ===")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_4B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)
    print(f"  GDN layers: {gdn_layers}")

    # Check if layer is a GDN layer
    if layer not in gdn_layers:
        nearest = min(gdn_layers, key=lambda x: abs(x - layer))
        print(f"  Layer {layer} not GDN, using nearest: {nearest}")
        layer = nearest

    corpus_path = Path(DATA_4B) / "states" / "corpus.npy"
    if not corpus_path.exists():
        return {"error": f"No corpus at {corpus_path}. Run 4B extraction first."}

    corpus_ids = np.load(str(corpus_path))
    n_seq = min(n_sequences, len(corpus_ids))
    seq_len = corpus_ids.shape[1]
    split_pos = seq_len // 2

    print(f"  {n_seq} sequences, split at {split_pos}")

    ckpt_base = Path(DATA_4B) / "checkpoints"
    results = {"layer": layer, "n_sequences": n_seq, "model": MODEL_4B}

    # Baseline PPL
    baseline_losses = []
    baseline_tokens = 0

    # Per-type losses
    sae_types_to_test = ["flat", "rank1", "bilinear"]
    type_losses = {t: [] for t in sae_types_to_test}
    type_tokens = {t: 0 for t in sae_types_to_test}

    # Load SAEs (one type at a time to save memory)
    for sae_type in sae_types_to_test:
        import glob
        pattern = f"{ckpt_base}/{sae_type}_L{layer}_s*/best.pt"
        candidates = glob.glob(pattern)
        if not candidates:
            # Try seed 42 specifically
            pattern2 = f"{ckpt_base}/{sae_type}_L{layer}_s42/best.pt"
            candidates = glob.glob(pattern2)
        if not candidates:
            print(f"  {sae_type}: no checkpoint found at {pattern}")
            continue

        ckpt_path = candidates[0]
        config_path = Path(ckpt_path).parent / "config.json"
        print(f"  {sae_type}: loading {ckpt_path}")

        sae, cfg, _ = load_sae_from_checkpoint(
            str(ckpt_path),
            str(config_path) if config_path.exists() else None,
            device="cuda",
        )
        sae.eval()

        for seq_i in range(n_seq):
            input_ids = torch.tensor(
                corpus_ids[seq_i:seq_i + 1], dtype=torch.long, device="cuda"
            )
            prefix = input_ids[:, :split_pos]
            suffix = input_ids[:, split_pos:]

            prefix_out = model(input_ids=prefix, use_cache=True)
            cache = prefix_out.past_key_values

            gdn_states = {}
            for idx in gdn_layers:
                lc = cache.layers[idx]
                if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                    gdn_states[idx] = lc.recurrent_states.clone()

            # Baseline (only compute once, on first type)
            if sae_type == sae_types_to_test[0]:
                with _patch_gdn_initial_states(model, gdn_layers, gdn_states):
                    baseline_out = model(
                        input_ids=suffix, past_key_values=cache,
                        use_cache=False, labels=suffix,
                    )
                n_tok = suffix.shape[1] - 1
                baseline_losses.append(baseline_out.loss.item() * n_tok)
                baseline_tokens += n_tok

            # Reconstruct all heads at target layer with this SAE
            # Re-run prefix for fresh cache
            prefix_out2 = model(input_ids=prefix, use_cache=True)
            cache2 = prefix_out2.past_key_values
            gdn_states2 = {}
            for idx in gdn_layers:
                lc = cache2.layers[idx]
                if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                    gdn_states2[idx] = lc.recurrent_states.clone()

            state = gdn_states2[layer]
            n_heads = state.shape[1]
            for h in range(n_heads):
                original = state[0, h].float()
                reconstructed = reconstruct_state_head(sae, original, sae_type)
                state[0, h] = reconstructed.to(state.dtype)

            with _patch_gdn_initial_states(model, gdn_layers, gdn_states2):
                recon_out = model(
                    input_ids=suffix, past_key_values=cache2,
                    use_cache=False, labels=suffix,
                )
            n_tok = suffix.shape[1] - 1
            type_losses[sae_type].append(recon_out.loss.item() * n_tok)
            type_tokens[sae_type] += n_tok

            if seq_i % 25 == 0:
                print(f"    {sae_type} seq {seq_i}/{n_seq}")

        # Free SAE memory
        del sae
        torch.cuda.empty_cache()

    if baseline_tokens > 0:
        baseline_avg = sum(baseline_losses) / baseline_tokens
        baseline_ppl = math.exp(baseline_avg)
        results["baseline_ppl"] = baseline_ppl
        results["baseline_loss"] = baseline_avg
        print(f"\n  Baseline PPL: {baseline_ppl:.4f}")

        results["sae_results"] = {}
        for sae_type in sae_types_to_test:
            if type_tokens[sae_type] > 0:
                avg = sum(type_losses[sae_type]) / type_tokens[sae_type]
                ppl = math.exp(avg)
                delta = (ppl - baseline_ppl) / baseline_ppl * 100
                results["sae_results"][sae_type] = {
                    "ppl": ppl, "loss": avg, "delta_pct": delta,
                }
                print(f"  {sae_type}: PPL={ppl:.4f} (Δ={delta:+.2f}%)")

    # Save
    out_dir = Path(DATA_4B) / "reviewer_experiments" / "e7_4b_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_4b_vol.commit()

    return results


# ===================================================================
# E9: Higher-rank bilinear decoder
# ===================================================================

@app.function(
    volumes={DATA: data_vol},
    gpu="A10G", image=image, timeout=7200, memory=16384,
)
def exp_e9_higher_rank(
    layer: int = 9,
    ranks: list[int] = [2, 4],
    seeds: list[int] = [0, 1, 42],
) -> dict[str, Any]:
    """Train BilinearSAE at rank-2 and rank-4, compare MSE."""
    import sys
    sys.path.insert(0, "/root")
    import numpy as np

    from train import train as train_sae

    print(f"=== E9: higher-rank bilinear, layer {layer}, ranks {ranks} ===")

    states_dir = f"{DATA}/states"
    results = {"layer": layer, "training": {}}

    for rank in ranks:
        for seed in seeds:
            key = f"bilinear_r{rank}_s{seed}"
            out_dir = f"{DATA}/checkpoints/higher_rank/{key}_L{layer}"

            print(f"  Training {key} at L{layer}...")
            t0 = time.time()
            out = train_sae(
                sae_type="bilinear",
                data_dir=states_dir,
                layer=layer, head=0,
                n_features=2048, k=32,
                lr=3e-4, lr_min=3e-5,
                batch_size=256, epochs=20,
                warmup_steps=50, norm_every=100,
                resample_every=250,
                rank=rank, seed=seed,
                output_dir=out_dir,
            )
            elapsed = time.time() - t0

            results["training"][key] = {
                "rank": rank,
                "seed": seed,
                "best_mse": out.get("best_mse", out.get("best_val_mse")),
                "n_dead": out.get("final_n_dead"),
                "time_s": elapsed,
            }
            print(f"    MSE={results['training'][key]['best_mse']:.6e}, "
                  f"dead={results['training'][key]['n_dead']}, {elapsed:.0f}s")

    # Also train rank-1 baseline for direct comparison (if not already there)
    for seed in seeds:
        key = f"bilinear_r1_s{seed}"
        out_dir = f"{DATA}/checkpoints/higher_rank/{key}_L{layer}"
        print(f"  Training {key} at L{layer} (baseline)...")
        t0 = time.time()
        out = train_sae(
            sae_type="bilinear",
            data_dir=states_dir,
            layer=layer, head=0,
            n_features=2048, k=32,
            lr=3e-4, lr_min=3e-5,
            batch_size=256, epochs=20,
            warmup_steps=50, norm_every=100,
            resample_every=250,
            rank=1, seed=seed,
            output_dir=out_dir,
        )
        elapsed = time.time() - t0
        results["training"][key] = {
            "rank": 1, "seed": seed,
            "best_mse": out.get("best_mse", out.get("best_val_mse")),
            "n_dead": out.get("final_n_dead"),
            "time_s": elapsed,
        }
        print(f"    MSE={results['training'][key]['best_mse']:.6e}, "
              f"dead={results['training'][key]['n_dead']}, {elapsed:.0f}s")

    # Summary
    print("\n  === Summary ===")
    for rank in [1] + ranks:
        mses = [v["best_mse"] for k, v in results["training"].items() if v["rank"] == rank]
        deads = [v["n_dead"] for k, v in results["training"].items() if v["rank"] == rank]
        if mses:
            print(f"  rank={rank}: MSE={np.mean(mses):.6e} ± {np.std(mses):.6e}, "
                  f"dead={np.mean(deads):.0f}")

    # Save
    out_dir_path = Path(DATA) / "reviewer_experiments" / "e9_higher_rank"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    with open(out_dir_path / f"L{layer}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


# ===================================================================
# ORCHESTRATOR
# ===================================================================

@app.local_entrypoint()
def main(experiment: str = "all"):
    t0 = time.time()
    handles = []

    if experiment in ("all", "e7"):
        print("Launching E7: 4B downstream PPL...")
        for layer in [9]:  # Start with L9, add L1/L17 if needed
            h = exp_e7_4b_downstream.spawn(layer=layer, n_sequences=200)
            handles.append((f"e7_4b_L{layer}", h))

    if experiment in ("all", "e9"):
        print("Launching E9: higher-rank bilinear...")
        for layer in [9, 17]:
            h = exp_e9_higher_rank.spawn(layer=layer, ranks=[2, 4], seeds=[0, 1, 42])
            handles.append((f"e9_rank_L{layer}", h))

    print(f"\n{len(handles)} jobs launched. Waiting...")

    results = {}
    failures: list[str] = []
    for name, handle in handles:
        try:
            result = handle.get()
            results[name] = result
            print(f"\n✓ {name} complete")

            if "baseline_ppl" in result:
                print(f"  Baseline: {result['baseline_ppl']:.4f}")
                for t, r in result.get("sae_results", {}).items():
                    print(f"  {t}: Δ={r['delta_pct']:+.2f}%")
            elif "training" in result:
                import numpy as np
                for rank in [1, 2, 4]:
                    mses = [v["best_mse"] for v in result["training"].values() if v["rank"] == rank]
                    if mses:
                        print(f"  rank={rank}: MSE={np.mean(mses):.6e}")

        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results[name] = {"error": str(e)}
            failures.append(f"{name}: {e}")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out_path = Path("results/data/9pager_overnight.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    if failures:
        raise RuntimeError("One or more overnight jobs failed: " + "; ".join(failures))

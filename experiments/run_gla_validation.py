#!/usr/bin/env python3
"""GLA model validation: extract states, spectral audit, train flat + bilinear SAEs.

Validates that the matrix SAE pipeline generalizes beyond Qwen3.5 GDN to
pure GLA (Gated Linear Attention) models.  Uses fla-hub/gla-1.3B-100B,
which has all 24 layers as linear attention with (256, 512) state matrices.

Usage:
    modal run experiments/run_gla_validation.py
    modal run experiments/run_gla_validation.py --stage extract
    modal run experiments/run_gla_validation.py --stage spectral
    modal run experiments/run_gla_validation.py --stage train
    modal run experiments/run_gla_validation.py --stage report
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from _modal_utils import code_sha

# Constants

MODEL_NAME = "fla-hub/gla-1.3B-100B"
LAYERS = [1, 12, 22]  # early / mid / late
N_SAMPLES = 5000
SEQ_LEN = 1024
BATCH_SIZE = 16  # 1.3B model, fits easily on A10G
N_FEATURES = 2048
K = 32
SEEDS = [0, 1, 2]
SAE_TYPES = ["flat", "bilinear"]
EPOCHS = 20


CODE_SHA = code_sha()

_ext_dir = Path(__file__).resolve().parent / "extraction"
_core_dir = Path(__file__).resolve().parent.parent / "core"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "MAX_JOBS": "4",
        "CC": "gcc",
        "CXX": "g++",
        "CUDAHOSTCXX": "g++",
    })
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install(
        "transformers>=5.0", "datasets", "numpy", "tqdm",
        "accelerate", "sentencepiece", "scipy", "scikit-learn",
        "einops", "ninja", "flash-linear-attention",
    )
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_core_dir / "types.py"), "/root/core/types.py", copy=True)
    .add_local_file(str(_core_dir / "__init__.py"), "/root/core/__init__.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA, "_CACHE_BUST": "v7"})
)

app = modal.App("gla-validation")
vol = modal.Volume.from_name("gla-validation-data", create_if_missing=True)
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
MODELS = "/models"
STATES_DIR = Path(f"{DATA}/gla_states")
CKPT_DIR = Path(f"{DATA}/gla_checkpoints")
RESULTS_DIR = Path(f"{DATA}/gla_results")

GPU_KW = dict(gpu="A10G", image=image, timeout=7200, memory=32768)


# Stage 1: Extract states from all 3 layers

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KW)
def extract_layer(layer: int) -> dict[str, Any]:
    """Extract GLA recurrent states for one layer."""
    import json, os, sys, time
    import numpy as np
    import torch
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import (
        get_gdn_layer_indices, load_corpus_tokens,
        setup_memmaps, probe_state_dims, extract_states,
    )

    # Load GLA model directly (not via AutoModel, which doesn't register FLA types)
    from fla.models.gla import GLAForCausalLM
    from transformers import AutoTokenizer
    import torch

    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = GLAForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    config = model.config
    gdn_layers = get_gdn_layer_indices(config)

    output_dir = STATES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / f"layer_{layer}"

    # Skip if already done
    layer_meta_path = layer_dir / "layer_metadata.json"
    if layer_meta_path.exists():
        existing = json.loads(layer_meta_path.read_text())
        if existing.get("n_samples", 0) >= N_SAMPLES and existing.get("model") == MODEL_NAME:
            print(f"Layer {layer} already extracted ({existing['n_samples']} samples). Skipping.")
            return existing

    all_layers = gdn_layers
    print(f"All linear attention layers: {len(all_layers)} (model has {config.num_hidden_layers} total)")

    n_heads, key_dim, val_dim = probe_state_dims(model, layer, tokenizer, "cuda")
    print(f"Layer {layer}: {n_heads} heads x ({key_dim}, {val_dim})")

    batches = load_corpus_tokens(tokenizer, None, SEQ_LEN, N_SAMPLES, BATCH_SIZE)
    actual_samples = sum(b.shape[0] for b in batches)
    print(f"Prepared {len(batches)} batches, {actual_samples} samples")

    memmaps = setup_memmaps(output_dir, [layer], n_heads, key_dim, val_dim, actual_samples)

    t0 = time.time()
    n_written = extract_states(model, config, batches, [layer], memmaps, "cuda")
    elapsed = time.time() - t0

    for mm in memmaps[layer]:
        mm.flush()

    layer_meta = {
        "model": MODEL_NAME,
        "layer": layer,
        "n_samples": n_written,
        "n_heads": n_heads,
        "key_head_dim": key_dim,
        "value_head_dim": val_dim,
        "state_shape_per_head": [n_written, key_dim, val_dim],
        "dtype": "float16",
        "seq_len": SEQ_LEN,
        "extraction_time_s": round(elapsed, 1),
    }
    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_meta_path.write_text(json.dumps(layer_meta, indent=2))
    vol.commit()

    print(f"Layer {layer}: {n_written} samples in {elapsed:.1f}s")
    return layer_meta


# Stage 2: Spectral audit (sigma_1 / sigma_2 per layer)

@app.function(volumes={DATA: vol}, **GPU_KW)
def spectral_audit() -> dict[str, Any]:
    """Compute sigma_1/sigma_2 ratio for each layer, head 0."""
    import json
    import numpy as np
    import torch

    vol.reload()
    results: dict[str, Any] = {}

    for layer in LAYERS:
        data_path = STATES_DIR / f"layer_{layer}" / "head_0.npy"
        if not data_path.exists():
            print(f"WARNING: {data_path} not found, skipping layer {layer}")
            continue

        states = np.load(str(data_path), mmap_mode="r").astype(np.float32)
        n, d_k, d_v = states.shape
        print(f"Layer {layer}: states shape = ({n}, {d_k}, {d_v})")

        # Compute SVD for batches of states, collect sigma_1 and sigma_2
        sigma1_list, sigma2_list = [], []
        batch_sz = 512
        for i in range(0, n, batch_sz):
            j = min(i + batch_sz, n)
            S = torch.linalg.svdvals(torch.from_numpy(states[i:j]).cuda()).cpu().numpy()
            sigma1_list.append(S[:, 0])
            sigma2_list.append(S[:, 1] if S.shape[1] > 1 else np.zeros(j - i))

        sigma1 = np.concatenate(sigma1_list)
        sigma2 = np.concatenate(sigma2_list)
        ratio = sigma1 / np.clip(sigma2, 1e-12, None)

        layer_result = {
            "sigma1_mean": float(sigma1.mean()),
            "sigma2_mean": float(sigma2.mean()),
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_std": float(ratio.std()),
            "n_samples": int(n),
            "d_k": d_k,
            "d_v": d_v,
        }
        results[f"layer_{layer}"] = layer_result
        print(
            f"  Layer {layer}: sigma1/sigma2 = {layer_result['ratio_mean']:.2f} "
            f"(median {layer_result['ratio_median']:.2f})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "spectral_audit.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()

    return results


# Stage 3: Train flat + bilinear SAEs (3 seeds, 3 layers, head 0)

@app.function(volumes={DATA: vol}, **GPU_KW)
def train_sae(
    sae_type: str,
    layer: int,
    seed: int,
) -> dict[str, Any]:
    """Train one SAE configuration."""
    import sys
    sys.path.insert(0, "/root")
    from train import train

    vol.reload()

    output_dir = str(CKPT_DIR / f"{sae_type}_L{layer}_H0_nf{N_FEATURES}_k{K}_s{seed}")
    result = train(
        sae_type=sae_type,
        data_dir=str(STATES_DIR),
        layer=layer,
        head=0,
        n_features=N_FEATURES,
        k=K,
        lr=3e-4,
        batch_size=256,
        epochs=EPOCHS,
        warmup_steps=50,
        resample_every=250,
        output_dir=output_dir,
        seed=seed,
        rank=1,
    )

    vol.commit()
    return result


# Stage 4: Report (aggregate and compare)

@app.function(volumes={DATA: vol}, image=image, timeout=600)
def report() -> dict[str, Any]:
    """Aggregate training results and spectral audit into a single report."""
    import json
    import numpy as np

    vol.reload()

    spectral_path = RESULTS_DIR / "spectral_audit.json"
    spectral = json.loads(spectral_path.read_text()) if spectral_path.exists() else {}

    # Collect training results
    train_results: list[dict] = []
    for sae_type in SAE_TYPES:
        for layer in LAYERS:
            for seed in SEEDS:
                config_path = CKPT_DIR / f"{sae_type}_L{layer}_H0_nf{N_FEATURES}_k{K}_s{seed}" / "config.json"
                if not config_path.exists():
                    print(f"Missing: {config_path}")
                    continue
                cfg = json.loads(config_path.read_text())
                train_results.append(cfg)

    # Build comparison table: per-layer, flat vs bilinear MSE
    comparison: dict[str, Any] = {}
    for layer in LAYERS:
        layer_key = f"layer_{layer}"
        flat_mses = []
        bilinear_mses = []
        for r in train_results:
            if r.get("layer") != layer or r.get("head", 0) != 0:
                continue
            tag = f"{r['sae_type']}_L{layer}_H0_nf{N_FEATURES}_k{K}_s{r.get('seed', 42)}"
            best_path = CKPT_DIR / tag / "best.pt"
            if not best_path.exists():
                continue
            import torch
            ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
            val_mse = ckpt.get("val_mse", float("inf"))
            if r["sae_type"] == "flat":
                flat_mses.append(val_mse)
            elif r["sae_type"] == "bilinear":
                bilinear_mses.append(val_mse)

        if flat_mses and bilinear_mses:
            flat_mean = float(np.mean(flat_mses))
            bilinear_mean = float(np.mean(bilinear_mses))
            advantage_pct = 100.0 * (flat_mean - bilinear_mean) / flat_mean if flat_mean > 0 else 0.0
            sigma_ratio = spectral.get(layer_key, {}).get("ratio_mean", float("nan"))

            comparison[layer_key] = {
                "flat_mse_mean": flat_mean,
                "flat_mse_std": float(np.std(flat_mses)),
                "bilinear_mse_mean": bilinear_mean,
                "bilinear_mse_std": float(np.std(bilinear_mses)),
                "bilinear_advantage_pct": round(advantage_pct, 2),
                "sigma1_sigma2_ratio": sigma_ratio,
                "n_seeds": len(flat_mses),
            }
            print(
                f"  Layer {layer}: flat={flat_mean:.4e}, bilinear={bilinear_mean:.4e}, "
                f"advantage={advantage_pct:.1f}%, sigma1/sigma2={sigma_ratio:.2f}"
            )
        else:
            print(f"  Layer {layer}: insufficient results (flat={len(flat_mses)}, bilinear={len(bilinear_mses)})")

    full_report = {
        "model": MODEL_NAME,
        "layers": LAYERS,
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "k": K,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "spectral_audit": spectral,
        "comparison": comparison,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "gla_validation_report.json"
    out_path.write_text(json.dumps(full_report, indent=2))
    vol.commit()

    print("\n=== GLA Validation Report ===")
    print(json.dumps(full_report, indent=2))
    return full_report


# Entrypoint

@app.local_entrypoint()
def main(
    stage: str = "all",
) -> None:
    """Run the GLA validation pipeline.

    Stages: extract, spectral, train, report, all
    """
    stages_to_run: list[str]
    if stage == "all":
        stages_to_run = ["extract", "spectral", "train", "report"]
    else:
        stages_to_run = [stage]

    # --- Extract ---
    if "extract" in stages_to_run:
        print(f"\n{'='*60}")
        print(f"STAGE: extract ({len(LAYERS)} layers in parallel)")
        print(f"{'='*60}")
        extract_futures = []
        for layer in LAYERS:
            extract_futures.append(extract_layer.spawn(layer))
        for fut in extract_futures:
            meta = fut.get()
            print(f"  Layer {meta.get('layer', '?')}: {meta.get('n_samples', 0)} samples, "
                  f"({meta.get('key_head_dim', '?')}, {meta.get('value_head_dim', '?')})")

    # --- Spectral ---
    if "spectral" in stages_to_run:
        print(f"\n{'='*60}")
        print("STAGE: spectral audit")
        print(f"{'='*60}")
        spectral_result = spectral_audit.remote()
        for layer_key, vals in spectral_result.items():
            print(f"  {layer_key}: sigma1/sigma2 = {vals['ratio_mean']:.2f}")

    # --- Train ---
    if "train" in stages_to_run:
        print(f"\n{'='*60}")
        configs = [(t, l, s) for t in SAE_TYPES for l in LAYERS for s in SEEDS]
        print(f"STAGE: train ({len(configs)} jobs)")
        print(f"{'='*60}")
        train_futures = []
        for sae_type, layer, seed in configs:
            train_futures.append(
                (sae_type, layer, seed, train_sae.spawn(sae_type, layer, seed))
            )
        for sae_type, layer, seed, fut in train_futures:
            result = fut.get()
            print(
                f"  {sae_type} L{layer} s{seed}: "
                f"best_mse={result.get('best_mse', '?'):.4e}, "
                f"time={result.get('total_time_s', '?')}s"
            )

    # --- Report ---
    if "report" in stages_to_run:
        print(f"\n{'='*60}")
        print("STAGE: report")
        print(f"{'='*60}")
        report.remote()

"""Train BatchTopK per-head checkpoints and evaluate matched all-head downstream fidelity.

This is the cleanest alternate sparsity-mechanism test already supported by the
codebase. It compares the canonical TopK checkpoints from the clean 0.8B volume
against freshly trained BatchTopK checkpoints under the same layer-9 per-head
matched downstream protocol.

Usage:
    modal run experiments/run_batchtopk_downstream.py
"""

from __future__ import annotations

import json
import math
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
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands("python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers>=5.0",
        "datasets",
        "numpy",
        "tqdm",
        "matplotlib",
        "accelerate",
        "sentencepiece",
        "scipy",
        "scikit-learn",
        "einops",
        "ninja",
        "flash-linear-attention",
    )
    .run_commands(
        f"python -m pip install --no-deps '{CAUSAL_CONV1D_WHEEL}'",
        f"python -m pip install --no-deps '{MAMBA_SSM_WHEEL}'",
    )
    .add_local_dir(str(_core_dir), "/root/core", copy=True)
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("batchtopk-downstream")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
MODELS = "/models"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
LAYER = 9
N_FEATURES = 2048
K = 32
SEED = 42
N_HEADS = 16

MATCHED_CHECKPOINT_ROOT = f"{DATA}/checkpoints/qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k"
BATCHTOPK_CHECKPOINT_ROOT = f"{DATA}/checkpoints/review_batchtopk_downstream"


@app.function(
    volumes={DATA: data_vol},
    gpu="A10G",
    image=image,
    timeout=7200,
    memory=32768,
)
def train_batchtopk_head(
    sae_type: str,
    head: int,
    layer: int = LAYER,
    seed: int = SEED,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    from train import train as train_sae

    out_dir = (
        f"{BATCHTOPK_CHECKPOINT_ROOT}/"
        f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_batchtopk_s{seed}"
    )
    print(f"Training {sae_type} BatchTopK, L{layer} H{head}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type=sae_type,
        data_dir=f"{DATA}/states",
        layer=layer,
        head=head,
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
        use_batchtopk=True,
        output_dir=out_dir,
    )
    elapsed = time.time() - t0
    data_vol.commit()
    return {
        "sae_type": sae_type,
        "head": head,
        "seed": seed,
        "best_mse": out.get("best_mse", out.get("best_val_mse")),
        "n_dead": out.get("final_n_dead"),
        "time_s": elapsed,
        "checkpoint_dir": out_dir,
    }


@app.function(
    volumes={DATA: data_vol, MODELS: model_vol},
    gpu="A10G",
    image=image,
    timeout=14400,
    memory=32768,
)
def evaluate_batchtopk_downstream(
    layer: int = LAYER,
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    import numpy as np
    from evaluate_downstream import (
        evaluate_downstream_perhead_matched,
        load_sae_from_checkpoint,
    )
    from extract_states import load_model_and_tokenizer

    print(f"=== BatchTopK downstream, L{layer}, n_sequences={n_sequences} ===")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()

    corpus_ids = np.load(f"{DATA}/states/corpus.npy", mmap_mode="r")
    actual = min(n_sequences, len(corpus_ids))
    batches = []
    batch_size = 8
    for start in range(0, actual, batch_size):
        end = min(start + batch_size, actual)
        batches.append(np.array(corpus_ids[start:end], dtype=np.int64))

    import torch
    corpus_batches = [torch.tensor(b, dtype=torch.long) for b in batches]

    sae_type_configs: dict[str, dict[int, tuple]] = {}

    # Canonical TopK baselines from the clean matched cohort.
    for sae_type in ["flat", "rank1", "bilinear"]:
        head_saes: dict[int, tuple] = {}
        for head in range(N_HEADS):
            ckpt_dir = (
                Path(MATCHED_CHECKPOINT_ROOT)
                / f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"
            )
            best_path = ckpt_dir / "best.pt"
            cfg_path = ckpt_dir / "config.json"
            sae, _, _ = load_sae_from_checkpoint(
                str(best_path),
                str(cfg_path),
                device="cuda",
            )
            head_saes[head] = (sae, sae_type)
        sae_type_configs[f"{sae_type}_topk"] = head_saes

    # Fresh BatchTopK checkpoints.
    for sae_type in ["flat", "rank1", "bilinear"]:
        head_saes = {}
        for head in range(N_HEADS):
            ckpt_dir = (
                Path(BATCHTOPK_CHECKPOINT_ROOT)
                / f"{sae_type}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_batchtopk_s{seed}"
            )
            best_path = ckpt_dir / "best.pt"
            cfg_path = ckpt_dir / "config.json"
            sae, _, _ = load_sae_from_checkpoint(
                str(best_path),
                str(cfg_path),
                device="cuda",
            )
            head_saes[head] = (sae, sae_type)
        sae_type_configs[f"{sae_type}_batchtopk"] = head_saes

    result = evaluate_downstream_perhead_matched(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=corpus_batches,
        layer_idx=layer,
        sae_type_configs=sae_type_configs,
        n_heads=N_HEADS,
        split_fraction=0.5,
        device="cuda",
    )
    result["experiment"] = "batchtopk_downstream"
    result["seed"] = seed
    result["n_features"] = N_FEATURES
    result["k"] = K
    result["n_sequences"] = actual

    out_dir = Path(DATA) / "reviewer_experiments" / "batchtopk_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_s{seed}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    data_vol.commit()
    return result


@app.local_entrypoint()
def main():
    t0 = time.time()
    handles = []

    for sae_type in ["flat", "rank1", "bilinear"]:
        for head in range(N_HEADS):
            handles.append(
                (
                    sae_type,
                    head,
                    train_batchtopk_head.spawn(sae_type=sae_type, head=head),
                )
            )

    print(f"Launched {len(handles)} BatchTopK training jobs. Waiting...")
    train_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for sae_type, head, handle in handles:
        try:
            result = handle.get()
            train_results.append(result)
            print(
                f"✓ {sae_type} H{head}: "
                f"MSE={result['best_mse']:.6e}, dead={result['n_dead']}, {result['time_s']:.0f}s"
            )
        except Exception as exc:
            failures.append(f"{sae_type} H{head}: {exc}")
            print(f"✗ {sae_type} H{head}: {exc}")

    if failures:
        raise RuntimeError("BatchTopK training failures: " + "; ".join(failures))

    print("\nTraining complete. Running downstream evaluation...")
    downstream = evaluate_batchtopk_downstream.remote(layer=LAYER, n_sequences=500, seed=SEED)

    summary = {
        "experiment": "batchtopk_downstream",
        "generated_at": time.time(),
        "train_results": train_results,
        "downstream": downstream,
    }

    out_path = Path("results/data/batchtopk_downstream.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

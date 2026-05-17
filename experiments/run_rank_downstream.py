"""Train higher-rank bilinear per-head checkpoints and evaluate downstream.

This turns the existing head-0 higher-rank observation into the clean
per-head matched downstream test the reviewer actually asked for.

Default run:
    modal run experiments/run_rank_downstream.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha


def _parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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

app = modal.App("rank-downstream")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
MODELS = "/models"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
N_FEATURES = 2048
K = 32
SEED = 42
N_HEADS = 16

MATCHED_CHECKPOINT_ROOT = f"{DATA}/checkpoints/qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k"
RANK_CHECKPOINT_ROOT = f"{DATA}/checkpoints/review_rank_downstream"


def _canonical_rank1_dir(layer: int, head: int, seed: int) -> Path:
    return Path(MATCHED_CHECKPOINT_ROOT) / f"bilinear_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"


def _rank_checkpoint_dir(layer: int, head: int, rank: int, seed: int) -> Path:
    return Path(RANK_CHECKPOINT_ROOT) / f"bilinear_r{rank}_L{layer}_H{head}_nf{N_FEATURES}_k{K}_s{seed}"


def _rank1_checkpoint_dir(layer: int, head: int, seed: int) -> Path:
    canonical = _canonical_rank1_dir(layer=layer, head=head, seed=seed)
    if (canonical / "best.pt").exists() and (canonical / "config.json").exists():
        return canonical
    return _rank_checkpoint_dir(layer=layer, head=head, rank=1, seed=seed)


def _load_existing_train_result(ckpt_dir: Path) -> dict[str, Any]:
    import torch

    best_path = ckpt_dir / "best.pt"
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    return {
        "best_mse": payload.get("val_mse"),
        "config": payload.get("config", {}),
    }


@app.function(
    volumes={DATA: data_vol},
    gpu="A10G",
    image=image,
    timeout=7200,
    memory=32768,
)
def train_rank_head(
    layer: int,
    head: int,
    rank: int = 2,
    seed: int = SEED,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    from train import train as train_sae

    if rank == 1:
        canonical_dir = _canonical_rank1_dir(layer=layer, head=head, seed=seed)
        canonical_best = canonical_dir / "best.pt"
        canonical_cfg = canonical_dir / "config.json"
        if canonical_best.exists() and canonical_cfg.exists():
            existing = _load_existing_train_result(canonical_dir)
            return {
                "layer": layer,
                "head": head,
                "rank": rank,
                "seed": seed,
                "best_mse": existing["best_mse"],
                "checkpoint_dir": str(canonical_dir),
                "skipped": True,
                "source": "canonical",
            }

    ckpt_dir = _rank_checkpoint_dir(layer=layer, head=head, rank=rank, seed=seed)
    best_path = ckpt_dir / "best.pt"
    cfg_path = ckpt_dir / "config.json"

    if best_path.exists() and cfg_path.exists():
        existing = _load_existing_train_result(ckpt_dir)
        return {
            "layer": layer,
            "head": head,
            "rank": rank,
            "seed": seed,
            "best_mse": existing["best_mse"],
            "checkpoint_dir": str(ckpt_dir),
            "skipped": True,
        }

    print(f"Training bilinear rank-{rank}, L{layer} H{head}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type="bilinear",
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
        rank=rank,
        seed=seed,
        output_dir=str(ckpt_dir),
    )
    elapsed = time.time() - t0
    data_vol.commit()
    return {
        "layer": layer,
        "head": head,
        "rank": rank,
        "seed": seed,
        "best_mse": out.get("best_mse", out.get("best_val_mse")),
        "n_dead": out.get("final_n_dead"),
        "time_s": elapsed,
        "checkpoint_dir": str(ckpt_dir),
        "skipped": False,
    }


@app.function(
    volumes={DATA: data_vol, MODELS: model_vol},
    gpu="A10G",
    image=image,
    timeout=14400,
    memory=32768,
)
def evaluate_rank_downstream(
    layer: int,
    ranks_csv: str = "1,2",
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from evaluate_downstream import (
        evaluate_downstream_perhead_matched,
        load_sae_from_checkpoint,
    )
    from extract_states import load_model_and_tokenizer

    ranks = _parse_int_csv(ranks_csv)
    if 1 not in ranks:
        ranks = [1] + ranks

    model, tokenizer, _ = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()

    corpus_ids = np.load(f"{DATA}/states/corpus.npy", mmap_mode="r")
    actual = min(n_sequences, len(corpus_ids))
    batches = []
    batch_size = 8
    for start in range(0, actual, batch_size):
        end = min(start + batch_size, actual)
        batches.append(np.array(corpus_ids[start:end], dtype=np.int64))
    corpus_batches = [torch.tensor(batch, dtype=torch.long) for batch in batches]

    sae_type_configs: dict[str, dict[int, tuple]] = {}
    train_mse_summary: dict[str, dict[str, float | int | None]] = {}

    for rank in ranks:
        tag = f"bilinear_r{rank}"
        head_saes: dict[int, tuple] = {}
        train_mses: list[float] = []
        for head in range(N_HEADS):
            if rank == 1:
                ckpt_dir = _rank1_checkpoint_dir(layer=layer, head=head, seed=seed)
            else:
                ckpt_dir = _rank_checkpoint_dir(layer=layer, head=head, rank=rank, seed=seed)

            best_path = ckpt_dir / "best.pt"
            cfg_path = ckpt_dir / "config.json"
            sae, _, train_mse = load_sae_from_checkpoint(
                str(best_path),
                str(cfg_path),
                device="cuda",
            )
            head_saes[head] = (sae, "bilinear")
            if train_mse is not None:
                train_mses.append(float(train_mse))

        sae_type_configs[tag] = head_saes
        train_mse_summary[tag] = {
            "rank": rank,
            "n_heads_loaded": len(head_saes),
            "mean_val_mse": _mean(train_mses),
        }

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
    result["experiment"] = "rank_downstream"
    result["seed"] = seed
    result["n_features"] = N_FEATURES
    result["k"] = K
    result["n_sequences"] = actual
    result["train_mse_summary"] = train_mse_summary
    result["ranks"] = ranks

    out_dir = Path(DATA) / "reviewer_experiments" / "rank_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_ranks_{'-'.join(str(rank) for rank in ranks)}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    data_vol.commit()
    return result


@app.local_entrypoint()
def main(
    layers: str = "9,17",
    ranks: str = "2",
    n_sequences: int = 500,
    seed: int = SEED,
):
    t0 = time.time()
    layers_list = _parse_int_csv(layers)
    requested_ranks = sorted(set(_parse_int_csv(ranks)))
    train_ranks = sorted({1, *[rank for rank in requested_ranks if rank > 1]})
    eval_ranks = sorted({1, *requested_ranks})

    handles = []
    for layer in layers_list:
        for rank in train_ranks:
            for head in range(N_HEADS):
                handles.append(
                    (
                        layer,
                        rank,
                        head,
                        train_rank_head.spawn(layer=layer, head=head, rank=rank, seed=seed),
                    )
                )

    print(f"Launched {len(handles)} higher-rank training jobs. Waiting...")
    train_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for layer, rank, head, handle in handles:
        try:
            result = handle.get()
            train_results.append(result)
            status = "skip" if result.get("skipped") else "done"
            print(
                f"✓ [{status}] L{layer} r{rank} H{head}: "
                f"MSE={result.get('best_mse'):.6e}"
            )
        except Exception as exc:
            failures.append(f"L{layer} r{rank} H{head}: {exc}")
            print(f"✗ L{layer} r{rank} H{head}: {exc}")

    if failures:
        raise RuntimeError("Higher-rank training failures: " + "; ".join(failures))

    print("\nTraining complete. Running downstream evaluations...")
    downstream_results: dict[str, Any] = {}
    for layer in layers_list:
        result = evaluate_rank_downstream.remote(
            layer=layer,
            ranks_csv=",".join(str(rank) for rank in eval_ranks),
            n_sequences=n_sequences,
            seed=seed,
        )
        downstream_results[f"L{layer}"] = result

    summary = {
        "experiment": "rank_downstream",
        "generated_at": time.time(),
        "layers": layers_list,
        "ranks": eval_ranks,
        "train_results": train_results,
        "downstream": downstream_results,
    }

    out_path = Path("results/data/rank_downstream.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

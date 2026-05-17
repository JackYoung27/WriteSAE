"""Train fresh per-head cohorts for downstream sensitivity sweeps.

This addresses reviewer ask #4 with matched all-head downstream curves over
either `k` or `nf`.

Defaults to the cheaper `k` sweep:
    modal run experiments/run_downstream_sensitivity.py
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

app = modal.App("downstream-sensitivity")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
MODELS = "/models"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
LAYER = 9
DEFAULT_N_FEATURES = 2048
DEFAULT_K = 32
SEED = 42
N_HEADS = 16

MATCHED_CHECKPOINT_ROOT = f"{DATA}/checkpoints/qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k"
SENSITIVITY_CHECKPOINT_ROOT = f"{DATA}/checkpoints/review_downstream_sensitivity"


def _sweep_root_tag(sweep_kind: str, values: list[int], base_n_features: int, base_k: int) -> str:
    values_slug = "-".join(str(value) for value in values)
    return f"{sweep_kind}_{values_slug}_nf{base_n_features}_k{base_k}"


def _canonical_checkpoint_dir(
    sae_type: str,
    layer: int,
    head: int,
    seed: int,
) -> Path:
    return Path(MATCHED_CHECKPOINT_ROOT) / f"{sae_type}_L{layer}_H{head}_nf{DEFAULT_N_FEATURES}_k{DEFAULT_K}_s{seed}"


def _sensitivity_checkpoint_dir(
    sweep_root_tag: str,
    sae_type: str,
    layer: int,
    head: int,
    n_features: int,
    k: int,
    seed: int,
) -> Path:
    return (
        Path(SENSITIVITY_CHECKPOINT_ROOT)
        / sweep_root_tag
        / f"{sae_type}_L{layer}_H{head}_nf{n_features}_k{k}_s{seed}"
    )


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
def train_sensitivity_head(
    sweep_root_tag: str,
    sae_type: str,
    head: int,
    layer: int = LAYER,
    n_features: int = DEFAULT_N_FEATURES,
    k: int = DEFAULT_K,
    seed: int = SEED,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    from train import train as train_sae

    ckpt_dir = _sensitivity_checkpoint_dir(
        sweep_root_tag=sweep_root_tag,
        sae_type=sae_type,
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        seed=seed,
    )
    best_path = ckpt_dir / "best.pt"
    cfg_path = ckpt_dir / "config.json"

    if best_path.exists() and cfg_path.exists():
        existing = _load_existing_train_result(ckpt_dir)
        return {
            "layer": layer,
            "head": head,
            "sae_type": sae_type,
            "n_features": n_features,
            "k": k,
            "seed": seed,
            "best_mse": existing["best_mse"],
            "checkpoint_dir": str(ckpt_dir),
            "skipped": True,
        }

    print(f"Training {sae_type}, L{layer} H{head}, nf={n_features}, k={k}, seed {seed}")
    t0 = time.time()
    out = train_sae(
        sae_type=sae_type,
        data_dir=f"{DATA}/states",
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
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
    elapsed = time.time() - t0
    data_vol.commit()
    return {
        "layer": layer,
        "head": head,
        "sae_type": sae_type,
        "n_features": n_features,
        "k": k,
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
def evaluate_sensitivity_downstream(
    sweep_kind: str = "k",
    values_csv: str = "16,32,64",
    sae_types_csv: str = "flat,rank1,bilinear",
    layer: int = LAYER,
    n_sequences: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    import gc
    import sys

    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    from evaluate_downstream import (
        evaluate_downstream_perhead_matched,
        load_sae_from_checkpoint,
    )
    from extract_states import load_model_and_tokenizer

    values = _parse_int_csv(values_csv)
    sae_types = [part.strip() for part in sae_types_csv.split(",") if part.strip()]

    if sweep_kind not in {"k", "nf"}:
        raise ValueError(f"Unsupported sweep_kind={sweep_kind!r}")

    sweep_root_tag = _sweep_root_tag(
        sweep_kind=sweep_kind,
        values=values,
        base_n_features=DEFAULT_N_FEATURES,
        base_k=DEFAULT_K,
    )

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

    train_mse_summary: dict[str, dict[str, float | int | None]] = {}

    baseline_eval = evaluate_downstream_perhead_matched(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=corpus_batches,
        layer_idx=layer,
        sae_type_configs={},
        n_heads=N_HEADS,
        split_fraction=0.5,
        device="cuda",
    )

    result: dict[str, Any] = dict(baseline_eval)
    result["sae_results"] = {}

    for sae_type in sae_types:
        for value in values:
            if sweep_kind == "k":
                n_features = DEFAULT_N_FEATURES
                k = value
                tag = f"{sae_type}_k{value}"
                use_canonical = value == DEFAULT_K
            else:
                n_features = value
                k = DEFAULT_K
                tag = f"{sae_type}_nf{value}"
                use_canonical = value == DEFAULT_N_FEATURES

            head_saes: dict[int, tuple] = {}
            train_mses: list[float] = []
            for head in range(N_HEADS):
                if use_canonical:
                    ckpt_dir = _canonical_checkpoint_dir(
                        sae_type=sae_type,
                        layer=layer,
                        head=head,
                        seed=seed,
                    )
                else:
                    ckpt_dir = _sensitivity_checkpoint_dir(
                        sweep_root_tag=sweep_root_tag,
                        sae_type=sae_type,
                        layer=layer,
                        head=head,
                        n_features=n_features,
                        k=k,
                        seed=seed,
                    )

                sae, _, train_mse = load_sae_from_checkpoint(
                    str(ckpt_dir / "best.pt"),
                    str(ckpt_dir / "config.json"),
                    device="cuda",
                )
                head_saes[head] = (sae, sae_type)
                if train_mse is not None:
                    train_mses.append(float(train_mse))

            train_mse_summary[tag] = {
                "n_features": n_features,
                "k": k,
                "n_heads_loaded": len(head_saes),
                "mean_val_mse": _mean(train_mses),
            }

            single_result = evaluate_downstream_perhead_matched(
                model=model,
                tokenizer=tokenizer,
                corpus_batches=corpus_batches,
                layer_idx=layer,
                sae_type_configs={tag: head_saes},
                n_heads=N_HEADS,
                split_fraction=0.5,
                device="cuda",
                baseline_result=result["baseline"],
            )
            result["sae_results"][tag] = single_result["sae_results"][tag]

            for sae, _ in head_saes.values():
                sae.cpu()
            del head_saes
            gc.collect()
            torch.cuda.empty_cache()

    result["experiment"] = "downstream_sensitivity"
    result["sweep_kind"] = sweep_kind
    result["values"] = values
    result["seed"] = seed
    result["n_sequences"] = actual
    result["train_mse_summary"] = train_mse_summary

    out_dir = Path(DATA) / "reviewer_experiments" / "downstream_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{sweep_root_tag}_L{layer}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    data_vol.commit()
    return result


@app.local_entrypoint()
def main(
    sweep_kind: str = "k",
    values: str = "16,32,64",
    sae_types: str = "flat,rank1,bilinear",
    layer: int = LAYER,
    n_sequences: int = 500,
    seed: int = SEED,
):
    t0 = time.time()
    values_list = _parse_int_csv(values)
    sae_type_list = [part.strip() for part in sae_types.split(",") if part.strip()]

    if sweep_kind not in {"k", "nf"}:
        raise ValueError(f"Unsupported sweep_kind={sweep_kind!r}")

    sweep_root_tag = _sweep_root_tag(
        sweep_kind=sweep_kind,
        values=values_list,
        base_n_features=DEFAULT_N_FEATURES,
        base_k=DEFAULT_K,
    )

    handles = []
    for sae_type in sae_type_list:
        for value in values_list:
            if sweep_kind == "k":
                n_features = DEFAULT_N_FEATURES
                k = value
                is_canonical = value == DEFAULT_K
            else:
                n_features = value
                k = DEFAULT_K
                is_canonical = value == DEFAULT_N_FEATURES

            if is_canonical:
                continue

            for head in range(N_HEADS):
                handles.append(
                    (
                        sae_type,
                        value,
                        head,
                        train_sensitivity_head.spawn(
                            sweep_root_tag=sweep_root_tag,
                            sae_type=sae_type,
                            head=head,
                            layer=layer,
                            n_features=n_features,
                            k=k,
                            seed=seed,
                        ),
                    )
                )

    print(f"Launched {len(handles)} sensitivity training jobs. Waiting...")
    train_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for sae_type, value, head, handle in handles:
        try:
            result = handle.get()
            train_results.append(result)
            status = "skip" if result.get("skipped") else "done"
            label = f"{sweep_kind}={value}"
            print(
                f"✓ [{status}] {sae_type} {label} H{head}: "
                f"MSE={result.get('best_mse'):.6e}"
            )
        except Exception as exc:
            failures.append(f"{sae_type} {sweep_kind}={value} H{head}: {exc}")
            print(f"✗ {sae_type} {sweep_kind}={value} H{head}: {exc}")

    if failures:
        raise RuntimeError("Downstream sensitivity training failures: " + "; ".join(failures))

    print("\nTraining complete. Running downstream evaluation...")
    downstream = evaluate_sensitivity_downstream.remote(
        sweep_kind=sweep_kind,
        values_csv=values,
        sae_types_csv=sae_types,
        layer=layer,
        n_sequences=n_sequences,
        seed=seed,
    )

    summary = {
        "experiment": "downstream_sensitivity",
        "generated_at": time.time(),
        "sweep_kind": sweep_kind,
        "values": values_list,
        "layer": layer,
        "train_results": train_results,
        "downstream": downstream,
    }

    out_path = Path(f"results/data/downstream_sensitivity_{sweep_kind}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved to {out_path}")

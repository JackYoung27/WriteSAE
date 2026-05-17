#!/usr/bin/env python3
"""Layer-level encoder-swap ablation: train all 5 SAE types across all 16 heads
on layers 1, 9, 17 to test whether bilinear encoder gains hold across the full
spectral concentration gradient.

Trains 5 SAE types x 16 heads x 3 layers x 3 seeds = 720 jobs total.

SAE types:
  flat          = linear encoder  + flat decoder
  rank1         = linear encoder  + rank-1 decoder
  bilinear      = bilinear encoder + rank-1 decoder (untied)
  bilinear_tied = bilinear encoder + rank-1 decoder (tied)
  bilinear_flat = bilinear encoder + flat decoder

Data source: volume matrix-sae-data-08b-clean (read-only)
  states/layer_{L}/head_{H}.npy  (5000, 128, 128) float16

Checkpoints: volume layer-encoder-swap-v1
  layer_{L}/head_{H}/{type}_s{seed}/best.pt

Usage:
    modal run layer_encoder_swap_ablation.py --stage train
    modal run layer_encoder_swap_ablation.py --stage analyze
    modal run layer_encoder_swap_ablation.py --stage all
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal
import numpy as np


def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__) or ".",
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


CODE_SHA = _code_sha()

# Experiment configuration

LAYERS = [1, 9, 17]
N_HEADS = 16
ALL_SAE_TYPES = ["flat", "rank1", "bilinear", "bilinear_tied", "bilinear_flat"]
SEEDS = [0, 1, 42]

# Spectral concentration labels for reporting.
SPECTRAL_LABELS = {
    1:  "~12 (concentrated)",
    9:  "6.8 (medium)",
    17: "3.0 (diffuse)",
}

# Hyperparameters (match the multihead sweep).
SAE_N_FEATURES = 2048
SAE_K = 32
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 3e-4
TRAIN_LR_MIN = 3e-5
TRAIN_WARMUP_STEPS = 50
TRAIN_RESAMPLE_EVERY = 250

TOTAL_JOBS = len(LAYERS) * N_HEADS * len(ALL_SAE_TYPES) * len(SEEDS)  # 720

# Modal setup

CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    "v1.6.1.post4/"
    "causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
MAMBA_SSM_WHEEL = (
    "https://github.com/state-spaces/mamba/releases/download/"
    "v2.3.1/"
    "mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
          "MAX_JOBS": "4", "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++"})
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands("python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126")
    .pip_install("transformers>=5.0", "datasets", "numpy", "tqdm", "accelerate", "scipy",
                 "einops", "ninja", "flash-linear-attention")
    .run_commands(
        f"python -m pip install --no-deps '{CAUSAL_CONV1D_WHEEL}'",
        f"python -m pip install --no-deps '{MAMBA_SSM_WHEEL}'",
    )
    .add_local_file("sae.py", "/root/sae.py", copy=True)
    .add_local_file("split_utils.py", "/root/split_utils.py", copy=True)
    .add_local_file("train.py", "/root/train.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("layer-encoder-swap-ablation")

data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1", create_if_missing=True)

DATA = "/data"
ABLATION = "/ablation"


# Stage 1: TRAIN

@app.function(
    volumes={DATA: data_vol, ABLATION: ablation_vol},
    image=image,
    gpu="A10G",
    timeout=1800,
    memory=16384,
)
def train_sae(
    layer: int,
    head: int,
    sae_type: str,
    seed: int,
    n_features: int = SAE_N_FEATURES,
    k: int = SAE_K,
) -> dict[str, Any]:
    """Train one SAE on one (layer, head) pair. Skips if checkpoint exists."""
    import sys
    sys.path.insert(0, "/root")
    import torch

    ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}"
    best_path = ckpt_dir / "best.pt"

    # Skip if already trained with matching config and code version.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        ckpt_code_sha = str(cfg.get("code_sha", ckpt.get("code_sha", "unknown")))
        if (
            cfg.get("sae_type") == sae_type
            and int(cfg.get("layer", layer)) == layer
            and int(cfg.get("head", head)) == head
            and int(cfg.get("n_features", n_features)) == n_features
            and int(cfg.get("k", k)) == k
            and ckpt_code_sha == CODE_SHA
        ):
            val_mse = ckpt.get("val_mse", ckpt.get("best_val_mse"))
            print(f"L{layer} H{head} {sae_type} s{seed}: already exists (val_mse={val_mse})")
            return {
                "layer": layer, "head": head, "sae_type": sae_type, "seed": seed,
                "best_mse": val_mse, "skipped": True,
            }
        print(
            f"L{layer} H{head} {sae_type} s{seed}: checkpoint exists but metadata/code_sha "
            "mismatch; retraining"
        )

    # Verify state data exists.
    data_path = Path(DATA) / "states" / f"layer_{layer}" / f"head_{head}.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"State data not found: {data_path}")

    from train import train

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        sae_type=sae_type,
        data_dir=f"{DATA}/states",
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        seed=seed,
        output_dir=str(ckpt_dir),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        lr_min=TRAIN_LR_MIN,
        warmup_steps=TRAIN_WARMUP_STEPS,
        resample_every=TRAIN_RESAMPLE_EVERY,
        log_every=25,
    )

    ablation_vol.commit()
    print(
        f"L{layer} H{head} {sae_type} s{seed}: "
        f"val_mse={result.get('best_mse', '?')}"
    )
    return {"layer": layer, "head": head, "sae_type": sae_type, "seed": seed, **result}


# Stage 2: ANALYZE

def _load_checkpoint_mse(ckpt_path: Path) -> float | None:
    """Load val_mse from a best.pt checkpoint."""
    import torch
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("val_mse", ckpt.get("best_val_mse"))


@app.function(
    volumes={ABLATION: ablation_vol},
    image=image,
    timeout=1800,
    memory=16384,
)
def analyze_results() -> dict[str, Any]:
    """Load all checkpoints and compute layer-average MSE per SAE type."""
    from scipy import stats

    # Collect per-(layer, head, type, seed) MSE values.
    all_mses: dict[int, dict[str, dict[int, list[float]]]] = {}
    # all_mses[layer][sae_type][head] = [mse_s0, mse_s1, mse_s42]
    missing: list[str] = []

    for layer in LAYERS:
        all_mses[layer] = {}
        for sae_type in ALL_SAE_TYPES:
            all_mses[layer][sae_type] = {}
            for head in range(N_HEADS):
                seed_vals = []
                for seed in SEEDS:
                    path = Path(ABLATION) / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}" / "best.pt"
                    val = _load_checkpoint_mse(path)
                    if val is not None:
                        seed_vals.append(float(val))
                    else:
                        missing.append(f"L{layer} H{head} {sae_type} s{seed}")
                all_mses[layer][sae_type][head] = seed_vals

    if missing:
        sample = ", ".join(missing[:15])
        extra = f", ... (+{len(missing) - 15} more)" if len(missing) > 15 else ""
        raise RuntimeError(f"Missing {len(missing)} checkpoints: {sample}{extra}")

    # -----------------------------------------------------------------------
    # Compute per-head mean (across seeds), then layer-average (across heads)
    # -----------------------------------------------------------------------
    # per_head_mean[layer][sae_type][head] = mean over seeds
    per_head_mean: dict[int, dict[str, dict[int, float]]] = {}
    # layer_avg[layer][sae_type] = mean over heads of per_head_mean
    layer_avg: dict[int, dict[str, float]] = {}
    layer_std: dict[int, dict[str, float]] = {}

    for layer in LAYERS:
        per_head_mean[layer] = {}
        layer_avg[layer] = {}
        layer_std[layer] = {}
        for sae_type in ALL_SAE_TYPES:
            head_means = []
            per_head_mean[layer][sae_type] = {}
            for head in range(N_HEADS):
                m = float(np.mean(all_mses[layer][sae_type][head]))
                per_head_mean[layer][sae_type][head] = m
                head_means.append(m)
            layer_avg[layer][sae_type] = float(np.mean(head_means))
            layer_std[layer][sae_type] = float(np.std(head_means))

    # -----------------------------------------------------------------------
    # Table 1: Layer-average MSE per type
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("TABLE 1: LAYER-AVERAGE MSE PER SAE TYPE")
    print(f"{'=' * 100}")
    hdr = f"{'Layer':>8}  {'Spectrum':>22}"
    for t in ALL_SAE_TYPES:
        hdr += f"  {t:>14}"
    print(hdr)
    print("-" * 100)

    for layer in LAYERS:
        row = f"{'L' + str(layer):>8}  {SPECTRAL_LABELS[layer]:>22}"
        for t in ALL_SAE_TYPES:
            row += f"  {layer_avg[layer][t]:.4e}"
        print(row)

    # -----------------------------------------------------------------------
    # Table 2: % advantage vs flat per type (positive = better)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("TABLE 2: % ADVANTAGE VS FLAT (positive = lower MSE than flat)")
    print(f"{'=' * 100}")
    hdr = f"{'Layer':>8}  {'Spectrum':>22}"
    for t in ALL_SAE_TYPES:
        if t == "flat":
            continue
        hdr += f"  {t:>14}"
    print(hdr)
    print("-" * 100)

    for layer in LAYERS:
        flat_val = layer_avg[layer]["flat"]
        row = f"{'L' + str(layer):>8}  {SPECTRAL_LABELS[layer]:>22}"
        for t in ALL_SAE_TYPES:
            if t == "flat":
                continue
            adv = (flat_val - layer_avg[layer][t]) / flat_val * 100.0
            row += f"  {adv:>+13.1f}%"
        print(row)

    # -----------------------------------------------------------------------
    # Key hypothesis tests (paired t-test across 16 heads)
    # -----------------------------------------------------------------------
    def _paired_test(layer: int, type_a: str, type_b: str) -> tuple[float, float, str]:
        """Paired t-test: type_a vs type_b across 16 heads.
        Returns (pct_advantage_of_a_over_b, p_value, significance_label).
        Positive pct means type_a has lower MSE."""
        a_vals = [per_head_mean[layer][type_a][h] for h in range(N_HEADS)]
        b_vals = [per_head_mean[layer][type_b][h] for h in range(N_HEADS)]
        diffs = [b - a for a, b in zip(a_vals, b_vals)]
        mean_b = float(np.mean(b_vals))
        pct = float(np.mean(diffs)) / mean_b * 100.0 if abs(mean_b) > 1e-12 else 0.0
        _, p_val = stats.ttest_rel(b_vals, a_vals)
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        return pct, float(p_val), sig

    comparisons = [
        ("bilinear_flat", "flat",          "Encoder effect: bilinear_flat vs flat (same decoder, different encoder)"),
        ("bilinear",      "bilinear_flat", "Decoder effect: bilinear vs bilinear_flat (same encoder, different decoder)"),
        ("bilinear",      "flat",          "Full bilinear vs flat"),
        ("rank1",         "flat",          "rank1 vs flat (decoder-only change)"),
    ]

    print(f"\n{'=' * 100}")
    print("KEY HYPOTHESIS TESTS (paired t-test across 16 heads)")
    print("Does bilinear_flat beat flat on low-sv1/sv2 layers?")
    print(f"{'=' * 100}")

    for type_a, type_b, desc in comparisons:
        print(f"\n  {desc}:")
        for layer in LAYERS:
            pct, p_val, sig = _paired_test(layer, type_a, type_b)
            winner = type_a if pct > 0 else type_b
            print(f"    L{layer:>2} ({SPECTRAL_LABELS[layer]:>22}): "
                  f"{pct:>+6.2f}% (p={p_val:.4f} {sig})  [{winner} wins]")

    # -----------------------------------------------------------------------
    # Per-head breakdown for L9 (the paper's main layer)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("PER-HEAD BREAKDOWN: L9 (paper's main layer)")
    print(f"{'=' * 100}")
    hdr = f"{'Head':>6}"
    for t in ALL_SAE_TYPES:
        hdr += f"  {t:>14}"
    hdr += f"  {'bf_vs_flat':>10}"
    print(hdr)
    print("-" * 100)

    for head in range(N_HEADS):
        row = f"{'H' + str(head):>6}"
        for t in ALL_SAE_TYPES:
            row += f"  {per_head_mean[9][t][head]:.4e}"
        flat_h = per_head_mean[9]["flat"][head]
        bf_h = per_head_mean[9]["bilinear_flat"][head]
        adv = (flat_h - bf_h) / flat_h * 100.0 if abs(flat_h) > 1e-12 else 0.0
        row += f"  {adv:>+9.1f}%"
        print(row)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    per_layer_results = []
    for layer in LAYERS:
        flat_val = layer_avg[layer]["flat"]
        type_results = {}
        for t in ALL_SAE_TYPES:
            adv = (flat_val - layer_avg[layer][t]) / flat_val * 100.0 if abs(flat_val) > 1e-12 else 0.0
            type_results[t] = {
                "layer_avg_mse": layer_avg[layer][t],
                "layer_std_mse": layer_std[layer][t],
                "advantage_vs_flat_pct": adv,
                "per_head_mean_mse": {str(h): per_head_mean[layer][t][h] for h in range(N_HEADS)},
            }
        per_layer_results.append({
            "layer": layer,
            "spectral_label": SPECTRAL_LABELS[layer],
            "mse_by_type": type_results,
        })

    hypothesis_tests = []
    for type_a, type_b, desc in comparisons:
        test_results = {}
        for layer in LAYERS:
            pct, p_val, sig = _paired_test(layer, type_a, type_b)
            test_results[f"L{layer}"] = {
                "pct_advantage": pct,
                "p_value": p_val,
                "significance": sig,
            }
        hypothesis_tests.append({
            "comparison": desc,
            "type_a": type_a,
            "type_b": type_b,
            "results": test_results,
        })

    output = {
        "experiment": "layer_encoder_swap_ablation",
        "hypothesis": (
            "bilinear encoder drives reconstruction gains on diffuse-spectrum layers; "
            "rank-1 decoder is secondary"
        ),
        "layers": LAYERS,
        "n_heads": N_HEADS,
        "sae_types": ALL_SAE_TYPES,
        "seeds": SEEDS,
        "spectral_labels": {str(k): v for k, v in SPECTRAL_LABELS.items()},
        "train_config": {
            "n_features": SAE_N_FEATURES,
            "k": SAE_K,
            "epochs": TRAIN_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "lr": TRAIN_LR,
            "lr_min": TRAIN_LR_MIN,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "resample_every": TRAIN_RESAMPLE_EVERY,
        },
        "per_layer": per_layer_results,
        "hypothesis_tests": hypothesis_tests,
        "code_sha": CODE_SHA,
    }

    # Save to volume.
    results_path = Path(ABLATION) / "layer_encoder_swap_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    ablation_vol.commit()

    return output


# Local entrypoint

@app.local_entrypoint()
def main(stage: str = "all", layer: int = -1):
    target_layers = [layer] if layer >= 0 else LAYERS

    if stage in ("train", "all"):
        n_jobs = len(target_layers) * N_HEADS * len(ALL_SAE_TYPES) * len(SEEDS)
        print(f"=== Training layer-level encoder-swap SAEs ===")
        print(f"{len(target_layers)} layers x {N_HEADS} heads x {len(ALL_SAE_TYPES)} types x {len(SEEDS)} seeds = {n_jobs} jobs")

        jobs: list[dict[str, Any]] = []
        for ly in target_layers:
            for head in range(N_HEADS):
                for sae_type in ALL_SAE_TYPES:
                    for seed in SEEDS:
                        jobs.append({
                            "layer": ly,
                            "head": head,
                            "sae_type": sae_type,
                            "seed": seed,
                            "call": train_sae.spawn(ly, head, sae_type, seed),
                        })
                        if len(jobs) % 100 == 0:
                            print(f"Spawned {len(jobs)}/{n_jobs} jobs")

        print(f"Spawned {len(jobs)}/{n_jobs} jobs")

        results = []
        failures = []
        for idx, job in enumerate(jobs, start=1):
            try:
                result = job["call"].get()
                results.append(result)
                if idx % 50 == 0 or idx == len(jobs):
                    print(f"  [{idx}/{n_jobs}] completed ({len(failures)} failures so far)")
            except Exception as exc:
                failures.append({
                    "layer": job["layer"],
                    "head": job["head"],
                    "sae_type": job["sae_type"],
                    "seed": job["seed"],
                    "error": str(exc),
                })
                print(
                    f"  [{idx}/{n_jobs}] FAILED: L{job['layer']} H{job['head']} "
                    f"{job['sae_type']} s{job['seed']}: {exc}"
                )

        n_ok = len(results)
        print(f"\nTraining complete: {n_ok}/{n_jobs} succeeded, {len(failures)} failed")
        if failures:
            local_out = Path(__file__).resolve().parent / "results" / "data"
            local_out.mkdir(parents=True, exist_ok=True)
            failures_path = local_out / "layer_encoder_swap_training_failures.json"
            with open(failures_path, "w") as f:
                json.dump({"failures": failures, "code_sha": CODE_SHA}, f, indent=2)
            print("Failed jobs:")
            for fail in failures[:20]:
                print(f"  L{fail['layer']} H{fail['head']} {fail['sae_type']} s{fail['seed']}: {fail['error']}")
            if len(failures) > 20:
                print(f"  ... and {len(failures) - 20} more")
            print(f"Saved failure summary to {failures_path}")

    if stage in ("analyze", "all"):
        print("\n=== Analyzing layer-level encoder-swap results ===")
        output = analyze_results.remote()

        # Save results locally.
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_path = os.path.join(local_out, "layer_encoder_swap_results.json")
        with open(local_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved local results to {local_path}")

#!/usr/bin/env python3
"""Alive-feature overlap analysis: flat vs bilinear SAEs across 16 heads.

Tests the hypothesis that flat SAEs have more overlapping alive features
across heads than bilinear SAEs. If flat Jaccard >> bilinear Jaccard,
the same atoms are alive everywhere, creating correlated null spaces.

Phase 1: Extract alive indices from Modal volume (flat, rank1) and local
         checkpoints (bilinear).
Phase 2: Compute pairwise Jaccard similarity across all 120 head pairs.

Usage:
    # Extract from Modal + compute locally
    modal run experiments/analysis/alive_overlap.py

    # If data already extracted (results/data/alive_overlap_indices.json exists)
    python experiments/analysis/alive_overlap.py --local-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

# Config
LAYER = 9
N_HEADS = 16
SAE_TYPES = ["flat", "bilinear", "rank1"]
SEED = 42
DEAD_THRESHOLD = 100  # steps_since_active >= 100 => dead
N_FEATURES = 2048

LOCAL_CKPT_DIR = Path("release/hf_model_qwen3_5_0_8b_l9/staging/"
                      "matrix-sae-qwen3_5-0_8b-l9/checkpoints")
OUTPUT_PATH = Path("results/data/alive_overlap_indices.json")

# Modal setup (minimal image - just need torch for checkpoint loading)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy")
)

app = modal.App("alive-overlap-analysis")
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")
ABLATION = "/ablation"


@app.function(
    volumes={ABLATION: ablation_vol},
    image=image,
    timeout=300,
    memory=4096,
)
def extract_alive_indices_modal(
    layer: int,
    sae_types: list[str],
    n_heads: int,
    seed: int,
    dead_threshold: int,
) -> dict[str, dict[int, list[int]]]:
    """Load checkpoints from Modal volume, return {sae_type: {head: [alive_indices]}}."""
    import torch

    results: dict[str, dict[int, list[int]]] = {}
    for sae_type in sae_types:
        results[sae_type] = {}
        for h in range(n_heads):
            ckpt_path = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type}_s{seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path}")
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"]
            ssa = sd["steps_since_active"]
            alive_mask = ssa < dead_threshold
            alive_idx = torch.where(alive_mask)[0].tolist()
            results[sae_type][h] = alive_idx
            print(f"  {sae_type} L{layer} H{h}: {len(alive_idx)}/{ssa.shape[0]} alive")
    return results


def extract_alive_indices_local(
    layer: int,
    sae_type: str,
    n_heads: int,
    seed: int,
    dead_threshold: int,
) -> dict[int, list[int]]:
    """Load checkpoints from local release dir."""
    import torch

    results: dict[int, list[int]] = {}
    for h in range(n_heads):
        tag = f"{sae_type}_L{layer}_H{h}_nf{N_FEATURES}_k32_s{seed}"
        ckpt_path = LOCAL_CKPT_DIR / tag / "best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        ssa = sd["steps_since_active"]
        alive_mask = ssa < dead_threshold
        alive_idx = torch.where(alive_mask)[0].tolist()
        results[h] = alive_idx
        print(f"  {sae_type} L{layer} H{h}: {len(alive_idx)}/{ssa.shape[0]} alive (local)")
    return results


def compute_jaccard_matrix(
    alive_sets: dict[int, set[int]],
    n_heads: int,
) -> dict[str, float | list[list[float]]]:
    """Compute pairwise Jaccard similarity matrix for alive feature sets."""
    heads = sorted(alive_sets.keys())
    n = len(heads)
    matrix = [[0.0] * n for _ in range(n)]
    pairwise_values = []

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            a = alive_sets[heads[i]]
            b = alive_sets[heads[j]]
            intersection = len(a & b)
            union = len(a | b)
            jaccard = intersection / union if union > 0 else 0.0
            matrix[i][j] = jaccard
            matrix[j][i] = jaccard
            pairwise_values.append(jaccard)

    import numpy as np
    arr = np.array(pairwise_values)
    return {
        "matrix": matrix,
        "heads": heads,
        "mean_jaccard": float(arr.mean()) if len(arr) > 0 else 0.0,
        "std_jaccard": float(arr.std()) if len(arr) > 0 else 0.0,
        "min_jaccard": float(arr.min()) if len(arr) > 0 else 0.0,
        "max_jaccard": float(arr.max()) if len(arr) > 0 else 0.0,
        "n_pairs": len(pairwise_values),
    }


def analyze(data: dict[str, dict[str, list[int]]]) -> dict:
    """Compute overlap statistics from alive index data."""
    import numpy as np

    results = {}
    for sae_type in SAE_TYPES:
        if sae_type not in data:
            continue
        head_data = data[sae_type]
        if not head_data:
            continue

        # Convert string keys back to int (JSON serialization)
        alive_sets = {int(h): set(indices) for h, indices in head_data.items()}
        alive_counts = {int(h): len(indices) for h, indices in head_data.items()}

        # Alive count stats
        counts = list(alive_counts.values())
        count_arr = np.array(counts)

        # Jaccard
        jaccard_result = compute_jaccard_matrix(alive_sets, N_HEADS)

        # Union and intersection across ALL heads
        all_sets = list(alive_sets.values())
        if all_sets:
            global_union = set.union(*all_sets)
            global_intersection = set.intersection(*all_sets) if len(all_sets) > 1 else all_sets[0]
        else:
            global_union = set()
            global_intersection = set()

        results[sae_type] = {
            "alive_counts": alive_counts,
            "mean_alive": float(count_arr.mean()),
            "std_alive": float(count_arr.std()),
            "min_alive": int(count_arr.min()),
            "max_alive": int(count_arr.max()),
            "global_union_size": len(global_union),
            "global_intersection_size": len(global_intersection),
            "jaccard": jaccard_result,
        }

    return results


def print_report(results: dict) -> None:
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("ALIVE FEATURE OVERLAP ANALYSIS: FLAT vs BILINEAR vs RANK1")
    print("=" * 70)

    for sae_type in SAE_TYPES:
        if sae_type not in results:
            continue
        r = results[sae_type]
        print(f"\n--- {sae_type.upper()} ---")
        print(f"  Alive per head:  mean={r['mean_alive']:.1f}  std={r['std_alive']:.1f}  "
              f"range=[{r['min_alive']}, {r['max_alive']}]")
        print(f"  Dead per head:   mean={N_FEATURES - r['mean_alive']:.1f}")
        print(f"  Global union:    {r['global_union_size']} / {N_FEATURES}")
        print(f"  Global intersect:{r['global_intersection_size']} / {N_FEATURES}")

        j = r["jaccard"]
        print(f"  Pairwise Jaccard: mean={j['mean_jaccard']:.4f}  std={j['std_jaccard']:.4f}  "
              f"range=[{j['min_jaccard']:.4f}, {j['max_jaccard']:.4f}]  (n={j['n_pairs']} pairs)")

    # Comparison
    if "flat" in results and "bilinear" in results:
        flat_j = results["flat"]["jaccard"]["mean_jaccard"]
        bi_j = results["bilinear"]["jaccard"]["mean_jaccard"]
        print(f"\n{'=' * 70}")
        print("HYPOTHESIS TEST: flat Jaccard >> bilinear Jaccard?")
        print(f"  flat mean Jaccard:     {flat_j:.4f}")
        print(f"  bilinear mean Jaccard: {bi_j:.4f}")
        ratio = flat_j / bi_j if bi_j > 0 else float("inf")
        print(f"  ratio (flat/bilinear): {ratio:.2f}x")
        if flat_j > bi_j * 1.5:
            print(f"  VERDICT: YES. Flat alive sets are {ratio:.1f}x more similar across heads.")
            print(f"  The same ~{results['flat']['mean_alive']:.0f} atoms activate everywhere,")
            print("  creating a shared null space that makes residuals correlate.")
        elif flat_j > bi_j:
            print(f"  VERDICT: WEAK. Flat has higher overlap ({ratio:.2f}x) but not dramatically.")
        else:
            print("  VERDICT: NO. Bilinear has equal or higher Jaccard overlap.")

    for sae_type in SAE_TYPES:
        if sae_type not in results:
            continue
        j = results[sae_type]["jaccard"]
        matrix = j["matrix"]
        heads = j["heads"]
        print(f"\n--- {sae_type.upper()} Jaccard matrix (heads 0-15) ---")
        # Header
        print("     " + " ".join(f"H{h:2d}" for h in heads))
        for i, h in enumerate(heads):
            row = " ".join(f"{matrix[i][j]:.2f}" for j in range(len(heads)))
            print(f"H{h:2d}  {row}")


@app.local_entrypoint()
def main():
    """Extract alive indices from Modal + local, then analyze."""
    # Check if we already have extracted data
    if OUTPUT_PATH.exists():
        print(f"Loading cached indices from {OUTPUT_PATH}")
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
    else:
        data = {}

        # Try local first for bilinear (all 16 heads available locally)
        print("Extracting bilinear alive indices from local checkpoints...")
        bilinear_local = extract_alive_indices_local(
            LAYER, "bilinear", N_HEADS, SEED, DEAD_THRESHOLD
        )
        if len(bilinear_local) == N_HEADS:
            data["bilinear"] = {str(h): indices for h, indices in bilinear_local.items()}
            print(f"  Got {len(bilinear_local)} bilinear heads locally")
            modal_types = ["flat", "rank1"]
        else:
            print(f"  Only {len(bilinear_local)} bilinear heads locally, fetching all from Modal")
            modal_types = SAE_TYPES

        # Fetch remaining from Modal
        print(f"Extracting {modal_types} alive indices from Modal volume...")
        modal_data = extract_alive_indices_modal.remote(
            layer=LAYER,
            sae_types=modal_types,
            n_heads=N_HEADS,
            seed=SEED,
            dead_threshold=DEAD_THRESHOLD,
        )
        for sae_type, head_data in modal_data.items():
            data[sae_type] = {str(h): indices for h, indices in head_data.items()}
            print(f"  Got {len(head_data)} {sae_type} heads from Modal")

        # Cache
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f)
        print(f"Saved indices to {OUTPUT_PATH}")

    # Analyze
    results = analyze(data)
    print_report(results)

    analysis_path = Path("results/data/alive_overlap_analysis.json")
    serializable = {}
    for sae_type, r in results.items():
        sr = dict(r)
        sr["alive_counts"] = {str(k): v for k, v in r["alive_counts"].items()}
        serializable[sae_type] = sr
    with open(analysis_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull analysis saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true",
                        help="Skip Modal, use cached indices only")
    args = parser.parse_args()

    if args.local_only:
        if not OUTPUT_PATH.exists():
            print(f"ERROR: {OUTPUT_PATH} not found. Run with Modal first.")
            sys.exit(1)
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        results = analyze(data)
        print_report(results)
    else:
        print("Use 'modal run' to execute. For cached data: python ... --local-only")

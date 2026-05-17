#!/usr/bin/env python3
"""Substitution test: per-firing 4-way forward KL (atom / ablate / random / native).

Canonical CLI entry point for the headline result of the matrix-sae paper:
"a single rank-1 SAE atom replaces the native cache write and beats matched-norm
ablation on 92.4% of n=4,851 firings at Qwen3.5-0.8B L9 H4."

For each natural firing of feature i at the chosen (layer, head), the script runs
four forward passes that differ only in the cache state at firing position t:

    native :  S_t  (unmodified)
    atom   :  S_t - beta_t * k_t v_t^T + ||beta_t * k_t v_t^T||_F * (v_i w_i^T) / ||v_i w_i^T||_F
    ablate :  S_t - beta_t * k_t v_t^T
    random :  S_t - beta_t * k_t v_t^T + matched-norm rank-1 from a non-firing atom

It records per-firing KL(p_native || p_cond) for cond in {atom, ablate, random}.
The canonical headline statistic is

    win_rate = mean[ KL_atom < KL_ablate ]   # over all firings, all features

which equals 92.4% (Wilson 95% CI [91.6, 93.1]) at Qwen3.5-0.8B L9 H4 on the
register firing pool (n=4,851 firings across L1/L9/L17 head 4).

Usage
-----

Single-feature, default layer/head:

    python scripts/clean_amplified_kl.py \
        --feature 412 \
        --sae-checkpoint <path-to>/writesae_L9_H4_nf2048_k32_s42/best.pt \
        --states-dir <path-to>/states/Qwen3.5-0.8B/L9 \
        --out out/clean_amplified_kl_F412.json

Layer sweep over the flagship register pool:

    python scripts/clean_amplified_kl.py \
        --layer 9 --head 4 --features 412 192 97 1361 53 63 87 1335 \
        --sae-checkpoint <path-to>/writesae_L9_H4_nf2048_k32_s42/best.pt \
        --states-dir <path-to>/states/Qwen3.5-0.8B/L9 \
        --out out/clean_amplified_kl_L9.json

The full firing pool that produces the 92.4% headline runs across L1, L9, L17 head 4
and takes ~10 minutes per cell on a single A10G with cached states. To reproduce
end-to-end (state extraction + SAE checkpoints + this test), see REPRODUCE.md
and `scripts/reproduce_headline.sh`.

Output JSON schema
------------------

    {
        "config": {"layer": int, "head": int, "feature_ids": [int], ...},
        "summary": {
            "n_firings": int,
            "win_rate_atom_vs_ablate": float,           # the 92.4% number
            "win_rate_atom_vs_ablate_ci95": [float, float],
            "median_kl_atom": float,
            "median_kl_ablate": float,
            "median_kl_random": float,
            "strict_chain_atom_lt_ablate_lt_random": float,
        },
        "records": [
            {
                "feature_id": int,
                "firing_position": int,
                "kl_atom": float,
                "kl_ablate": float,
                "kl_random": float,
            },
            ...
        ]
    }

Implementation
--------------

The 4-way forward-KL kernel lives at
`paper-9pager/src/flagship/akash_forward_kl_4way.py` (Akash-native batch runner).
This script is the importable, single-feature, single-cell CLI. Both produce the
same JSON schema and feed the same Wilson CI computation in `experiments/analysis/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path resolution: this script ships under both `scripts/` (top-level repo) and
# `release/anon-mirror/scripts/`. The substitution-test kernel lives under
# `paper-9pager/src/flagship/`. Resolve relative to either location.
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
_CANDIDATE_ROOTS = [
    _THIS.parents[1],                                    # scripts/ at repo root
    _THIS.parents[2],                                    # release/anon-mirror/scripts/
]
_REPO_ROOT = next((p for p in _CANDIDATE_ROOTS if (p / "core").exists()), _CANDIDATE_ROOTS[0])
sys.path.insert(0, str(_REPO_ROOT))
_FLAGSHIP_DIR = _REPO_ROOT / "paper-9pager" / "src" / "flagship"
if _FLAGSHIP_DIR.exists():
    sys.path.insert(0, str(_FLAGSHIP_DIR))


def kl_div(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Per-token KL(softmax(a) || softmax(b)) returned as a Python float.

    Used by the per-firing 4-way kernel; exposed at module level so downstream
    tests can import a single canonical implementation.
    """
    log_pa = F.log_softmax(logits_a.float(), dim=-1)
    log_pb = F.log_softmax(logits_b.float(), dim=-1)
    pa = log_pa.exp()
    return float((pa * (log_pa - log_pb)).sum(dim=-1).item())


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion (matches the paper's reporting)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def run_clean_amplified_kl(
    sae_checkpoint: Path,
    states_dir: Path,
    layer: int,
    head: int,
    feature_ids: list[int],
    n_firings_cap: int | None = None,
    seed: int = 2026,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Run the 4-way substitution test on the given features.

    Delegates the per-firing kernel to `flagship.akash_forward_kl_4way` for
    consistency with the published numbers. This wrapper is the single-cell
    CLI; the Akash batch runner handles multi-cell sweeps.
    """
    # Loaded via sys.path.insert at module top; flagship dir not a proper package.
    from akash_forward_kl_4way import run_per_firing_kl  # type: ignore[import-not-found]

    records = run_per_firing_kl(
        sae_checkpoint=sae_checkpoint,
        states_dir=states_dir,
        layer=layer,
        head=head,
        feature_ids=feature_ids,
        n_firings_cap=n_firings_cap,
        seed=seed,
        device=device,
    )

    n = len(records)
    if n == 0:
        return {
            "config": {
                "layer": layer, "head": head, "feature_ids": feature_ids,
                "sae_checkpoint": str(sae_checkpoint), "seed": seed,
            },
            "summary": {"n_firings": 0},
            "records": [],
        }

    kl_atom = np.array([r["kl_atom"] for r in records])
    kl_ablate = np.array([r["kl_ablate"] for r in records])
    kl_random = np.array([r["kl_random"] for r in records])

    wins_atom_vs_ablate = int((kl_atom < kl_ablate).sum())
    strict_chain = int(((kl_atom < kl_ablate) & (kl_ablate < kl_random)).sum())

    win_rate = wins_atom_vs_ablate / n
    ci_lo, ci_hi = _wilson_ci(wins_atom_vs_ablate, n)

    return {
        "config": {
            "layer": layer, "head": head, "feature_ids": feature_ids,
            "sae_checkpoint": str(sae_checkpoint), "seed": seed,
        },
        "summary": {
            "n_firings": n,
            "win_rate_atom_vs_ablate": win_rate,
            "win_rate_atom_vs_ablate_ci95": [ci_lo, ci_hi],
            "median_kl_atom": float(np.median(kl_atom)),
            "median_kl_ablate": float(np.median(kl_ablate)),
            "median_kl_random": float(np.median(kl_random)),
            "strict_chain_atom_lt_ablate_lt_random": strict_chain / n,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-firing 4-way forward KL substitution test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sae-checkpoint", type=Path, required=True,
                        help="Path to writesae_L{layer}_H{head}_*.pt checkpoint.")
    parser.add_argument("--states-dir", type=Path, required=True,
                        help="Directory of cached GDN state .npy shards for this layer.")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--features", type=int, nargs="+", default=[412],
                        help="Feature ids to test. Default: F412 (paper exemplar).")
    parser.add_argument("--n-firings-cap", type=int, default=None,
                        help="Cap firings per feature (default: all natural firings).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON path.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    result = run_clean_amplified_kl(
        sae_checkpoint=args.sae_checkpoint,
        states_dir=args.states_dir,
        layer=args.layer,
        head=args.head,
        feature_ids=args.features,
        n_firings_cap=args.n_firings_cap,
        seed=args.seed,
    )

    args.out.write_text(json.dumps(result, indent=2))

    s = result["summary"]
    if s.get("n_firings", 0) > 0:
        print(f"n_firings = {s['n_firings']}")
        print(f"win_rate (atom vs ablate) = {s['win_rate_atom_vs_ablate']:.4f} "
              f"(Wilson 95% CI [{s['win_rate_atom_vs_ablate_ci95'][0]:.4f}, "
              f"{s['win_rate_atom_vs_ablate_ci95'][1]:.4f}])")
        print(f"strict chain (atom < ablate < random) = "
              f"{s['strict_chain_atom_lt_ablate_lt_random']:.4f}")
        print(f"median KL: atom={s['median_kl_atom']:.4f}, "
              f"ablate={s['median_kl_ablate']:.4f}, random={s['median_kl_random']:.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

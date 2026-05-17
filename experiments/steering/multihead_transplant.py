#!/usr/bin/env python3
"""Multi-head recurrent state transplant experiment.

Tests whether transplanting the FULL layer-9 recurrent state (all 16 heads)
from a boundary context to a non-boundary context moves p('.') more than
single-head interventions. Then ablates heads to find the minimal subset
that carries the boundary signal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _get_period_token_id(tokenizer) -> int:
    """Get the token ID for '.' in the tokenizer."""
    ids = tokenizer.encode(".", add_special_tokens=False)
    return ids[-1]


def _load_benchmark_rows(benchmark_path: str | Path) -> list[dict]:
    """Load benchmark rows and filter for '.' target token."""
    data = json.loads(Path(benchmark_path).read_text())
    rows = data.get("accepted_rows", [])
    dot_rows = [r for r in rows if r.get("readout_text") == "."]
    return dot_rows


def _forward_and_extract_states(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    up_to_pos: int,
    device: str = "cuda",
) -> tuple[Any, torch.Tensor]:
    """Forward through input_ids token-by-token up to up_to_pos, return (cache, logits_at_pos).

    Returns the cache after processing tokens [0..up_to_pos] and the logits
    produced after processing token at up_to_pos.
    """
    cache = None
    for pos in range(up_to_pos + 1):
        step_ids = input_ids[:, pos:pos + 1].to(device)
        if cache is None:
            out = model(input_ids=step_ids, use_cache=True)
        else:
            out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    logits = out.logits[:, -1, :].float().detach().cpu()
    return cache, logits


@torch.no_grad()
def run_multihead_transplant(
    model,
    tokenizer,
    layer_idx: int,
    rows: list[dict],
    corpus_sequences: list[torch.Tensor],
    heads_to_transplant: list[int] | None = None,
    device: str = "cuda",
) -> list[dict]:
    """Run transplant experiment for given rows.

    For each row:
    1. Forward donor sequence, extract recurrent states at layer_idx for all heads at donor_position
    2. Forward recipient sequence token-by-token with use_cache=True
    3. At use_position, patch cache.layers[layer_idx].recurrent_states[0, heads, :, :]
    4. Get next-token logits, compute p('.')
    5. Also compute baseline (no patch) p('.')

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        layer_idx: Which GDN layer to transplant states from.
        rows: Benchmark rows (each has donor_prompt_index, recipient_prompt_index, etc.)
        corpus_sequences: List of token tensors, indexed by prompt_index.
        heads_to_transplant: Which heads to transplant. None = all 16.
        device: CUDA device.

    Returns:
        List of per-row result dicts.
    """
    period_id = _get_period_token_id(tokenizer)
    n_heads = 16  # Qwen3.5-0.8B has 16 heads per GDN layer

    if heads_to_transplant is None:
        heads_to_transplant = list(range(n_heads))

    results = []

    for row_i, row in enumerate(rows):
        donor_idx = row["donor_prompt_index"]
        recipient_idx = row["recipient_prompt_index"]

        if donor_idx >= len(corpus_sequences) or recipient_idx >= len(corpus_sequences):
            continue

        donor_seq = corpus_sequences[donor_idx].to(device)
        recipient_seq = corpus_sequences[recipient_idx].to(device)

        # decision_pos is where the model predicts '.' as the next token.
        # write_pos is where the donor writes the boundary signal.
        # We patch at decision_pos - 1 and measure logits at decision_pos.
        write_pos = row["write_pos"]
        decision_pos = row["decision_pos"]

        if write_pos >= donor_seq.shape[1] or decision_pos >= recipient_seq.shape[1]:
            continue

        # 1. Forward donor to write_pos, extract all heads' recurrent states
        donor_cache, _ = _forward_and_extract_states(
            model, donor_seq, layer_idx, write_pos, device,
        )
        donor_states = donor_cache.layers[layer_idx].recurrent_states.clone()
        # shape: (1, n_heads, d_k, d_v)
        del donor_cache
        torch.cuda.empty_cache()

        # 2. Baseline: forward recipient to decision_pos, get p('.') without patching
        # Logits at decision_pos predict the token at decision_pos+1 (which is '.')
        baseline_cache, baseline_logits = _forward_and_extract_states(
            model, recipient_seq, layer_idx, decision_pos, device,
        )
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        baseline_p_dot = float(baseline_probs[0, period_id].item())
        del baseline_cache
        torch.cuda.empty_cache()

        # 3. Transplant: forward recipient to decision_pos - 1, patch, run one more step
        patch_pos = decision_pos - 1
        if patch_pos < 0:
            continue
        transplant_cache, _ = _forward_and_extract_states(
            model, recipient_seq, layer_idx, patch_pos, device,
        )

        # Patch selected heads
        layer_cache = transplant_cache.layers[layer_idx]
        original_dtype = layer_cache.recurrent_states.dtype
        for h in heads_to_transplant:
            layer_cache.recurrent_states[0, h] = donor_states[0, h].to(original_dtype)

        # Run one more step: feed token at decision_pos, get logits predicting decision_pos+1
        step_ids = recipient_seq[:, decision_pos:decision_pos + 1]
        out = model(input_ids=step_ids, past_key_values=transplant_cache, use_cache=True)
        transplant_cache = out.past_key_values

        transplant_logits = out.logits[:, -1, :].float().detach().cpu()
        transplant_probs = torch.softmax(transplant_logits, dim=-1)
        transplant_p_dot = float(transplant_probs[0, period_id].item())
        del transplant_cache
        torch.cuda.empty_cache()

        delta = transplant_p_dot - baseline_p_dot

        result = {
            "row_index": row_i,
            "donor_prompt_index": donor_idx,
            "recipient_prompt_index": recipient_idx,
            "write_pos": write_pos,
            "decision_pos": decision_pos,
            "patch_pos": patch_pos,
            "heads_transplanted": heads_to_transplant,
            "baseline_p_dot": baseline_p_dot,
            "transplant_p_dot": transplant_p_dot,
            "delta_p_dot": delta,
        }
        results.append(result)
        print(
            f"  Row {row_i}: baseline={baseline_p_dot:.4f} "
            f"transplant={transplant_p_dot:.4f} delta={delta:+.4f} "
            f"heads={heads_to_transplant}",
            flush=True,
        )

    return results


def run_full_experiment(
    model,
    tokenizer,
    layer_idx: int,
    benchmark_path: str | Path,
    corpus_sequences: list[torch.Tensor],
    device: str = "cuda",
    max_rows: int = 32,
) -> dict[str, Any]:
    """Run all three phases of the multi-head transplant experiment.

    Phase 1: All-heads transplant
    Phase 2: Leave-one-out head ablation
    Phase 3: Minimal head set (top-4 by contribution)
    """
    rows = _load_benchmark_rows(benchmark_path)
    if not rows:
        raise ValueError(f"No '.' rows found in {benchmark_path}")

    rows = rows[:max_rows]
    n_heads = 16
    print(f"[multihead-transplant] {len(rows)} dot-rows, layer={layer_idx}", flush=True)

    # ---- Phase 1: All-heads transplant ----
    print("\n=== Phase 1: All-heads transplant ===", flush=True)
    t0 = time.perf_counter()
    phase1_results = run_multihead_transplant(
        model, tokenizer, layer_idx, rows, corpus_sequences,
        heads_to_transplant=None,
        device=device,
    )
    phase1_time = time.perf_counter() - t0

    if not phase1_results:
        raise ValueError("No valid rows produced results in Phase 1")

    phase1_deltas = [r["delta_p_dot"] for r in phase1_results]
    phase1_baselines = [r["baseline_p_dot"] for r in phase1_results]
    phase1_transplants = [r["transplant_p_dot"] for r in phase1_results]
    mean_all_heads_delta = float(np.mean(phase1_deltas))

    phase1_summary = {
        "rows": phase1_results,
        "mean_baseline_p": float(np.mean(phase1_baselines)),
        "mean_transplant_p": float(np.mean(phase1_transplants)),
        "mean_delta": mean_all_heads_delta,
        "median_delta": float(np.median(phase1_deltas)),
        "std_delta": float(np.std(phase1_deltas)),
        "n_positive": sum(1 for d in phase1_deltas if d > 0),
        "n_rows": len(phase1_results),
        "time_s": round(phase1_time, 1),
    }
    print(
        f"\nPhase 1 summary: mean_delta={mean_all_heads_delta:+.4f} "
        f"n_positive={phase1_summary['n_positive']}/{len(phase1_results)}",
        flush=True,
    )

    # ---- Phase 2: Leave-one-out head ablation ----
    print("\n=== Phase 2: Leave-one-out head ablation ===", flush=True)
    t0 = time.perf_counter()
    head_contributions: dict[str, dict] = {}

    for h in range(n_heads):
        heads_without_h = [i for i in range(n_heads) if i != h]
        print(f"\n  Leave out head {h}:", flush=True)
        loo_results = run_multihead_transplant(
            model, tokenizer, layer_idx, rows, corpus_sequences,
            heads_to_transplant=heads_without_h,
            device=device,
        )
        loo_deltas = [r["delta_p_dot"] for r in loo_results]
        mean_delta_without = float(np.mean(loo_deltas)) if loo_deltas else 0.0
        # Contribution = how much delta drops when this head is removed
        contribution = mean_all_heads_delta - mean_delta_without

        head_contributions[str(h)] = {
            "mean_delta_without": mean_delta_without,
            "contribution": contribution,
            "n_positive_without": sum(1 for d in loo_deltas if d > 0),
        }
        print(
            f"  Head {h}: delta_without={mean_delta_without:+.4f} "
            f"contribution={contribution:+.4f}",
            flush=True,
        )

    phase2_time = time.perf_counter() - t0

    # Rank heads by contribution (largest positive contribution = most important)
    sorted_heads = sorted(
        head_contributions.items(),
        key=lambda x: x[1]["contribution"],
        reverse=True,
    )
    top_heads = [int(h) for h, _ in sorted_heads[:4]]

    phase2_summary = {
        "head_contributions": head_contributions,
        "top_heads_by_contribution": top_heads,
        "all_heads_ranked": [int(h) for h, _ in sorted_heads],
        "time_s": round(phase2_time, 1),
    }
    print(
        f"\nPhase 2 summary: top-4 heads = {top_heads}",
        flush=True,
    )

    # ---- Phase 3: Minimal head set ----
    print(f"\n=== Phase 3: Minimal head set (heads {top_heads}) ===", flush=True)
    t0 = time.perf_counter()
    phase3_results = run_multihead_transplant(
        model, tokenizer, layer_idx, rows, corpus_sequences,
        heads_to_transplant=top_heads,
        device=device,
    )
    phase3_time = time.perf_counter() - t0

    phase3_deltas = [r["delta_p_dot"] for r in phase3_results]
    mean_minimal_delta = float(np.mean(phase3_deltas)) if phase3_deltas else 0.0
    fraction_of_full = (
        mean_minimal_delta / mean_all_heads_delta
        if abs(mean_all_heads_delta) > 1e-8 else 0.0
    )

    phase3_summary = {
        "heads_used": top_heads,
        "rows": phase3_results,
        "mean_delta": mean_minimal_delta,
        "median_delta": float(np.median(phase3_deltas)) if phase3_deltas else 0.0,
        "n_positive": sum(1 for d in phase3_deltas if d > 0),
        "n_rows": len(phase3_results),
        "fraction_of_full": fraction_of_full,
        "time_s": round(phase3_time, 1),
    }
    print(
        f"\nPhase 3 summary: mean_delta={mean_minimal_delta:+.4f} "
        f"fraction_of_full={fraction_of_full:.2f}",
        flush=True,
    )

    return {
        "experiment": "multihead_transplant",
        "layer": layer_idx,
        "n_heads": n_heads,
        "period_token_id": _get_period_token_id(tokenizer),
        "n_benchmark_rows": len(rows),
        "phase1_all_heads": phase1_summary,
        "phase2_leave_one_out": phase2_summary,
        "phase3_minimal_set": phase3_summary,
    }

#!/usr/bin/env python3
"""Hierarchical causal faithfulness experiment for bilinear SAE on GDN recurrent states.

Tests whether the bilinear SAE captures the causal content of L9H4 recurrent states
by comparing three transplant conditions:
  1. Full-state transplant (ceiling)
  2. SAE-reconstruction transplant (what SAE captures)
  3. Feature-difference transplant (what top-k features explain)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _get_period_token_ids(tokenizer) -> list[int]:
    """Get all token IDs that decode to '.'."""
    ids = tokenizer.encode(".", add_special_tokens=False)
    return ids


def _p_period(logits: torch.Tensor, period_ids: list[int]) -> float:
    """Compute p('.') from logits (1, vocab) by summing over all period token IDs."""
    probs = F.softmax(logits.float(), dim=-1)
    return float(probs[0, period_ids].sum().item())


@torch.no_grad()
def _run_sequence_to_position(
    model,
    input_ids: torch.Tensor,
    target_pos: int,
) -> Any:
    """Run model token-by-token to target_pos, return cache after that position.

    Returns (cache, logits_at_target_pos).
    logits_at_target_pos: (1, vocab_size) float tensor on CPU.
    """
    cache = None
    last_logits = None
    for pos in range(target_pos + 1):
        step_ids = input_ids[:, pos : pos + 1]
        if cache is None:
            out = model(input_ids=step_ids, use_cache=True)
        else:
            out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :].float().detach().cpu()
    return cache, last_logits


@torch.no_grad()
def _run_one_more_step(
    model,
    input_ids: torch.Tensor,
    pos: int,
    cache: Any,
) -> torch.Tensor:
    """Run one more token through the model at position pos using existing cache.

    Returns logits (1, vocab_size) on CPU.
    """
    step_ids = input_ids[:, pos : pos + 1]
    out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :].float().detach().cpu()


def _deep_copy_cache(cache):
    """Create a deep copy of the KV cache so we can patch one without affecting the other.

    HybridMambaAttentionDynamicCache stores layers as a list of layer caches.
    Each layer cache has key_states, value_states, and recurrent_states.
    We need to clone the recurrent_states tensors at minimum.
    """
    import copy
    return copy.deepcopy(cache)


@torch.no_grad()
def run_hierarchical_transplant(
    model,
    tokenizer,
    sae,
    layer_idx: int,
    head_idx: int,
    row_specs: list[dict[str, Any]],
    top_k: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the hierarchical transplant experiment across all row specs.

    Each row_spec must contain:
      - recipient_input_ids: (1, seq_len) tensor
      - recipient_decision_pos: int, position where '.' is the next token
      - donor_input_ids: (1, seq_len) tensor
      - donor_pos: int, position to extract donor state from
      - row_id: int
      - context_snippet: str
    """
    period_ids = _get_period_token_ids(tokenizer)
    print(f"Period token IDs: {period_ids}")
    print(f"Running {len(row_specs)} rows, layer={layer_idx}, head={head_idx}, top_k={top_k}")

    results = []

    for spec in row_specs:
        row_id = spec["row_id"]
        r_ids = spec["recipient_input_ids"].to(device)
        d_ids = spec["donor_input_ids"].to(device)
        decision_pos = spec["recipient_decision_pos"]
        donor_pos = spec["donor_pos"]

        print(f"\n--- Row {row_id} ---")
        print(f"  recipient decision_pos={decision_pos}, donor_pos={donor_pos}")

        # decision_pos is the position whose next-token prediction is '.'.
        # Run to decision_pos: the logits from this step predict token at decision_pos+1.
        # The recurrent state in the cache after this step is the state that
        # determines the next-token distribution (including p('.')).

        # Step 1: Run recipient to decision_pos
        r_cache, baseline_logits = _run_sequence_to_position(model, r_ids, decision_pos)

        # Extract recipient state at decision_pos
        r_state = r_cache.layers[layer_idx].recurrent_states[0, head_idx].clone()

        # Baseline p('.') comes directly from the logits at decision_pos
        baseline_p = _p_period(baseline_logits, period_ids)
        print(f"  baseline p('.')={baseline_p:.4f}")

        # Step 2: Run donor to donor_pos to get donor state
        d_cache, _ = _run_sequence_to_position(model, d_ids, donor_pos)
        d_state = d_cache.layers[layer_idx].recurrent_states[0, head_idx].clone()
        del d_cache
        torch.cuda.empty_cache()

        # For conditions 1-3, we need to patch the state BEFORE the model
        # processes the token at decision_pos. So we run to decision_pos - 1,
        # patch the state, then run one more step (feeding token at decision_pos)
        # to get the logits that predict the token at decision_pos+1 (the '.').
        r_cache_pre, _ = _run_sequence_to_position(model, r_ids, decision_pos - 1)

        # --- Condition 1: Full-state transplant ---
        c1_cache = _deep_copy_cache(r_cache_pre)
        c1_cache.layers[layer_idx].recurrent_states[0, head_idx] = d_state.to(
            c1_cache.layers[layer_idx].recurrent_states.dtype
        )
        full_logits = _run_one_more_step(model, r_ids, decision_pos, c1_cache)
        full_p = _p_period(full_logits, period_ids)
        del c1_cache
        print(f"  full transplant p('.')={full_p:.4f}")

        # --- Condition 2: SAE-reconstruction transplant ---
        d_state_f32 = d_state.float()
        d_coeffs = sae.encode(d_state_f32.unsqueeze(0))  # (1, n_features)
        d_recon = sae._decode(d_coeffs).squeeze(0)  # (128, 128)

        c2_cache = _deep_copy_cache(r_cache_pre)
        c2_cache.layers[layer_idx].recurrent_states[0, head_idx] = d_recon.to(
            c2_cache.layers[layer_idx].recurrent_states.dtype
        ).to(device)
        sae_logits = _run_one_more_step(model, r_ids, decision_pos, c2_cache)
        sae_p = _p_period(sae_logits, period_ids)
        del c2_cache
        print(f"  SAE recon transplant p('.')={sae_p:.4f}")

        # --- Condition 3: Feature-difference transplant ---
        r_state_f32 = r_state.float()
        r_coeffs = sae.encode(r_state_f32.unsqueeze(0))  # (1, n_features)

        diff = (d_coeffs - r_coeffs).abs().squeeze(0)  # (n_features,)
        _, topk_indices = torch.topk(diff, k=min(top_k, diff.shape[0]))
        topk_indices_list = topk_indices.tolist()
        topk_diffs_list = [float(diff[i].item()) for i in topk_indices_list]

        hybrid_coeffs = r_coeffs.clone()
        hybrid_coeffs[0, topk_indices] = d_coeffs[0, topk_indices]
        hybrid_recon = sae._decode(hybrid_coeffs).squeeze(0)  # (128, 128)

        c3_cache = _deep_copy_cache(r_cache_pre)
        c3_cache.layers[layer_idx].recurrent_states[0, head_idx] = hybrid_recon.to(
            c3_cache.layers[layer_idx].recurrent_states.dtype
        ).to(device)
        feat_logits = _run_one_more_step(model, r_ids, decision_pos, c3_cache)
        feat_p = _p_period(feat_logits, period_ids)
        del c3_cache
        del r_cache_pre
        print(f"  feature-diff transplant p('.')={feat_p:.4f}")

        full_delta = full_p - baseline_p
        sae_delta = sae_p - baseline_p
        feat_delta = feat_p - baseline_p

        sae_faithfulness = (sae_delta / full_delta) if abs(full_delta) > 1e-6 else float("nan")
        feat_ratio = (feat_delta / full_delta) if abs(full_delta) > 1e-6 else float("nan")

        row_result = {
            "row_id": row_id,
            "context_snippet": spec["context_snippet"],
            "use_position": decision_pos,
            "baseline_p_period": round(baseline_p, 6),
            "full_transplant_p_period": round(full_p, 6),
            "sae_recon_transplant_p_period": round(sae_p, 6),
            "feature_diff_transplant_p_period": round(feat_p, 6),
            "full_transplant_delta": round(full_delta, 6),
            "sae_recon_delta": round(sae_delta, 6),
            "feature_diff_delta": round(feat_delta, 6),
            "sae_causal_faithfulness": round(sae_faithfulness, 4) if sae_faithfulness == sae_faithfulness else None,
            "feature_decomposition_ratio": round(feat_ratio, 4) if feat_ratio == feat_ratio else None,
            "top_k_feature_indices": topk_indices_list,
            "top_k_feature_diffs": [round(d, 4) for d in topk_diffs_list],
        }
        results.append(row_result)

        # Clean up
        del r_cache
        torch.cuda.empty_cache()

        faith_str = f"  faith={sae_faithfulness:.3f}" if sae_faithfulness == sae_faithfulness else ""
        print(f"  deltas: full={full_delta:+.4f} sae={sae_delta:+.4f} feat={feat_delta:+.4f}{faith_str}")

    # Summary statistics
    full_deltas = [r["full_transplant_delta"] for r in results]
    sae_deltas = [r["sae_recon_delta"] for r in results]
    feat_deltas = [r["feature_diff_delta"] for r in results]
    faithfulness_vals = [r["sae_causal_faithfulness"] for r in results if r["sae_causal_faithfulness"] is not None]
    feat_ratios = [r["feature_decomposition_ratio"] for r in results if r["feature_decomposition_ratio"] is not None]

    summary = {
        "mean_full_delta": round(sum(full_deltas) / len(full_deltas), 6) if full_deltas else 0.0,
        "mean_sae_delta": round(sum(sae_deltas) / len(sae_deltas), 6) if sae_deltas else 0.0,
        "mean_feature_delta": round(sum(feat_deltas) / len(feat_deltas), 6) if feat_deltas else 0.0,
        "mean_sae_faithfulness": round(sum(faithfulness_vals) / len(faithfulness_vals), 4) if faithfulness_vals else None,
        "mean_feature_ratio": round(sum(feat_ratios) / len(feat_ratios), 4) if feat_ratios else None,
        "n_rows_full_positive": sum(1 for d in full_deltas if d > 0),
        "n_rows_sae_positive": sum(1 for d in sae_deltas if d > 0),
    }

    return {"rows": results, "summary": summary}


def print_results_table(result: dict[str, Any]) -> None:
    """Print a clean ASCII table of results."""
    rows = result["rows"]
    summary = result["summary"]

    print("\n" + "=" * 78)
    print("Hierarchical Causal Faithfulness: L9H4 Bilinear SAE")
    print("=" * 78)
    header = f"{'Row':>4} | {'Baseline':>8} | {'Full':>8} | {'SAE':>8} | {'Feat-k':>8} | {'Faith':>6} | {'Feat-R':>6}"
    print(header)
    print("-" * len(header))

    for r in rows:
        faith = f"{r['sae_causal_faithfulness']:.2f}" if r["sae_causal_faithfulness"] is not None else "  N/A"
        feat_r = f"{r['feature_decomposition_ratio']:.2f}" if r["feature_decomposition_ratio"] is not None else "  N/A"
        print(
            f"{r['row_id']:>4} | {r['baseline_p_period']:>8.4f} | "
            f"{r['full_transplant_p_period']:>8.4f} | "
            f"{r['sae_recon_transplant_p_period']:>8.4f} | "
            f"{r['feature_diff_transplant_p_period']:>8.4f} | "
            f"{faith:>6} | {feat_r:>6}"
        )

    print("-" * len(header))
    mean_faith = f"{summary['mean_sae_faithfulness']:.2f}" if summary["mean_sae_faithfulness"] is not None else "  N/A"
    mean_feat = f"{summary['mean_feature_ratio']:.2f}" if summary["mean_feature_ratio"] is not None else "  N/A"
    print(
        f"{'AVG':>4} | {'':>8} | "
        f"{summary['mean_full_delta']:>+8.4f} | "
        f"{summary['mean_sae_delta']:>+8.4f} | "
        f"{summary['mean_feature_delta']:>+8.4f} | "
        f"{mean_faith:>6} | {mean_feat:>6}"
    )
    print(f"\nFull positive: {summary['n_rows_full_positive']}/{len(rows)}")
    print(f"SAE positive:  {summary['n_rows_sae_positive']}/{len(rows)}")
    print()


def build_row_specs_from_benchmark(
    benchmark_path: str,
    corpus_sequences: list[torch.Tensor],
    tokenizer,
    baseline_p_range: tuple[float, float] = (0.3, 0.7),
) -> list[dict[str, Any]]:
    """Select qualifying rows from the factor_trace_benchmark data.

    Selects rows where:
      - actual_next_token is '.' (period, token_id 13 for Qwen)
      - baseline_actual_token_prob is in baseline_p_range
    """
    data = json.loads(Path(benchmark_path).read_text())
    accepted = data.get("accepted_rows", [])

    period_ids = set(_get_period_token_ids(tokenizer))

    specs = []
    for i, row in enumerate(accepted):
        token_id = row.get("actual_next_token")
        if token_id not in period_ids:
            continue
        bp = row.get("baseline_actual_token_prob", 0.0)
        if not (baseline_p_range[0] <= bp <= baseline_p_range[1]):
            continue

        recipient_idx = row["recipient_prompt_index"]
        donor_idx = row["donor_prompt_index"]
        decision_pos = row["decision_pos"]

        if recipient_idx >= len(corpus_sequences) or donor_idx >= len(corpus_sequences):
            print(f"  Skipping row {i}: prompt index out of range")
            continue

        r_ids = corpus_sequences[recipient_idx]
        d_ids = corpus_sequences[donor_idx]

        if r_ids.shape[-1] <= decision_pos + 1:
            print(f"  Skipping row {i}: recipient sequence too short for decision_pos={decision_pos}")
            continue
        if d_ids.shape[-1] <= decision_pos:
            print(f"  Skipping row {i}: donor sequence too short for donor_pos={decision_pos}")
            continue

        # Use the same decision_pos for donor state extraction.
        # The donor was selected for having a strong boundary signal, so its
        # recurrent state at this position will carry that information.
        donor_pos = min(decision_pos, int(d_ids.shape[-1]) - 1)

        # Context snippet from recipient
        snippet_ids = r_ids[0, max(0, decision_pos - 25) : decision_pos + 1].tolist()
        snippet = tokenizer.decode(snippet_ids, skip_special_tokens=False)[:50]

        specs.append({
            "row_id": i,
            "recipient_input_ids": r_ids,
            "donor_input_ids": d_ids,
            "recipient_decision_pos": decision_pos,
            "donor_pos": donor_pos,
            "context_snippet": snippet,
            "benchmark_baseline_p": bp,
        })

    print(f"Selected {len(specs)} qualifying rows from {len(accepted)} accepted benchmark rows")
    return specs

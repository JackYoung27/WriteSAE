#!/usr/bin/env python3
"""Circuit-level ablation: test whether SAE features CAUSE property-specific model behavior.

For each interpretable feature F correlated with text property P:
  1. Split corpus into HIGH-P (top quartile) and LOW-P (bottom quartile) groups
  2. Zero feature F in the SAE reconstruction during inference
  3. Measure PPL change on HIGH vs LOW groups separately
  4. If ablating F hurts PPL on HIGH-P texts more than LOW-P texts, that is causal evidence

Also builds a cross-property specificity matrix: rows = features, columns = property groups,
cells = PPL change when that feature is ablated on that group. Diagonal dominance means
features encode property-specific information in the recurrent state.

Usage: called from run_modal.py via --stage circuit-ablation.
"""

import time

import numpy as np
import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from experiments.analysis.evaluate_downstream import (
    _get_gdn_layer_indices,
    _patch_gdn_initial_states,
)
from experiments.analysis.probe_features import compute_text_properties


# Feature-level ablation: encode -> zero feature -> decode


def reconstruct_with_ablation(
    sae,
    state_head: torch.Tensor,
    sae_type: str,
    ablate_features: list[int] | None = None,
) -> torch.Tensor:
    """Reconstruct state through SAE, optionally zeroing specific features.

    Args:
        sae: trained SAE model (on GPU, eval mode)
        state_head: (d_k, d_v) single head state
        sae_type: "flat", "rank1", "bilinear", or "bilinear_tied"
        ablate_features: feature indices to zero before decoding (None = no ablation)

    Returns:
        Reconstructed (d_k, d_v) tensor with ablated features removed
    """
    d_k, d_v = state_head.shape

    if sae_type == "flat":
        x = state_head.reshape(1, d_k * d_v)
        coeffs = sae.encode(x).clone()  # (1, n_features)
    else:
        x = state_head.unsqueeze(0)  # (1, d_k, d_v)
        coeffs = sae.encode(x).clone()  # (1, n_features)

    if ablate_features:
        coeffs[:, ablate_features] = 0.0

    recon = sae._decode(coeffs)

    if sae_type == "flat":
        return recon.reshape(d_k, d_v)
    else:
        return recon.squeeze(0)


# Single-sequence PPL with feature ablation


@torch.no_grad()
def compute_ppl_with_feature_ablation(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int,
    sae,
    sae_type: str,
    ablate_features: list[int],
    head_idx: int = 0,
) -> dict:
    """Run prefix->suffix split with specific features zeroed in the SAE reconstruction.

    Same protocol as compute_split_perplexity but ablates individual features
    instead of comparing SAE-vs-no-SAE reconstruction.

    Returns dict with loss, perplexity, n_tokens.
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    assert split_pos > 0 and split_pos < seq_len

    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos:]

    # Pass 1: prefix to build caches
    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    # Extract all GDN states
    gdn_states = {}
    for idx in gdn_layer_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            gdn_states[idx] = layer_cache.recurrent_states.clone()

    if target_layer_idx in gdn_states:
        state = gdn_states[target_layer_idx]
        for b in range(batch_size):
            original_head = state[b, head_idx].float()
            reconstructed = reconstruct_with_ablation(
                sae, original_head, sae_type, ablate_features=ablate_features,
            )
            state[b, head_idx] = reconstructed.to(state.dtype)

    # Pass 2: suffix with patched states
    with _patch_gdn_initial_states(model, gdn_layer_indices, gdn_states):
        suffix_out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=False,
            labels=suffix,
        )

    loss = suffix_out.loss
    n_tokens = suffix.shape[1] - 1

    return {
        "loss": loss.item(),
        "perplexity": float(torch.exp(loss).item()),
        "n_tokens": n_tokens,
    }


# Corpus stratification by text properties


def stratify_corpus_by_properties(
    texts: list[str],
    properties: list[str],
    quartile_size: float = 0.25,
) -> dict[str, dict[str, list[int]]]:
    """Split corpus indices into HIGH and LOW groups per property.

    Args:
        texts: raw text strings
        properties: which properties to stratify by
        quartile_size: fraction for top/bottom groups (0.25 = top/bottom 25%)

    Returns:
        {property_name: {"high": [indices], "low": [indices]}}
    """
    N = len(texts)
    print(f"Computing text properties for {N} sequences...")
    t0 = time.time()

    all_props = []
    for text in texts:
        all_props.append(compute_text_properties(text))

    print(f"  Properties computed in {time.time() - t0:.1f}s")

    # For each target property, find top/bottom quartile indices
    groups: dict[str, dict[str, list[int]]] = {}
    n_per_group = max(int(N * quartile_size), 1)

    for prop in properties:
        values = np.array([p[prop] for p in all_props])
        sorted_indices = np.argsort(values)
        low_indices = sorted_indices[:n_per_group].tolist()
        high_indices = sorted_indices[-n_per_group:].tolist()

        groups[prop] = {
            "high": high_indices,
            "low": low_indices,
            "high_mean": float(values[high_indices].mean()),
            "low_mean": float(values[low_indices].mean()),
        }
        print(f"  {prop}: LOW mean={groups[prop]['low_mean']:.4f}, "
              f"HIGH mean={groups[prop]['high_mean']:.4f}, "
              f"n_per_group={n_per_group}")

    return groups


# Select target features from probe results


def select_target_features(
    probe_results: dict,
    n_per_property: int = 3,
    min_rho: float = 0.15,
) -> list[dict]:
    """Pick the top features per property from probe_features.json.

    Returns list of dicts: {feature_idx, property, rho, rank}
    """
    features = probe_results.get("probe", {}).get("features", [])

    # Group by best_property, sort by |rho| descending
    by_property: dict[str, list[dict]] = {}
    for feat in features:
        prop = feat["best_property"]
        rho = abs(feat["best_rho"])
        if rho < min_rho:
            continue
        if prop not in by_property:
            by_property[prop] = []
        by_property[prop].append({
            "feature_idx": feat["feature_idx"],
            "property": prop,
            "rho": feat["best_rho"],
            "abs_rho": rho,
            "frequency": feat.get("frequency", 0),
        })

    targets = []
    for prop, feats in by_property.items():
        feats.sort(key=lambda x: x["abs_rho"], reverse=True)
        for rank, f in enumerate(feats[:n_per_property]):
            f["rank"] = rank
            targets.append(f)

    targets.sort(key=lambda x: x["abs_rho"], reverse=True)
    return targets


# Main experiment: targeted feature ablation


@torch.no_grad()
def run_circuit_ablation(
    model,
    corpus_batches: list[torch.Tensor],
    texts: list[str],
    layer_idx: int,
    sae,
    sae_type: str,
    target_features: list[dict],
    head_idx: int = 0,
    split_fraction: float = 0.5,
    quartile_size: float = 0.25,
    device: str = "cuda",
) -> dict:
    """Run targeted feature ablation experiment.

    For each target feature (correlated with property P):
      1. Measure baseline PPL on HIGH-P and LOW-P text groups (SAE recon, no ablation)
      2. Measure ablated PPL on both groups (feature zeroed)
      3. Compute delta_high and delta_low
      4. Causal evidence = delta_high >> delta_low

    Args:
        model: HuggingFace CausalLM
        corpus_batches: token tensors (batch, seq_len)
        texts: raw text strings matching corpus_batches
        layer_idx: GDN layer index
        sae: trained SAE
        sae_type: SAE type string
        target_features: from select_target_features()
        head_idx: which head the SAE reconstructs
        split_fraction: prefix/suffix split point
        quartile_size: fraction for high/low groups
        device: compute device

    Returns:
        dict with per-feature ablation results and cross-property matrix
    """
    seq_len = corpus_batches[0].shape[1]
    split_pos = int(seq_len * split_fraction)
    gdn_layers = _get_gdn_layer_indices(model)

    # Flatten batches to individual sequences
    all_seqs = []
    for batch in corpus_batches:
        for i in range(batch.shape[0]):
            all_seqs.append(batch[i:i + 1])  # (1, seq_len)

    n_seqs = len(all_seqs)
    assert len(texts) >= n_seqs, f"Need {n_seqs} texts, have {len(texts)}"
    texts = texts[:n_seqs]

    # Collect unique target properties
    target_props = sorted(set(f["property"] for f in target_features))
    print(f"\nCircuit ablation: {len(target_features)} features across {len(target_props)} properties")
    print(f"  {n_seqs} sequences, split at {split_pos}/{seq_len}")
    print(f"  Quartile size: {quartile_size} ({int(n_seqs * quartile_size)} seqs per group)")

    # Stratify corpus
    groups = stratify_corpus_by_properties(texts, target_props, quartile_size)

    # Collect all unique group indices we need to evaluate
    group_indices: dict[str, set[int]] = {}  # "prop_high" -> {seq indices}
    for prop in target_props:
        group_indices[f"{prop}_high"] = set(groups[prop]["high"])
        group_indices[f"{prop}_low"] = set(groups[prop]["low"])

    # For each group, compute baseline PPL (full SAE reconstruction, no ablation)
    # and ablated PPL for each target feature.
    # To save GPU time: precompute baseline PPL per sequence once,
    # then only compute ablated PPL for sequences in relevant groups.

    print("\nPhase 1: Baseline PPL per sequence (SAE recon, no feature ablation)")
    baseline_loss_per_seq = {}
    baseline_tokens_per_seq = {}

    # Identify which sequences we need (union of all groups)
    needed_indices = set()
    for idxs in group_indices.values():
        needed_indices.update(idxs)
    needed_sorted = sorted(needed_indices)

    t0 = time.time()
    for idx in tqdm(needed_sorted, desc="Baseline"):
        seq = all_seqs[idx].to(device)
        r = compute_ppl_with_feature_ablation(
            model, seq, split_pos, gdn_layers,
            target_layer_idx=layer_idx,
            sae=sae, sae_type=sae_type,
            ablate_features=[],  # empty = no ablation, just SAE recon
            head_idx=head_idx,
        )
        baseline_loss_per_seq[idx] = r["loss"]
        baseline_tokens_per_seq[idx] = r["n_tokens"]
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    baseline_time = time.time() - t0
    print(f"  Baseline: {len(needed_sorted)} sequences in {baseline_time:.1f}s")

    # Identify which features also contribute to the cross-property matrix.
    # Top 1 per property (by |rho|), capped at 8 for time budget.
    matrix_feat_set: set[int] = set()
    seen_props_for_matrix: set[str] = set()
    for feat in target_features:
        if feat["property"] not in seen_props_for_matrix and len(matrix_feat_set) < 8:
            matrix_feat_set.add(feat["feature_idx"])
            seen_props_for_matrix.add(feat["property"])

    # Phase 2: Per-feature ablation (merged with cross-property matrix)
    # For matrix features, evaluate ALL needed sequences (union of all property groups).
    # For non-matrix features, evaluate only the matched property HIGH/LOW groups.
    # This avoids redundant forward passes between phases 2 and 3.
    print(f"\nPhase 2: Ablating {len(target_features)} features "
          f"({len(matrix_feat_set)} contribute to cross-property matrix)")
    feature_results = []
    matrix_results = []

    def _group_ppl(indices, loss_dict):
        if not indices:
            return float("nan")
        total_loss = sum(loss_dict[i] * baseline_tokens_per_seq[i] for i in indices)
        total_tokens = sum(baseline_tokens_per_seq[i] for i in indices)
        return float(np.exp(total_loss / max(total_tokens, 1)))

    def _group_avg_loss(indices, loss_dict):
        if not indices:
            return float("nan")
        total_loss = sum(loss_dict[i] * baseline_tokens_per_seq[i] for i in indices)
        total_tokens = sum(baseline_tokens_per_seq[i] for i in indices)
        return total_loss / max(total_tokens, 1)

    for fi, feat_info in enumerate(target_features):
        feat_idx = feat_info["feature_idx"]
        feat_prop = feat_info["property"]
        feat_rho = feat_info["rho"]
        is_matrix_feat = feat_idx in matrix_feat_set

        tag = " [matrix]" if is_matrix_feat else ""
        print(f"\n  [{fi+1}/{len(target_features)}] Feature {feat_idx} "
              f"({feat_prop}, rho={feat_rho:.3f}){tag}")

        if is_matrix_feat:
            # Evaluate on ALL property groups for cross-property matrix
            eval_set = set(needed_sorted)
        else:
            # Only evaluate on matched property groups
            eval_set = set(groups[feat_prop]["high"]) | set(groups[feat_prop]["low"])
        eval_indices = sorted(eval_set)

        ablated_loss_per_seq = {}
        t0_feat = time.time()

        for idx in tqdm(eval_indices, desc=f"  F{feat_idx}", leave=False):
            seq = all_seqs[idx].to(device)
            r = compute_ppl_with_feature_ablation(
                model, seq, split_pos, gdn_layers,
                target_layer_idx=layer_idx,
                sae=sae, sae_type=sae_type,
                ablate_features=[feat_idx],
                head_idx=head_idx,
            )
            ablated_loss_per_seq[idx] = r["loss"]

        feat_time = time.time() - t0_feat

        high_indices = groups[feat_prop]["high"]
        low_indices = groups[feat_prop]["low"]

        baseline_ppl_high = _group_ppl(high_indices, baseline_loss_per_seq)
        baseline_ppl_low = _group_ppl(low_indices, baseline_loss_per_seq)
        ablated_ppl_high = _group_ppl(high_indices, ablated_loss_per_seq)
        ablated_ppl_low = _group_ppl(low_indices, ablated_loss_per_seq)

        baseline_loss_high = _group_avg_loss(high_indices, baseline_loss_per_seq)
        baseline_loss_low = _group_avg_loss(low_indices, baseline_loss_per_seq)
        ablated_loss_high = _group_avg_loss(high_indices, ablated_loss_per_seq)
        ablated_loss_low = _group_avg_loss(low_indices, ablated_loss_per_seq)

        delta_ppl_high = ablated_ppl_high - baseline_ppl_high
        delta_ppl_low = ablated_ppl_low - baseline_ppl_low
        delta_loss_high = ablated_loss_high - baseline_loss_high
        delta_loss_low = ablated_loss_low - baseline_loss_low

        selectivity = delta_loss_high - delta_loss_low

        result = {
            "feature_idx": feat_idx,
            "property": feat_prop,
            "rho": feat_rho,
            "baseline_ppl_high": baseline_ppl_high,
            "baseline_ppl_low": baseline_ppl_low,
            "ablated_ppl_high": ablated_ppl_high,
            "ablated_ppl_low": ablated_ppl_low,
            "delta_ppl_high": delta_ppl_high,
            "delta_ppl_low": delta_ppl_low,
            "delta_loss_high": delta_loss_high,
            "delta_loss_low": delta_loss_low,
            "selectivity": selectivity,
            "n_high": len(high_indices),
            "n_low": len(low_indices),
            "time_s": round(feat_time, 1),
        }
        feature_results.append(result)

        print(f"    HIGH-{feat_prop}: PPL {baseline_ppl_high:.2f} -> {ablated_ppl_high:.2f} "
              f"(delta={delta_ppl_high:+.3f})")
        print(f"    LOW-{feat_prop}:  PPL {baseline_ppl_low:.2f} -> {ablated_ppl_low:.2f} "
              f"(delta={delta_ppl_low:+.3f})")
        print(f"    Selectivity (delta_loss_high - delta_loss_low): {selectivity:+.4f}")

        # Compute cross-property matrix row (reusing same ablated losses)
        if is_matrix_feat:
            cross_deltas = {}
            for prop in target_props:
                hi = groups[prop]["high"]
                lo = groups[prop]["low"]
                bl_hi = _group_avg_loss(hi, baseline_loss_per_seq)
                bl_lo = _group_avg_loss(lo, baseline_loss_per_seq)
                ab_hi = _group_avg_loss(hi, ablated_loss_per_seq)
                ab_lo = _group_avg_loss(lo, ablated_loss_per_seq)
                cross_deltas[prop] = {
                    "delta_loss_high": ab_hi - bl_hi,
                    "delta_loss_low": ab_lo - bl_lo,
                    "selectivity": (ab_hi - bl_hi) - (ab_lo - bl_lo),
                }
                print(f"    cross({prop}): delta_high={ab_hi - bl_hi:+.4f}, "
                      f"delta_low={ab_lo - bl_lo:+.4f}")
            matrix_results.append({
                "feature_idx": feat_idx,
                "property": feat_prop,
                "rho": feat_rho,
                "cross_property": cross_deltas,
            })

        torch.cuda.empty_cache()

    n_causal = sum(1 for r in feature_results if r["selectivity"] > 0)
    mean_selectivity = np.mean([r["selectivity"] for r in feature_results])
    mean_delta_high = np.mean([r["delta_loss_high"] for r in feature_results])
    mean_delta_low = np.mean([r["delta_loss_low"] for r in feature_results])

    # Matrix diagonal dominance
    diag_dom_scores = []
    for row in matrix_results:
        own_prop = row["property"]
        cross = row.get("cross_property", {})
        if own_prop in cross:
            own_sel = cross[own_prop]["selectivity"]
            other_sels = [v["selectivity"] for k, v in cross.items() if k != own_prop]
            if other_sels:
                mean_other = np.mean(other_sels)
                diag_dom_scores.append(own_sel - mean_other)

    output = {
        "layer": layer_idx,
        "head": head_idx,
        "sae_type": sae_type,
        "n_sequences": n_seqs,
        "split_pos": split_pos,
        "quartile_size": quartile_size,
        "n_target_features": len(target_features),
        "target_properties": target_props,
        "summary": {
            "n_causal_positive": n_causal,
            "n_total": len(feature_results),
            "causal_fraction": n_causal / max(len(feature_results), 1),
            "mean_selectivity": float(mean_selectivity),
            "mean_delta_loss_high": float(mean_delta_high),
            "mean_delta_loss_low": float(mean_delta_low),
            "diagonal_dominance": float(np.mean(diag_dom_scores)) if diag_dom_scores else None,
        },
        "property_groups": {
            prop: {"high_mean": g["high_mean"], "low_mean": g["low_mean"],
                   "n_high": len(g["high"]), "n_low": len(g["low"])}
            for prop, g in groups.items()
        },
        "feature_results": feature_results,
        "cross_property_matrix": matrix_results,
    }

    return output


# Report formatting


def format_circuit_ablation_report(results: dict) -> str:
    """Format circuit ablation results as a readable report."""
    lines = []
    lines.append("Circuit Ablation Results")
    lines.append("=" * 70)

    s = results["summary"]
    lines.append(f"Layer {results['layer']}, head {results['head']}, "
                 f"{results['n_sequences']} sequences")
    lines.append(f"Causal features: {s['n_causal_positive']}/{s['n_total']} "
                 f"({s['causal_fraction']*100:.1f}%) show selective PPL damage")
    lines.append(f"Mean selectivity: {s['mean_selectivity']:+.4f}")
    lines.append(f"Mean delta_loss: HIGH={s['mean_delta_loss_high']:+.4f}, "
                 f"LOW={s['mean_delta_loss_low']:+.4f}")
    if s["diagonal_dominance"] is not None:
        lines.append(f"Cross-property diagonal dominance: {s['diagonal_dominance']:+.4f}")

    lines.append("")
    lines.append(f"{'Feature':>8} {'Property':<20} {'rho':>6} "
                 f"{'dPPL_high':>10} {'dPPL_low':>10} {'Selectivity':>12}")
    lines.append("-" * 72)

    for r in sorted(results["feature_results"], key=lambda x: x["selectivity"], reverse=True):
        lines.append(
            f"{r['feature_idx']:>8} {r['property']:<20} {r['rho']:>+.3f} "
            f"{r['delta_ppl_high']:>+10.4f} {r['delta_ppl_low']:>+10.4f} "
            f"{r['selectivity']:>+12.4f}"
        )

    # Cross-property matrix
    if results["cross_property_matrix"]:
        props = results["target_properties"]
        lines.append("")
        lines.append("Cross-Property Selectivity Matrix")
        lines.append("(rows=features, cols=properties, cells=selectivity)")
        lines.append("")

        # Header
        header = f"{'Feature':>8} {'Own':>6}"
        for p in props:
            header += f" {p[:12]:>12}"
        lines.append(header)
        lines.append("-" * (20 + 13 * len(props)))

        for row in results["cross_property_matrix"]:
            line = f"{row['feature_idx']:>8} {row['property'][:6]:>6}"
            for p in props:
                sel = row["cross_property"].get(p, {}).get("selectivity", 0)
                marker = " *" if p == row["property"] else "  "
                line += f" {sel:>+10.4f}{marker}"
            lines.append(line)

    return "\n".join(lines)

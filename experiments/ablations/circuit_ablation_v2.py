#!/usr/bin/env python3
"""Circuit ablation v2: GROUP ablation with dose-response curves.

v1 zeroed individual features and got weak results (selectivity ~0.0004) because
each feature carries ~1/32 of the reconstruction (k=32 active per input). The signal
is real but distributed.

v2 fixes this by ablating GROUPS of features correlated with each property simultaneously.
For each property P:
  1. Collect all features with |rho| > threshold for P, sorted by |rho| descending.
  2. Ablate top-1, top-2, top-4, top-8, and ALL correlated features at once.
  3. Measure PPL on HIGH-P vs LOW-P text groups at each dose level.

The expected result: PPL on HIGH-P text rises steeply as more P-correlated features
are zeroed, while PPL on LOW-P text stays flat. The widening gap is causal evidence
that these features encode property-specific information.

Usage: called from run_modal.py via --stage circuit-ablation-v2.
"""

import time

import numpy as np
import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from circuit_ablation import (
    compute_ppl_with_feature_ablation,
    stratify_corpus_by_properties,
)
from evaluate_downstream import _get_gdn_layer_indices


# Build per-property feature groups from probe results


def build_property_feature_groups(
    probe_results: dict,
    min_rho: float = 0.10,
    min_features: int = 4,
) -> dict[str, list[dict]]:
    """For each property, collect all features with |rho| >= min_rho, sorted by |rho| desc.

    Uses the 'all_correlations' field from each feature (all 57 properties), not just
    best_property. A feature can appear in multiple property groups.

    Args:
        probe_results: output of probe_features_modal
        min_rho: minimum |rho| to include a feature for a property
        min_features: skip properties with fewer than this many correlated features

    Returns:
        {property_name: [{feature_idx, rho, abs_rho}, ...]} sorted by abs_rho desc
    """
    features = probe_results.get("probe", {}).get("features", [])

    # For each property, gather (feature_idx, rho) pairs
    by_property: dict[str, list[dict]] = {}

    for feat in features:
        feat_idx = feat["feature_idx"]
        # Use all_correlations if available, else significant_correlations_bonferroni
        corrs = feat.get("all_correlations", feat.get("significant_correlations_bonferroni", {}))

        for prop, data in corrs.items():
            rho = data["rho"]
            if abs(rho) < min_rho:
                continue
            if prop not in by_property:
                by_property[prop] = []
            by_property[prop].append({
                "feature_idx": feat_idx,
                "rho": rho,
                "abs_rho": abs(rho),
            })

    # Sort each group by |rho| descending, filter out small groups
    result = {}
    for prop, feats in by_property.items():
        feats.sort(key=lambda x: x["abs_rho"], reverse=True)
        if len(feats) >= min_features:
            result[prop] = feats

    return result


# Main: dose-response group ablation


@torch.no_grad()
def run_group_ablation(
    model,
    corpus_batches: list[torch.Tensor],
    texts: list[str],
    layer_idx: int,
    sae,
    sae_type: str,
    property_groups: dict[str, list[dict]],
    head_idx: int = 0,
    split_fraction: float = 0.5,
    quartile_size: float = 0.25,
    dose_levels: list[int] | None = None,
    max_properties: int = 8,
    device: str = "cuda",
) -> dict:
    """Run dose-response group ablation.

    For each property, ablate increasing numbers of correlated features and measure
    the PPL impact on high-property vs low-property text.

    Args:
        model: HuggingFace CausalLM
        corpus_batches: token tensors (batch, seq_len)
        texts: raw text strings matching corpus_batches
        layer_idx: GDN layer index
        sae: trained SAE
        sae_type: SAE type string
        property_groups: from build_property_feature_groups()
        head_idx: which head the SAE reconstructs
        split_fraction: prefix/suffix split point
        quartile_size: fraction for high/low groups
        dose_levels: list of group sizes to ablate (e.g. [1, 2, 4, 8, 16, all])
        max_properties: cap on number of properties to test (pick those with most features)
        device: compute device

    Returns:
        dict with dose-response curves per property
    """
    if dose_levels is None:
        dose_levels = [1, 2, 4, 8, 16]  # "all" is always appended

    seq_len = corpus_batches[0].shape[1]
    split_pos = int(seq_len * split_fraction)
    gdn_layers = _get_gdn_layer_indices(model)

    # Flatten batches
    all_seqs = []
    for batch in corpus_batches:
        for i in range(batch.shape[0]):
            all_seqs.append(batch[i:i + 1])
    n_seqs = len(all_seqs)
    assert len(texts) >= n_seqs, f"Need {n_seqs} texts, have {len(texts)}"
    texts = texts[:n_seqs]

    # Select top properties by number of correlated features
    sorted_props = sorted(property_groups.keys(), key=lambda p: len(property_groups[p]), reverse=True)
    target_props = sorted_props[:max_properties]
    print(f"\nGroup ablation v2: {len(target_props)} properties, {n_seqs} sequences")
    for prop in target_props:
        n_feat = len(property_groups[prop])
        top_rho = property_groups[prop][0]["abs_rho"]
        print(f"  {prop}: {n_feat} features (top |rho|={top_rho:.3f})")

    # Stratify corpus
    groups = stratify_corpus_by_properties(texts, target_props, quartile_size)

    # Collect all sequence indices we need
    needed_indices = set()
    for prop in target_props:
        needed_indices.update(groups[prop]["high"])
        needed_indices.update(groups[prop]["low"])
    needed_sorted = sorted(needed_indices)

    # Phase 1: Baseline PPL (SAE reconstruction, no ablation)
    print(f"\nPhase 1: Baseline PPL for {len(needed_sorted)} sequences")
    baseline_loss = {}
    baseline_tokens = {}
    t0 = time.time()

    for idx in tqdm(needed_sorted, desc="Baseline"):
        seq = all_seqs[idx].to(device)
        r = compute_ppl_with_feature_ablation(
            model, seq, split_pos, gdn_layers,
            target_layer_idx=layer_idx,
            sae=sae, sae_type=sae_type,
            ablate_features=[],  # no ablation
            head_idx=head_idx,
        )
        baseline_loss[idx] = r["loss"]
        baseline_tokens[idx] = r["n_tokens"]

    baseline_time = time.time() - t0
    print(f"  {len(needed_sorted)} sequences in {baseline_time:.1f}s")

    # Helper: compute weighted average loss for a group
    def _avg_loss(indices, loss_dict):
        if not indices:
            return float("nan")
        total = sum(loss_dict[i] * baseline_tokens[i] for i in indices if i in loss_dict)
        tokens = sum(baseline_tokens[i] for i in indices if i in loss_dict)
        return total / max(tokens, 1)

    def _ppl(indices, loss_dict):
        return float(np.exp(_avg_loss(indices, loss_dict)))

    # Phase 2: Dose-response curves
    results_by_property = {}
    total_ablation_time = 0

    for prop in target_props:
        feat_group = property_groups[prop]
        n_total = len(feat_group)

        # Build dose levels: [1, 2, 4, 8, 16, all]
        actual_doses = sorted(set(d for d in dose_levels if d <= n_total))
        if n_total not in actual_doses:
            actual_doses.append(n_total)

        high_indices = groups[prop]["high"]
        low_indices = groups[prop]["low"]
        eval_indices = sorted(set(high_indices) | set(low_indices))

        baseline_loss_high = _avg_loss(high_indices, baseline_loss)
        baseline_loss_low = _avg_loss(low_indices, baseline_loss)
        baseline_ppl_high = _ppl(high_indices, baseline_loss)
        baseline_ppl_low = _ppl(low_indices, baseline_loss)

        print(f"\nProperty: {prop} ({n_total} features, {len(eval_indices)} eval sequences)")
        print(f"  Baseline: HIGH loss={baseline_loss_high:.4f} PPL={baseline_ppl_high:.2f}, "
              f"LOW loss={baseline_loss_low:.4f} PPL={baseline_ppl_low:.2f}")
        print(f"  Doses: {actual_doses}")

        dose_curve = []

        # Dose 0 = baseline (no ablation)
        dose_curve.append({
            "n_ablated": 0,
            "feature_indices": [],
            "loss_high": baseline_loss_high,
            "loss_low": baseline_loss_low,
            "ppl_high": baseline_ppl_high,
            "ppl_low": baseline_ppl_low,
            "delta_loss_high": 0.0,
            "delta_loss_low": 0.0,
            "selectivity": 0.0,
        })

        for dose in actual_doses:
            # Select top-dose features by |rho|
            ablate_feats = [f["feature_idx"] for f in feat_group[:dose]]

            t0_dose = time.time()
            ablated_loss = {}

            for idx in tqdm(eval_indices, desc=f"  {prop} dose={dose}", leave=False):
                seq = all_seqs[idx].to(device)
                r = compute_ppl_with_feature_ablation(
                    model, seq, split_pos, gdn_layers,
                    target_layer_idx=layer_idx,
                    sae=sae, sae_type=sae_type,
                    ablate_features=ablate_feats,
                    head_idx=head_idx,
                )
                ablated_loss[idx] = r["loss"]

            dose_time = time.time() - t0_dose
            total_ablation_time += dose_time

            abl_loss_high = _avg_loss(high_indices, ablated_loss)
            abl_loss_low = _avg_loss(low_indices, ablated_loss)
            abl_ppl_high = _ppl(high_indices, ablated_loss)
            abl_ppl_low = _ppl(low_indices, ablated_loss)

            delta_loss_high = abl_loss_high - baseline_loss_high
            delta_loss_low = abl_loss_low - baseline_loss_low
            selectivity = delta_loss_high - delta_loss_low

            dose_curve.append({
                "n_ablated": dose,
                "feature_indices": ablate_feats,
                "loss_high": abl_loss_high,
                "loss_low": abl_loss_low,
                "ppl_high": abl_ppl_high,
                "ppl_low": abl_ppl_low,
                "delta_loss_high": delta_loss_high,
                "delta_loss_low": delta_loss_low,
                "selectivity": selectivity,
                "time_s": round(dose_time, 1),
            })

            pct_high = 100 * (abl_ppl_high - baseline_ppl_high) / max(baseline_ppl_high, 1e-6)
            pct_low = 100 * (abl_ppl_low - baseline_ppl_low) / max(baseline_ppl_low, 1e-6)
            print(f"  dose={dose:>3}: HIGH PPL {baseline_ppl_high:.2f}->{abl_ppl_high:.2f} "
                  f"({pct_high:+.2f}%), LOW PPL {baseline_ppl_low:.2f}->{abl_ppl_low:.2f} "
                  f"({pct_low:+.2f}%), selectivity={selectivity:+.4f} [{dose_time:.1f}s]")

            torch.cuda.empty_cache()

        results_by_property[prop] = {
            "n_features_total": n_total,
            "top_rho": feat_group[0]["abs_rho"],
            "feature_rhos": [f["rho"] for f in feat_group],
            "high_mean": groups[prop]["high_mean"],
            "low_mean": groups[prop]["low_mean"],
            "n_high": len(high_indices),
            "n_low": len(low_indices),
            "baseline_loss_high": baseline_loss_high,
            "baseline_loss_low": baseline_loss_low,
            "baseline_ppl_high": baseline_ppl_high,
            "baseline_ppl_low": baseline_ppl_low,
            "dose_curve": dose_curve,
        }

    # Summary: find property with largest max selectivity
    best_prop = None
    best_sel = -float("inf")
    for prop, data in results_by_property.items():
        max_sel = max(d["selectivity"] for d in data["dose_curve"])
        if max_sel > best_sel:
            best_sel = max_sel
            best_prop = prop

    # Compute mean selectivity at max dose across properties
    max_dose_sels = []
    for prop, data in results_by_property.items():
        max_dose_sels.append(data["dose_curve"][-1]["selectivity"])

    output = {
        "layer": layer_idx,
        "head": head_idx,
        "sae_type": sae_type,
        "n_sequences": n_seqs,
        "split_pos": split_pos,
        "quartile_size": quartile_size,
        "n_properties": len(target_props),
        "target_properties": target_props,
        "summary": {
            "best_property": best_prop,
            "best_selectivity": float(best_sel),
            "mean_max_dose_selectivity": float(np.mean(max_dose_sels)),
            "median_max_dose_selectivity": float(np.median(max_dose_sels)),
            "n_positive_selectivity": sum(1 for s in max_dose_sels if s > 0),
            "n_total": len(max_dose_sels),
            "total_ablation_time_s": round(total_ablation_time, 1),
        },
        "property_results": results_by_property,
    }

    return output


# Report formatting


def format_group_ablation_report(results: dict) -> str:
    """Format dose-response ablation results."""
    lines = []
    lines.append("Group Ablation v2: Dose-Response Curves")
    lines.append("=" * 70)

    s = results["summary"]
    lines.append(f"Layer {results['layer']}, head {results['head']}, "
                 f"{results['n_sequences']} sequences, {results['n_properties']} properties")
    lines.append(f"Best property: {s['best_property']} (selectivity={s['best_selectivity']:+.4f})")
    lines.append(f"Mean max-dose selectivity: {s['mean_max_dose_selectivity']:+.4f}")
    lines.append(f"Positive selectivity: {s['n_positive_selectivity']}/{s['n_total']}")

    for prop, data in results["property_results"].items():
        lines.append("")
        lines.append(f"Property: {prop} ({data['n_features_total']} features, "
                     f"top |rho|={data['top_rho']:.3f})")
        lines.append(f"  {'Dose':>6} {'PPL_high':>10} {'PPL_low':>10} "
                     f"{'dLoss_high':>11} {'dLoss_low':>11} {'Selectivity':>12}")
        lines.append("  " + "-" * 64)

        for d in data["dose_curve"]:
            lines.append(
                f"  {d['n_ablated']:>6} {d['ppl_high']:>10.3f} {d['ppl_low']:>10.3f} "
                f"{d['delta_loss_high']:>+11.5f} {d['delta_loss_low']:>+11.5f} "
                f"{d['selectivity']:>+12.5f}"
            )

    return "\n".join(lines)

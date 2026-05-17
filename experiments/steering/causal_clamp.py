#!/usr/bin/env python3
"""Causal feature clamping: clamp a feature HIGH, generate text, measure property shift.

This turns correlational probing claims into causal evidence. For each
interpretable feature F correlated with text property P:
  1. Take N prompts (first 512 tokens of OpenWebText sequences)
  2. Encode the GDN state at the last prompt token through the SAE
  3. Clamp feature F's activation to its 95th percentile value
  4. Decode back, patch the GDN state, generate 256 tokens
  5. Also generate without patching (baseline)
  6. Compute text property P on both generations
  7. Report: mean shift, Cohen's d, paired t-test p-value

If clamping F shifts the generated text toward P, that is causal evidence
that F encodes P in the recurrent state.
"""

import time

import numpy as np
import torch
from scipy import stats as scipy_stats

from experiments.analysis.probe_features import compute_text_properties


FORMAT_PROPERTIES = (
    "newline_density",
    "paragraph_count",
    "line_count",
    "list_ratio",
    "semicolon_colon_density",
)

NUMERIC_SURFACE_PROPERTIES = (
    "number_count",
    "digit_density",
)

LAYOUT_PROPERTIES = (
    "last_sent_boundary_pos",
    "newline_density",
    "paragraph_count",
    "line_count",
)

PROPERTY_FAMILIES = {
    "document_format": FORMAT_PROPERTIES,
    "numeric_surface": NUMERIC_SURFACE_PROPERTIES,
    "layout": LAYOUT_PROPERTIES,
}

DEFAULT_DOSE_LEVELS = [1, 2, 4, 8, 16]


def _get_gdn_layer_indices(model) -> list[int]:
    """Get indices of GDN (linear_attention) layers from model config."""
    config = getattr(model.config, "text_config", model.config)
    return [i for i, layer_type in enumerate(config.layer_types) if layer_type == "linear_attention"]


# Precompute activation percentiles from training data


def compute_activation_percentiles(
    sae,
    states: torch.Tensor,
    feature_indices: list[int],
    percentile: float = 95.0,
    batch_size: int = 512,
) -> dict[int, float]:
    """Compute the Pth percentile activation for each target feature.

    Args:
        sae: trained SAE (on GPU, eval mode)
        states: (N, d_k, d_v) training states
        feature_indices: which features to compute percentiles for
        percentile: which percentile (default 95th)
        batch_size: encoding batch size

    Returns:
        {feature_idx: percentile_value}
    """
    device = next(sae.parameters()).device
    N = states.shape[0]

    all_acts = []
    for i in range(0, N, batch_size):
        batch = states[i : i + batch_size].to(device)
        with torch.no_grad():
            acts = sae.encode(batch)
        all_acts.append(acts.cpu().numpy())
    act_matrix = np.concatenate(all_acts, axis=0)  # (N, n_features)

    result = {}
    for fi in feature_indices:
        vals = act_matrix[:, fi]
        nonzero = vals[vals > 0]
        if len(nonzero) >= 10:
            # Enough nonzero activations: use percentile of nonzero values
            result[fi] = float(np.percentile(nonzero, percentile))
        elif len(nonzero) > 0:
            # Few nonzero: use max activation as clamp value
            result[fi] = float(nonzero.max())
        else:
            # Feature never fires on this data: use overall 99.9th percentile
            # as a reasonable "high activation" value
            result[fi] = float(np.percentile(act_matrix[act_matrix > 0], 90)) if (act_matrix > 0).any() else 0.01

    return result


def compute_activation_matrix(
    sae,
    states: torch.Tensor,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode a batch of states and return the full activation matrix."""
    device = next(sae.parameters()).device
    n_states = states.shape[0]

    all_acts = []
    for start in range(0, n_states, batch_size):
        batch = states[start : start + batch_size].to(device)
        with torch.no_grad():
            acts = sae.encode(batch)
        all_acts.append(acts.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def find_alive_feature_indices(
    sae,
    states: torch.Tensor,
    batch_size: int = 512,
) -> list[int]:
    """Return feature indices that activate at least once on the provided states."""
    act_matrix = compute_activation_matrix(sae, states, batch_size=batch_size)
    alive_mask = (act_matrix > 0).any(axis=0)
    return np.nonzero(alive_mask)[0].astype(int).tolist()


def select_primary_family_property(
    probe_results: dict,
    family_properties: list[str] | tuple[str, ...],
) -> str:
    """Pick the family property with the strongest probe footprint."""
    summary = probe_results.get("probe", {}).get("property_summary", {})
    ranked = []
    for prop in family_properties:
        entry = summary.get(prop)
        if not entry:
            continue
        ranked.append((
            entry.get("n_correlated_features", 0),
            entry.get("max_abs_rho", 0.0),
            entry.get("mean_abs_rho", 0.0),
            prop,
        ))
    if ranked:
        return max(ranked)[-1]
    return family_properties[0]


def select_top_family_features(
    probe_results: dict,
    family_properties: list[str] | tuple[str, ...],
    n_features: int = 16,
    min_rho: float = 0.10,
) -> list[dict]:
    """Select the strongest features for a property family using all correlations."""
    features = probe_results.get("probe", {}).get("features", [])
    ranked = []

    for feat in features:
        corrs = feat.get("all_correlations", feat.get("significant_correlations_bonferroni", {}))
        family_corrs = []
        for prop in family_properties:
            data = corrs.get(prop)
            if not data:
                continue
            rho = float(data["rho"])
            family_corrs.append((prop, rho))
        if not family_corrs:
            continue

        best_property, best_rho = max(family_corrs, key=lambda item: abs(item[1]))
        if abs(best_rho) < min_rho:
            continue

        ranked.append({
            "feature_idx": int(feat["feature_idx"]),
            "property": best_property,
            "rho": float(best_rho),
            "family_support": len(family_corrs),
            "family_mean_abs_rho": float(np.mean([abs(rho) for _, rho in family_corrs])),
        })

    ranked.sort(
        key=lambda item: (abs(item["rho"]), item["family_support"], item["family_mean_abs_rho"]),
        reverse=True,
    )
    return ranked[:n_features]


# Reconstruct with feature updates


def reconstruct_with_feature_updates(
    sae,
    state_head: torch.Tensor,
    sae_type: str,
    set_updates: dict[int, float] | None = None,
    clamp_updates: dict[int, float] | None = None,
    ablate_features: list[int] | None = None,
) -> torch.Tensor:
    """Encode through the SAE, edit feature activations, and decode back."""
    d_k, d_v = state_head.shape

    if sae_type == "flat":
        x = state_head.reshape(1, d_k * d_v)
        coeffs = sae.encode(x).clone()
    else:
        x = state_head.unsqueeze(0)
        coeffs = sae.encode(x).clone()

    if ablate_features:
        coeffs[:, ablate_features] = 0.0

    if set_updates:
        set_indices = sorted(set_updates)
        set_values = torch.as_tensor(
            [set_updates[idx] for idx in set_indices],
            device=coeffs.device,
            dtype=coeffs.dtype,
        )
        coeffs[:, set_indices] = set_values.unsqueeze(0)

    if clamp_updates:
        clamp_indices = sorted(clamp_updates)
        clamp_values = torch.as_tensor(
            [clamp_updates[idx] for idx in clamp_indices],
            device=coeffs.device,
            dtype=coeffs.dtype,
        )
        coeffs[:, clamp_indices] = torch.maximum(
            coeffs[:, clamp_indices],
            clamp_values.unsqueeze(0),
        )

    recon = sae._decode(coeffs)
    if sae_type == "flat":
        return recon.reshape(d_k, d_v)
    return recon.squeeze(0)


def reconstruct_with_clamp(
    sae,
    state_head: torch.Tensor,
    sae_type: str,
    clamp_feature: int,
    clamp_value: float,
) -> torch.Tensor:
    """Single-feature clamp wrapper around reconstruct_with_feature_updates."""
    return reconstruct_with_feature_updates(
        sae=sae,
        state_head=state_head,
        sae_type=sae_type,
        clamp_updates={clamp_feature: clamp_value},
    )




@torch.no_grad()
def generate_with_feature_updates(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
    gen_len: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int,
    sae,
    sae_type: str,
    head_idx: int,
    set_updates: dict[int, float] | None = None,
    clamp_updates: dict[int, float] | None = None,
    ablate_features: list[int] | None = None,
) -> str:
    """Run prefix to build state, optionally edit features, then generate tokens.

    Args:
        model: CausalLM
        tokenizer: tokenizer for decoding
        input_ids: (1, seq_len) full token ids
        prompt_len: number of prompt tokens to process before generating
        gen_len: number of tokens to generate
        gdn_layer_indices: all GDN layer indices
        target_layer_idx: which layer to clamp
        sae: trained SAE
        sae_type: SAE type string
        head_idx: which head
        set_updates: feature->value map for exact value edits
        clamp_updates: feature->floor map for clamping HIGH
        ablate_features: feature indices to zero before decoding

    Returns:
        Generated text string (gen_len tokens)
    """
    device = input_ids.device
    prompt = input_ids[:, :prompt_len]

    # Forward pass on prompt to build caches
    prefix_out = model(input_ids=prompt, use_cache=True)
    cache = prefix_out.past_key_values

    # Apply clamping directly to the cache's recurrent state.
    # For token-by-token generation (seq_len=1), the GDN layer reads
    # recurrent_states from the cache directly, so we patch in-place.
    if set_updates or clamp_updates or ablate_features:
        layer_cache = cache.layers[target_layer_idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            state = layer_cache.recurrent_states
            original_head = state[0, head_idx].float()
            clamped = reconstruct_with_feature_updates(
                sae,
                original_head,
                sae_type,
                set_updates=set_updates,
                clamp_updates=clamp_updates,
                ablate_features=ablate_features,
            )
            state[0, head_idx] = clamped.to(state.dtype)

    # Autoregressive generation (greedy decoding)
    next_token_logits = prefix_out.logits[:, -1, :]
    generated_ids = []

    for _ in range(gen_len):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token)

        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

    all_ids = torch.cat(generated_ids, dim=-1)
    text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
    return text


@torch.no_grad()
def generate_with_clamp(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
    gen_len: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int,
    sae,
    sae_type: str,
    head_idx: int,
    clamp_feature: int | None,
    clamp_value: float | None,
) -> str:
    """Single-feature clamp wrapper around generate_with_feature_updates."""
    clamp_updates = None
    if clamp_feature is not None and clamp_value is not None:
        clamp_updates = {clamp_feature: clamp_value}
    return generate_with_feature_updates(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        prompt_len=prompt_len,
        gen_len=gen_len,
        gdn_layer_indices=gdn_layer_indices,
        target_layer_idx=target_layer_idx,
        sae=sae,
        sae_type=sae_type,
        head_idx=head_idx,
        clamp_updates=clamp_updates,
    )


# Main causal clamping experiment


def select_top_features_by_rho(
    probe_results: dict,
    n_features: int = 5,
    allowed_properties: set[str] | None = None,
) -> list[dict]:
    """Select top N features by |best_rho| from probe results.

    Returns list of {feature_idx, property, rho}.
    """
    features = probe_results.get("probe", {}).get("features", [])
    if allowed_properties is not None:
        features = [
            feat for feat in features
            if feat.get("best_property") in allowed_properties
        ]
    sorted_feats = sorted(features, key=lambda x: abs(x.get("best_rho", 0)), reverse=True)

    result = []
    for f in sorted_feats[:n_features]:
        result.append({
            "feature_idx": f["feature_idx"],
            "property": f["best_property"],
            "rho": f["best_rho"],
        })
    return result


# Logit-based causal intervention


LOGIT_TEST_PROPERTIES = {"code_density", "dialogue_ratio", "word_entropy"}


def _single_token_ids(tokenizer, texts: list[str]) -> list[int]:
    """Return token ids for strings that map to exactly one token."""
    token_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.append(int(ids[0]))
    return sorted(set(token_ids))


def build_property_token_groups(tokenizer) -> dict[str, dict]:
    """Build token groups for direct logit-mass tests."""
    code_strings = [
        "{", "}", "(", ")", "[", "]", "=", ";", ":",
        " def", " class", " import", " return", " self",
        "def", "class", "import", "return",
    ]
    dialogue_strings = [
        '"', "“", "”", "'", " said", " asked", " replied",
        " whispered", " shouted", " says", " asked", " replied",
        "said", "asked", "replied",
    ]

    groups = {
        "code_density": {
            "metric": "token_mass",
            "token_ids": _single_token_ids(tokenizer, code_strings),
            "strings": code_strings,
        },
        "dialogue_ratio": {
            "metric": "token_mass",
            "token_ids": _single_token_ids(tokenizer, dialogue_strings),
            "strings": dialogue_strings,
        },
        "word_entropy": {
            "metric": "entropy",
            "token_ids": [],
            "strings": [],
        },
    }
    return groups


@torch.no_grad()
def next_token_logits_with_clamp(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    target_layer_idx: int,
    sae,
    sae_type: str,
    head_idx: int,
    clamp_feature: int | None = None,
    clamp_value: float | None = None,
) -> torch.Tensor:
    """Return next-token logits at a prefix/suffix split, optionally with clamping.

    The suffix is one token long. For Qwen3.5 GDN layers this means the model
    consumes the cached recurrent state directly, so patching the cache state
    after the prefix gives the immediate effect on the next-token distribution.
    """
    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos : split_pos + 1]
    assert suffix.shape[1] == 1

    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    if clamp_feature is not None and clamp_value is not None:
        layer_cache = cache.layers[target_layer_idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            state = layer_cache.recurrent_states
            original_head = state[0, head_idx].float()
            clamped = reconstruct_with_clamp(
                sae, original_head, sae_type,
                clamp_feature=clamp_feature,
                clamp_value=clamp_value,
            )
            state[0, head_idx] = clamped.to(state.dtype)

    out = model(input_ids=suffix, past_key_values=cache, use_cache=False)
    return out.logits[:, 0, :].float()


def metric_from_logits(logits: torch.Tensor, metric_name: str, token_ids: list[int]) -> float:
    """Compute a scalar metric from logits for a given property."""
    if metric_name == "entropy":
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
        return float(entropy.item())

    if metric_name == "token_mass":
        if not token_ids:
            return 0.0
        probs = torch.softmax(logits, dim=-1)
        group_mass = probs[:, token_ids].sum(dim=-1)
        return float(group_mass.item())

    raise ValueError(f"Unknown metric: {metric_name}")


def default_split_positions(
    seq_len: int,
    prompt_len: int,
    positions_per_prompt: int,
) -> list[int]:
    """Choose split positions within a sequence for direct logit tests."""
    upper = min(prompt_len, seq_len - 2)
    lower = min(128, upper)
    if upper <= 1:
        return []
    if positions_per_prompt <= 1 or upper <= lower + 8:
        return [max(1, upper)]
    positions = np.linspace(lower, upper, positions_per_prompt, dtype=int)
    deduped = sorted({int(p) for p in positions if 1 <= int(p) < seq_len - 1})
    return deduped


@torch.no_grad()
def run_logit_causal_intervention(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    states: torch.Tensor,
    layer_idx: int,
    sae,
    sae_type: str,
    target_features: list[dict],
    head_idx: int = 0,
    prompt_len: int = 512,
    positions_per_prompt: int = 1,
    n_prompts: int = 50,
    activation_percentile: float = 95.0,
    device: str = "cuda",
) -> dict:
    """Measure direct next-token logit effects from clamping SAE features high."""
    groups = build_property_token_groups(tokenizer)

    all_seqs: list[torch.Tensor] = []
    for batch in corpus_batches:
        for i in range(batch.shape[0]):
            all_seqs.append(batch[i : i + 1])
    all_seqs = all_seqs[:n_prompts]
    actual_prompts = len(all_seqs)

    feature_indices = [f["feature_idx"] for f in target_features]
    percentiles = compute_activation_percentiles(
        sae,
        states,
        feature_indices,
        percentile=activation_percentile,
    )

    feature_results = []

    for feat_info in target_features:
        feat_idx = feat_info["feature_idx"]
        feat_prop = feat_info["property"]
        feat_rho = feat_info["rho"]
        group = groups.get(feat_prop)
        if group is None:
            continue

        clamp_val = percentiles[feat_idx]
        baseline_metrics: list[float] = []
        clamped_metrics: list[float] = []
        sampled_positions: list[int] = []
        t0_feat = time.time()

        for seq in all_seqs:
            seq = seq.to(device)
            seq_len = int(seq.shape[1])
            split_positions = default_split_positions(
                seq_len=seq_len,
                prompt_len=prompt_len,
                positions_per_prompt=positions_per_prompt,
            )
            for split_pos in split_positions:
                baseline_logits = next_token_logits_with_clamp(
                    model=model,
                    input_ids=seq,
                    split_pos=split_pos,
                    target_layer_idx=layer_idx,
                    sae=sae,
                    sae_type=sae_type,
                    head_idx=head_idx,
                    clamp_feature=None,
                    clamp_value=None,
                )
                clamped_logits = next_token_logits_with_clamp(
                    model=model,
                    input_ids=seq,
                    split_pos=split_pos,
                    target_layer_idx=layer_idx,
                    sae=sae,
                    sae_type=sae_type,
                    head_idx=head_idx,
                    clamp_feature=feat_idx,
                    clamp_value=clamp_val,
                )
                baseline_metrics.append(
                    metric_from_logits(baseline_logits, group["metric"], group["token_ids"])
                )
                clamped_metrics.append(
                    metric_from_logits(clamped_logits, group["metric"], group["token_ids"])
                )
                sampled_positions.append(split_pos)

        baseline_arr = np.array(baseline_metrics, dtype=np.float64)
        clamped_arr = np.array(clamped_metrics, dtype=np.float64)
        diffs = clamped_arr - baseline_arr

        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-8
        if len(diffs) > 1 and std_diff > 1e-8:
            t_stat, p_value = scipy_stats.ttest_rel(clamped_arr, baseline_arr)
        else:
            t_stat, p_value = 0.0, 1.0

        mean_shift = float(np.mean(diffs)) if len(diffs) else 0.0
        direction_aligned = (feat_rho > 0 and mean_shift > 0) or (feat_rho < 0 and mean_shift < 0)

        result = {
            "feature_idx": feat_idx,
            "property": feat_prop,
            "rho": feat_rho,
            "metric": group["metric"],
            "token_group_size": len(group["token_ids"]),
            "token_group_examples": group["strings"][:8],
            "clamp_value": clamp_val,
            "n_prompts": actual_prompts,
            "n_positions": len(diffs),
            "mean_baseline": float(np.mean(baseline_arr)) if len(baseline_arr) else 0.0,
            "mean_clamped": float(np.mean(clamped_arr)) if len(clamped_arr) else 0.0,
            "mean_shift": mean_shift,
            "std_shift": std_diff,
            "cohens_d": mean_shift / max(std_diff, 1e-8),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "direction_aligned": direction_aligned,
            "significant_p05": bool(p_value < 0.05),
            "sampled_positions": sampled_positions,
            "time_s": round(time.time() - t0_feat, 1),
        }
        feature_results.append(result)

    n_significant = sum(1 for r in feature_results if r["significant_p05"])
    n_aligned = sum(1 for r in feature_results if r["direction_aligned"])
    n_both = sum(
        1 for r in feature_results
        if r["significant_p05"] and r["direction_aligned"]
    )

    output = {
        "layer": layer_idx,
        "head": head_idx,
        "sae_type": sae_type,
        "prompt_len": prompt_len,
        "positions_per_prompt": positions_per_prompt,
        "n_prompts": actual_prompts,
        "n_features": len(feature_results),
        "activation_percentile": activation_percentile,
        "summary": {
            "n_significant_p05": n_significant,
            "n_direction_aligned": n_aligned,
            "n_significant_and_aligned": n_both,
            "fraction_causal": n_both / max(len(feature_results), 1),
            "mean_abs_cohens_d": float(np.mean([abs(r["cohens_d"]) for r in feature_results])) if feature_results else 0.0,
        },
        "feature_results": feature_results,
    }
    return output


def _property_matrix(
    texts: list[str],
    properties: list[str] | tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Compute aligned property arrays for a list of texts."""
    rows = [compute_text_properties(text) for text in texts]
    return {
        prop: np.array([row[prop] for row in rows], dtype=np.float64)
        for prop in properties
    }


def _family_score(
    property_arrays: dict[str, np.ndarray],
    properties: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Combine several properties into one within-corpus family score."""
    z_rows = []
    for prop in properties:
        values = property_arrays[prop]
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-8:
            z_rows.append(np.zeros_like(values))
        else:
            z_rows.append((values - mean) / std)
    return np.mean(np.stack(z_rows, axis=0), axis=0)


def summarize_shift(
    baseline_values: list[float] | np.ndarray,
    edited_values: list[float] | np.ndarray,
) -> dict[str, float | bool]:
    """Paired summary statistics for one scalar readout."""
    baseline_arr = np.array(baseline_values, dtype=np.float64)
    edited_arr = np.array(edited_values, dtype=np.float64)
    diffs = edited_arr - baseline_arr
    mean_shift = float(np.mean(diffs)) if len(diffs) else 0.0
    std_shift = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    if len(diffs) > 1 and std_shift > 1e-8:
        t_stat, p_value = scipy_stats.ttest_rel(edited_arr, baseline_arr)
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "mean_baseline": float(np.mean(baseline_arr)) if len(baseline_arr) else 0.0,
        "mean_edited": float(np.mean(edited_arr)) if len(edited_arr) else 0.0,
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "cohens_d": mean_shift / max(std_shift, 1e-8),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05),
    }


@torch.no_grad()
def compute_family_activation_profile(
    texts: list[str],
    states: torch.Tensor,
    sae,
    feature_group: list[dict],
    family_properties: list[str] | tuple[str, ...],
    n_quantiles: int = 5,
    batch_size: int = 512,
) -> dict:
    """Compute group activation as a function of a property-family score."""
    n_points = min(len(texts), states.shape[0])
    if n_points == 0:
        raise ValueError("Need at least one text/state pair for activation profile")

    feature_indices = [int(item["feature_idx"]) for item in feature_group]
    act_matrix = compute_activation_matrix(sae, states[:n_points], batch_size=batch_size)
    group_activation = act_matrix[:, feature_indices].sum(axis=1)

    property_arrays = _property_matrix(texts[:n_points], family_properties)
    family_score = _family_score(property_arrays, family_properties)

    quantile_edges = np.quantile(family_score, np.linspace(0, 1, n_quantiles + 1))
    rows = []
    for qi in range(n_quantiles):
        lo = quantile_edges[qi]
        hi = quantile_edges[qi + 1]
        if qi == n_quantiles - 1:
            mask = (family_score >= lo) & (family_score <= hi)
        else:
            mask = (family_score >= lo) & (family_score < hi)
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        rows.append({
            "quantile_idx": qi,
            "n_samples": int(len(idx)),
            "family_score_mean": float(np.mean(family_score[idx])),
            "activation_mean": float(np.mean(group_activation[idx])),
            "activation_stderr": float(np.std(group_activation[idx], ddof=1) / max(np.sqrt(len(idx)), 1.0)) if len(idx) > 1 else 0.0,
            "property_means": {
                prop: float(np.mean(values[idx]))
                for prop, values in property_arrays.items()
            },
        })

    return {
        "n_samples": int(n_points),
        "feature_indices": feature_indices,
        "family_properties": list(family_properties),
        "quantiles": rows,
        "feature_group_summary": {
            "n_features": len(feature_indices),
            "top_feature_indices": feature_indices[:8],
        },
    }


@torch.no_grad()
def run_group_causal_clamp(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    states: torch.Tensor,
    layer_idx: int,
    sae,
    sae_type: str,
    feature_group: list[dict],
    family_properties: list[str] | tuple[str, ...],
    main_property: str,
    head_idx: int = 0,
    prompt_len: int = 512,
    gen_len: int = 128,
    n_prompts: int = 24,
    activation_percentile: float = 95.0,
    dose_levels: list[int] | None = None,
    n_random_groups: int = 4,
    random_seed: int = 0,
    device: str = "cuda",
) -> dict:
    """Clamp a feature group high, then measure family-aligned text shifts."""
    gdn_layers = _get_gdn_layer_indices(model)
    if dose_levels is None:
        dose_levels = DEFAULT_DOSE_LEVELS

    all_seqs = []
    for batch in corpus_batches:
        for i in range(batch.shape[0]):
            all_seqs.append(batch[i : i + 1])
    all_seqs = all_seqs[:n_prompts]
    actual_prompts = len(all_seqs)
    if actual_prompts == 0:
        raise ValueError("No prompts available for group causal clamping")

    sorted_group = sorted(feature_group, key=lambda item: abs(item["rho"]), reverse=True)
    main_property_rhos = [item["rho"] for item in sorted_group if item["property"] == main_property]
    if main_property_rhos:
        main_direction = float(np.sign(np.mean(main_property_rhos)))
    else:
        main_direction = float(np.sign(np.mean([item["rho"] for item in sorted_group]))) if sorted_group else 1.0
    if abs(main_direction) < 1e-8:
        main_direction = 1.0
    max_dose = min(len(sorted_group), max(dose_levels)) if sorted_group else 0
    target_indices = [int(item["feature_idx"]) for item in sorted_group[:max_dose]]

    alive_features = find_alive_feature_indices(sae, states)
    alive_pool = [idx for idx in alive_features if idx not in set(target_indices)]
    if len(alive_pool) < max(max_dose, 1):
        raise ValueError("Not enough alive features for matched random controls")

    percentiles = compute_activation_percentiles(
        sae,
        states,
        target_indices,
        percentile=activation_percentile,
    )

    baseline_texts = []
    baseline_props = []
    for seq in all_seqs:
        seq = seq.to(device)
        text = generate_with_feature_updates(
            model=model,
            tokenizer=tokenizer,
            input_ids=seq,
            prompt_len=prompt_len,
            gen_len=gen_len,
            gdn_layer_indices=gdn_layers,
            target_layer_idx=layer_idx,
            sae=sae,
            sae_type=sae_type,
            head_idx=head_idx,
        )
        baseline_texts.append(text)
        baseline_props.append(compute_text_properties(text))

    baseline_property_values = {
        prop: np.array([row[prop] for row in baseline_props], dtype=np.float64)
        for prop in family_properties
    }

    actual_doses = sorted(set(d for d in dose_levels if d <= len(sorted_group)))
    if len(sorted_group) not in actual_doses:
        actual_doses.append(len(sorted_group))

    target_results = []
    for dose in actual_doses:
        chosen = sorted_group[:dose]
        clamp_updates = {
            int(item["feature_idx"]): float(percentiles[int(item["feature_idx"])])
            for item in chosen
        }
        edited_props = []
        for seq in all_seqs:
            seq = seq.to(device)
            text = generate_with_feature_updates(
                model=model,
                tokenizer=tokenizer,
                input_ids=seq,
                prompt_len=prompt_len,
                gen_len=gen_len,
                gdn_layer_indices=gdn_layers,
                target_layer_idx=layer_idx,
                sae=sae,
                sae_type=sae_type,
                head_idx=head_idx,
                clamp_updates=clamp_updates,
            )
            edited_props.append(compute_text_properties(text))

        property_shifts = {
            prop: summarize_shift(
                baseline_property_values[prop],
                [row[prop] for row in edited_props],
            )
            for prop in family_properties
        }
        target_results.append({
            "dose": int(dose),
            "feature_indices": [int(item["feature_idx"]) for item in chosen],
            "property_shifts": property_shifts,
            "main_property_shift": float(property_shifts[main_property]["mean_shift"]),
            "main_property_aligned_shift": float(main_direction * property_shifts[main_property]["mean_shift"]),
            "main_property_cohens_d": float(property_shifts[main_property]["cohens_d"]),
        })

    rng = np.random.default_rng(random_seed)
    random_controls = []
    control_dose = target_results[-1]["dose"]
    for control_idx in range(n_random_groups):
        sampled = rng.choice(alive_pool, size=control_dose, replace=False).tolist()
        sampled_percentiles = compute_activation_percentiles(
            sae,
            states,
            sampled,
            percentile=activation_percentile,
        )
        clamp_updates = {int(idx): float(sampled_percentiles[int(idx)]) for idx in sampled}
        edited_props = []
        for seq in all_seqs:
            seq = seq.to(device)
            text = generate_with_feature_updates(
                model=model,
                tokenizer=tokenizer,
                input_ids=seq,
                prompt_len=prompt_len,
                gen_len=gen_len,
                gdn_layer_indices=gdn_layers,
                target_layer_idx=layer_idx,
                sae=sae,
                sae_type=sae_type,
                head_idx=head_idx,
                clamp_updates=clamp_updates,
            )
            edited_props.append(compute_text_properties(text))

        property_shifts = {
            prop: summarize_shift(
                baseline_property_values[prop],
                [row[prop] for row in edited_props],
            )
            for prop in family_properties
        }
        random_controls.append({
            "control_idx": int(control_idx),
            "feature_indices": [int(idx) for idx in sampled],
            "property_shifts": property_shifts,
            "main_property_shift": float(property_shifts[main_property]["mean_shift"]),
            "main_property_aligned_shift": float(main_direction * property_shifts[main_property]["mean_shift"]),
            "main_property_cohens_d": float(property_shifts[main_property]["cohens_d"]),
        })

    main_shifts = [row["main_property_aligned_shift"] for row in target_results]
    dose_ordered = all(
        later >= earlier - 1e-6
        for earlier, later in zip(main_shifts, main_shifts[1:])
    )
    random_main = np.array([row["main_property_aligned_shift"] for row in random_controls], dtype=np.float64)
    random_mean = float(np.mean(random_main)) if len(random_main) else 0.0
    random_std = float(np.std(random_main, ddof=1)) if len(random_main) > 1 else 0.0
    target_best = float(max(main_shifts)) if main_shifts else 0.0
    ratio_vs_random = target_best / max(abs(random_mean), 1e-8)

    return {
        "layer": layer_idx,
        "head": head_idx,
        "sae_type": sae_type,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "n_prompts": actual_prompts,
        "family_properties": list(family_properties),
        "main_property": main_property,
        "main_property_direction": main_direction,
        "feature_group": [
            {
                "feature_idx": int(item["feature_idx"]),
                "property": item["property"],
                "rho": float(item["rho"]),
            }
            for item in sorted_group
        ],
        "dose_results": target_results,
        "random_controls": random_controls,
        "summary": {
            "dose_ordered_main_shift": bool(dose_ordered),
            "best_main_property_shift": target_best,
            "best_main_property_cohens_d": float(max([row["main_property_cohens_d"] for row in target_results], default=0.0)),
            "random_main_property_mean_shift": random_mean,
            "random_main_property_std_shift": random_std,
            "ratio_vs_random_mean": ratio_vs_random,
            "target_beats_random_2x": bool(target_best >= 2.0 * max(abs(random_mean), 1e-8)),
        },
    }


def format_logit_causal_report(results: dict) -> str:
    """Format logit-based causal results as a readable report."""
    lines = []
    lines.append("Logit-Based Causal Intervention Results")
    lines.append("=" * 70)

    s = results["summary"]
    lines.append(
        f"Layer {results['layer']}, head {results['head']}, "
        f"{results['n_prompts']} prompts x {results['positions_per_prompt']} split positions"
    )
    lines.append(f"Features tested: {results['n_features']}")
    lines.append(f"Significant (p<0.05): {s['n_significant_p05']}/{results['n_features']}")
    lines.append(f"Direction-aligned: {s['n_direction_aligned']}/{results['n_features']}")
    lines.append(
        f"Causal (sig + aligned): {s['n_significant_and_aligned']}/{results['n_features']} "
        f"({s['fraction_causal'] * 100:.0f}%)"
    )
    lines.append(f"Mean |Cohen's d|: {s['mean_abs_cohens_d']:.3f}")
    lines.append("")
    lines.append(
        f"{'Feature':>8} {'Property':<16} {'Metric':<10} {'Shift':>9} "
        f"{'d':>7} {'p':>8} {'Aligned':>8}"
    )
    lines.append("-" * 76)

    for r in sorted(results["feature_results"], key=lambda x: abs(x["cohens_d"]), reverse=True):
        aligned = "yes" if r["direction_aligned"] else "no"
        sig_mark = "*" if r["significant_p05"] else ""
        lines.append(
            f"{r['feature_idx']:>8} {r['property']:<16} {r['metric']:<10} "
            f"{r['mean_shift']:>+9.4f} {r['cohens_d']:>+7.3f} {r['p_value']:>8.4f}{sig_mark} {aligned:>7}"
        )

    return "\n".join(lines)


def format_group_causal_report(results: dict) -> str:
    """Format grouped causal clamp results as a readable report."""
    lines = []
    summary = results["summary"]
    lines.append("Grouped Format Clamp Results")
    lines.append("=" * 70)
    lines.append(
        f"Layer {results['layer']}, head {results['head']}, "
        f"{results['n_prompts']} prompts x {results['gen_len']} tokens"
    )
    lines.append(
        f"Family={','.join(results['family_properties'])} "
        f"main_property={results['main_property']}"
    )
    lines.append(
        f"Dose-ordered main shift: {summary['dose_ordered_main_shift']} | "
        f"best shift={summary['best_main_property_shift']:+.4f} | "
        f"best d={summary['best_main_property_cohens_d']:+.3f}"
    )
    lines.append(
        f"Random mean={summary['random_main_property_mean_shift']:+.4f} "
        f"(std={summary['random_main_property_std_shift']:.4f}) | "
        f"ratio={summary['ratio_vs_random_mean']:.2f}x"
    )
    lines.append("")
    lines.append(f"{'Dose':>6} {'Shift':>9} {'d':>7} {'p':>8}")
    lines.append("-" * 36)
    for row in results["dose_results"]:
        main_stats = row["property_shifts"][results["main_property"]]
        lines.append(
            f"{row['dose']:>6} {main_stats['mean_shift']:>+9.4f} "
            f"{main_stats['cohens_d']:>+7.3f} {main_stats['p_value']:>8.4f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def run_causal_clamp(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    texts: list[str],
    states: torch.Tensor,
    layer_idx: int,
    sae,
    sae_type: str,
    target_features: list[dict],
    head_idx: int = 0,
    prompt_len: int = 512,
    gen_len: int = 256,
    n_prompts: int = 50,
    activation_percentile: float = 95.0,
    device: str = "cuda",
) -> dict:
    """Run the causal clamping experiment.

    For each target feature:
      1. Compute 95th percentile activation from training data
      2. For each prompt, generate with and without clamping
      3. Compute text properties on both generations
      4. Report mean shift, Cohen's d, paired t-test

    Args:
        model: CausalLM
        tokenizer: tokenizer
        corpus_batches: tokenized sequences (batch, seq_len)
        texts: raw text strings
        states: (N, d_k, d_v) training states for percentile computation
        layer_idx: GDN layer index
        sae: trained SAE
        sae_type: SAE type string
        target_features: list of {feature_idx, property, rho}
        head_idx: which head
        prompt_len: tokens to use as prompt
        gen_len: tokens to generate
        n_prompts: number of prompts to test
        activation_percentile: percentile for clamp value
        device: compute device

    Returns:
        dict with per-feature results
    """
    gdn_layers = _get_gdn_layer_indices(model)

    # Flatten batches
    all_seqs = []
    for batch in corpus_batches:
        for i in range(batch.shape[0]):
            all_seqs.append(batch[i:i + 1])
    all_seqs = all_seqs[:n_prompts]
    actual_prompts = len(all_seqs)
    print(f"Causal clamping: {len(target_features)} features, "
          f"{actual_prompts} prompts, prompt_len={prompt_len}, gen_len={gen_len}")

    # Compute activation percentiles for all target features
    feature_indices = [f["feature_idx"] for f in target_features]
    print(f"Computing {activation_percentile}th percentile activations...")
    t0 = time.time()
    percentiles = compute_activation_percentiles(
        sae, states, feature_indices,
        percentile=activation_percentile,
    )
    print(f"  Percentiles computed in {time.time() - t0:.1f}s")
    for fi, pval in percentiles.items():
        print(f"  Feature {fi}: p{activation_percentile:.0f} = {pval:.4f}")

    feature_results = []

    for feat_info in target_features:
        feat_idx = feat_info["feature_idx"]
        feat_prop = feat_info["property"]
        feat_rho = feat_info["rho"]
        clamp_val = percentiles[feat_idx]

        print(f"\nFeature {feat_idx} ({feat_prop}, rho={feat_rho:.3f}, "
              f"clamp={clamp_val:.4f})")

        baseline_props = []
        clamped_props = []
        t0_feat = time.time()

        for pi, seq in enumerate(all_seqs):
            seq = seq.to(device)

            # Baseline generation (no clamping)
            baseline_text = generate_with_clamp(
                model, tokenizer, seq, prompt_len, gen_len,
                gdn_layers, layer_idx, sae, sae_type, head_idx,
                clamp_feature=None, clamp_value=None,
            )

            # Clamped generation
            clamped_text = generate_with_clamp(
                model, tokenizer, seq, prompt_len, gen_len,
                gdn_layers, layer_idx, sae, sae_type, head_idx,
                clamp_feature=feat_idx, clamp_value=clamp_val,
            )

            bp = compute_text_properties(baseline_text)
            cp = compute_text_properties(clamped_text)
            baseline_props.append(bp[feat_prop])
            clamped_props.append(cp[feat_prop])

            if (pi + 1) % 10 == 0:
                mean_shift = np.mean(np.array(clamped_props) - np.array(baseline_props))
                print(f"  [{pi+1}/{actual_prompts}] running mean shift: {mean_shift:+.4f}")

            torch.cuda.empty_cache()

        feat_time = time.time() - t0_feat

        baseline_arr = np.array(baseline_props)
        clamped_arr = np.array(clamped_props)
        diffs = clamped_arr - baseline_arr

        mean_baseline = float(np.mean(baseline_arr))
        mean_clamped = float(np.mean(clamped_arr))
        mean_shift = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-8

        # Cohen's d (paired)
        cohens_d = mean_shift / max(std_diff, 1e-8)

        # Paired t-test
        if len(diffs) > 1 and std_diff > 1e-8:
            t_stat, p_value = scipy_stats.ttest_rel(clamped_arr, baseline_arr)
        else:
            t_stat, p_value = 0.0, 1.0

        # Direction check: does shift align with rho sign?
        # Positive rho means higher activation -> higher property value
        # Clamping HIGH should increase property value if rho > 0
        direction_aligned = (feat_rho > 0 and mean_shift > 0) or (feat_rho < 0 and mean_shift < 0)

        result = {
            "feature_idx": feat_idx,
            "property": feat_prop,
            "rho": feat_rho,
            "clamp_value": clamp_val,
            "n_prompts": actual_prompts,
            "mean_baseline": mean_baseline,
            "mean_clamped": mean_clamped,
            "mean_shift": mean_shift,
            "std_shift": std_diff,
            "cohens_d": cohens_d,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "direction_aligned": direction_aligned,
            "significant_p05": p_value < 0.05,
            "significant_p01": p_value < 0.01,
            "time_s": round(feat_time, 1),
        }
        feature_results.append(result)

        print(f"  baseline={mean_baseline:.4f}, clamped={mean_clamped:.4f}, "
              f"shift={mean_shift:+.4f}")
        print(f"  Cohen's d={cohens_d:+.3f}, t={t_stat:.3f}, p={p_value:.4f}, "
              f"aligned={direction_aligned}")

    # Summary statistics
    n_significant = sum(1 for r in feature_results if r["significant_p05"])
    n_aligned = sum(1 for r in feature_results if r["direction_aligned"])
    n_both = sum(1 for r in feature_results
                 if r["significant_p05"] and r["direction_aligned"])
    mean_abs_d = float(np.mean([abs(r["cohens_d"]) for r in feature_results]))

    output = {
        "layer": layer_idx,
        "head": head_idx,
        "sae_type": sae_type,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "n_prompts": actual_prompts,
        "n_features": len(target_features),
        "activation_percentile": activation_percentile,
        "summary": {
            "n_significant_p05": n_significant,
            "n_direction_aligned": n_aligned,
            "n_significant_and_aligned": n_both,
            "fraction_causal": n_both / max(len(feature_results), 1),
            "mean_abs_cohens_d": mean_abs_d,
        },
        "feature_results": feature_results,
    }

    return output


# Report formatting


def format_causal_clamp_report(results: dict) -> str:
    """Format causal clamping results as a readable report."""
    lines = []
    lines.append("Causal Feature Clamping Results")
    lines.append("=" * 70)

    s = results["summary"]
    lines.append(f"Layer {results['layer']}, head {results['head']}, "
                 f"{results['n_prompts']} prompts x {results['gen_len']} tokens")
    lines.append(f"Features tested: {results['n_features']}")
    lines.append(f"Significant (p<0.05): {s['n_significant_p05']}/{results['n_features']}")
    lines.append(f"Direction-aligned: {s['n_direction_aligned']}/{results['n_features']}")
    lines.append(f"Causal (sig + aligned): {s['n_significant_and_aligned']}/{results['n_features']} "
                 f"({s['fraction_causal']*100:.0f}%)")
    lines.append(f"Mean |Cohen's d|: {s['mean_abs_cohens_d']:.3f}")

    lines.append("")
    lines.append(f"{'Feature':>8} {'Property':<22} {'rho':>6} "
                 f"{'Shift':>8} {'d':>7} {'p':>8} {'Aligned':>8}")
    lines.append("-" * 75)

    for r in sorted(results["feature_results"],
                    key=lambda x: abs(x["cohens_d"]), reverse=True):
        aligned_str = "yes" if r["direction_aligned"] else "no"
        sig_str = "*" if r["significant_p05"] else ""
        lines.append(
            f"{r['feature_idx']:>8} {r['property']:<22} {r['rho']:>+.3f} "
            f"{r['mean_shift']:>+.4f} {r['cohens_d']:>+.3f} "
            f"{r['p_value']:>8.4f}{sig_str} {aligned_str:>7}"
        )

    return "\n".join(lines)

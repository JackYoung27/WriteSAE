#!/usr/bin/env python3
"""Feature ablation vs random direction: causal evidence that SAE features carry real info.

For each alive SAE feature:
  1. Zero that feature in the SAE reconstruction
  2. Measure PPL on N corpus sequences (suffix perplexity with patched state)
  3. Record delta_loss = ablated_loss - baseline_loss

For M random unit vectors in R^(d_k * d_v):
  1. Project state onto that direction, zero the component
  2. Same PPL measurement
  3. Record delta_loss

If alive features cause larger |delta_loss| than random directions, the SAE has
discovered meaningful structure in the state, not arbitrary directions.

Statistical test: Mann-Whitney U on |delta_loss| distributions (alive vs random),
plus Cohen's d effect size.
"""

import time

import numpy as np
import torch
from tqdm import tqdm

from experiments.ablations.circuit_ablation import reconstruct_with_ablation
from experiments.analysis.evaluate_downstream import (
    _get_gdn_layer_indices,
    _patch_gdn_initial_states,
)


@torch.no_grad()
def find_alive_feature_indices(
    sae,
    states: torch.Tensor,
    sae_type: str,
    batch_size: int = 128,
    device: str = "cuda",
) -> list[int]:
    """Return feature indices that activate at least once on the provided states."""
    was_training = sae.training
    sae.eval()

    alive_mask = None
    n_states = states.shape[0]

    for start in range(0, n_states, batch_size):
        batch = states[start:start + batch_size].to(device)
        if sae_type == "flat":
            coeffs = sae.encode(batch.reshape(batch.shape[0], -1))
        else:
            coeffs = sae.encode(batch)
        batch_alive = (coeffs.abs() > 0).any(dim=0)
        alive_mask = batch_alive if alive_mask is None else (alive_mask | batch_alive)

    if was_training:
        sae.train()

    if alive_mask is None:
        return []
    return torch.nonzero(alive_mask, as_tuple=False).squeeze(1).cpu().tolist()


@torch.no_grad()
def compute_ppl_with_random_direction_ablation(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int,
    random_direction: torch.Tensor,
    head_idx: int = 0,
) -> dict:
    """Measure PPL after projecting out a random direction from the raw state.

    Args:
        model: CausalLM
        input_ids: (1, seq_len)
        split_pos: prefix/suffix boundary
        gdn_layer_indices: all GDN layers
        target_layer_idx: layer to ablate
        random_direction: (d_k * d_v,) unit vector
        head_idx: head to ablate

    Returns:
        dict with loss, perplexity, n_tokens
    """
    device = input_ids.device

    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos:]

    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    gdn_states = {}
    for idx in gdn_layer_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            gdn_states[idx] = layer_cache.recurrent_states.clone()

    if target_layer_idx in gdn_states:
        state = gdn_states[target_layer_idx]
        d_k, d_v = state.shape[2], state.shape[3]
        head_state = state[0, head_idx].float()  # (d_k, d_v)
        flat = head_state.reshape(-1)  # (d_k * d_v,)

        # Project out the random direction: x' = x - (x . d) * d
        proj = torch.dot(flat, random_direction)
        flat_ablated = flat - proj * random_direction
        state[0, head_idx] = flat_ablated.reshape(d_k, d_v).to(state.dtype)

    with _patch_gdn_initial_states(model, gdn_layer_indices, gdn_states):
        suffix_out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=False,
            labels=suffix,
        )

    return {
        "loss": suffix_out.loss.item(),
        "perplexity": float(torch.exp(suffix_out.loss).item()),
        "n_tokens": suffix.shape[1] - 1,
    }


@torch.no_grad()
def compute_ppl_baseline_sae(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int,
    sae,
    sae_type: str,
    head_idx: int = 0,
) -> dict:
    """Baseline PPL with SAE reconstruction (no feature ablation)."""
    from circuit_ablation import compute_ppl_with_feature_ablation
    return compute_ppl_with_feature_ablation(
        model, input_ids, split_pos, gdn_layer_indices,
        target_layer_idx=target_layer_idx,
        sae=sae, sae_type=sae_type,
        ablate_features=[],  # no ablation
        head_idx=head_idx,
    )


@torch.no_grad()
def compute_ppl_no_sae(
    model,
    input_ids: torch.Tensor,
    split_pos: int,
    gdn_layer_indices: list[int],
    target_layer_idx: int,
    head_idx: int = 0,
) -> dict:
    """Baseline PPL with original state (no SAE, no ablation)."""
    device = input_ids.device

    prefix = input_ids[:, :split_pos]
    suffix = input_ids[:, split_pos:]

    prefix_out = model(input_ids=prefix, use_cache=True)
    cache = prefix_out.past_key_values

    gdn_states = {}
    for idx in gdn_layer_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, "recurrent_states") and layer_cache.recurrent_states is not None:
            gdn_states[idx] = layer_cache.recurrent_states.clone()

    with _patch_gdn_initial_states(model, gdn_layer_indices, gdn_states):
        suffix_out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=False,
            labels=suffix,
        )

    return {
        "loss": suffix_out.loss.item(),
        "perplexity": float(torch.exp(suffix_out.loss).item()),
        "n_tokens": suffix.shape[1] - 1,
    }


def run_feature_vs_random_ablation(
    model,
    corpus_seqs: list[torch.Tensor],
    layer_idx: int,
    sae,
    sae_type: str,
    alive_feature_indices: list[int],
    head_idx: int = 0,
    split_fraction: float = 0.5,
    n_random: int = 100,
    d_k: int = 128,
    d_v: int = 128,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Run alive-feature ablation vs random-direction ablation.

    Args:
        model: CausalLM
        corpus_seqs: list of (1, seq_len) tensors
        layer_idx: GDN layer
        sae: trained SAE
        sae_type: SAE type
        alive_feature_indices: indices of alive features in the SAE
        head_idx: head the SAE was trained on
        split_fraction: prefix/suffix split
        n_random: number of random directions to test
        d_k, d_v: state dimensions
        device: cuda
        seed: RNG seed for random directions

    Returns:
        dict with distributions, statistics, per-feature results
    """
    from circuit_ablation import compute_ppl_with_feature_ablation

    seq_len = corpus_seqs[0].shape[1]
    split_pos = int(seq_len * split_fraction)
    gdn_layers = _get_gdn_layer_indices(model)
    n_seqs = len(corpus_seqs)

    print(f"Feature vs Random ablation: {len(alive_feature_indices)} alive features, "
          f"{n_random} random directions, {n_seqs} sequences")

    # Phase 1: Baseline PPL per sequence (SAE reconstruction, no ablation)
    print(f"\nPhase 1: Baseline PPL ({n_seqs} sequences)")
    baseline_losses = []
    baseline_tokens = []
    t0 = time.time()

    for i, seq in enumerate(tqdm(corpus_seqs, desc="Baseline")):
        seq = seq.to(device)
        r = compute_ppl_baseline_sae(
            model, seq, split_pos, gdn_layers,
            target_layer_idx=layer_idx,
            sae=sae, sae_type=sae_type, head_idx=head_idx,
        )
        baseline_losses.append(r["loss"])
        baseline_tokens.append(r["n_tokens"])
        if (i + 1) % 10 == 0 or (i + 1) == n_seqs:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            remaining = (n_seqs - i - 1) / max(rate, 1e-9)
            print(f"  Baseline [{i+1}/{n_seqs}] "
                  f"loss={r['loss']:.6f} ppl={r['perplexity']:.4f} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    baseline_time = time.time() - t0
    total_tokens = sum(baseline_tokens)
    baseline_avg_loss = sum(l * t for l, t in zip(baseline_losses, baseline_tokens)) / total_tokens
    baseline_ppl = float(np.exp(baseline_avg_loss))
    print(f"  Baseline PPL: {baseline_ppl:.4f} (loss={baseline_avg_loss:.6f}, {baseline_time:.1f}s)")

    # Phase 2: Per-alive-feature ablation
    print(f"\nPhase 2: Ablating {len(alive_feature_indices)} alive features")
    feature_delta_losses = []
    feature_results = []
    t0 = time.time()

    for fi, feat_idx in enumerate(alive_feature_indices):
        losses_ablated = []
        for seq in corpus_seqs:
            seq = seq.to(device)
            r = compute_ppl_with_feature_ablation(
                model, seq, split_pos, gdn_layers,
                target_layer_idx=layer_idx,
                sae=sae, sae_type=sae_type,
                ablate_features=[feat_idx],
                head_idx=head_idx,
            )
            losses_ablated.append(r["loss"])

        avg_ablated_loss = sum(
            l * t for l, t in zip(losses_ablated, baseline_tokens)
        ) / total_tokens
        delta_loss = avg_ablated_loss - baseline_avg_loss
        feature_delta_losses.append(delta_loss)
        feature_results.append({
            "feature_idx": feat_idx,
            "avg_ablated_loss": avg_ablated_loss,
            "delta_loss": delta_loss,
            "abs_delta_loss": abs(delta_loss),
        })

        if (fi + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            remaining = (len(alive_feature_indices) - fi - 1) / rate
            print(f"  [{fi+1}/{len(alive_feature_indices)}] "
                  f"F{feat_idx} dL={delta_loss:+.6f} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
            torch.cuda.empty_cache()

    feature_time = time.time() - t0
    print(f"  Feature ablation: {len(alive_feature_indices)} features in {feature_time:.1f}s")

    # Phase 3: Random direction ablation
    print(f"\nPhase 3: Ablating {n_random} random directions")
    rng = np.random.RandomState(seed)
    random_delta_losses = []
    random_results = []
    t0 = time.time()

    for ri in range(n_random):
        # Generate random unit vector in R^(d_k * d_v)
        vec = rng.randn(d_k * d_v).astype(np.float32)
        vec /= np.linalg.norm(vec)
        direction = torch.from_numpy(vec).to(device)

        losses_ablated = []
        for seq in corpus_seqs:
            seq = seq.to(device)
            r = compute_ppl_with_random_direction_ablation(
                model, seq, split_pos, gdn_layers,
                target_layer_idx=layer_idx,
                random_direction=direction,
                head_idx=head_idx,
            )
            losses_ablated.append(r["loss"])

        avg_ablated_loss = sum(
            l * t for l, t in zip(losses_ablated, baseline_tokens)
        ) / total_tokens
        delta_loss = avg_ablated_loss - baseline_avg_loss
        random_delta_losses.append(delta_loss)
        random_results.append({
            "random_idx": ri,
            "delta_loss": delta_loss,
            "abs_delta_loss": abs(delta_loss),
        })

        if (ri + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (ri + 1) / elapsed
            remaining = (n_random - ri - 1) / rate
            print(f"  [{ri+1}/{n_random}] "
                  f"dL={delta_loss:+.6f} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
            torch.cuda.empty_cache()

    random_time = time.time() - t0
    print(f"  Random ablation: {n_random} directions in {random_time:.1f}s")

    # Phase 4: Statistical comparison
    from scipy import stats

    feat_abs = np.array([abs(d) for d in feature_delta_losses])
    rand_abs = np.array([abs(d) for d in random_delta_losses])

    # Mann-Whitney U test on |delta_loss|
    u_stat, u_pval = stats.mannwhitneyu(feat_abs, rand_abs, alternative="greater")

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(feat_abs) - 1) * feat_abs.std(ddof=1)**2 +
         (len(rand_abs) - 1) * rand_abs.std(ddof=1)**2) /
        (len(feat_abs) + len(rand_abs) - 2)
    )
    cohens_d = (feat_abs.mean() - rand_abs.mean()) / max(pooled_std, 1e-12)

    # Welch's t-test
    t_stat, t_pval = stats.ttest_ind(feat_abs, rand_abs, equal_var=False, alternative="greater")

    # Signed delta losses (not abs) for understanding direction
    feat_signed = np.array(feature_delta_losses)
    rand_signed = np.array(random_delta_losses)

    summary = {
        "n_alive_features": len(alive_feature_indices),
        "n_random_directions": n_random,
        "n_sequences": n_seqs,
        "baseline_ppl": baseline_ppl,
        "baseline_avg_loss": baseline_avg_loss,
        "alive_features": {
            "mean_abs_delta_loss": float(feat_abs.mean()),
            "median_abs_delta_loss": float(np.median(feat_abs)),
            "std_abs_delta_loss": float(feat_abs.std()),
            "mean_signed_delta_loss": float(feat_signed.mean()),
            "max_abs_delta_loss": float(feat_abs.max()),
            "pct_positive_delta": float((feat_signed > 0).mean() * 100),
        },
        "random_directions": {
            "mean_abs_delta_loss": float(rand_abs.mean()),
            "median_abs_delta_loss": float(np.median(rand_abs)),
            "std_abs_delta_loss": float(rand_abs.std()),
            "mean_signed_delta_loss": float(rand_signed.mean()),
            "max_abs_delta_loss": float(rand_abs.max()),
            "pct_positive_delta": float((rand_signed > 0).mean() * 100),
        },
        "comparison": {
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_pval),
            "welch_t": float(t_stat),
            "welch_p": float(t_pval),
            "cohens_d": float(cohens_d),
            "ratio_mean_abs": float(feat_abs.mean() / max(rand_abs.mean(), 1e-12)),
        },
        "timing": {
            "baseline_s": round(baseline_time, 1),
            "feature_ablation_s": round(feature_time, 1),
            "random_ablation_s": round(random_time, 1),
        },
    }

    feature_results.sort(key=lambda x: x["abs_delta_loss"], reverse=True)

    print(f"\n{'='*70}")
    print(f"RESULTS: Feature Ablation vs Random Directions")
    print(f"{'='*70}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Alive features ({len(alive_feature_indices)}):")
    print(f"  Mean |dL|: {feat_abs.mean():.6f}")
    print(f"  Median |dL|: {np.median(feat_abs):.6f}")
    print(f"  % positive dL: {(feat_signed > 0).mean()*100:.1f}%")
    print(f"Random directions ({n_random}):")
    print(f"  Mean |dL|: {rand_abs.mean():.6f}")
    print(f"  Median |dL|: {np.median(rand_abs):.6f}")
    print(f"  % positive dL: {(rand_signed > 0).mean()*100:.1f}%")
    print(f"Comparison:")
    print(f"  Ratio (alive/random): {feat_abs.mean() / max(rand_abs.mean(), 1e-12):.2f}x")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Mann-Whitney p: {u_pval:.2e}")
    print(f"  Welch's t p: {t_pval:.2e}")
    print(f"\nTop-10 most impactful features:")
    for r in feature_results[:10]:
        print(f"  F{r['feature_idx']:>5}: dL={r['delta_loss']:+.6f}")

    return {
        "summary": summary,
        "feature_results": feature_results,
        "random_results": random_results,
        "alive_feature_indices": alive_feature_indices,
    }

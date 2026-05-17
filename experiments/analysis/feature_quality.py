"""Per-sample MSE distribution and per-feature specificity analysis.

Two questions this answers:
1. Does bilinear's larger alive dictionary (2048 vs ~298) produce tighter
   per-sample MSE? Lower worst-case = better coverage of the input space.
2. Are bilinear features more specialized? Each should fire on ~1.6% of inputs
   (32/2048) vs flat's ~10.7% (32/298). Lower frequency = more specific atoms.

Usage: called from run_modal.py via --stage feature-quality.
"""
from __future__ import annotations

import numpy as np
import torch


def gini_coefficient(freq: np.ndarray) -> float:
    """Gini coefficient of a frequency array. 0=uniform, 1=concentrated."""
    if len(freq) == 0 or freq.sum() == 0:
        return 0.0
    sorted_freq = np.sort(freq)
    n = len(sorted_freq)
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_freq).sum() / (n * sorted_freq.sum())) - (n + 1) / n)


def analyze_sae(
    sae: torch.nn.Module,
    val_data: torch.Tensor,
    k: int,
    is_flat: bool,
    batch_size: int = 512,
) -> dict:
    """Run full feature quality analysis on one SAE.

    Args:
        sae: trained SAE model (on GPU, eval mode)
        val_data: validation states, shape (N, d_k, d_v) as float32
        k: top-k sparsity
        is_flat: whether this is a FlatSAE (needs reshaping)
        batch_size: forward pass batch size

    Returns:
        dict with all computed metrics
    """
    device = next(sae.parameters()).device
    n_samples = val_data.shape[0]
    n_features = int(sae.n_features)

    # Accumulators
    per_sample_mse = np.zeros(n_samples, dtype=np.float64)
    feature_fire_count = np.zeros(n_features, dtype=np.int64)
    feature_mse_sum = np.zeros(n_features, dtype=np.float64)
    feature_mse_count = np.zeros(n_features, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = val_data[start:end].to(device)
            bs = batch.shape[0]

            inp = batch.reshape(bs, -1) if is_flat else batch
            out = sae(inp)

            # Per-sample MSE (mean over dimensions, not reduced over batch)
            x_flat = batch.reshape(bs, -1)
            r_flat = out.reconstruction.reshape(bs, -1)
            sample_mse = ((r_flat - x_flat) ** 2).mean(dim=-1).cpu().numpy()
            per_sample_mse[start:end] = sample_mse

            # Which features fired in each sample (top-k nonzero entries)
            coeffs = out.coefficients  # (bs, n_features)
            active_mask = (coeffs.abs() > 0)  # (bs, n_features)

            # Per-feature fire count
            fire_count_batch = active_mask.sum(dim=0).cpu().numpy()
            feature_fire_count += fire_count_batch

            # Per-feature conditional MSE:
            # for each feature, accumulate the MSE of samples where it fired
            # Expand sample_mse to (bs, n_features) and mask
            sample_mse_t = torch.from_numpy(sample_mse).to(device).unsqueeze(1)  # (bs, 1)
            masked_mse = (active_mask.float() * sample_mse_t)  # (bs, nf)
            feature_mse_sum += masked_mse.sum(dim=0).cpu().numpy()
            feature_mse_count += active_mask.sum(dim=0).cpu().numpy()

            del batch, inp, out, coeffs, active_mask, masked_mse, sample_mse_t
            torch.cuda.empty_cache()

    # Per-sample MSE distribution
    mse_stats = {
        "mean": float(np.mean(per_sample_mse)),
        "median": float(np.median(per_sample_mse)),
        "std": float(np.std(per_sample_mse)),
        "p90": float(np.percentile(per_sample_mse, 90)),
        "p95": float(np.percentile(per_sample_mse, 95)),
        "p99": float(np.percentile(per_sample_mse, 99)),
        "max": float(np.max(per_sample_mse)),
        "min": float(np.min(per_sample_mse)),
        "cv": float(np.std(per_sample_mse) / max(np.mean(per_sample_mse), 1e-12)),
    }

    # Feature frequency (fraction of val samples each feature fires on)
    feature_freq = feature_fire_count / max(n_samples, 1)

    # Alive = fired at least once
    alive_mask = feature_fire_count > 0
    n_alive = int(alive_mask.sum())
    n_dead = n_features - n_alive

    alive_freq = feature_freq[alive_mask]
    freq_stats = {
        "n_alive": n_alive,
        "n_dead": n_dead,
        "alive_pct": float(n_alive / n_features * 100),
        "dead_pct": float(n_dead / n_features * 100),
    }
    if n_alive > 0:
        freq_stats.update({
            "mean_freq": float(np.mean(alive_freq)),
            "median_freq": float(np.median(alive_freq)),
            "std_freq": float(np.std(alive_freq)),
            "min_freq": float(np.min(alive_freq)),
            "max_freq": float(np.max(alive_freq)),
            "expected_freq": float(k / n_alive),
        })
    else:
        freq_stats.update({
            "mean_freq": 0.0, "median_freq": 0.0, "std_freq": 0.0,
            "min_freq": 0.0, "max_freq": 0.0, "expected_freq": 0.0,
        })

    # Per-feature conditional MSE (avg MSE on samples where feature is active)
    cond_mse = np.zeros(n_features, dtype=np.float64)
    valid_cond = feature_mse_count > 0
    cond_mse[valid_cond] = feature_mse_sum[valid_cond] / feature_mse_count[valid_cond]
    alive_cond_mse = cond_mse[alive_mask]
    cond_mse_stats = {}
    if n_alive > 0:
        cond_mse_stats = {
            "mean": float(np.mean(alive_cond_mse)),
            "median": float(np.median(alive_cond_mse)),
            "std": float(np.std(alive_cond_mse)),
            "p90": float(np.percentile(alive_cond_mse, 90)),
            "p95": float(np.percentile(alive_cond_mse, 95)),
            "max": float(np.max(alive_cond_mse)),
        }

    # Gini coefficient of feature usage
    gini = gini_coefficient(feature_fire_count[alive_mask].astype(np.float64))

    return {
        "mse_distribution": mse_stats,
        "feature_frequency": freq_stats,
        "conditional_mse": cond_mse_stats,
        "gini_coefficient": float(gini),
        "n_samples": n_samples,
        "n_features": n_features,
        "k": k,
    }


def format_summary_table(results: dict[str, dict]) -> str:
    """Format results as a readable comparison table."""
    lines = []
    lines.append(f"{'SAE':<40} {'MSE':>8} {'p95':>8} {'p99':>8} {'CV':>6} "
                 f"{'Alive':>6} {'Dead%':>6} {'Freq':>7} {'Gini':>5} {'CondMSE':>8}")
    lines.append("-" * 115)
    for tag, r in sorted(results.items()):
        mse = r["mse_distribution"]
        freq = r["feature_frequency"]
        cond = r.get("conditional_mse", {})
        lines.append(
            f"{tag:<40} "
            f"{mse['mean']:>8.5f} "
            f"{mse['p95']:>8.5f} "
            f"{mse['p99']:>8.5f} "
            f"{mse['cv']:>6.3f} "
            f"{freq['n_alive']:>6d} "
            f"{freq['dead_pct']:>5.1f}% "
            f"{freq.get('mean_freq', 0):>7.4f} "
            f"{r['gini_coefficient']:>5.3f} "
            f"{cond.get('mean', 0):>8.5f}"
        )
    return "\n".join(lines)

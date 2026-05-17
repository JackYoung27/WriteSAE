#!/usr/bin/env python3
"""Feature interpretability analysis for matrix SAEs.

For each SAE checkpoint, produces per-feature activation statistics, top
contexts, decoder factors, and outer-product data.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae import FlatSAE, MatrixSAE, BilinearMatrixSAE, build_sae_from_config  # noqa: E402


class FactorAtomMeta(TypedDict):
    rank: int
    method: str
    rank1_retained_energy: float


class TopContext(TypedDict):
    rank: int
    sample_idx: int
    activation: float
    text: str


class FeatureBase(TypedDict):
    feature_idx: int
    frequency: float
    mean_activation: float
    max_activation: float
    top_contexts: list[TopContext]


class FeatureEntry(FeatureBase, total=False):
    v_dec: list[float]
    w_dec: list[float]
    decoder_rank: int
    decoder_vector_method: str
    decoder_rank1_retained_energy: float
    v_norm: float
    w_norm: float
    v_sparsity: float
    w_sparsity: float
    v_dec_components: list[list[float]]
    w_dec_components: list[list[float]]
    v_enc: list[float]
    w_enc: list[float]
    encoder_rank: int
    encoder_vector_method: str
    encoder_rank1_retained_energy: float
    v_enc_components: list[list[float]]
    w_enc_components: list[list[float]]
    v_enc_dec_cosine: float
    w_enc_dec_cosine: float
    enc_dec_atom_cosine: float


class SummaryBase(TypedDict):
    sae_type: str
    sae_path: str
    layer: int
    head: int
    n_samples: int
    n_features_total: int
    n_features_alive: int
    dead_fraction: float
    mean_l0: float
    has_rank1_structure: bool


class SummaryEntry(SummaryBase, total=False):
    mean_v_enc_dec_cosine: float
    mean_w_enc_dec_cosine: float
    mean_enc_dec_atom_cosine: float


class ComparisonBase(TypedDict):
    sae_type: str
    layer: int
    n_alive: int
    dead_frac: float
    mean_l0: float
    has_rank1: bool


class ComparisonRow(ComparisonBase, total=False):
    n_features: int | None
    expansion_factor: int | None
    k: int | None
    seed: int | None
    top_feat_mean_freq: float
    top_feat_mean_max: float
    selectivity_ratio: float


class ComparisonResult(TypedDict):
    models: list[ComparisonRow]


def load_sae(path: str) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load SAE checkpoint. Returns (model, config_dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.eval(), {}
    cfg = ckpt.get("config", {})
    sd = ckpt["model_state_dict"]
    model = build_sae_from_config(cfg, state_dict=sd)
    model.load_state_dict(sd)
    return model.eval(), cfg


def load_states(data_dir: str, layer: int, head: int, n_samples: int | None = None) -> torch.Tensor:
    """Load GDN states from memmap. Returns (N, d_k, d_v) float32 tensor."""
    npy_path = os.path.join(data_dir, f"layer_{layer}", f"head_{head}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"No state data at {npy_path}")
    raw = np.load(npy_path, mmap_mode="r")
    if n_samples is not None:
        raw = raw[:n_samples]
    return torch.from_numpy(raw.astype(np.float32))


def load_texts(data_dir: str) -> list[str]:
    """Load text sequences. Returns empty list if not available."""
    tp = os.path.join(data_dir, "texts.json")
    if os.path.exists(tp):
        return json.load(open(tp))
    return []


@torch.no_grad()
def _encode_sae(sae: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return sae.encode(batch)  # type: ignore[operator]


def _factor_atom(
    v_factors: np.ndarray,
    w_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, FactorAtomMeta]:
    if v_factors.ndim == 1:
        return v_factors, w_factors, {"rank": 1, "method": "direct", "rank1_retained_energy": 1.0}

    atom = np.einsum("rk,rv->kv", v_factors, w_factors)
    u, s, vt = np.linalg.svd(atom, full_matrices=False)
    top_sv = float(s[0]) if len(s) else 0.0
    scale = np.sqrt(top_sv) if top_sv > 0 else 0.0
    total_energy = float(np.square(s).sum())
    retained = float((s[0] ** 2) / total_energy) if total_energy > 0 else 1.0
    return (
        u[:, 0] * scale,
        vt[0] * scale,
        {
            "rank": int(v_factors.shape[0]),
            "method": "svd_rank1_approx",
            "rank1_retained_energy": retained,
        },
    )


def _flatten_atom(v_factors: np.ndarray, w_factors: np.ndarray) -> np.ndarray:
    if v_factors.ndim == 1:
        return np.outer(v_factors, w_factors).reshape(-1)
    return np.einsum("rk,rv->kv", v_factors, w_factors).reshape(-1)


@torch.no_grad()
def compute_activations(
    sae: torch.nn.Module, states: torch.Tensor, batch_size: int = 512,
) -> torch.Tensor:
    """Encode all states through the SAE. Returns (N, n_features) activation matrix."""
    device = next(sae.parameters()).device
    parts = []
    for i in range(0, len(states), batch_size):
        batch = states[i : i + batch_size].to(device)
        acts = _encode_sae(sae, batch)
        parts.append(acts.cpu())
    return torch.cat(parts, dim=0)


def activation_stats(acts: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-feature statistics from the activation matrix."""
    a = acts.numpy()
    return {
        "mean": a.mean(0),
        "frequency": (a > 0).astype(np.float32).mean(0),
        "max": a.max(0),
        "std": a.std(0),
    }


def top_features_by_activity(acts: torch.Tensor, n: int = 20) -> list[int]:
    """Return indices of the N most frequently active features."""
    freq = (acts > 0).float().mean(0)
    return torch.topk(freq, min(n, len(freq))).indices.tolist()


def top_activating_contexts(
    acts: torch.Tensor, texts: list[str], feature_idx: int, n: int = 10,
) -> list[TopContext]:
    """Find the N samples where feature_idx activates most strongly."""
    col = acts[:, feature_idx]
    active_idx = torch.nonzero(col > 0, as_tuple=True)[0]
    if active_idx.numel() == 0:
        return []
    active_vals = col[active_idx]
    k = min(n, int(active_idx.numel()))
    topk = torch.topk(active_vals, k)
    results = []
    for rank, (val, idx) in enumerate(zip(topk.values, topk.indices)):
        sample_idx = int(active_idx[idx])
        results.append({
            "rank": rank,
            "sample_idx": sample_idx,
            "activation": float(val),
            "text": texts[sample_idx][:500] if sample_idx < len(texts) else "",
        })
    return results


def extract_feature_vectors(sae: torch.nn.Module) -> dict[str, np.ndarray | None]:
    """Extract raw decoder factors from the SAE.

    Returns dict with keys:
      'v_dec_factors': (n_features, rank, d_k) or (n_features, d_k) or None
      'w_dec_factors': (n_features, rank, d_v) or (n_features, d_v) or None
      'decoder_weight': (d_in, n_features) for flat SAEs, else None
    """
    if isinstance(sae, BilinearMatrixSAE):
        return {
            "v_dec_factors": sae.V_dec.detach().cpu().numpy(),
            "w_dec_factors": sae.W_dec.detach().cpu().numpy(),
            "v_enc_factors": sae._v_enc.detach().cpu().numpy(),
            "w_enc_factors": sae._w_enc.detach().cpu().numpy(),
            "decoder_weight": None,
        }
    elif isinstance(sae, MatrixSAE):
        return {
            "v_dec_factors": sae.V.detach().cpu().numpy(),
            "w_dec_factors": sae.W.detach().cpu().numpy(),
            "v_enc_factors": None,
            "w_enc_factors": None,
            "decoder_weight": None,
        }
    elif isinstance(sae, FlatSAE):
        return {
            "v_dec_factors": None,
            "w_dec_factors": None,
            "v_enc_factors": None,
            "w_enc_factors": None,
            "decoder_weight": sae.decoder.weight.detach().cpu().numpy(),
        }
    return {
        "v_dec_factors": None,
        "w_dec_factors": None,
        "v_enc_factors": None,
        "w_enc_factors": None,
        "decoder_weight": None,
    }


def run_interpretability(
    sae_path: str,
    data_dir: str,
    layer: int,
    head: int = 0,
    n_top_features: int = 20,
    n_top_contexts: int = 10,
    n_samples: int | None = None,
    output_dir: str = "interpret_output",
    device: str = "cpu",
    feature_indices: list[int] | None = None,
) -> dict[str, object]:
    """Full interpretability pipeline for one SAE checkpoint.

    Args:
        feature_indices: If provided, analyze these specific features instead
            of selecting the top-N by activation frequency.

    Returns a dict with all analysis results, also saved as JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading SAE from {sae_path}")
    sae, cfg = load_sae(sae_path)
    sae = sae.to(device)
    sae_type = cfg.get("sae_type", type(sae).__name__)

    print(f"Loading states: layer={layer}, head={head}")
    states = load_states(data_dir, layer, head, n_samples=n_samples)
    texts = load_texts(data_dir)
    print(f"  States: {states.shape}, Texts: {len(texts)}")

    print("Computing activations...")
    acts = compute_activations(sae, states, batch_size=512)
    print(f"  Activations: {acts.shape}")

    stats = activation_stats(acts)
    n_alive = int((stats["max"] > 1e-8).sum())
    n_total = acts.shape[1]
    print(f"  Alive features: {n_alive}/{n_total} ({100 * n_alive / n_total:.1f}%)")

    if feature_indices is not None:
        top_feat_indices = [i for i in feature_indices if 0 <= i < n_total]
        skipped = len(feature_indices) - len(top_feat_indices)
        if skipped > 0:
            print(f"  WARNING: {skipped} feature indices out of range [0, {n_total})")
        print(f"  Using {len(top_feat_indices)} specified features: {top_feat_indices}")
    else:
        top_feat_indices = top_features_by_activity(acts, n=n_top_features)
        print(f"  Top {len(top_feat_indices)} features by frequency: {top_feat_indices[:10]}...")

    vecs = extract_feature_vectors(sae)
    v_dec_factors = vecs["v_dec_factors"]
    w_dec_factors = vecs["w_dec_factors"]
    v_enc_factors = vecs["v_enc_factors"]
    w_enc_factors = vecs["w_enc_factors"]
    has_rank1 = v_dec_factors is not None and w_dec_factors is not None

    features: list[FeatureEntry] = []
    for fi in top_feat_indices:
        entry: FeatureEntry = {
            "feature_idx": fi,
            "frequency": float(stats["frequency"][fi]),
            "mean_activation": float(stats["mean"][fi]),
            "max_activation": float(stats["max"][fi]),
            "top_contexts": [],
        }

        if texts:
            entry["top_contexts"] = top_activating_contexts(acts, texts, fi, n=n_top_contexts)
        else:
            entry["top_contexts"] = []

        if has_rank1:
            assert v_dec_factors is not None and w_dec_factors is not None
            v, w, dec_meta = _factor_atom(v_dec_factors[fi], w_dec_factors[fi])
            entry["v_dec"] = v.tolist()
            entry["w_dec"] = w.tolist()
            entry["decoder_rank"] = dec_meta["rank"]
            entry["decoder_vector_method"] = dec_meta["method"]
            entry["decoder_rank1_retained_energy"] = dec_meta["rank1_retained_energy"]
            entry["v_norm"] = float(np.linalg.norm(v))
            entry["w_norm"] = float(np.linalg.norm(w))
            entry["v_sparsity"] = float((np.abs(v) < 0.01 * np.abs(v).max()).mean())
            entry["w_sparsity"] = float((np.abs(w) < 0.01 * np.abs(w).max()).mean())
            if dec_meta["rank"] > 1:
                entry["v_dec_components"] = np.asarray(v_dec_factors[fi]).tolist()
                entry["w_dec_components"] = np.asarray(w_dec_factors[fi]).tolist()
            if v_enc_factors is not None and w_enc_factors is not None:
                v_enc, w_enc, enc_meta = _factor_atom(v_enc_factors[fi], w_enc_factors[fi])
                entry["v_enc"] = v_enc.tolist()
                entry["w_enc"] = w_enc.tolist()
                entry["encoder_rank"] = enc_meta["rank"]
                entry["encoder_vector_method"] = enc_meta["method"]
                entry["encoder_rank1_retained_energy"] = enc_meta["rank1_retained_energy"]
                if enc_meta["rank"] > 1:
                    entry["v_enc_components"] = np.asarray(v_enc_factors[fi]).tolist()
                    entry["w_enc_components"] = np.asarray(w_enc_factors[fi]).tolist()
                entry["v_enc_dec_cosine"] = float(
                    np.dot(v, v_enc) / (np.linalg.norm(v) * np.linalg.norm(v_enc) + 1e-12))
                entry["w_enc_dec_cosine"] = float(
                    np.dot(w, w_enc) / (np.linalg.norm(w) * np.linalg.norm(w_enc) + 1e-12))
                dec_atom = _flatten_atom(v_dec_factors[fi], w_dec_factors[fi])
                enc_atom = _flatten_atom(v_enc_factors[fi], w_enc_factors[fi])
                entry["enc_dec_atom_cosine"] = float(
                    np.dot(dec_atom, enc_atom)
                    / (np.linalg.norm(dec_atom) * np.linalg.norm(enc_atom) + 1e-12)
                )

        features.append(entry)

    summary: SummaryEntry = {
        "sae_type": str(sae_type),
        "sae_path": sae_path,
        "layer": layer,
        "head": head,
        "n_samples": int(states.shape[0]),
        "n_features_total": n_total,
        "n_features_alive": n_alive,
        "dead_fraction": 1.0 - n_alive / n_total,
        "mean_l0": float((acts > 0).float().sum(1).mean()),
        "has_rank1_structure": has_rank1,
    }

    if has_rank1 and v_enc_factors is not None and w_enc_factors is not None:
        all_v_cos = [cast(float, f["v_enc_dec_cosine"]) for f in features if "v_enc_dec_cosine" in f]
        all_w_cos = [cast(float, f["w_enc_dec_cosine"]) for f in features if "w_enc_dec_cosine" in f]
        all_atom_cos = [cast(float, f["enc_dec_atom_cosine"]) for f in features if "enc_dec_atom_cosine" in f]
        if all_v_cos:
            summary["mean_v_enc_dec_cosine"] = float(np.mean(all_v_cos))
            summary["mean_w_enc_dec_cosine"] = float(np.mean(all_w_cos))
        if all_atom_cos:
            summary["mean_enc_dec_atom_cosine"] = float(np.mean(all_atom_cos))

    result = {"summary": summary, "features": features, "config": cfg}

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=2)

    np.savez(os.path.join(output_dir, "activation_stats.npz"), **stats)

    if has_rank1:
        fv = {}
        for feat in features:
            if "v_dec" not in feat or "w_dec" not in feat:
                continue
            fi = feat["feature_idx"]
            fv[str(fi)] = {"v": feat["v_dec"], "w": feat["w_dec"]}
        with open(os.path.join(output_dir, "feature_vectors.json"), "w") as f:
            json.dump(fv, f, indent=2)

    if texts:
        top_examples = {}
        for feat in features:
            fi = feat["feature_idx"]
            top_examples[str(fi)] = feat["top_contexts"]
        with open(os.path.join(output_dir, "top_examples.json"), "w") as f:
            json.dump(top_examples, f, indent=2)

    print(f"Results saved to {output_dir}")
    return result


def compare_saes(
    results: list[dict[str, object]], output_dir: str = "interpret_output",
) -> ComparisonResult:
    """Compare feature quality across SAE types."""
    os.makedirs(output_dir, exist_ok=True)
    rows: list[ComparisonRow] = []
    for r in results:
        s = cast(SummaryEntry, r["summary"])
        cfg = cast(dict[str, object], r.get("config", {}))  # config from run_interpretability if available
        feats = cast(list[FeatureEntry], r["features"])
        row: ComparisonRow = {
            "sae_type": s["sae_type"],
            "layer": s["layer"],
            "n_features": cast(int | None, s.get("n_features_total", cfg.get("n_features"))),
            "expansion_factor": cast(int | None, cfg.get("expansion_factor")),
            "k": cast(int | None, cfg.get("k")),
            "seed": cast(int | None, cfg.get("seed")),
            "n_alive": s["n_features_alive"],
            "dead_frac": round(s["dead_fraction"], 4),
            "mean_l0": round(s["mean_l0"], 1),
            "has_rank1": s["has_rank1_structure"],
        }

        if feats:
            freqs = [f["frequency"] for f in feats]
            maxes = [f["max_activation"] for f in feats]
            row["top_feat_mean_freq"] = round(float(np.mean(freqs)), 4)
            row["top_feat_mean_max"] = round(float(np.mean(maxes)), 4)

            if feats[0]["top_contexts"]:
                spreads = []
                for f in feats:
                    ctxs = f["top_contexts"]
                    mean_activation = f["mean_activation"]
                    if ctxs and mean_activation > 1e-12:
                        top_mean = float(np.mean([c["activation"] for c in ctxs]))
                        spreads.append(top_mean / mean_activation)
                if spreads:
                    row["selectivity_ratio"] = round(float(np.mean(spreads)), 2)

        rows.append(row)

    comparison: ComparisonResult = {"models": rows}
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== SAE Comparison ===")
    print(f"{'Type':>15} {'Layer':>5} {'Alive':>7} {'Dead%':>7} {'L0':>6} {'Select':>8}")
    print("-" * 55)
    for row in rows:
        print(f"{row['sae_type']:>15} {row['layer']:>5} {row['n_alive']:>7} "
              f"{row['dead_frac']*100:>6.1f}% {row['mean_l0']:>6.1f} "
              f"{row.get('selectivity_ratio', 0):>8.1f}")

    return comparison


def main() -> None:
    p = argparse.ArgumentParser(description="Feature interpretability analysis for matrix SAEs")
    p.add_argument("--sae_checkpoint", required=True, help="Path to SAE checkpoint (.pt)")
    p.add_argument("--data_dir", required=True, help="Directory with extracted states")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--n_top_features", type=int, default=20)
    p.add_argument("--n_top_contexts", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=None, help="Limit samples (None=all)")
    p.add_argument("--output_dir", default="interpret_output")
    p.add_argument("--device", default="cpu")
    p.add_argument("--feature_indices", type=int, nargs="+", default=None,
                   help="Specific feature indices to analyze (overrides n_top_features)")
    args = p.parse_args()

    run_interpretability(
        sae_path=args.sae_checkpoint,
        data_dir=args.data_dir,
        layer=args.layer,
        head=args.head,
        n_top_features=args.n_top_features,
        n_top_contexts=args.n_top_contexts,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        device=args.device,
        feature_indices=args.feature_indices,
    )


if __name__ == "__main__":
    main()

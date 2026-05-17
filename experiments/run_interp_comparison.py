"""Compare interpretability of bilinear vs flat SAE features.

Run the Spearman probing pipeline on both bilinear and flat per-head
checkpoints at L9 H4. Report: n_interpretable, interpretable_fraction,
top properties, and whether bilinear features are more interpretable.

Usage:
    modal run experiments/run_interp_comparison.py
"""
import json
import time
from pathlib import Path
from typing import Any

import modal

from _modal_utils import CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL, code_sha


CODE_SHA = code_sha()

_ext_dir = Path(__file__).resolve().parent / "extraction"
_core_dir = Path(__file__).resolve().parent.parent / "core"
_analysis_dir = Path(__file__).resolve().parent / "analysis"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "MAX_JOBS": "4", "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++",
    })
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands("python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers>=5.0", "datasets", "numpy", "tqdm",
        "matplotlib", "accelerate", "sentencepiece", "scipy", "scikit-learn",
        "einops", "ninja", "flash-linear-attention",
    )
    .run_commands(
        f"python -m pip install --no-deps '{CAUSAL_CONV1D_WHEEL}'",
        f"python -m pip install --no-deps '{MAMBA_SSM_WHEEL}'",
    )
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "types.py"), "/root/core/types.py", copy=True)
    .add_local_file(str(_core_dir / "__init__.py"), "/root/core/__init__.py", copy=True)
    .add_local_file(str(_analysis_dir / "probe_features.py"), "/root/probe_features.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("interp-comparison")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
ABLATION = "/ablation"
MODELS = "/models"


@app.function(
    volumes={DATA: data_vol, ABLATION: ablation_vol, MODELS: model_vol},
    image=image, timeout=3600, memory=16384,
)
def compare_interpretability(
    layer: int = 9,
    head: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Spearman probing on bilinear and flat SAEs at same head."""
    import os, sys
    sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    import numpy as np
    import torch

    from sae import build_sae_from_config
    from probe_features import probe_features

    print(f"=== Interpretability comparison: L{layer} H{head} ===")

    states_dir = Path(DATA) / "states"
    meta = json.loads((states_dir / "metadata.json").read_text())
    n_total = meta["n_samples"]
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    data = np.lib.format.open_memmap(
        str(head_path), mode="r", dtype=np.float16,
        shape=(n_total, key_dim, val_dim),
    )
    states = torch.from_numpy(data[:n_total].astype(np.float32))
    print(f"  Loaded {states.shape[0]} states")

    # Decode texts from corpus.npy
    print("  Decoding texts from corpus.npy...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    corpus = np.load(str(states_dir / "corpus.npy"))[:n_total]
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in corpus]
    n_use = min(states.shape[0], len(texts))
    states = states[:n_use]
    texts = texts[:n_use]
    print(f"  {n_use} samples with texts")

    results = {}

    for sae_type in ["bilinear", "flat"]:
        print(f"\n  --- {sae_type} ---")

        ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}"
        best_path = ckpt_dir / "best.pt"
        config_path = ckpt_dir / "config.json"

        if not best_path.exists():
            print(f"  Checkpoint not found: {best_path}")
            continue

        cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
        ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", {})

        sae = build_sae_from_config(config=cfg, state_dict=sd)
        sae.load_state_dict(sd)
        sae.eval()

        # Use the correct API: probe_features(sae, states, texts)
        # For flat SAE, states need reshaping inside probe_features
        # Check if probe_features handles this or if we need to pass flat states
        input_states = states
        if sae_type == "flat":
            input_states = states.reshape(n_use, -1)

        print(f"  Running Spearman probing on {n_use} samples...")
        probe_result = probe_features(
            sae=sae,
            states=input_states,
            texts=texts,
        )

        n_interp = probe_result.get("n_interpretable", 0)
        n_alive = probe_result.get("n_alive", 0)
        frac = probe_result.get("interpretable_fraction", 0)
        print(f"  Interpretable: {n_interp}/{n_alive} ({frac*100:.1f}%)")

        prop_summary = probe_result.get("property_summary", {})
        top_props = sorted(prop_summary.items(),
                          key=lambda x: x[1].get("n_correlated_features", 0),
                          reverse=True)[:5]
        for prop_name, stats in top_props:
            print(f"    {prop_name}: {stats.get('n_correlated_features', 0)} features, "
                  f"max|rho|={stats.get('max_abs_rho', 0):.3f}")

        # Per-feature data for downstream analysis
        per_feature = probe_result.get("features", [])

        results[sae_type] = {
            "n_alive": int(n_alive),
            "n_dead": int(probe_result.get("n_dead", 0)),
            "n_interpretable": n_interp,
            "n_interpretable_bonferroni": int(probe_result.get("n_interpretable_bonferroni", 0)),
            "interpretable_fraction": frac,
            "interpretable_fraction_bonferroni": float(probe_result.get("interpretable_fraction_bonferroni", 0)),
            "top_properties": {k: v for k, v in top_props},
            "features": per_feature,
        }

        del sae

    # Summary comparison
    if "bilinear" in results and "flat" in results:
        b = results["bilinear"]
        f = results["flat"]
        print(f"\n  === COMPARISON ===")
        print(f"  bilinear: {b['n_interpretable']}/{b['n_alive']} interpretable ({b['interpretable_fraction']*100:.1f}%)")
        print(f"  flat:     {f['n_interpretable']}/{f['n_alive']} interpretable ({f['interpretable_fraction']*100:.1f}%)")
        ratio = b['interpretable_fraction'] / max(f['interpretable_fraction'], 1e-6)
        print(f"  Ratio: {ratio:.2f}x")
        results["comparison"] = {
            "bilinear_frac": b['interpretable_fraction'],
            "flat_frac": f['interpretable_fraction'],
            "ratio": ratio,
        }

    # Per-feature analysis: monovariance, top features, property diversity
    if "bilinear" in results and "flat" in results:
        print(f"\n  === PER-FEATURE ANALYSIS ===")
        for sae_type in ["bilinear", "flat"]:
            feats = results[sae_type].get("features", [])
            # Monovariant: exactly 1 Bonferroni-significant correlation
            monovariant = [f for f in feats if f.get("n_significant_bonferroni") == 1]
            # Interpretable (Bonferroni): at least 1 significant correlation
            interpretable = [f for f in feats if f.get("n_significant_bonferroni", 0) >= 1]

            print(f"\n  --- {sae_type} per-feature ---")
            print(f"  Monovariant features (n_sig_bonf == 1): {len(monovariant)}")
            print(f"  Interpretable features (n_sig_bonf >= 1): {len(interpretable)}")

            # Mean |rho| among interpretable features
            if interpretable:
                mean_abs_rho = sum(abs(f["best_rho"]) for f in interpretable) / len(interpretable)
                print(f"  Mean |best_rho| among interpretable: {mean_abs_rho:.4f}")
            else:
                mean_abs_rho = 0.0
                print(f"  Mean |best_rho| among interpretable: N/A (none)")

            # Property diversity: distinct best_property values among interpretable
            props_covered = set(f["best_property"] for f in interpretable if f.get("best_property"))
            print(f"  Property diversity (distinct best_property): {len(props_covered)}")
            if props_covered:
                print(f"    Properties: {sorted(props_covered)}")

            # Top 5 monovariant features by |rho|
            mono_sorted = sorted(monovariant, key=lambda f: abs(f["best_rho"]), reverse=True)
            print(f"  Top 5 monovariant features:")
            for rank, mf in enumerate(mono_sorted[:5], 1):
                # Get the single significant property from bonferroni dict
                sig_bonf = mf.get("significant_correlations_bonferroni", {})
                if sig_bonf:
                    prop_name = next(iter(sig_bonf))
                    rho_val = sig_bonf[prop_name]["rho"]
                else:
                    prop_name = mf["best_property"]
                    rho_val = mf["best_rho"]
                print(f"    {rank}. feature {mf['feature_idx']}: "
                      f"{prop_name} rho={rho_val:+.4f} freq={mf['frequency']:.3f}")

            # Store analysis in results
            results[sae_type]["monovariant_count"] = len(monovariant)
            results[sae_type]["mean_abs_rho_interpretable"] = float(mean_abs_rho)
            results[sae_type]["property_diversity"] = len(props_covered)
            results[sae_type]["properties_covered"] = sorted(props_covered)
            results[sae_type]["top_monovariant"] = [
                {
                    "feature_idx": mf["feature_idx"],
                    "property": next(iter(mf.get("significant_correlations_bonferroni", {})), mf["best_property"]),
                    "rho": float(next(iter(mf.get("significant_correlations_bonferroni", {}).values()), {}).get("rho", mf["best_rho"])),
                    "frequency": mf["frequency"],
                }
                for mf in mono_sorted[:5]
            ]

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "interp_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_H{head}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


@app.local_entrypoint()
def main():
    t0 = time.time()
    result = compare_interpretability.remote(layer=9, head=4, seed=42)
    print(f"\nDone in {time.time() - t0:.0f}s")

    out_path = Path("results/data/interp_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {out_path}")

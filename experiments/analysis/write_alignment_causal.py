#!/usr/bin/env python3
"""Causal test: do write-aligned SAE features steer more effectively than non-aligned features?

The paper reports 17.5% of bilinear decoder atoms align with actual GDN write vectors
(combined alignment > 0.3). A reviewer asks whether this alignment is causally relevant
or merely geometric coincidence.

Protocol:
  1. Load the write alignment data (memory_alignment_L9_H4.json) to split features into
     write-aligned (combined > 0.3) and non-aligned (combined < 0.1) groups.
  2. From each group, select the top 10 by boundary-differential activation (same
     criterion as the boost experiment).
  3. Boost each group at 5x and 10x, single head (H4 only).
  4. Compare newline shift: if write-aligned features steer harder, alignment identifies
     causally relevant features.

Conditions per prompt (7 total):
  1. baseline: no intervention
  2. write_aligned_5x: boost top-10 write-aligned features at 5x
  3. write_aligned_10x: boost top-10 write-aligned features at 10x
  4. non_aligned_5x: boost top-10 non-aligned features at 5x
  5. non_aligned_10x: boost top-10 non-aligned features at 10x
  6. random_5x: boost 10 random alive features at 5x
  7. random_10x: boost 10 random alive features at 10x

Uses the same 20 instruction prompts, model (Qwen3.5-4B-Base), layer 9, head 4,
generation parameters (400 tokens, temperature 0.7) as the existing experiments.

Usage:
    modal run --detach write_alignment_causal.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

# Modal infrastructure


def _current_code_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (FileNotFoundError, OSError):
        sha = ""
    return sha or os.environ.get("MATRIX_SAE_CODE_SHA", "unknown")


CURRENT_CODE_SHA = _current_code_sha()
CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    "v1.6.1.post4/"
    "causal_conv1d-1.6.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
MAMBA_SSM_WHEEL = (
    "https://github.com/state-spaces/mamba/releases/download/"
    "v2.3.1/"
    "mamba_ssm-2.3.1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "MAX_JOBS": "4",
        "CC": "gcc",
        "CXX": "g++",
        "CUDAHOSTCXX": "g++",
    })
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install(
        "transformers>=5.0", "datasets", "numpy", "tqdm",
        "matplotlib", "wandb", "accelerate", "sentencepiece", "scipy", "scikit-learn",
        "einops", "ninja", "flash-linear-attention",
    )
    .run_commands(
        f"python -m pip install --no-deps '{CAUSAL_CONV1D_WHEEL}'",
        f"python -m pip install --no-deps '{MAMBA_SSM_WHEEL}'",
        "python -c \""
        "import causal_conv1d; "
        "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; "
        "from fla.modules import FusedRMSNormGated; "
        "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; "
        "print('ALL_FASTPATH_IMPORTS_OK')\"",
    )
    .add_local_file("extract_states.py", "/root/extract_states.py", copy=True)
    .add_local_file("sae.py", "/root/sae.py", copy=True)
    .add_local_file("split_utils.py", "/root/split_utils.py", copy=True)
    .add_local_file("generation_intervention.py", "/root/generation_intervention.py", copy=True)
    .add_local_file("memory_alignment.py", "/root/memory_alignment.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CURRENT_CODE_SHA})
)

app = modal.App("matrix-sae-write-alignment-causal")

vol = modal.Volume.from_name(
    os.environ.get("MATRIX_SAE_MODAL_DATA_VOLUME", "").strip() or "matrix-sae-data",
    create_if_missing=True,
)
model_vol = modal.Volume.from_name(
    os.environ.get("MATRIX_SAE_MODAL_MODEL_VOLUME", "").strip() or "hf-model-cache",
    create_if_missing=True,
)

DATA = "/data"
MODELS = "/models"

# Prompts (identical to existing experiments)

QUALITATIVE_PROMPTS = [
    "Write a paragraph about the history of bridges.",
    "Describe the fall of the Roman Empire in a few sentences.",
    "Summarize the key events of the French Revolution.",
    "Explain how photosynthesis works.",
    "Describe what happens inside a star during nuclear fusion.",
    "Explain why the sky is blue.",
    "Describe the process of making bread from scratch.",
    "Explain how to make a simple tomato sauce.",
    "Write a paragraph about the history of chocolate.",
    "Describe what a visitor would see walking through the streets of Tokyo.",
    "Write a paragraph about the geography of Iceland.",
    "Explain how a computer processor executes instructions.",
    "Describe how the internet routes data between computers.",
    "Write a paragraph about the invention of the printing press.",
    "Describe the water cycle from ocean to rainfall.",
    "Explain how birds migrate thousands of miles each year.",
    "Write a paragraph about the ecosystem of a coral reef.",
    "Explain why we have seasons on Earth.",
    "Describe how human memory works.",
    "Write a paragraph about the construction of the Great Wall of China.",
]


# SAE checkpoint resolution (from existing experiments)


def _normalize_corpus_source(corpus_source: str) -> str:
    aliases = {
        "openwebtext": "openwebtext", "owt": "openwebtext",
        "skylion007/openwebtext": "openwebtext",
    }
    return aliases.get(corpus_source.strip().lower(), corpus_source)


def _states_dir(corpus_source: str) -> Path:
    slug = _normalize_corpus_source(corpus_source)
    if slug == "openwebtext":
        return Path(f"{DATA}/states")
    return Path(f"{DATA}/states_{slug}")


def _experiment_tag(model_name: str, seq_len: int, n_samples: int, corpus_source: str = "openwebtext") -> str:
    model_slug = model_name.split("/")[-1].lower().replace(".", "_")
    tag = f"{model_slug}_sl{seq_len}_ns{n_samples}"
    slug = _normalize_corpus_source(corpus_source)
    if slug != "openwebtext":
        tag = f"{tag}_{slug}"
    return tag


def _resolve_sae_checkpoint(
    *, layer: int, head: int, n_features_target: int,
    corpus_source: str = "openwebtext",
    preferred_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
) -> tuple[Path, dict, str]:
    """Find the best SAE checkpoint for a given layer/head."""
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag

    best_ckpt = None
    best_cfg = None
    best_tag = None

    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoint root at {ckpt_root}")

    for d in sorted(ckpt_root.iterdir()):
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type not in preferred_types:
            continue
        candidate_key = (
            preferred_types.index(sae_type) if sae_type in preferred_types else 99,
            0 if cfg.get("seed") == 42 else 1,
            cfg.get("seed", 999),
        )
        current_key = None
        if best_cfg is not None:
            best_sae_type = best_cfg.get("sae_type", "")
            current_key = (
                preferred_types.index(best_sae_type) if best_sae_type in preferred_types else 99,
                0 if best_cfg.get("seed") == 42 else 1,
                best_cfg.get("seed", 999),
            )
        if current_key is None or candidate_key < current_key:
            best_ckpt = bp
            best_cfg = cfg
            best_tag = d.name

    if best_ckpt is None:
        raise FileNotFoundError(f"No checkpoint for L{layer}H{head} nf={n_features_target}")
    return best_ckpt, best_cfg, best_tag


# Main experiment


@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=14400, memory=32768,
)
def run_write_alignment_causal(
    layer: int = 9,
    head: int = 4,
    n_tokens: int = 400,
    n_boundary_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-4B-Base",
    temperature: float = 0.7,
    random_seed: int = 42,
    period_token_id: int = 13,
    alignment_threshold_high: float = 0.3,
    alignment_threshold_low: float = 0.1,
    boost_scales: tuple[float, ...] = (5.0, 10.0),
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
) -> dict:
    """Test whether write-aligned features steer more effectively than non-aligned features.

    Single-head experiment (H4 only, where alignment data exists).

    Feature selection pipeline:
      1. Load alignment data to partition features into write-aligned and non-aligned.
      2. Load extracted states to compute boundary-differential activation per feature.
      3. From each alignment group, select top n_boundary_features by |boundary_diff|.
      4. Boost each group at each dose; compare newline shift.
    """
    import sys
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer
    from generation_intervention import (
        select_boundary_features_fast,
        generate_with_intervention,
        compute_generation_stats,
    )

    vol.reload()
    t0 = time.time()
    rng = np.random.RandomState(random_seed)

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # The memory_alignment stage saves to /data/memory_alignment/L{layer}_H{head}/alignment_results.json
    candidates = [
        Path(f"{DATA}/memory_alignment/L{layer}_H{head}/alignment_results.json"),
        Path(f"{DATA}/analysis/memory_alignment_L{layer}_H{head}.json"),
        Path(f"{DATA}/memory_alignment_L{layer}_H{head}.json"),
    ]
    alignment_path = None
    for candidate in candidates:
        if candidate.exists():
            alignment_path = candidate
            print(f"Found alignment data at: {alignment_path}")
            break
    if alignment_path is None:
        raise FileNotFoundError(
            f"No alignment data found. Checked: {[str(c) for c in candidates]}"
        )

    alignment_data = json.loads(alignment_path.read_text())
    alignment_results = alignment_data["results"]

    # Compute combined alignment = mean(|k_cos|, |v_cos|)
    alignment_map: dict[int, float] = {}
    for r in alignment_results:
        if r.get("alive", False):
            combined = (r.get("mean_abs_k_cos", 0) + r.get("mean_abs_v_cos", 0)) / 2
            alignment_map[r["feature"]] = combined

    write_aligned_set = {f for f, c in alignment_map.items() if c > alignment_threshold_high}
    non_aligned_set = {f for f, c in alignment_map.items() if c < alignment_threshold_low}
    all_alive_set = set(alignment_map.keys())

    print(f"Alignment data loaded: {len(alignment_map)} alive features")
    print(f"Write-aligned (combined > {alignment_threshold_high}): {len(write_aligned_set)}")
    print(f"Non-aligned (combined < {alignment_threshold_low}): {len(non_aligned_set)}")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    corpus_source = "openwebtext"
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer, head=head, n_features_target=n_features_target,
        corpus_source=corpus_source, preferred_types=sae_types,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae_model = build_sae_from_config(
        cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae_model.load_state_dict(ckpt["model_state_dict"])
    sae_model = sae_model.cuda().eval()
    sae_type = infer_sae_type(cfg, ckpt["model_state_dict"])
    print(f"Loaded SAE: {resolved_tag} (type={sae_type})")

    # ---------------------------------------------------------------
    # Load corpus, build boundary mask, compute boundary-differential
    # ---------------------------------------------------------------
    corpus_path = states_dir / "corpus.npy"
    corpus_arr = np.load(str(corpus_path), mmap_mode="r")
    n_corpus_seqs = corpus_arr.shape[0]
    seq_len = corpus_arr.shape[1]

    corpus_full = np.array(corpus_arr)
    boundary_mask = np.zeros((n_corpus_seqs, seq_len), dtype=bool)
    boundary_mask[:, :-1] = (corpus_full[:, 1:] == period_token_id)

    head_states_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_states_path.exists():
        raise FileNotFoundError(f"No extracted states at {head_states_path}")

    head_states = np.load(str(head_states_path), mmap_mode="r")
    n_states = head_states.shape[0]
    n_use_seqs = min(n_states // seq_len, n_corpus_seqs)
    flat_mask = boundary_mask[:n_use_seqs].reshape(-1)
    states_flat = head_states[:n_use_seqs * seq_len]

    boundary_indices = np.where(flat_mask)[0]
    nonboundary_indices = np.where(~flat_mask)[0]
    n_sample = min(len(boundary_indices), len(nonboundary_indices), 10000)

    b_idx = rng.choice(boundary_indices, size=n_sample, replace=False)
    nb_idx = rng.choice(nonboundary_indices, size=n_sample, replace=False)

    print(f"Corpus: {n_corpus_seqs} seqs x {seq_len} tokens")
    print(f"Boundary positions: {int(boundary_mask.sum())}")
    print(f"Sampled: {n_sample} boundary + {n_sample} non-boundary states")

    # Get boundary-differential ranking for ALL features
    all_ranked = select_boundary_features_fast(
        sae_model, sae_type,
        np.array(states_flat[b_idx]), np.array(states_flat[nb_idx]),
        n_features=n_features_target,  # rank all features
    )

    # Build lookup: feature_idx -> rank info
    ranked_lookup = {f["feature_idx"]: f for f in all_ranked}

    # ---------------------------------------------------------------
    # Select features for each group (boundary-differential within group)
    # ---------------------------------------------------------------

    def _select_top_boundary_diff(feature_set: set[int], n: int, sign: str = "positive") -> list[dict]:
        """From a set of feature indices, select top n by boundary-differential."""
        candidates = []
        for fidx in feature_set:
            info = ranked_lookup.get(fidx)
            if info is None:
                continue
            if sign == "positive" and info["mean_diff"] <= 0:
                continue
            if sign == "negative" and info["mean_diff"] >= 0:
                continue
            candidates.append(info)
        candidates.sort(key=lambda x: abs(x["mean_diff"]), reverse=True)
        return candidates[:n]

    write_aligned_features = _select_top_boundary_diff(write_aligned_set, n_boundary_features, sign="positive")
    non_aligned_features = _select_top_boundary_diff(non_aligned_set, n_boundary_features, sign="positive")

    # Random alive features (excluding both groups)
    excluded = set(f["feature_idx"] for f in write_aligned_features + non_aligned_features)
    random_pool = [idx for idx in all_alive_set if idx not in excluded]
    random_selection = rng.choice(random_pool, size=n_boundary_features, replace=False).tolist()
    random_features = [ranked_lookup.get(fi, {"feature_idx": fi, "mean_boundary_nonzero": 0.01}) for fi in random_selection]

    print("\n--- Feature Selection ---")
    print(f"Write-aligned group ({len(write_aligned_features)} features):")
    for f in write_aligned_features:
        alignment = alignment_map.get(f["feature_idx"], 0)
        print(f"  F{f['feature_idx']}: boundary_diff={f['mean_diff']:+.4f}, alignment={alignment:.3f}")

    print(f"\nNon-aligned group ({len(non_aligned_features)} features):")
    for f in non_aligned_features:
        alignment = alignment_map.get(f["feature_idx"], 0)
        print(f"  F{f['feature_idx']}: boundary_diff={f['mean_diff']:+.4f}, alignment={alignment:.3f}")

    print(f"\nRandom control ({len(random_features)} features):")
    for f in random_features:
        alignment = alignment_map.get(f["feature_idx"], 0)
        print(f"  F{f['feature_idx']}: alignment={alignment:.3f}")

    if len(write_aligned_features) < n_boundary_features:
        print(f"\nWARNING: Only {len(write_aligned_features)} write-aligned boundary-differential "
              f"features found (need {n_boundary_features})")
    if len(non_aligned_features) < n_boundary_features:
        print(f"\nWARNING: Only {len(non_aligned_features)} non-aligned boundary-differential "
              f"features found (need {n_boundary_features})")

    # ---------------------------------------------------------------
    # Compute calibrated push values (same calibration as boost experiment)
    # ---------------------------------------------------------------

    def _make_push_dict(features: list[dict], strength: float) -> dict[int, float]:
        """Build additive push dict: feature_idx -> push_value * strength."""
        pushes = {}
        for f in features:
            fi = f["feature_idx"]
            # Use mean nonzero boundary activation as base push
            base = max(abs(float(f.get("mean_boundary_nonzero", 0.0))), 0.01)
            pushes[fi] = base * strength
        return pushes

    def _make_random_push_dict(features: list[dict], strength: float) -> dict[int, float]:
        """Build push dict for random features using their own mean nonzero."""
        pushes = {}
        for f in features:
            fi = f["feature_idx"]
            base = max(abs(float(f.get("mean_boundary_nonzero", f.get("mean_nonboundary_nonzero", 0.01)))), 0.01)
            pushes[fi] = base * strength
        return pushes

    rand_indices = [f["feature_idx"] for f in random_features]

    conditions: dict[str, dict[int, dict[int, float]]] = {"baseline": {}}

    for scale in boost_scales:
        scale_tag = f"{int(scale)}x" if scale == int(scale) else f"{scale:g}x"

        # Write-aligned boost
        conditions[f"write_aligned_{scale_tag}"] = {
            head: _make_push_dict(write_aligned_features, scale),
        }
        # Non-aligned boost
        conditions[f"non_aligned_{scale_tag}"] = {
            head: _make_push_dict(non_aligned_features, scale),
        }
        # Random boost
        conditions[f"random_{scale_tag}"] = {
            head: _make_random_push_dict(random_features, scale),
        }

    cond_names = list(conditions.keys())
    print(f"\nConditions: {cond_names}")
    print(f"Boost scales: {boost_scales}")

    # Report calibration values
    for label, features in [("write_aligned", write_aligned_features),
                            ("non_aligned", non_aligned_features),
                            ("random", random_features)]:
        base_pushes = [max(abs(float(f.get("mean_boundary_nonzero",
                       f.get("mean_nonboundary_nonzero", 0.01)))), 0.01) for f in features]
        if base_pushes:
            print(f"  {label} base push: min={min(base_pushes):.4f}, max={max(base_pushes):.4f}, "
                  f"mean={np.mean(base_pushes):.4f}")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"\nLoading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    raw_prompts = list(QUALITATIVE_PROMPTS)
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompts = []
        for p in raw_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            text = text.replace("<think>\n\n</think>\n\n", "")
            formatted_prompts.append(text)
        print(f"Formatted {len(formatted_prompts)} prompts with chat template")
    else:
        formatted_prompts = raw_prompts

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    device = next(model.parameters()).device
    sae_per_head = {head: sae_model}
    all_results: list[dict] = []
    total_runs = len(formatted_prompts) * len(conditions)
    run_idx = 0
    t_gen = time.time()

    for i, (raw_prompt, fmt_prompt) in enumerate(zip(raw_prompts, formatted_prompts)):
        prompt_ids = tokenizer(fmt_prompt, return_tensors="pt")["input_ids"].to(device)
        entry = {
            "prompt_idx": i,
            "prompt_text": raw_prompt,
            "prompt_tokens": int(prompt_ids.shape[1]),
        }

        for cond_name, updates_per_head in conditions.items():
            gen_ids, meta = generate_with_intervention(
                model=model,
                tokenizer=tokenizer,
                sae_type=sae_type,
                layer_idx=layer,
                prompt_ids=prompt_ids,
                n_tokens=n_tokens,
                temperature=temperature,
                sae_per_head=sae_per_head if updates_per_head else {},
                feature_updates_per_head=updates_per_head,
                additive=True,
                additive_min_zero=False,
            )
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_stats = compute_generation_stats(gen_text)

            entry[cond_name] = {
                "text": gen_text,
                "stats": gen_stats,
                "n_generated": meta["n_generated"],
                "mean_intervention_norm": meta["mean_intervention_norm"],
            }
            run_idx += 1

        elapsed = time.time() - t_gen
        rate = run_idx / elapsed if elapsed > 0 else 0
        remaining = (total_runs - run_idx) / rate if rate > 0 else 0

        base_nl = entry["baseline"]["stats"]["n_newlines"]
        wa_nl = entry.get("write_aligned_10x", {}).get("stats", {}).get("n_newlines", 0)
        na_nl = entry.get("non_aligned_10x", {}).get("stats", {}).get("n_newlines", 0)
        print(
            f"  [{i+1}/{len(formatted_prompts)}] {elapsed:.0f}s elapsed, "
            f"{remaining:.0f}s remaining | "
            f"newlines: baseline={base_nl:.0f} write_aligned_10x={wa_nl:.0f} non_aligned_10x={na_nl:.0f}"
        )
        all_results.append(entry)

    total_time = time.time() - t0

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    from scipy import stats as sp_stats

    stat_names = list(compute_generation_stats("test text.").keys())
    condition_means: dict[str, dict[str, float]] = {}
    per_cond_arrays: dict[str, dict[str, list[float]]] = {}

    for cond_name in cond_names:
        means = {}
        arrays = {}
        for stat in stat_names:
            vals = [r[cond_name]["stats"][stat] for r in all_results]
            means[stat] = float(np.mean(vals))
            arrays[stat] = vals
        condition_means[cond_name] = means
        per_cond_arrays[cond_name] = arrays

    # ---------------------------------------------------------------
    # Statistical tests
    # ---------------------------------------------------------------
    n_bootstrap = 10000

    def bootstrap_ci(vals_a, vals_b, n_boot=n_bootstrap, alpha=0.05):
        a = np.array(vals_a)
        b = np.array(vals_b)
        diffs = a - b
        observed = float(np.mean(diffs))
        boot_means = np.array([
            np.mean(rng.choice(diffs, size=len(diffs), replace=True))
            for _ in range(n_boot)
        ])
        ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        if observed >= 0:
            p = 2 * float(np.mean(boot_means < 0))
        else:
            p = 2 * float(np.mean(boot_means > 0))
        return observed, ci_lo, ci_hi, min(p, 1.0)

    # Key comparisons
    comparisons = []
    for scale in boost_scales:
        tag = f"{int(scale)}x" if scale == int(scale) else f"{scale:g}x"
        comparisons.append((f"write_aligned_{tag}_vs_baseline", f"write_aligned_{tag}", "baseline"))
        comparisons.append((f"non_aligned_{tag}_vs_baseline", f"non_aligned_{tag}", "baseline"))
        comparisons.append((f"random_{tag}_vs_baseline", f"random_{tag}", "baseline"))
        # Direct comparison: write-aligned vs non-aligned
        comparisons.append((f"write_aligned_vs_non_aligned_{tag}", f"write_aligned_{tag}", f"non_aligned_{tag}"))

    tests: dict[str, dict[str, dict]] = {}
    for comparison, cond_a, cond_b in comparisons:
        comp = {}
        for stat in stat_names:
            vals_a = per_cond_arrays[cond_a][stat]
            vals_b = per_cond_arrays[cond_b][stat]

            t_stat, t_pval = sp_stats.ttest_rel(vals_a, vals_b)
            diff = np.array(vals_a) - np.array(vals_b)
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

            obs, ci_lo, ci_hi, boot_p = bootstrap_ci(vals_a, vals_b)

            direction = "higher" if mean_diff > 0 else "lower"
            comp[stat] = {
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "mean_diff": mean_diff,
                "direction": direction,
                "t_stat": float(t_stat),
                "t_pvalue": float(t_pval),
                "cohens_d": cohens_d,
                "bootstrap_mean_diff": obs,
                "bootstrap_ci_95": [ci_lo, ci_hi],
                "bootstrap_pvalue": boot_p,
            }
        tests[comparison] = comp

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    analysis_dir = Path(f"{DATA}/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "write_alignment_causal",
        "layer": layer,
        "head": head,
        "n_prompts": len(formatted_prompts),
        "n_tokens": n_tokens,
        "n_boundary_features": n_boundary_features,
        "temperature": temperature,
        "sae_type": sae_type,
        "model_name": model_name,
        "alignment_threshold_high": alignment_threshold_high,
        "alignment_threshold_low": alignment_threshold_low,
        "boost_scales": list(boost_scales),
        "total_time_s": total_time,
        "n_write_aligned_available": len(write_aligned_set),
        "n_non_aligned_available": len(non_aligned_set),
        "write_aligned_features": [
            {"feature_idx": f["feature_idx"],
             "mean_diff": f["mean_diff"],
             "alignment": alignment_map.get(f["feature_idx"], 0)}
            for f in write_aligned_features
        ],
        "non_aligned_features": [
            {"feature_idx": f["feature_idx"],
             "mean_diff": f["mean_diff"],
             "alignment": alignment_map.get(f["feature_idx"], 0)}
            for f in non_aligned_features
        ],
        "random_features": [
            {"feature_idx": fi,
             "alignment": alignment_map.get(fi, 0)}
            for fi in rand_indices
        ],
        "conditions": condition_means,
        "condition_names": cond_names,
        "tests": tests,
        "per_prompt": all_results,
    }

    filename = f"write_alignment_causal_L{layer}_H{head}.json"
    out_path = analysis_dir / filename
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")
    result["output_path"] = str(out_path)

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"WRITE ALIGNMENT CAUSAL TEST: L{layer}H{head}")
    print(f"{'='*80}")

    for cond_name in cond_names:
        m = condition_means[cond_name]
        print(f"  {cond_name:25s}: newlines={m['n_newlines']:.1f}  paragraphs={m['n_paragraphs']:.1f}  "
              f"words={m['n_words']:.0f}  mean_word_len={m['mean_word_length']:.2f}")

    print("\n--- Key Statistical Tests ---")
    key_stats = ["n_newlines", "n_paragraphs"]
    for comp_name, comp_data in tests.items():
        print(f"\n  {comp_name}:")
        for stat in key_stats:
            s = comp_data[stat]
            sig = "*" if s["t_pvalue"] < 0.05 else ""
            print(f"    {stat:22s}: diff={s['mean_diff']:+.2f}  d={s['cohens_d']:+.3f}  "
                  f"p={s['t_pvalue']:.4f}{sig}  "
                  f"boot_ci=[{s['bootstrap_ci_95'][0]:+.2f}, {s['bootstrap_ci_95'][1]:+.2f}]")

    # The key question
    print(f"\n{'='*80}")
    print("KEY QUESTION: Do write-aligned features steer harder?")
    print(f"{'='*80}")
    for scale in boost_scales:
        tag = f"{int(scale)}x" if scale == int(scale) else f"{scale:g}x"
        wa_nl = condition_means[f"write_aligned_{tag}"]["n_newlines"]
        na_nl = condition_means[f"non_aligned_{tag}"]["n_newlines"]
        rand_nl = condition_means[f"random_{tag}"]["n_newlines"]
        base_nl = condition_means["baseline"]["n_newlines"]
        wa_shift = wa_nl - base_nl
        na_shift = na_nl - base_nl
        rand_shift = rand_nl - base_nl

        direct = tests[f"write_aligned_vs_non_aligned_{tag}"]
        nl_direct = direct["n_newlines"]
        print(f"\n  At {tag} dose:")
        print(f"    Write-aligned newlines: {wa_nl:.1f} (shift {wa_shift:+.1f})")
        print(f"    Non-aligned newlines:   {na_nl:.1f} (shift {na_shift:+.1f})")
        print(f"    Random newlines:        {rand_nl:.1f} (shift {rand_shift:+.1f})")
        print(f"    Write-aligned vs non-aligned: diff={nl_direct['mean_diff']:+.2f}, "
              f"p={nl_direct['t_pvalue']:.4f}, d={nl_direct['cohens_d']:+.3f}")

    print(f"\nTotal time: {total_time:.0f}s")
    return result


# Local entrypoint


@app.local_entrypoint()
def main():
    result = run_write_alignment_causal.remote()

    local_out = os.path.join(os.path.dirname(__file__), "results", "data")
    os.makedirs(local_out, exist_ok=True)
    local_name = os.path.basename(result.get("output_path", ""))
    if not local_name:
        local_name = "write_alignment_causal_L9_H4.json"
    local_path = os.path.join(local_out, local_name)
    with open(local_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved locally to {local_path}")

    tests = result.get("tests", {})
    conds = result.get("conditions", {})

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    for tag in ["10x"]:
        direct_key = f"write_aligned_vs_non_aligned_{tag}"
        if direct_key in tests:
            nl = tests[direct_key].get("n_newlines", {})
            p = nl.get("t_pvalue", 1.0)
            d = nl.get("cohens_d", 0)
            wa_nl = conds.get(f"write_aligned_{tag}", {}).get("n_newlines", 0)
            na_nl = conds.get(f"non_aligned_{tag}", {}).get("n_newlines", 0)

            if p < 0.05:
                if wa_nl < na_nl:
                    print(f"  Write-aligned features steer HARDER at {tag} (p={p:.4f}, d={d:+.3f})")
                    print("  Alignment metric identifies causally relevant features.")
                else:
                    print(f"  Non-aligned features steer harder at {tag} (p={p:.4f}, d={d:+.3f})")
                    print("  Alignment metric does NOT predict causal relevance.")
            else:
                print(f"  No significant difference at {tag} (p={p:.4f}, d={d:+.3f})")
                print(f"  Inconclusive with {result.get('n_prompts', 20)} prompts.")

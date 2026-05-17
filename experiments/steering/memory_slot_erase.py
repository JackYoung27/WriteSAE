#!/usr/bin/env python3
"""Memory Slot Erase: zero write-aligned SAE features during generation and measure effect.

Reverse of the additive boost experiment. Instead of amplifying boundary-correlated
features, zero them. If erasing write-aligned features disrupts document structure
(increases newlines, changes paragraph counts), that demonstrates bidirectional
causal control through the SAE representation.

Three conditions per prompt:
  1. baseline: no intervention
  2. erase_aligned: zero top-10 write-aligned features (from memory_alignment) across all 32 heads
  3. erase_random: zero 10 random alive features across all 32 heads (matched control)

Uses the same 20 instruction prompts, model (Qwen3.5-4B-Base), layer (9), generation
parameters (400 tokens, temperature 0.7) as the boost experiment for direct comparison.

Usage:
    modal run --detach memory_slot_erase.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import modal

# Modal infrastructure (mirrors run_modal.py)


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

app = modal.App("matrix-sae-erase")

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

# Prompts (identical to boost experiment)

QUALITATIVE_PROMPTS = [
    # History
    "Write a paragraph about the history of bridges.",
    "Describe the fall of the Roman Empire in a few sentences.",
    "Summarize the key events of the French Revolution.",
    # Science
    "Explain how photosynthesis works.",
    "Describe what happens inside a star during nuclear fusion.",
    "Explain why the sky is blue.",
    # Cooking
    "Describe the process of making bread from scratch.",
    "Explain how to make a simple tomato sauce.",
    "Write a paragraph about the history of chocolate.",
    # Travel
    "Describe what a visitor would see walking through the streets of Tokyo.",
    "Write a paragraph about the geography of Iceland.",
    # Technology
    "Explain how a computer processor executes instructions.",
    "Describe how the internet routes data between computers.",
    "Write a paragraph about the invention of the printing press.",
    # Nature
    "Describe the water cycle from ocean to rainfall.",
    "Explain how birds migrate thousands of miles each year.",
    "Write a paragraph about the ecosystem of a coral reef.",
    # General knowledge
    "Explain why we have seasons on Earth.",
    "Describe how human memory works.",
    "Write a paragraph about the construction of the Great Wall of China.",
]

# Top-10 write-aligned features from memory_alignment_L9_H4.json
# Ranked by combined alignment score = sqrt(|k_cos| * |v_cos|)
WRITE_ALIGNED_FEATURES = [758, 927, 1903, 1901, 361, 710, 1582, 62, 1012, 460]
WRITE_ALIGNED_SCORES = {
    758: 0.7684, 927: 0.7684, 1903: 0.6976, 1901: 0.6627,
    361: 0.6583, 710: 0.6478, 1582: 0.6456, 62: 0.6444,
    1012: 0.6441, 460: 0.6374,
}


# SAE checkpoint resolution (simplified from run_modal.py)


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


# Main experiment function


@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=14400, memory=32768,
)
def run_erase_experiment(
    layer: int = 9,
    n_heads: int = 32,
    n_tokens: int = 400,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-4B-Base",
    temperature: float = 0.7,
    random_seed: int = 42,
    n_erase_features: int = 10,
) -> dict:
    """Run the memory slot erase experiment.

    For each of 20 prompts, generate 400 tokens under 3 conditions:
      1. baseline: no intervention
      2. erase_aligned: zero top-10 write-aligned features across all 32 heads
      3. erase_random: zero 10 random alive features across all 32 heads
    """
    import sys
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer
    from generation_intervention import (
        generate_with_intervention,
        compute_generation_stats,
    )

    vol.reload()
    t0 = time.time()
    rng = np.random.RandomState(random_seed)

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    corpus_source = "openwebtext"
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    # ---------------------------------------------------------------
    # Load per-head SAEs and find random alive features
    # ---------------------------------------------------------------
    sae_per_head: dict[int, object] = {}
    random_features_per_head: dict[int, list[int]] = {}
    sae_type = None
    n_loaded = 0

    for h in range(n_heads):
        try:
            ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
                layer=layer, head=h, n_features_target=n_features_target,
                corpus_source="openwebtext",
            )
        except FileNotFoundError:
            print(f"  H{h}: no checkpoint found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sae_model = build_sae_from_config(
            cfg, state_dict=ckpt["model_state_dict"],
            default_d_k=key_dim, default_d_v=val_dim,
        )
        sae_model.load_state_dict(ckpt["model_state_dict"])
        sae_model = sae_model.cuda().eval()
        this_sae_type = infer_sae_type(cfg, ckpt["model_state_dict"])
        if sae_type is None:
            sae_type = this_sae_type

        head_states_path = states_dir / f"layer_{layer}" / f"head_{h}.npy"
        if head_states_path.exists():
            head_states = np.load(str(head_states_path), mmap_mode="r")
            sample_size = min(head_states.shape[0], 5000)
            sample_idx = rng.choice(head_states.shape[0], size=sample_size, replace=False)
            sample_states = np.array(head_states[sample_idx])
            sample_tensor = torch.tensor(sample_states, dtype=torch.float32, device="cuda")
            if this_sae_type == "flat":
                sample_tensor = sample_tensor.reshape(sample_tensor.shape[0], -1)

            alive_mask = None
            for start in range(0, len(sample_tensor), 512):
                batch = sample_tensor[start:start + 512]
                coeffs = sae_model.encode(batch)
                batch_alive = (coeffs.abs() > 0).any(dim=0)
                alive_mask = batch_alive if alive_mask is None else (alive_mask | batch_alive)

            alive_indices = torch.nonzero(alive_mask, as_tuple=True)[0].cpu().numpy().tolist()
        else:
            # Fallback: treat all features as potentially alive
            alive_indices = list(range(n_features_target))

        # Pick random alive features, excluding write-aligned targets
        alive_non_aligned = [i for i in alive_indices if i not in set(WRITE_ALIGNED_FEATURES)]
        if len(alive_non_aligned) >= n_erase_features:
            random_feats = rng.choice(alive_non_aligned, size=n_erase_features, replace=False).tolist()
        else:
            random_feats = rng.choice(alive_indices, size=n_erase_features, replace=False).tolist()

        sae_per_head[h] = sae_model
        random_features_per_head[h] = random_feats
        n_loaded += 1

        print(f"  H{h}: loaded ({resolved_tag}), alive={len(alive_indices)}, "
              f"random={random_feats[:3]}...")

    if n_loaded == 0:
        raise FileNotFoundError(f"No per-head SAE checkpoints for layer {layer}.")

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")
    print(f"Write-aligned targets: {WRITE_ALIGNED_FEATURES}")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"Loading model: {model_name}")
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
    # Erase = set coefficient to 0. In the multiplicative intervention
    # framework, this is feature_updates[fi] = 0.0 (multiply by zero).
    # But we use additive=False (multiplicative mode) with scale=0.

    def _build_erase_updates(feature_list: list[int]) -> dict[int, dict[int, float]]:
        """Build per-head updates that zero specified feature indices."""
        return {
            h: {fi: 0.0 for fi in feature_list}
            for h in sae_per_head
        }

    def _build_random_erase_updates() -> dict[int, dict[int, float]]:
        """Build per-head updates that zero each head's random features."""
        return {
            h: {fi: 0.0 for fi in random_features_per_head[h]}
            for h in sae_per_head
        }

    conditions: dict[str, dict[int, dict[int, float]]] = {
        "baseline": {},
        "erase_aligned": _build_erase_updates(WRITE_ALIGNED_FEATURES),
        "erase_random": _build_random_erase_updates(),
    }

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    device = next(model.parameters()).device
    all_results: list[dict] = []
    cond_names = list(conditions.keys())
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
                additive=False,  # multiplicative: scale=0.0 zeros the coefficient
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
        erase_nl = entry["erase_aligned"]["stats"]["n_newlines"]
        rand_nl = entry["erase_random"]["stats"]["n_newlines"]
        print(
            f"  [{i+1}/{len(formatted_prompts)}] {elapsed:.0f}s elapsed, "
            f"{remaining:.0f}s remaining | "
            f"newlines: baseline={base_nl:.0f} erase_aligned={erase_nl:.0f} erase_random={rand_nl:.0f}"
        )
        all_results.append(entry)

    total_time = time.time() - t0

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
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
    # Bootstrap CIs and p-values
    # ---------------------------------------------------------------
    from scipy import stats as sp_stats

    n_bootstrap = 10000

    def bootstrap_ci(vals_a, vals_b, n_boot=n_bootstrap, alpha=0.05):
        """Bootstrap CI for mean(a) - mean(b), paired."""
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
        # Two-sided bootstrap p-value: fraction of bootstrap samples on wrong side of 0
        if observed >= 0:
            p = 2 * float(np.mean(boot_means < 0))
        else:
            p = 2 * float(np.mean(boot_means > 0))
        p = min(p, 1.0)
        return observed, ci_lo, ci_hi, p

    tests: dict[str, dict[str, dict]] = {}
    for comparison, cond_a, cond_b in [
        ("erase_aligned_vs_baseline", "erase_aligned", "baseline"),
        ("erase_random_vs_baseline", "erase_random", "baseline"),
        ("erase_aligned_vs_erase_random", "erase_aligned", "erase_random"),
    ]:
        comp = {}
        for stat in stat_names:
            vals_a = per_cond_arrays[cond_a][stat]
            vals_b = per_cond_arrays[cond_b][stat]

            # Paired t-test
            t_stat, t_pval = sp_stats.ttest_rel(vals_a, vals_b)
            # Cohen's d
            diff = np.array(vals_a) - np.array(vals_b)
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

            # Bootstrap
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
        "experiment": "memory_slot_erase",
        "layer": layer,
        "n_heads": n_loaded,
        "n_prompts": len(formatted_prompts),
        "n_tokens": n_tokens,
        "temperature": temperature,
        "sae_type": sae_type,
        "model_name": model_name,
        "intervention_mode": "multiplicative_zero",
        "total_time_s": total_time,
        "write_aligned_features": WRITE_ALIGNED_FEATURES,
        "write_aligned_scores": WRITE_ALIGNED_SCORES,
        "random_features_per_head": {
            str(h): feats for h, feats in random_features_per_head.items()
        },
        "conditions": condition_means,
        "condition_names": cond_names,
        "tests": tests,
        "per_prompt": all_results,
    }

    out_path = analysis_dir / "memory_slot_erase_4B_L9.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"MEMORY SLOT ERASE: L{layer}, {n_loaded} heads, {len(formatted_prompts)} prompts")
    print(f"Write-aligned features: {WRITE_ALIGNED_FEATURES}")
    print(f"{'='*80}")

    for cond_name in cond_names:
        m = condition_means[cond_name]
        print(f"\n  {cond_name:25s}: paragraphs={m['n_paragraphs']:.1f}  "
              f"newlines={m['n_newlines']:.1f}  sentences={m['n_sentences']:.1f}  "
              f"words={m['n_words']:.0f}  mean_word_len={m['mean_word_length']:.2f}")

    print(f"\n--- Statistical Tests ---")
    key_stats = ["n_newlines", "n_paragraphs", "n_sentences", "n_words", "mean_word_length"]
    for comp_name, comp_data in tests.items():
        print(f"\n  {comp_name}:")
        for stat in key_stats:
            s = comp_data[stat]
            sig = "*" if s["t_pvalue"] < 0.05 else ""
            print(f"    {stat:22s}: diff={s['mean_diff']:+.2f}  d={s['cohens_d']:+.3f}  "
                  f"p={s['t_pvalue']:.4f}{sig}  "
                  f"boot_ci=[{s['bootstrap_ci_95'][0]:+.2f}, {s['bootstrap_ci_95'][1]:+.2f}]  "
                  f"boot_p={s['bootstrap_pvalue']:.4f}")

    print(f"\nTotal time: {total_time:.0f}s")
    return result


# Local entrypoint


@app.local_entrypoint()
def main():
    result = run_erase_experiment.remote()

    local_out = os.path.join(os.path.dirname(__file__), "results", "data")
    os.makedirs(local_out, exist_ok=True)
    local_path = os.path.join(local_out, "memory_slot_erase_4B_L9.json")
    with open(local_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved locally to {local_path}")

    tests = result.get("tests", {})
    conds = result.get("conditions", {})

    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    if "baseline" in conds and "erase_aligned" in conds:
        base_nl = conds["baseline"]["n_newlines"]
        erase_nl = conds["erase_aligned"]["n_newlines"]
        rand_nl = conds.get("erase_random", {}).get("n_newlines", 0)
        print(f"  Newlines:    baseline={base_nl:.1f}  erase_aligned={erase_nl:.1f}  erase_random={rand_nl:.1f}")

        base_para = conds["baseline"]["n_paragraphs"]
        erase_para = conds["erase_aligned"]["n_paragraphs"]
        rand_para = conds.get("erase_random", {}).get("n_paragraphs", 0)
        print(f"  Paragraphs:  baseline={base_para:.1f}  erase_aligned={erase_para:.1f}  erase_random={rand_para:.1f}")

        base_wl = conds["baseline"]["mean_word_length"]
        erase_wl = conds["erase_aligned"]["mean_word_length"]
        rand_wl = conds.get("erase_random", {}).get("mean_word_length", 0)
        print(f"  Word length: baseline={base_wl:.2f}  erase_aligned={erase_wl:.2f}  erase_random={rand_wl:.2f}")

    comp = tests.get("erase_aligned_vs_baseline", {})
    if comp:
        nl_test = comp.get("n_newlines", {})
        print(f"\n  Erase vs Baseline (newlines): diff={nl_test.get('mean_diff', 0):+.2f}  "
              f"p={nl_test.get('t_pvalue', 1):.4f}  d={nl_test.get('cohens_d', 0):+.3f}")

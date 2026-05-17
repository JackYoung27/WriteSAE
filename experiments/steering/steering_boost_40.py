#!/usr/bin/env python3
"""Steering boost experiment with 40 prompts (power replication).

Replicates the 20-prompt additive boost experiment
(generation_intervention_qualitative_additive_4B_Base_L9.json) with double the
sample size. The original 20 prompts are preserved; 20 new prompts are appended.

Protocol (identical to original):
  - Model: Qwen/Qwen3.5-4B-Base, layer 9, temperature 0.7, 400 tokens
  - Per-head boundary-differential feature selection (10 features per head, 32 heads)
  - Additive push calibrated from mean_boundary activation
  - Conditions: baseline, boundary_boost_{2x,5x,10x}, random_boost_10x

Statistics computed inline: paired t-test, bootstrap CI, Cohen's d for all
condition pairs. Bonferroni threshold for 6 primary tests: p < 0.0083.

Usage:
    modal run --detach steering_boost_40.py
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

app = modal.App("matrix-sae-boost-40")

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

# 40 prompts: original 20 + 20 new

PROMPTS_ORIGINAL_20 = [
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

PROMPTS_NEW_20 = [
    # History
    "Describe the rise and fall of the Mongol Empire.",
    "Write a paragraph about the Industrial Revolution in Britain.",
    "Summarize the events leading to the American Civil War.",
    # Science
    "Explain how vaccines train the immune system.",
    "Describe how tectonic plates cause earthquakes.",
    "Explain the greenhouse effect and how it warms the Earth.",
    # Cooking
    "Describe how cheese is made from milk.",
    "Explain the process of fermenting vegetables into kimchi.",
    # Travel
    "Describe what a visitor would experience in Marrakech's old city.",
    "Write a paragraph about the landscape of Patagonia.",
    # Technology
    "Explain how GPS satellites determine your location.",
    "Describe how a search engine indexes and ranks web pages.",
    "Write a paragraph about the development of the transistor.",
    # Nature
    "Describe how bees organize a hive and produce honey.",
    "Explain how deep ocean currents circulate around the globe.",
    "Write a paragraph about the symbiosis between clownfish and anemones.",
    # General knowledge
    "Explain how time zones work and why they exist.",
    "Describe how a musical instrument produces sound.",
    "Write a paragraph about the history of writing systems.",
    "Explain how the human eye perceives color.",
]

ALL_40_PROMPTS = PROMPTS_ORIGINAL_20 + PROMPTS_NEW_20

# SAE checkpoint resolution (from memory_slot_suppress.py)


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
def run_boost_experiment(
    layer: int = 9,
    n_heads: int = 32,
    n_tokens: int = 400,
    n_boundary_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-4B-Base",
    temperature: float = 0.7,
    random_seed: int = 123,
    period_token_id: int = 13,
    boost_strengths: tuple[float, ...] = (2.0, 5.0, 10.0),
    random_control_strength: float = 10.0,
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
) -> dict:
    """Run the additive boost experiment with 40 prompts.

    For each prompt, generate 400 tokens under baseline, one boost condition
    per requested dose, and one random matched-dose control.

    Feature selection: per-head boundary-differential analysis. Encode boundary
    vs non-boundary states through each head's SAE, rank features by absolute
    mean activation difference, take top n_boundary_features.

    Push calibration: push = sign(mean_diff) * |mean_boundary| * strength.
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
    corpus_source = "openwebtext"
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")
    corpus_arr = np.load(str(corpus_path), mmap_mode="r")
    n_corpus_seqs = corpus_arr.shape[0]
    seq_len = corpus_arr.shape[1]

    corpus_full = np.array(corpus_arr)
    boundary_mask = np.zeros((n_corpus_seqs, seq_len), dtype=bool)
    boundary_mask[:, :-1] = (corpus_full[:, 1:] == period_token_id)

    print(f"Corpus: {n_corpus_seqs} seqs x {seq_len} tokens")
    print(f"Boundary positions: {int(boundary_mask.sum())}")

    # ---------------------------------------------------------------
    # Load per-head SAEs, select boundary features, compute push values
    # ---------------------------------------------------------------
    sae_per_head: dict[int, object] = {}
    boundary_features_per_head: dict[int, list[int]] = {}
    random_features_per_head: dict[int, list[int]] = {}
    boundary_push_per_head: dict[int, dict[int, float]] = {}
    random_push_per_head: dict[int, dict[int, float]] = {}
    feature_details_per_head: dict[str, list[dict]] = {}
    sae_type = None
    n_loaded = 0

    for h in range(n_heads):
        try:
            ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
                layer=layer, head=h, n_features_target=n_features_target,
                corpus_source="openwebtext", preferred_types=sae_types,
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

        # Load extracted states for boundary feature selection
        head_states_path = states_dir / f"layer_{layer}" / f"head_{h}.npy"
        if not head_states_path.exists():
            print(f"  H{h}: no extracted states, skipping")
            continue

        head_states = np.load(str(head_states_path), mmap_mode="r")
        n_states = head_states.shape[0]
        n_use_seqs = min(n_states // seq_len, n_corpus_seqs)
        flat_mask = boundary_mask[:n_use_seqs].reshape(-1)
        states_flat = head_states[:n_use_seqs * seq_len]

        boundary_indices = np.where(flat_mask)[0]
        nonboundary_indices = np.where(~flat_mask)[0]
        n_sample = min(len(boundary_indices), len(nonboundary_indices), 10000)
        if n_sample == 0:
            print(f"  H{h}: no boundary positions found, skipping")
            continue

        b_idx = rng.choice(boundary_indices, size=n_sample, replace=False)
        nb_idx = rng.choice(nonboundary_indices, size=n_sample, replace=False)

        # Boundary features (top by absolute mean_diff)
        top_features = select_boundary_features_fast(
            sae_model, this_sae_type,
            np.array(states_flat[b_idx]), np.array(states_flat[nb_idx]),
            n_features=n_boundary_features,
        )
        feat_indices = [f["feature_idx"] for f in top_features]

        # Random alive features (for control condition)
        sample_size = min(n_use_seqs * seq_len, 5000)
        sample_idx = rng.choice(n_use_seqs * seq_len, size=sample_size, replace=False)
        sample_states = np.array(head_states[sample_idx])
        sample_tensor = torch.tensor(sample_states, dtype=torch.float32, device="cuda")
        if this_sae_type == "flat":
            sample_tensor = sample_tensor.reshape(sample_tensor.shape[0], -1)

        all_coeffs_list = []
        alive_mask = None
        for start in range(0, len(sample_tensor), 512):
            batch = sample_tensor[start:start + 512]
            coeffs = sae_model.encode(batch)
            batch_alive = (coeffs.abs() > 0).any(dim=0)
            alive_mask = batch_alive if alive_mask is None else (alive_mask | batch_alive)
            all_coeffs_list.append(coeffs.detach().cpu())

        alive_indices = torch.nonzero(alive_mask, as_tuple=True)[0].cpu().numpy().tolist()
        alive_non_boundary = [i for i in alive_indices if i not in set(feat_indices)]
        n_random = n_boundary_features
        if len(alive_non_boundary) >= n_random:
            random_feats = rng.choice(alive_non_boundary, size=n_random, replace=False).tolist()
        else:
            random_feats = rng.choice(alive_indices, size=n_random, replace=False).tolist()

        # Compute base additive push values (1x strength)
        all_coeffs_cat = torch.cat(all_coeffs_list, dim=0)
        mean_acts = all_coeffs_cat.mean(dim=0).numpy()

        # Boundary features: push = sign(mean_diff) * |mean_boundary|
        b_push: dict[int, float] = {}
        for f in top_features:
            fi = f["feature_idx"]
            sign = 1.0 if f["mean_diff"] >= 0 else -1.0
            b_push[fi] = sign * abs(f["mean_boundary"])
        boundary_push_per_head[h] = b_push

        # Random features: push = |mean_activation|
        r_push: dict[int, float] = {}
        for fi in random_feats:
            r_push[fi] = abs(float(mean_acts[fi]))
        random_push_per_head[h] = r_push

        sae_per_head[h] = sae_model
        boundary_features_per_head[h] = feat_indices
        random_features_per_head[h] = random_feats
        feature_details_per_head[str(h)] = top_features
        n_loaded += 1

        b_vals = list(b_push.values())
        r_vals = list(r_push.values())
        print(f"  H{h}: boundary={feat_indices[:3]}... random={random_feats[:3]}... "
              f"boundary_push=[{min(b_vals):.4f},{max(b_vals):.4f}] "
              f"random_push=[{min(r_vals):.4f},{max(r_vals):.4f}]")

    if n_loaded == 0:
        raise FileNotFoundError(f"No per-head SAE checkpoints for layer {layer}.")

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")
    total_features = sum(len(v) for v in boundary_features_per_head.values())
    print(f"Total boundary features across all heads: {total_features}")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    raw_prompts = list(ALL_40_PROMPTS)
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
    conditions: dict[str, dict[int, dict[int, float]]] = {"baseline": {}}

    for strength in boost_strengths:
        label = f"boundary_boost_{strength:.0f}x" if strength == int(strength) else f"boundary_boost_{strength}x"
        cond: dict[int, dict[int, float]] = {}
        for h, push_dict in boundary_push_per_head.items():
            cond[h] = {fi: pv * strength for fi, pv in push_dict.items()}
        conditions[label] = cond

    # Random control at the matched dose
    random_label = (
        f"random_boost_{random_control_strength:.0f}x"
        if float(random_control_strength).is_integer()
        else f"random_boost_{random_control_strength:g}x"
    )
    random_cond: dict[int, dict[int, float]] = {}
    for h, push_dict in random_push_per_head.items():
        random_cond[h] = {fi: pv * random_control_strength for fi, pv in push_dict.items()}
    conditions[random_label] = random_cond

    cond_names = list(conditions.keys())
    print(f"\nConditions: {cond_names}")
    print(f"Boost strengths: {boost_strengths}")
    print(f"Random control dose: {random_control_strength}x")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    device = next(model.parameters()).device
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
        strongest_label = f"boundary_boost_{max(boost_strengths):.0f}x"
        boost_nl = entry.get(strongest_label, {}).get("stats", {}).get("n_newlines", 0)
        rand_nl = entry.get(random_label, {}).get("stats", {}).get("n_newlines", 0)
        print(
            f"  [{i+1}/{len(formatted_prompts)}] {elapsed:.0f}s elapsed, "
            f"{remaining:.0f}s remaining | "
            f"newlines: baseline={base_nl:.0f} {strongest_label}={boost_nl:.0f} {random_label}={rand_nl:.0f}"
        )
        all_results.append(entry)

    total_time = time.time() - t0

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    import numpy as np
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
    # Bootstrap CIs and p-values
    # ---------------------------------------------------------------
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
        if observed >= 0:
            p = 2 * float(np.mean(boot_means < 0))
        else:
            p = 2 * float(np.mean(boot_means > 0))
        p = min(p, 1.0)
        return observed, ci_lo, ci_hi, p

    comparisons = []
    for cond_name in cond_names:
        if cond_name == "baseline":
            continue
        comparisons.append((f"{cond_name}_vs_baseline", cond_name, "baseline"))
    # Strongest boost vs matched random
    strongest_label = f"boundary_boost_{max(boost_strengths):.0f}x"
    if strongest_label in cond_names and random_label in cond_names:
        comparisons.append((f"{strongest_label}_vs_{random_label}", strongest_label, random_label))

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
    # Subset analysis: first-20 vs second-20 consistency
    # ---------------------------------------------------------------
    subset_tests = {}
    for subset_name, idx_range in [("original_20", range(0, 20)), ("new_20", range(20, 40))]:
        subset_comp = {}
        for stat in stat_names:
            for cond_a_label in [strongest_label]:
                vals_a = [all_results[i][cond_a_label]["stats"][stat] for i in idx_range]
                vals_b = [all_results[i]["baseline"]["stats"][stat] for i in idx_range]
                t_stat, t_pval = sp_stats.ttest_rel(vals_a, vals_b)
                diff = np.array(vals_a) - np.array(vals_b)
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1))
                cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0
                subset_comp[stat] = {
                    "mean_diff": mean_diff,
                    "t_pvalue": float(t_pval),
                    "cohens_d": cohens_d,
                }
        subset_tests[subset_name] = subset_comp

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    analysis_dir = Path(f"{DATA}/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "steering_boost_40",
        "layer": layer,
        "n_heads": n_loaded,
        "n_prompts": len(formatted_prompts),
        "n_tokens": n_tokens,
        "temperature": temperature,
        "sae_type": sae_type,
        "model_name": model_name,
        "intervention_mode": "additive",
        "boost_strengths": list(boost_strengths),
        "random_control_strength": float(random_control_strength),
        "n_boundary_features": n_boundary_features,
        "total_time_s": total_time,
        "boundary_features_per_head": {
            str(h): feats for h, feats in boundary_features_per_head.items()
        },
        "random_features_per_head": {
            str(h): feats for h, feats in random_features_per_head.items()
        },
        "boundary_push_per_head": {
            str(h): {str(fi): v for fi, v in pushes.items()}
            for h, pushes in boundary_push_per_head.items()
        },
        "random_push_per_head": {
            str(h): {str(fi): v for fi, v in pushes.items()}
            for h, pushes in random_push_per_head.items()
        },
        "feature_details_per_head": feature_details_per_head,
        "conditions": condition_means,
        "condition_names": cond_names,
        "tests": tests,
        "subset_tests": subset_tests,
        "per_prompt": all_results,
    }
    filename = f"steering_boost_40_4B_L{layer}.json"
    out_path = analysis_dir / filename
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")
    result["output_path"] = str(out_path)

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"STEERING BOOST (40 prompts): L{layer}, {n_loaded} heads, {len(formatted_prompts)} prompts")
    print(f"{'='*80}")

    for cond_name in cond_names:
        m = condition_means[cond_name]
        print(f"\n  {cond_name:25s}: paragraphs={m['n_paragraphs']:.1f}  "
              f"newlines={m['n_newlines']:.1f}  sentences={m['n_sentences']:.1f}  "
              f"words={m['n_words']:.0f}  mean_word_len={m['mean_word_length']:.2f}")

    print(f"\n--- Statistical Tests (Bonferroni threshold: p < 0.0083) ---")
    key_stats = ["n_newlines", "n_paragraphs", "n_sentences", "n_words", "mean_word_length"]
    for comp_name, comp_data in tests.items():
        print(f"\n  {comp_name}:")
        for stat in key_stats:
            s = comp_data[stat]
            sig = "***" if s["t_pvalue"] < 0.001 else "**" if s["t_pvalue"] < 0.0083 else "*" if s["t_pvalue"] < 0.05 else ""
            print(f"    {stat:22s}: diff={s['mean_diff']:+.2f}  d={s['cohens_d']:+.3f}  "
                  f"p={s['t_pvalue']:.4f}{sig}  "
                  f"boot_ci=[{s['bootstrap_ci_95'][0]:+.2f}, {s['bootstrap_ci_95'][1]:+.2f}]  "
                  f"boot_p={s['bootstrap_pvalue']:.4f}")

    print(f"\n--- Subset Consistency ---")
    for subset_name, subset_data in subset_tests.items():
        print(f"\n  {subset_name} ({strongest_label} vs baseline):")
        for stat in key_stats:
            s = subset_data[stat]
            print(f"    {stat:22s}: diff={s['mean_diff']:+.2f}  d={s['cohens_d']:+.3f}  p={s['t_pvalue']:.4f}")

    print(f"\nTotal time: {total_time:.0f}s")
    return result


# Local entrypoint


@app.local_entrypoint()
def main():
    result = run_boost_experiment.remote()

    local_out = os.path.join(os.path.dirname(__file__), "results", "data")
    os.makedirs(local_out, exist_ok=True)
    local_name = os.path.basename(result.get("output_path", "steering_boost_40_4B_L9.json"))
    local_path = os.path.join(local_out, local_name)
    with open(local_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved locally to {local_path}")

    tests = result.get("tests", {})
    conds = result.get("conditions", {})

    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    if "baseline" in conds:
        base_nl = conds["baseline"]["n_newlines"]
        base_para = conds["baseline"]["n_paragraphs"]
        base_wl = conds["baseline"]["mean_word_length"]
        print(f"  Baseline:    newlines={base_nl:.1f}  paragraphs={base_para:.1f}  word_len={base_wl:.2f}")

        for cond_name in [c for c in conds if c != "baseline"]:
            nl = conds[cond_name]["n_newlines"]
            para = conds[cond_name]["n_paragraphs"]
            wl = conds[cond_name]["mean_word_length"]
            print(f"  {cond_name:25s}: newlines={nl:.1f}  paragraphs={para:.1f}  word_len={wl:.2f}")

    for comp_name in ["boundary_boost_10x_vs_baseline", "boundary_boost_10x_vs_random_boost_10x"]:
        comp = tests.get(comp_name, {})
        if comp:
            nl_test = comp.get("n_newlines", {})
            para_test = comp.get("n_paragraphs", {})
            print(f"\n  {comp_name}:")
            print(f"    newlines:    diff={nl_test.get('mean_diff', 0):+.2f}  "
                  f"p={nl_test.get('t_pvalue', 1):.4f}  d={nl_test.get('cohens_d', 0):+.3f}")
            print(f"    paragraphs:  diff={para_test.get('mean_diff', 0):+.2f}  "
                  f"p={para_test.get('t_pvalue', 1):.4f}  d={para_test.get('cohens_d', 0):+.3f}")

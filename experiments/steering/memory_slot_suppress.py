#!/usr/bin/env python3
"""Memory Slot Suppress: clipped additive suppression on boundary-differential SAE features.

Reverse of the additive boost experiment. Instead of amplifying boundary-correlated
features with a positive push, subtract from those coefficients and clamp at zero.
That matches the SAE's nonnegative TopK+ReLU code space: suppression becomes an
actual feature ablation rather than an out-of-distribution sign flip.

Uses per-head boundary-differential feature selection (same as the successful boost
experiment) so each head's SAE features are selected independently. The default
family keeps only positive boundary-differential features, which makes "suppress"
mean "move opposite the boundary direction."

Conditions per prompt:
  1. baseline: no intervention
  2. suppress_{dose}: subtract calibrated boundary activation, then clamp at zero
  3. random_suppress_{dose}: same protocol with random alive features

Uses the same 20 instruction prompts, model (Qwen3.5-4B-Base), layer (9), generation
parameters (400 tokens, temperature 0.7) as the boost experiment for direct comparison.

Usage:
    modal run --detach memory_slot_suppress.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

# Modal infrastructure (mirrors memory_slot_erase.py)


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

app = modal.App("matrix-sae-suppress")

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


# SAE checkpoint resolution (from memory_slot_erase.py)


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


def _format_strength_slug(strengths: tuple[float, ...]) -> str:
    parts = []
    for strength in strengths:
        if float(strength).is_integer():
            parts.append(f"{int(strength)}x")
        else:
            parts.append(f"{strength:g}x")
    return "_".join(parts)


def _calibration_value(feature_info: dict, calibration: str) -> float:
    if calibration == "mean_nonzero":
        return max(abs(float(feature_info.get("mean_boundary_nonzero", 0.0))), 0.01)
    if calibration == "mean_boundary":
        return max(abs(float(feature_info.get("mean_boundary", 0.0))), 0.01)
    raise ValueError(f"Unknown calibration mode: {calibration}")


# Main experiment function


@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    gpu="A10G", image=image, timeout=14400, memory=32768,
)
def run_suppress_experiment(
    layer: int = 9,
    n_heads: int = 32,
    n_tokens: int = 400,
    n_boundary_features: int = 5,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-4B-Base",
    temperature: float = 0.7,
    random_seed: int = 123,
    period_token_id: int = 13,
    suppress_strengths: tuple[float, ...] = (1.0, 2.0, 5.0),
    boundary_sign: str = "positive",
    calibration: str = "mean_nonzero",
    clip_min_zero: bool = True,
    random_control_strength: float = 5.0,
    result_tag: str = "",
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
) -> dict:
    """Run the memory slot suppression experiment.

    For each of 20 prompts, generate 400 tokens under a baseline, one suppress
    condition per requested dose, and one random matched-dose control.

    Default settings target the clean operating range:
      1. baseline: no intervention
      2. suppress_1x / suppress_2x / suppress_5x
      3. random_suppress_5x

    Feature selection uses per-head boundary-differential analysis (same as
    the successful boost experiment): encode boundary vs non-boundary states
    through each head's SAE, rank features by mean activation difference,
    keep the requested sign family, and take the top n_boundary_features.
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
    boundary_push_per_head: dict[int, dict[int, float]] = {}  # base push (1x)
    random_push_per_head: dict[int, dict[int, float]] = {}    # base push (1x)
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

        # Boundary-differential feature selection (same as boost experiment),
        # then keep the requested sign family.
        ranked_features = select_boundary_features_fast(
            sae_model, this_sae_type,
            np.array(states_flat[b_idx]), np.array(states_flat[nb_idx]),
            n_features=n_features_target,
        )
        if boundary_sign == "positive":
            top_features = [f for f in ranked_features if f["mean_diff"] > 0]
        elif boundary_sign == "negative":
            top_features = [f for f in ranked_features if f["mean_diff"] < 0]
        elif boundary_sign == "mixed":
            top_features = ranked_features
        else:
            raise ValueError(f"Unknown boundary_sign: {boundary_sign}")
        top_features = top_features[:n_boundary_features]
        if len(top_features) < n_boundary_features:
            print(
                f"  H{h}: only found {len(top_features)} {boundary_sign} features; "
                f"requested {n_boundary_features}"
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

        # Compute base additive push values (1x strength).
        all_coeffs_cat = torch.cat(all_coeffs_list, dim=0)
        mean_acts = all_coeffs_cat.mean(dim=0).numpy()

        b_push: dict[int, float] = {}
        for f in top_features:
            fi = f["feature_idx"]
            sign = 1.0 if f["mean_diff"] >= 0 else -1.0
            b_push[fi] = sign * _calibration_value(f, calibration)
        boundary_push_per_head[h] = b_push

        r_push: dict[int, float] = {}
        for fi in random_feats:
            acts = all_coeffs_cat[:, fi].numpy()
            nonzero = acts[acts > 0]
            if calibration == "mean_nonzero" and len(nonzero) > 0:
                r_push[fi] = max(abs(float(nonzero.mean())), 0.01)
            else:
                r_push[fi] = max(abs(float(mean_acts[fi])), 0.01)
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
    # Suppression = move opposite the boundary direction, then clamp at zero.
    conditions: dict[str, dict[int, dict[int, float]]] = {"baseline": {}}

    for strength in suppress_strengths:
        label = f"suppress_{strength:.0f}x" if strength == int(strength) else f"suppress_{strength}x"
        cond: dict[int, dict[int, float]] = {}
        for h, push_dict in boundary_push_per_head.items():
            # Negate: if boost adds +push*strength, suppress subtracts it
            cond[h] = {fi: -pv * strength for fi, pv in push_dict.items()}
        conditions[label] = cond

    # Random control at the requested matched dose
    max_strength = random_control_strength
    random_label = (
        f"random_suppress_{max_strength:.0f}x"
        if float(max_strength).is_integer()
        else f"random_suppress_{max_strength:g}x"
    )
    random_cond: dict[int, dict[int, float]] = {}
    for h, push_dict in random_push_per_head.items():
        random_cond[h] = {fi: -pv * max_strength for fi, pv in push_dict.items()}
    conditions[random_label] = random_cond

    cond_names = list(conditions.keys())
    print(f"\nConditions: {cond_names}")
    print(f"Suppress strengths: {suppress_strengths}")
    print(f"Random control dose: {max_strength}x")
    print(f"Boundary sign family: {boundary_sign}")
    print(f"Calibration: {calibration}")
    print(f"Clamp at zero: {clip_min_zero}")

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
                additive=True,  # additive mode (no compounding)
                additive_min_zero=clip_min_zero,
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
        strongest_strength = max(suppress_strengths) if suppress_strengths else 0.0
        strongest_label = (
            f"suppress_{strongest_strength:.0f}x"
            if float(strongest_strength).is_integer()
            else f"suppress_{strongest_strength:g}x"
        )
        s2_nl = entry.get("suppress_2x", {}).get("stats", {}).get("n_newlines", 0)
        strongest_nl = entry.get(strongest_label, {}).get("stats", {}).get("n_newlines", 0)
        print(
            f"  [{i+1}/{len(formatted_prompts)}] {elapsed:.0f}s elapsed, "
            f"{remaining:.0f}s remaining | "
            f"newlines: baseline={base_nl:.0f} suppress_2x={s2_nl:.0f} {strongest_label}={strongest_nl:.0f}"
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

    # Build comparisons: each suppress condition vs baseline, plus random vs baseline
    comparisons = []
    for cond_name in cond_names:
        if cond_name == "baseline":
            continue
        comparisons.append((f"{cond_name}_vs_baseline", cond_name, "baseline"))
    # Also compare the strongest suppress condition against the matched random control.
    strongest_strength = max(suppress_strengths) if suppress_strengths else 0.0
    strongest_label = (
        f"suppress_{strongest_strength:.0f}x"
        if float(strongest_strength).is_integer()
        else f"suppress_{strongest_strength:g}x"
    )
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
    # ---------------------------------------------------------------
    analysis_dir = Path(f"{DATA}/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "memory_slot_suppress",
        "layer": layer,
        "n_heads": n_loaded,
        "n_prompts": len(formatted_prompts),
        "n_tokens": n_tokens,
        "temperature": temperature,
        "sae_type": sae_type,
        "model_name": model_name,
        "intervention_mode": "additive_negative",
        "clip_min_zero": clip_min_zero,
        "suppress_strengths": list(suppress_strengths),
        "random_control_strength": float(max_strength),
        "boundary_sign": boundary_sign,
        "calibration": calibration,
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
        "per_prompt": all_results,
    }
    strength_slug = _format_strength_slug(suppress_strengths)
    sign_slug = {"positive": "pos", "negative": "neg", "mixed": "mix"}[boundary_sign]
    clip_slug = "clip0" if clip_min_zero else "signed"
    tag_bits = [f"top{n_boundary_features}", sign_slug, calibration, clip_slug, strength_slug]
    if result_tag:
        tag_bits.append(result_tag)
    filename = f"memory_slot_suppress_4B_L{layer}_{'_'.join(tag_bits)}.json"
    out_path = analysis_dir / filename
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")
    result["output_path"] = str(out_path)

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"MEMORY SLOT SUPPRESS: L{layer}, {n_loaded} heads, {len(formatted_prompts)} prompts")
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
def main(
    n_boundary_features: int = 5,
    suppress_strengths: str = "1,2,5",
    boundary_sign: str = "positive",
    calibration: str = "mean_nonzero",
    clip_min_zero: bool = True,
    random_control_strength: float = 5.0,
    result_tag: str = "",
):
    strengths = tuple(
        float(part.strip())
        for part in suppress_strengths.split(",")
        if part.strip()
    )
    result = run_suppress_experiment.remote(
        n_boundary_features=n_boundary_features,
        suppress_strengths=strengths,
        boundary_sign=boundary_sign,
        calibration=calibration,
        clip_min_zero=clip_min_zero,
        random_control_strength=random_control_strength,
        result_tag=result_tag,
    )

    local_out = os.path.join(os.path.dirname(__file__), "results", "data")
    os.makedirs(local_out, exist_ok=True)
    local_name = os.path.basename(result.get("output_path", "")) if result.get("output_path") else ""
    if not local_name:
        strength_slug = _format_strength_slug(strengths)
        sign_slug = {"positive": "pos", "negative": "neg", "mixed": "mix"}[boundary_sign]
        clip_slug = "clip0" if clip_min_zero else "signed"
        tag_bits = [f"top{n_boundary_features}", sign_slug, calibration, clip_slug, strength_slug]
        if result_tag:
            tag_bits.append(result_tag)
        local_name = f"memory_slot_suppress_4B_L9_{'_'.join(tag_bits)}.json"
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

        display_conditions = [name for name in conds if name != "baseline"]
        for cond_name in display_conditions:
            if cond_name in conds:
                nl = conds[cond_name]["n_newlines"]
                para = conds[cond_name]["n_paragraphs"]
                wl = conds[cond_name]["mean_word_length"]
                print(f"  {cond_name:20s}: newlines={nl:.1f}  paragraphs={para:.1f}  word_len={wl:.2f}")

    strongest_strength = max(strengths) if strengths else 0.0
    strongest_label = (
        f"suppress_{strongest_strength:.0f}x"
        if float(strongest_strength).is_integer()
        else f"suppress_{strongest_strength:g}x"
    )
    random_label = (
        f"random_suppress_{random_control_strength:.0f}x"
        if float(random_control_strength).is_integer()
        else f"random_suppress_{random_control_strength:g}x"
    )

    comp = tests.get(f"{strongest_label}_vs_baseline", {})
    if comp:
        nl_test = comp.get("n_newlines", {})
        para_test = comp.get("n_paragraphs", {})
        print(f"\n  {strongest_label} vs baseline:")
        print(f"    newlines:    diff={nl_test.get('mean_diff', 0):+.2f}  "
              f"p={nl_test.get('t_pvalue', 1):.4f}  d={nl_test.get('cohens_d', 0):+.3f}")
        print(f"    paragraphs:  diff={para_test.get('mean_diff', 0):+.2f}  "
              f"p={para_test.get('t_pvalue', 1):.4f}  d={para_test.get('cohens_d', 0):+.3f}")

    comp2 = tests.get(f"{strongest_label}_vs_{random_label}", {})
    if comp2:
        nl_test = comp2.get("n_newlines", {})
        print(f"\n  {strongest_label} vs {random_label} (specificity):")
        print(f"    newlines:    diff={nl_test.get('mean_diff', 0):+.2f}  "
              f"p={nl_test.get('t_pvalue', 1):.4f}  d={nl_test.get('cohens_d', 0):+.3f}")

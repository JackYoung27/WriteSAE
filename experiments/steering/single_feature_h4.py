#!/usr/bin/env python3
"""Single-feature intervention demo for H4 (boundary head).

Tests whether individual boundary features on L9H4 produce visible
structural changes (paragraph breaks, sentence endings) when boosted
with additive intervention during instruction-prompted generation.

Usage:
    modal run --detach single_feature_h4.py
"""

import json
import os
import subprocess
from pathlib import Path

import modal


def _current_code_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (FileNotFoundError, OSError):
        sha = ""
    return sha or "unknown"


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
        "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++",
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
    .add_local_file("train.py", "/root/train.py", copy=True)
    .add_local_file("generation_intervention.py", "/root/generation_intervention.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CURRENT_CODE_SHA})
)

app = modal.App("matrix-sae-h4-demo")
vol = modal.Volume.from_name("matrix-sae-data", create_if_missing=True)
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

DATA = "/data"
MODELS = "/models"

# 10 instruction prompts (same set used in the H12 experiment)
PROMPTS = [
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
]

# Target features on H4 (boundary head)
# F121: sentence boundary rho=0.306
# F9:   boundary rho=0.281
# F137: boundary rho=0.238
# F212, F382: additional boundary features
TARGET_FEATURES = [121, 9, 137, 212, 382]


@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    gpu="L4", image=image, timeout=7200, memory=32768,
)
def single_feature_h4_demo(
    layer: int = 9,
    head: int = 4,
    n_tokens: int = 400,
    temperature: float = 0.7,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    n_features_target: int = 2048,
    boost_multiplier: float = 2.0,
) -> dict:
    """Run single-feature additive intervention on L9H4 boundary features.

    For each of 5 target features x 10 prompts, generates text under 3 conditions:
      1. baseline (no intervention)
      2. boost (+2x mean_act additive push per step)
      3. suppress (-2x mean_act additive push per step)

    Returns full text + document statistics for each combination.
    """
    import sys
    import time

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

    # --- Load metadata ---
    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    # --- Resolve SAE checkpoint for L9H4 ---
    def _experiment_tag(m, sl, ns):
        slug = m.split("/")[-1].lower().replace(".", "_")
        return f"{slug}_sl{sl}_ns{ns}"

    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag

    best_ckpt = None
    best_cfg = None
    preferred_types = ("bilinear", "bilinear_tied", "rank1")

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
        best_ckpt = bp
        best_cfg = cfg
        break

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No SAE checkpoint for layer={layer}, head={head}, nf={n_features_target}"
        )

    print(f"SAE checkpoint: {best_ckpt}")
    print(f"SAE type: {best_cfg.get('sae_type')}")

    # --- Load SAE ---
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae_type = infer_sae_type(best_cfg, ckpt["model_state_dict"])
    sae_model = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae_model.load_state_dict(ckpt["model_state_dict"])
    sae_model = sae_model.cuda().eval()

    # --- Compute mean activations for target features ---
    # Encode a sample of extracted states to get mean activation per feature
    head_states_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_states_path.exists():
        raise FileNotFoundError(f"No extracted states at {head_states_path}")

    head_states = np.load(str(head_states_path), mmap_mode="r")
    n_total = head_states.shape[0]
    sample_size = min(n_total, 10000)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, size=sample_size, replace=False)
    sample_states = np.array(head_states[sample_idx])

    # Encode in batches
    all_coeffs = []
    batch_size = 512
    for start in range(0, len(sample_states), batch_size):
        batch = torch.tensor(
            sample_states[start:start + batch_size],
            dtype=torch.float32, device="cuda",
        )
        if sae_type == "flat":
            batch = batch.reshape(batch.shape[0], -1)
        coeffs = sae_model.encode(batch)
        all_coeffs.append(coeffs.detach().cpu().numpy())
    all_coeffs_np = np.concatenate(all_coeffs, axis=0)

    # Mean activation for each target feature
    mean_acts = {}
    for fi in TARGET_FEATURES:
        acts = all_coeffs_np[:, fi]
        # Mean over nonzero (active) positions for a meaningful push
        nonzero = acts[acts > 0]
        if len(nonzero) > 0:
            mean_acts[fi] = float(np.mean(nonzero))
        else:
            mean_acts[fi] = float(np.mean(np.abs(acts))) if np.any(acts != 0) else 0.1

    print(f"\nTarget features mean activations:")
    for fi in TARGET_FEATURES:
        alive_frac = float((all_coeffs_np[:, fi] > 0).mean())
        print(f"  F{fi}: mean_act={mean_acts[fi]:.4f}, alive_frac={alive_frac:.3f}")

    # --- Load model ---
    print(f"\nLoading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")
    device = next(model.parameters()).device

    # --- Format prompts ---
    formatted_prompts = []
    for p in PROMPTS:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = text.replace("<think>\n\n</think>\n\n", "")
        formatted_prompts.append(text)

    print(f"\nFormatted {len(formatted_prompts)} prompts")
    print(f"Example: {formatted_prompts[0][:200]}")

    # --- Run generation: 5 features x 3 conditions x 10 prompts ---
    all_results = []
    total_runs = len(TARGET_FEATURES) * 3 * len(PROMPTS)
    run_count = 0
    t_gen = time.time()

    for fi in TARGET_FEATURES:
        push_value = boost_multiplier * mean_acts[fi]
        print(f"\n{'='*60}")
        print(f"Feature F{fi}: push={push_value:.4f} ({boost_multiplier}x mean_act={mean_acts[fi]:.4f})")
        print(f"{'='*60}")

        feature_results = {
            "feature_idx": fi,
            "mean_act": mean_acts[fi],
            "push_value": push_value,
            "per_prompt": [],
        }

        for pi, (raw_prompt, fmt_prompt) in enumerate(zip(PROMPTS, formatted_prompts)):
            prompt_ids = tokenizer(fmt_prompt, return_tensors="pt")["input_ids"].to(device)
            entry = {"prompt_idx": pi, "prompt_text": raw_prompt}

            conditions = {
                "baseline": {},  # no intervention
                "boost": {fi: +push_value},
                "suppress": {fi: -push_value},
            }

            for cond_name, feat_updates in conditions.items():
                gen_ids, meta = generate_with_intervention(
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae_model,
                    sae_type=sae_type,
                    layer_idx=layer,
                    head_idx=head,
                    prompt_ids=prompt_ids,
                    feature_updates=feat_updates if feat_updates else None,
                    n_tokens=n_tokens,
                    temperature=temperature,
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
                run_count += 1

            elapsed = time.time() - t_gen
            rate = run_count / elapsed if elapsed > 0 else 1
            remaining = (total_runs - run_count) / rate if rate > 0 else 0

            bp = entry["baseline"]["stats"]["n_paragraphs"]
            bop = entry["boost"]["stats"]["n_paragraphs"]
            sp = entry["suppress"]["stats"]["n_paragraphs"]
            bn = entry["baseline"]["stats"]["n_newlines"]
            bon = entry["boost"]["stats"]["n_newlines"]
            sn = entry["suppress"]["stats"]["n_newlines"]
            print(
                f"  F{fi} [{pi+1}/{len(PROMPTS)}] {elapsed:.0f}s/{remaining:.0f}s rem | "
                f"paras: {bp:.0f}/{bop:.0f}/{sp:.0f}  newlines: {bn:.0f}/{bon:.0f}/{sn:.0f}"
            )
            feature_results["per_prompt"].append(entry)

        all_results.append(feature_results)

    total_time = time.time() - t0

    # --- Aggregate statistics per feature ---
    stat_names = list(compute_generation_stats("test text.").keys())
    feature_summaries = []

    for feat_result in all_results:
        fi = feat_result["feature_idx"]
        summary = {"feature_idx": fi, "mean_act": feat_result["mean_act"], "push_value": feat_result["push_value"]}
        for cond in ["baseline", "boost", "suppress"]:
            means = {}
            for stat in stat_names:
                vals = [e[cond]["stats"][stat] for e in feat_result["per_prompt"]]
                means[stat] = float(np.mean(vals))
            summary[cond] = means
        feature_summaries.append(summary)

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print(f"SINGLE-FEATURE DEMO: L{layer}H{head} (boundary head)")
    print(f"5 features x 3 conditions x {len(PROMPTS)} prompts = {total_runs} generations")
    print(f"Total time: {total_time:.0f}s")
    print(f"{'='*80}")

    header = f"{'Feature':>10} | {'mean_act':>8} | {'Stat':>20} | {'Baseline':>9} | {'Boost':>9} | {'Suppress':>9} | {'B-Base':>8} | {'S-Base':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    key_stats = ["n_paragraphs", "n_newlines", "n_sentences", "period_density", "n_words"]
    for fs in feature_summaries:
        for si, stat in enumerate(key_stats):
            base = fs["baseline"][stat]
            boost = fs["boost"][stat]
            supp = fs["suppress"][stat]
            feat_label = f"F{fs['feature_idx']}" if si == 0 else ""
            act_label = f"{fs['mean_act']:.4f}" if si == 0 else ""
            print(
                f"{feat_label:>10} | {act_label:>8} | {stat:>20} | {base:>9.2f} | "
                f"{boost:>9.2f} | {supp:>9.2f} | {boost-base:>+8.2f} | {supp-base:>+8.2f}"
            )
        print("-" * len(header))

    # --- Print example texts ---
    print(f"\n{'='*80}")
    print("EXAMPLE COMPARISONS (F121, first 3 prompts)")
    print(f"{'='*80}")
    f121_results = [r for r in all_results if r["feature_idx"] == 121]
    if f121_results:
        for entry in f121_results[0]["per_prompt"][:3]:
            print(f"\n--- Prompt {entry['prompt_idx']}: {entry['prompt_text']} ---")
            for cond in ["baseline", "boost", "suppress"]:
                text = entry[cond]["text"]
                paras = entry[cond]["stats"]["n_paragraphs"]
                nls = entry[cond]["stats"]["n_newlines"]
                display = text[:500].replace("\n\n", "\n\n[PARA]\n\n").replace("\n", "\\n\n")
                print(f"\n  [{cond}] (paragraphs={paras:.0f}, newlines={nls:.0f}):")
                print(f"  {display[:500]}")

    # --- Save results ---
    result = {
        "experiment": "single_feature_h4_demo",
        "layer": layer,
        "head": head,
        "target_features": TARGET_FEATURES,
        "n_prompts": len(PROMPTS),
        "n_tokens": n_tokens,
        "boost_multiplier": boost_multiplier,
        "temperature": temperature,
        "sae_type": sae_type,
        "model_name": model_name,
        "total_time_s": total_time,
        "mean_acts": {str(k): v for k, v in mean_acts.items()},
        "feature_summaries": feature_summaries,
        "per_feature": all_results,
    }

    analysis_dir = Path(f"{DATA}/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / f"single_feature_demo_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    return result


@app.local_entrypoint()
def main():
    result = single_feature_h4_demo.remote()
    local_out = os.path.join(os.path.dirname(__file__), "results", "data")
    os.makedirs(local_out, exist_ok=True)
    local_name = "single_feature_demo_L9_H4.json"
    with open(os.path.join(local_out, local_name), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nLocal copy saved to results/data/{local_name}")
    print(f"Total time: {result.get('total_time_s', 0):.0f}s")

"""k-head progressive replacement curve: PPL vs number of heads replaced.

Tests whether flat SAE errors compound superlinearly across heads while
bilinear errors stay linear. By default this uses the clean per-head matched
checkpoints from `matrix-sae-data-08b-clean`; the older encoder-swap ablation
checkpoints remain available as a fallback for provenance checks.

Usage:
    modal run experiments/run_khead_curve.py
    modal run experiments/run_khead_curve.py --checkpoint-source ablation
"""
import json
import math
import os
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
    .add_local_dir(str(_core_dir), "/root/core", copy=True)
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .add_local_file(str(_analysis_dir / "probe_features.py"), "/root/probe_features.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("khead-curve")
data_vol = modal.Volume.from_name("matrix-sae-data-08b-clean")
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
ablation_vol = modal.Volume.from_name("layer-encoder-swap-v1")

DATA = "/data"
MODELS = "/models"
ABLATION = "/ablation"
MODEL_08B = "Qwen/Qwen3.5-0.8B"
MATCHED_CHECKPOINT_ROOT = f"{DATA}/checkpoints/qwen3_5-0_8b_sl1024_ns5000_ultrachat_200k"


@app.function(
    volumes={DATA: data_vol, MODELS: model_vol, ABLATION: ablation_vol},
    gpu="A10G", image=image, timeout=14400, memory=32768,
)
def khead_replacement_curve(
    layer: int = 9,
    k_values: str = "1,2,4,8,12,16",
    sae_types: str = "rank1,flat",
    n_subsets: int = 5,
    n_sequences: int = 200,
    seed: int = 42,
    checkpoint_source: str = "matched",
) -> dict[str, Any]:
    """Progressive k-head replacement: replace k heads, measure PPL.

    For each k, sample n_subsets random subsets of k heads.
    Compare bilinear vs flat scaling curves.
    """
    import sys
    sys.path.insert(0, "/root")
    import torch
    import numpy as np

    from evaluate_downstream import (
        load_sae_from_checkpoint,
        reconstruct_state_head,
        _patch_gdn_initial_states,
    )
    from extract_states import load_model_and_tokenizer, get_gdn_layer_indices

    k_values_list = [int(part.strip()) for part in k_values.split(",") if part.strip()]
    sae_type_list = [part.strip() for part in sae_types.split(",") if part.strip()]

    print(f"=== k-head replacement curve, L{layer}, source={checkpoint_source} ===")
    print(f"  k values: {k_values_list}, {n_subsets} subsets each, {n_sequences} sequences")
    print(f"  SAE types: {sae_type_list}")

    model, tokenizer, config = load_model_and_tokenizer(MODEL_08B, device="cuda")
    model.eval()
    gdn_layers = get_gdn_layer_indices(config)

    corpus_ids = np.load(f"{DATA}/states/corpus.npy")
    n_seq = min(n_sequences, len(corpus_ids))
    seq_len = corpus_ids.shape[1]
    split_pos = seq_len // 2

    rng = np.random.default_rng(seed)

    results = {
        "layer": layer,
        "k_values": k_values_list,
        "sae_types": sae_type_list,
        "n_subsets": n_subsets,
        "checkpoint_source": checkpoint_source,
    }

    for sae_type in sae_type_list:
        print(f"\n  === {sae_type} ===")

        head_saes = {}
        for h in range(16):
            if checkpoint_source == "matched":
                ckpt_dir = Path(MATCHED_CHECKPOINT_ROOT) / f"{sae_type}_L{layer}_H{h}_nf2048_k32_s{seed}"
            elif checkpoint_source == "ablation":
                ckpt_dir = Path(ABLATION) / f"layer_{layer}" / f"head_{h}" / f"{sae_type}_s{seed}"
            else:
                raise ValueError(f"Unknown checkpoint_source={checkpoint_source}")

            best_path = ckpt_dir / "best.pt"
            if not best_path.exists():
                print(f"    missing checkpoint for H{h}: {best_path}")
                continue
            sae, cfg, _ = load_sae_from_checkpoint(
                str(best_path),
                str(ckpt_dir / "config.json") if (ckpt_dir / "config.json").exists() else None,
                device="cuda",
            )
            sae.eval()
            head_saes[h] = (sae, sae_type)

        if len(head_saes) < 16:
            print(f"  Only {len(head_saes)}/16 SAEs, skipping")
            continue

        curve = {}
        for k in k_values_list:
            subset_ppls = []

            if k == 16:
                subsets = [list(range(16))]  # only one way to pick all 16
            else:
                subsets = [sorted(rng.choice(16, size=k, replace=False).tolist())
                           for _ in range(n_subsets)]

            for subset_idx, head_subset in enumerate(subsets):
                total_loss = 0.0
                total_tokens = 0

                for seq_i in range(n_seq):
                    input_ids = torch.tensor(
                        corpus_ids[seq_i:seq_i + 1], dtype=torch.long, device="cuda"
                    )
                    prefix = input_ids[:, :split_pos]
                    suffix = input_ids[:, split_pos:]

                    # Fresh prefix
                    prefix_out = model(input_ids=prefix, use_cache=True)
                    cache = prefix_out.past_key_values
                    gdn_states = {}
                    for idx in gdn_layers:
                        lc = cache.layers[idx]
                        if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                            gdn_states[idx] = lc.recurrent_states.clone()

                    # Replace only the k selected heads
                    state = gdn_states[layer]
                    for h in head_subset:
                        sae, stype = head_saes[h]
                        original = state[0, h].float()
                        reconstructed = reconstruct_state_head(sae, original, stype)
                        state[0, h] = reconstructed.to(state.dtype)

                    with _patch_gdn_initial_states(model, gdn_layers, gdn_states):
                        out = model(
                            input_ids=suffix, past_key_values=cache,
                            use_cache=False, labels=suffix,
                        )
                    n_tok = suffix.shape[1] - 1
                    total_loss += out.loss.item() * n_tok
                    total_tokens += n_tok

                ppl = math.exp(total_loss / total_tokens)
                subset_ppls.append(ppl)

                if k <= 4:
                    print(f"    k={k} subset {subset_idx} ({head_subset[:4]}...): PPL={ppl:.4f}")

            curve[k] = {
                "mean_ppl": float(np.mean(subset_ppls)),
                "std_ppl": float(np.std(subset_ppls)),
                "all_ppls": [float(p) for p in subset_ppls],
            }
            print(f"  k={k}: PPL={np.mean(subset_ppls):.4f} ± {np.std(subset_ppls):.4f}")

        results[sae_type] = curve

        # Free SAEs
        del head_saes
        torch.cuda.empty_cache()

    # Compute baseline (no replacement)
    print("\n  Computing baseline...")
    total_loss = 0.0
    total_tokens = 0
    for seq_i in range(n_seq):
        input_ids = torch.tensor(corpus_ids[seq_i:seq_i + 1], dtype=torch.long, device="cuda")
        prefix = input_ids[:, :split_pos]
        suffix = input_ids[:, split_pos:]
        prefix_out = model(input_ids=prefix, use_cache=True)
        cache = prefix_out.past_key_values
        gdn_states = {}
        for idx in gdn_layers:
            lc = cache.layers[idx]
            if hasattr(lc, "recurrent_states") and lc.recurrent_states is not None:
                gdn_states[idx] = lc.recurrent_states.clone()
        with _patch_gdn_initial_states(model, gdn_layers, gdn_states):
            out = model(input_ids=suffix, past_key_values=cache, use_cache=False, labels=suffix)
        n_tok = suffix.shape[1] - 1
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    baseline_ppl = math.exp(total_loss / total_tokens)
    results["baseline_ppl"] = baseline_ppl
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Summary: compute deltas
    print("\n  === PPL delta (%) vs baseline ===")
    for sae_type in sae_type_list:
        if sae_type not in results:
            continue
        print(f"  {sae_type}:")
        for k in k_values_list:
            mean_ppl = results[sae_type][k]["mean_ppl"]
            delta = (mean_ppl - baseline_ppl) / baseline_ppl * 100
            print(f"    k={k:2d}: Δ={delta:+.3f}%")

    # Save
    out_dir = Path(DATA) / "reviewer_experiments" / "khead_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"L{layer}_{checkpoint_source}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    data_vol.commit()

    return results


@app.local_entrypoint()
def main(checkpoint_source: str = "matched"):
    t0 = time.time()
    result = khead_replacement_curve.remote(
        layer=9, k_values="1,2,4,8,12,16", sae_types="rank1,flat",
        n_subsets=5, n_sequences=200, seed=42, checkpoint_source=checkpoint_source,
    )

    print(f"\nDone in {time.time() - t0:.0f}s")

    suffix = "" if checkpoint_source == "matched" else f"_{checkpoint_source}"
    out_path = Path(f"results/data/khead_curve{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {out_path}")

"""Parameter-matched FlatSAE control experiment.

Trains a FlatSAE with nf=256 (~8.4M params) to match the BilinearSAE's
parameter count (8.4M at nf=16384). This isolates whether the bilinear
advantage comes from rank-1 geometry or parameter efficiency.

FlatSAE  nf=256:   8,405,248 params (encoder 256x16384 + decoder 16384x256 + biases)
BilinearSAE nf=16384: 8,421,376 params (V_enc + W_enc + V_dec + W_dec + biases)
Ratio: 99.8% -- effectively matched.

Usage:
    MATRIX_SAE_MODAL_APP_NAME=matrix-sae-clean \
    MATRIX_SAE_MODAL_DATA_VOLUME=matrix-sae-data-clean \
    MATRIX_SAE_MODAL_MODEL_VOLUME=hf-model-cache-clean \
    modal run run_param_matched_control.py
"""

import json
import os
import subprocess
from pathlib import Path

import modal

# ── Parameter-matched configuration ──────────────────────────────────────────
# BilinearSAE at nf=16384, rank=1, d_k=d_v=128 has 8,421,376 params.
# FlatSAE params = 2 * d_in * nf + nf + d_in  where d_in = 128*128 = 16384.
# Solving: nf = (8_421_376 - 16384) / (2*16384 + 1) ≈ 256.5 → nf=256.
# FlatSAE at nf=256: 8,405,248 params (99.8% match).
PARAM_MATCHED_NF = 256

LAYER = 9
HEAD = 0
SEED = 42
K = 32
EPOCHS = 20
BATCH_SIZE = 256
LR = 3e-4
LR_MIN = 3e-5
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
N_DOWNSTREAM_SEQUENCES = 500
SEQ_LEN = 1024


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


def _modal_name(env_key: str, default: str) -> str:
    value = os.environ.get(env_key, "").strip()
    return value or default


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
        "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, "
        "fused_recurrent_gated_delta_rule; print('ALL_FASTPATH_IMPORTS_OK')\"",
    )
    .add_local_file("extract_states.py", "/root/extract_states.py", copy=True)
    .add_local_file("sae.py", "/root/sae.py", copy=True)
    .add_local_file("split_utils.py", "/root/split_utils.py", copy=True)
    .add_local_file("train.py", "/root/train.py", copy=True)
    .add_local_file("evaluate_downstream.py", "/root/evaluate_downstream.py", copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CURRENT_CODE_SHA})
)

app = modal.App(_modal_name("MATRIX_SAE_MODAL_APP_NAME", "matrix-sae"))
vol = modal.Volume.from_name(
    _modal_name("MATRIX_SAE_MODAL_DATA_VOLUME", "matrix-sae-data"),
    create_if_missing=True,
)
model_vol = modal.Volume.from_name(
    _modal_name("MATRIX_SAE_MODAL_MODEL_VOLUME", "hf-model-cache"),
    create_if_missing=True,
)

DATA = "/data"
MODELS = "/models"


def _corpus_slug(corpus_source: str) -> str:
    return corpus_source.strip().lower()


def _states_dir(corpus_source: str) -> Path:
    slug = _corpus_slug(corpus_source)
    if slug == "openwebtext":
        return Path(f"{DATA}/states")
    return Path(f"{DATA}/states_{slug}")


def _experiment_tag(model_name: str, seq_len: int, n_samples: int, corpus_source: str) -> str:
    model_slug = model_name.split("/")[-1].lower().replace(".", "_")
    tag = f"{model_slug}_sl{seq_len}_ns{n_samples}"
    slug = _corpus_slug(corpus_source)
    if slug != "openwebtext":
        tag = f"{tag}_{slug}"
    return tag


# ── Stage 1: Train parameter-matched FlatSAE ────────────────────────────────

@app.function(
    volumes={DATA: vol},
    gpu="L4",
    image=image,
    timeout=7200,
    memory=32768,
)
def train_param_matched_flat(
    seed: int = SEED,
    corpus_source: str = "openwebtext",
) -> dict:
    """Train FlatSAE with nf=256 to match BilinearSAE's 8.4M parameter budget."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from train import train

    vol.reload()

    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())

    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    output_dir = (
        f"{DATA}/checkpoints/{exp_tag}/"
        f"flat_L{LAYER}_H{HEAD}_nf{PARAM_MATCHED_NF}_k{K}_s{seed}"
    )

    result = train(
        sae_type="flat",
        data_dir=str(states_dir),
        layer=LAYER,
        head=HEAD,
        n_features=PARAM_MATCHED_NF,
        k=K,
        lr=LR,
        lr_min=LR_MIN,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        warmup_steps=50,
        resample_every=250,
        output_dir=output_dir,
        seed=seed,
        rank=1,
    )

    vol.commit()
    print(f"\nTrained parameter-matched FlatSAE: nf={PARAM_MATCHED_NF}")
    print(f"  val MSE = {result['best_mse']:.6e}")
    print(f"  dead features = {result['final_n_dead']}")
    print(f"  training time = {result['total_time_s']:.0f}s")
    return result


# ── Stage 2: Downstream single-head perplexity patching ─────────────────────

@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    gpu="L4",
    image=image,
    timeout=7200,
    memory=32768,
)
def evaluate_param_matched_downstream(
    seed: int = SEED,
    corpus_source: str = "openwebtext",
) -> dict:
    """Single-head downstream eval: parameter-matched FlatSAE vs existing BilinearSAE."""
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from evaluate_downstream import (
        load_sae_from_checkpoint,
        evaluate_downstream,
        format_results_table,
    )

    vol.reload()

    t0 = time.time()

    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}")
    meta = json.loads(meta_path.read_text())

    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer, config = load_model_and_tokenizer(MODEL_NAME, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if LAYER not in gdn_layers:
        raise ValueError(f"Layer {LAYER} is not a GDN layer. Valid: {gdn_layers}")

    corpus_path = states_dir / "corpus.npy"
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), 8, n_samples=N_DOWNSTREAM_SEQUENCES)
    else:
        from extract_states import load_corpus_tokens
        batches = load_corpus_tokens(tokenizer, None, SEQ_LEN, N_DOWNSTREAM_SEQUENCES, 8)
    actual = sum(b.shape[0] for b in batches)
    print(f"Loaded {actual} sequences for downstream eval")

    # Collect SAE checkpoints to evaluate
    sae_configs = []

    # 1. Parameter-matched FlatSAE (nf=256)
    pm_tag = f"flat_L{LAYER}_H{HEAD}_nf{PARAM_MATCHED_NF}_k{K}_s{seed}"
    pm_dir = ckpt_root / pm_tag
    pm_cfg_path = pm_dir / "config.json"
    pm_best_path = pm_dir / "best.pt"
    if pm_cfg_path.exists() and pm_best_path.exists():
        sae, cfg, train_mse = load_sae_from_checkpoint(
            str(pm_best_path), str(pm_cfg_path), device="cuda",
        )
        n_params = sum(p.numel() for p in sae.parameters())
        sae_configs.append({
            "tag": f"flat_nf{PARAM_MATCHED_NF} ({n_params/1e6:.1f}M params)",
            "sae": sae,
            "sae_type": "flat",
            "train_mse": train_mse,
        })
        print(f"Loaded parameter-matched FlatSAE: {n_params:,} params, val_mse={train_mse:.4e}")
    else:
        print(f"WARNING: parameter-matched FlatSAE not found at {pm_dir}")

    # 2. Existing BilinearSAE checkpoints for comparison (try nf=2048 and nf=16384)
    for nf_compare in [2048, 16384]:
        for sae_type in ["bilinear", "bilinear_tied"]:
            tag = f"{sae_type}_L{LAYER}_H{HEAD}_nf{nf_compare}_k{K}_s{seed}"
            d = ckpt_root / tag
            cfg_path = d / "config.json"
            best_path = d / "best.pt"
            if cfg_path.exists() and best_path.exists():
                try:
                    sae, cfg, train_mse = load_sae_from_checkpoint(
                        str(best_path), str(cfg_path), device="cuda",
                    )
                    n_params = sum(p.numel() for p in sae.parameters())
                    sae_configs.append({
                        "tag": f"{sae_type}_nf{nf_compare} ({n_params/1e6:.1f}M params)",
                        "sae": sae,
                        "sae_type": sae_type,
                        "train_mse": train_mse,
                    })
                    print(f"Loaded {sae_type} nf={nf_compare}: {n_params:,} params, val_mse={train_mse:.4e}")
                except Exception as e:
                    print(f"WARN: failed to load {tag}: {e}")

    # 3. Existing FlatSAE nf=2048 for reference
    for nf_compare in [2048, 16384]:
        flat_tag = f"flat_L{LAYER}_H{HEAD}_nf{nf_compare}_k{K}_s{seed}"
        d = ckpt_root / flat_tag
        cfg_path = d / "config.json"
        best_path = d / "best.pt"
        if cfg_path.exists() and best_path.exists():
            try:
                sae, cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
                n_params = sum(p.numel() for p in sae.parameters())
                sae_configs.append({
                    "tag": f"flat_nf{nf_compare} ({n_params/1e6:.1f}M params)",
                    "sae": sae,
                    "sae_type": "flat",
                    "train_mse": train_mse,
                })
                print(f"Loaded flat nf={nf_compare}: {n_params:,} params, val_mse={train_mse:.4e}")
            except Exception as e:
                print(f"WARN: failed to load {flat_tag}: {e}")

    # Also try expansion-factor-based naming (ef1 = nf=16384)
    for sae_type in ["flat", "bilinear", "bilinear_tied", "rank1"]:
        ef_tag = f"{sae_type}_L{LAYER}_H{HEAD}_ef1_k{K}_s{seed}"
        d = ckpt_root / ef_tag
        cfg_path = d / "config.json"
        best_path = d / "best.pt"
        if cfg_path.exists() and best_path.exists():
            try:
                sae, cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
                n_params = sum(p.numel() for p in sae.parameters())
                sae_configs.append({
                    "tag": f"{sae_type}_ef1 ({n_params/1e6:.1f}M params)",
                    "sae": sae,
                    "sae_type": cfg["sae_type"],
                    "train_mse": train_mse,
                })
                print(f"Loaded {sae_type} ef=1: {n_params:,} params, val_mse={train_mse:.4e}")
            except Exception as e:
                print(f"WARN: failed to load {ef_tag}: {e}")

    if not sae_configs:
        return {"error": "no SAE checkpoints found"}

    print(f"\nEvaluating {len(sae_configs)} checkpoints + baseline")

    results = evaluate_downstream(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        layer_idx=LAYER,
        sae_configs=sae_configs,
        head_idx=HEAD,
        split_fraction=0.5,
        device="cuda",
    )
    results["model"] = MODEL_NAME
    results["experiment"] = "parameter_matched_control"
    results["param_matched_nf"] = PARAM_MATCHED_NF
    results["total_time_s"] = round(time.time() - t0, 1)

    table = format_results_table(results)
    print(f"\n{table}")

    out_dir = Path(f"{DATA}/downstream_eval") / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"layer_{LAYER}_param_matched_control_s{seed}.json"
    table_path = out_dir / f"layer_{LAYER}_param_matched_control_s{seed}_table.txt"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    table_path.write_text(table)

    vol.commit()
    print(f"\nResults saved to {results_path}")
    return results


# ── Entrypoint ───────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    stage: str = "all",
    seed: int = SEED,
    corpus_source: str = "openwebtext",
):
    """Run parameter-matched FlatSAE control experiment.

    Stages:
        train     - Train FlatSAE with nf=256 (~8.4M params)
        evaluate  - Run downstream perplexity patching
        all       - Train then evaluate (default)
    """
    import time

    t0 = time.time()
    print(f"Parameter-matched FlatSAE control: stage={stage}, seed={seed}")
    print(f"  nf={PARAM_MATCHED_NF}, layer={LAYER}, head={HEAD}, k={K}")
    print(f"  FlatSAE params ≈ {2 * 16384 * PARAM_MATCHED_NF + PARAM_MATCHED_NF + 16384:,}")
    print(f"  BilinearSAE params ≈ {4 * 16384 * 128 + 16384 + 16384:,}")

    if stage in ("train", "all"):
        train_result = train_param_matched_flat.remote(seed=seed, corpus_source=corpus_source)
        print(f"\nTraining result:")
        print(f"  val MSE = {train_result['best_mse']:.6e}")
        print(f"  dead features = {train_result['final_n_dead']}")
        print(f"  time = {train_result['total_time_s']:.0f}s")

    if stage in ("evaluate", "all"):
        eval_result = evaluate_param_matched_downstream.remote(
            seed=seed, corpus_source=corpus_source,
        )
        if "error" in eval_result:
            print(f"\nDownstream eval error: {eval_result['error']}")
        else:
            print(f"\nDownstream eval completed in {eval_result.get('total_time_s', 0):.0f}s")

    print(f"\nTotal wall clock: {time.time() - t0:.0f}s")

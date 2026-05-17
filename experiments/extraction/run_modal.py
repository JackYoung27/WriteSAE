# Modal deployment for matrix SAE pipeline: extract states, train SAEs, run analysis.
#
# Usage:
#     modal run run_modal.py                                  # full pipeline
#     modal run run_modal.py --stage extract --n-samples 5000 # extract only
#     modal run run_modal.py --stage extract-texts            # save texts for existing states
#     modal run run_modal.py --stage train                    # train single
#     modal run run_modal.py --stage sweep                    # train all configs
#     modal run run_modal.py --stage nf-sweep                 # sweep dictionary sizes
#     modal run run_modal.py --stage batchtopk-sweep --layers 9  # BatchTopK activation sweep
#     modal run run_modal.py --stage analyze                  # analyze only
#     modal run run_modal.py --stage interpret --layers 9     # feature interpretability
#     modal run run_modal.py --stage interpret-s0             # interpret S0-aligned features
#     modal run run_modal.py --stage s0                       # S0 decomposition
#     modal run run_modal.py --stage s0-shift --layer 9 --n-samples 200  # S0 activation shift
#     modal run run_modal.py --stage s0-proper-null                      # proper null test for S0
#     modal run run_modal.py --stage temporal --layer 9 --n-samples 500  # temporal dynamics
#     modal run run_modal.py --stage baselines --layers 9     # PCA/NMF baselines
#     modal run run_modal.py --stage evaluate-allheads --layers 9  # all-heads downstream eval
#     modal run run_modal.py --stage train-allheads --layer 9      # train SAEs for all 16 heads
#     modal run run_modal.py --stage evaluate-perhead --layers 9   # per-head matched downstream eval
#     modal run run_modal.py --stage train --sae-type rank1 --rank 2  # rank-2 SAE
#     modal run run_modal.py --stage train --sae-type flat --n-features 2048  # small dictionary
#     modal run run_modal.py --stage probe-features --layer 9   # statistical probing + vocab projection
#     modal run run_modal.py --stage circuit-ablation --layer 9 # causal feature ablation (single)
#     modal run run_modal.py --stage circuit-ablation-v2 --layer 9 # group ablation dose-response
#     modal run run_modal.py --stage feature-vs-random --layer 9  # alive features vs random directions
#     modal run run_modal.py --stage causal-clamp --layer 9    # Q3: clamp feature HIGH, measure text shift
#     modal run run_modal.py --stage causal-logit --layer 9    # direct next-token logit intervention
#     modal run run_modal.py --stage mechanistic-profile --layer 9 --head 12 --family document_format  # head-specific family profile
#     modal run run_modal.py --stage mechanistic-clamp --layer 9 --head 12 --family document_format    # grouped family clamp vs random
#     modal run run_modal.py --stage factor-trace --layer 9 --head 4    # token-level write-to-use traces
#     modal run run_modal.py --stage factor-trace-intervention --layer 9 --head 4  # write-time zeroing, use-time KL
#     modal run run_modal.py --stage factor-trace-transplant --layer 9 --head 4    # donor->recipient value transplant with boundary readouts
#     modal run run_modal.py --stage nonsparse-baseline --layer 9  # Q5: TopK-disabled/ReLU-only control
#     modal run run_modal.py --stage probe-stability --layer 9 # Q1: probe on 3 subsets, Jaccard similarity
#     modal run run_modal.py --stage memory-alignment --layer 9 --head 4  # SAE decoder vs GDN write alignment
#     modal run run_modal.py --stage generation-intervention-qualitative --layer 9  # qualitative demo with instruction prompts
#     modal run run_modal.py --stage single-feature-demo --layer 9               # single-feature additive steering demo
#
# Isolated reruns:
#   - MATRIX_SAE_MODAL_APP_NAME=matrix-sae-clean
#   - MATRIX_SAE_MODAL_DATA_VOLUME=matrix-sae-data-clean
#   - MATRIX_SAE_MODAL_MODEL_VOLUME=hf-model-cache-clean
#
# Performance (pipelined "all" stage):
#   - Extraction: 3 layers run on 3 GPUs in parallel (~15 min vs ~46 min sequential)
#   - Training: jobs launch per-layer as soon as that layer's extraction finishes
#   - Batch size 8 for extraction (was 4), halves forward pass count
#   - Estimated wall clock: ~20 min (was ~55 min sequential)

import json
import os
import subprocess
from collections.abc import Mapping
from typing import cast
from pathlib import Path

import modal


def _current_code_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (FileNotFoundError, OSError):
        sha = ""
    return sha or os.environ.get("MATRIX_SAE_CODE_SHA", "unknown")


def _modal_name(env_key: str, default: str) -> str:
    """Read a Modal resource name override from the environment."""
    value = os.environ.get(env_key, "").strip()
    return value or default


CURRENT_CODE_SHA = _current_code_sha()

# Source dirs resolved relative to this file so the image build works regardless of CWD.
_ext_dir = Path(__file__).resolve().parent
_repo_root = _ext_dir.parent.parent
_core_dir = _repo_root / "core"
_analysis_dir = _repo_root / "experiments" / "analysis"
_ablation_dir = _repo_root / "experiments" / "ablations"
_steering_dir = _repo_root / "experiments" / "steering"
_scripts_dir = _repo_root / "scripts"

# Pre-built wheels: cu12 + torch 2.8 + CXX11 ABI TRUE + cp312 + x86_64.
# PyTorch 2.8 still has c10::cuda::CUDAStream::query(), so these wheels load
# without the undefined-symbol error that breaks cu13+torch2.10 wheels.
# See: github.com/state-spaces/mamba/issues/891
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
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
            "MAX_JOBS": "4",
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
        }
    )
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
        "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}'); "
        "print('ALL_FASTPATH_IMPORTS_OK')\"",
    )
    .add_local_file(str(_ext_dir / "extract_states.py"), "/root/extract_states.py", copy=True)
    .add_local_file(str(_core_dir / "sae.py"), "/root/sae.py", copy=True)
    .add_local_file(str(_core_dir / "split_utils.py"), "/root/split_utils.py", copy=True)
    .add_local_file(str(_core_dir / "train.py"), "/root/train.py", copy=True)
    .add_local_file(str(_analysis_dir / "analyze.py"), "/root/analyze.py", copy=True)
    .add_local_file(str(_analysis_dir / "interpret.py"), "/root/interpret.py", copy=True)
    .add_local_file(str(_scripts_dir / "visualize.py"), "/root/visualize.py", copy=True)
    .add_local_file(str(_analysis_dir / "baselines.py"), "/root/baselines.py", copy=True)
    .add_local_file(str(_analysis_dir / "evaluate_downstream.py"), "/root/evaluate_downstream.py", copy=True)
    .add_local_file(str(_analysis_dir / "feature_quality.py"), "/root/feature_quality.py", copy=True)
    .add_local_file(str(_analysis_dir / "probe_features.py"), "/root/probe_features.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation.py"), "/root/circuit_ablation.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation_v2.py"), "/root/circuit_ablation_v2.py", copy=True)
    .add_local_file(str(_ablation_dir / "circuit_ablation_random.py"), "/root/circuit_ablation_random.py", copy=True)
    .add_local_file(str(_steering_dir / "causal_clamp.py"), "/root/causal_clamp.py", copy=True)
    .add_local_file(str(_analysis_dir / "factor_trace.py"), "/root/factor_trace.py", copy=True)
    .add_local_file(str(_steering_dir / "hierarchical_transplant.py"), "/root/hierarchical_transplant.py", copy=True)
    .add_local_file(str(_steering_dir / "multihead_transplant.py"), "/root/multihead_transplant.py", copy=True)
    .add_local_file(str(_steering_dir / "generation_intervention.py"), "/root/generation_intervention.py", copy=True)
    .add_local_file(str(_analysis_dir / "memory_alignment.py"), "/root/memory_alignment.py", copy=True)
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
GPU_KWARGS = dict(gpu="L4", image=image, timeout=7200, memory=32768)  # L4 24GB, $0.80/hr
GPU_KWARGS_A10G = dict(gpu="A10G", image=image, timeout=7200, memory=32768)  # A10G 24GB, $1.10/hr
GPU_KWARGS_A100 = dict(gpu="A100", image=image, timeout=7200, memory=32768)  # A100 40GB, $3.40/hr
EXTRACT_BATCH_SIZE = 32  # 0.8B model is 1.7GB, plenty of room on L4


def _model_is_large(model_name: str) -> bool:
    """True for models >2B params (need bigger GPU and smaller batch size)."""
    import re

    name = model_name.lower()
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)b\b", name):
        try:
            size_b = float(match.group(1))
        except ValueError:
            continue
        if size_b > 2.0:
            return True
    return False


def _extract_batch_size(model_name: str) -> int:
    """Batch size for extraction: smaller for bigger models to fit in GPU memory."""
    if _model_is_large(model_name):
        return 8
    return 32


# -- Sweep config builder (pure function, reused by train_sweep and run_all) --

def _normalize_corpus_source(corpus_source: str | None) -> str:
    source = (corpus_source or "openwebtext").strip().lower()
    aliases = {
        "openwebtext": "openwebtext",
        "owt": "openwebtext",
        "skylion007/openwebtext": "openwebtext",
        "ultrachat": "ultrachat_200k",
        "ultrachat_200k": "ultrachat_200k",
        "huggingfaceh4/ultrachat_200k": "ultrachat_200k",
    }
    if source not in aliases:
        raise ValueError(
            f"Unsupported corpus_source={corpus_source!r}. "
            "Use openwebtext or ultrachat_200k."
        )
    return aliases[source]


def _corpus_slug(corpus_source: str | None) -> str:
    return _normalize_corpus_source(corpus_source)


def _corpus_label(corpus_source: str | None) -> str:
    source = _normalize_corpus_source(corpus_source)
    if source == "openwebtext":
        return "Skylion007/openwebtext (streaming)"
    return "HuggingFaceH4/ultrachat_200k (train_sft, flattened chat)"


def _states_dir(corpus_source: str | None) -> Path:
    slug = _corpus_slug(corpus_source)
    if slug == "openwebtext":
        return Path(f"{DATA}/states")
    return Path(f"{DATA}/states_{slug}")


def _guard_shared_state_namespace(
    output_dir: Path,
    model_name: str,
    seq_len: int,
    n_samples: int,
    corpus_source: str,
) -> None:
    """Refuse to mix incompatible extractions into the same shared states root.

    OpenWebText extractions currently share a single `/data/states` namespace.
    Reusing that namespace across different sample counts can silently overwrite
    state tensors or metadata that downstream training jobs memory-map. Allow an
    explicit escape hatch for intentional migrations or manual recovery.
    """
    if os.environ.get("MATRIX_SAE_ALLOW_STATE_OVERWRITE", "").strip() == "1":
        return

    corpus_source = _normalize_corpus_source(corpus_source)
    unified_meta_path = output_dir / "metadata.json"
    if unified_meta_path.exists():
        existing = json.loads(unified_meta_path.read_text())
        if (
            existing.get("model") == model_name
            and existing.get("seq_len") == seq_len
            and existing.get("corpus_source", "openwebtext") == corpus_source
            and existing.get("n_samples") not in (None, n_samples)
        ):
            raise RuntimeError(
                "Refusing to overwrite shared extracted states with a different "
                f"sample count (existing n_samples={existing.get('n_samples')}, "
                f"requested n_samples={n_samples}) at {output_dir}. "
                "Set MATRIX_SAE_ALLOW_STATE_OVERWRITE=1 to override."
            )

    for layer_meta_path in sorted(output_dir.glob("layer_*/layer_metadata.json")):
        existing = json.loads(layer_meta_path.read_text())
        if (
            existing.get("model") == model_name
            and existing.get("seq_len") == seq_len
            and existing.get("corpus_source", "openwebtext") == corpus_source
            and existing.get("n_samples") not in (None, n_samples)
        ):
            raise RuntimeError(
                "Refusing to mix layer states with mismatched sample counts in "
                f"{output_dir} (saw {existing.get('n_samples')} in {layer_meta_path}, "
                f"requested {n_samples}). "
                "Set MATRIX_SAE_ALLOW_STATE_OVERWRITE=1 to override."
            )


def _analysis_dir(corpus_source: str | None) -> Path:
    slug = _corpus_slug(corpus_source)
    if slug == "openwebtext":
        return Path(f"{DATA}/analysis")
    return Path(f"{DATA}/analysis_{slug}")


def _figures_dir(corpus_source: str | None) -> Path:
    slug = _corpus_slug(corpus_source)
    if slug == "openwebtext":
        return Path(f"{DATA}/figures")
    return Path(f"{DATA}/figures_{slug}")


def _checkpoint_root_for_corpus(corpus_source: str | None, model_name: str | None = None) -> tuple[Path, dict]:
    source = _normalize_corpus_source(corpus_source)
    meta_path = _states_dir(source) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    if model_name is not None and meta.get("model") != model_name:
        raise FileNotFoundError(
            f"Metadata at {meta_path} is for model={meta.get('model')}, expected {model_name}"
        )
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], source)
    return Path(f"{DATA}/checkpoints") / exp_tag, meta


def _resolve_sae_checkpoint(
    *,
    layer: int,
    head: int,
    n_features_target: int,
    checkpoint_tag: str | None = None,
    corpus_source: str = "openwebtext",
    preferred_types: tuple[str, ...] = ("rank1", "bilinear"),
) -> tuple[Path, dict, str]:
    """Resolve a checkpoint by exact tag or by the existing rank1/bilinear preference."""
    search_sources = [corpus_source]
    if corpus_source != "openwebtext":
        search_sources = ["openwebtext", corpus_source]

    roots: list[Path] = []
    best_ckpt = None
    best_cfg = None
    best_tag = None

    for source in search_sources:
        try:
            ckpt_root, _ = _checkpoint_root_for_corpus(source)
        except FileNotFoundError:
            continue
        roots.append(ckpt_root)
        if checkpoint_tag:
            tag_dir = ckpt_root / checkpoint_tag
            cp, bp = tag_dir / "config.json", tag_dir / "best.pt"
            if cp.exists() and bp.exists():
                cfg = json.loads(cp.read_text())
                if (
                    cfg.get("layer") == layer
                    and cfg.get("head", 0) == head
                    and cfg.get("n_features") == n_features_target
                ):
                    return bp, cfg, tag_dir.name

    for ckpt_root in roots:
        if not ckpt_root.exists():
            continue
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
            if checkpoint_tag and d.name != checkpoint_tag:
                continue
            candidate_key = (
                0 if sae_type == preferred_types[0] else 1,
                0 if cfg.get("seed") == 42 else 1,
                cfg.get("seed", 999),
            )
            current_key = None
            if best_cfg is not None:
                current_key = (
                    0 if best_cfg.get("sae_type", "") == preferred_types[0] else 1,
                    0 if best_cfg.get("seed") == 42 else 1,
                    best_cfg.get("seed", 999),
                )
            if current_key is None or candidate_key < current_key:
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name

    if best_ckpt is None or best_cfg is None or best_tag is None:
        tag_msg = f", checkpoint_tag={checkpoint_tag}" if checkpoint_tag else ""
        raise FileNotFoundError(
            f"No checkpoint for layer={layer}, head={head}, nf={n_features_target}{tag_msg}"
        )
    return best_ckpt, best_cfg, best_tag


def _experiment_tag(
    model_name: str,
    seq_len: int,
    n_samples: int,
    corpus_source: str = "openwebtext",
) -> str:
    model_slug = model_name.split("/")[-1].lower().replace(".", "_")
    tag = f"{model_slug}_sl{seq_len}_ns{n_samples}"
    corpus_slug = _corpus_slug(corpus_source)
    if corpus_slug != "openwebtext":
        tag = f"{tag}_{corpus_slug}"
    return tag


def _flatten_ultrachat_messages(messages: list[dict[str, object]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        if role == "assistant":
            role_label = "Assistant"
        elif role == "system":
            role_label = "System"
        else:
            role_label = "User"

        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
            content_text = "\n".join(parts)
        else:
            content_text = str(content).strip()

        if content_text:
            lines.append(f"{role_label}: {content_text}")
    return "\n".join(lines)


def _materialize_ultrachat_corpus(
    tokenizer,
    output_dir: Path,
    n_samples: int,
    seq_len: int,
) -> tuple[Path, dict]:
    from datasets import load_dataset

    text_path = output_dir / "corpus_ultrachat_200k.txt"
    meta_path = output_dir / "corpus_ultrachat_200k.json"
    target_tokens = n_samples * seq_len * 2

    if text_path.exists() and meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text())
            if cached_meta.get("token_count", 0) >= target_tokens:
                print(
                    "Reusing cached UltraChat corpus:"
                    f" {cached_meta.get('conversation_count', 0)} conversations,"
                    f" {cached_meta.get('token_count', 0):,} tokens"
                )
                return text_path, cached_meta
        except (json.JSONDecodeError, ValueError):
            pass

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    texts = []
    token_count = 0
    conversation_count = 0
    for example in ds:
        if not isinstance(example, Mapping):
            continue
        raw_messages = example.get("messages", [])
        if not isinstance(raw_messages, list):
            continue
        flattened = _flatten_ultrachat_messages(raw_messages)
        if not flattened:
            continue
        token_count += len(tokenizer.encode(flattened, add_special_tokens=False))
        texts.append(flattened)
        conversation_count += 1
        if token_count >= target_tokens:
            break

    if token_count < seq_len:
        raise ValueError(
            "UltraChat materialization produced too few tokens "
            f"({token_count}) for seq_len={seq_len}"
        )

    text_path.write_text("\n\n".join(texts))
    meta = {
        "corpus_source": "ultrachat_200k",
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "conversation_count": conversation_count,
        "token_count": token_count,
        "target_tokens": target_tokens,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"Materialized UltraChat corpus: {conversation_count} conversations, "
        f"{token_count:,} tokens"
    )
    return text_path, meta

def _build_sweep_configs(layers: list[int], seeds: list[int] = [0, 1, 2]) -> list[dict]:
    # 3 seeds for error bars. ef=1 fits L4 at batch=256. ef=2 needs batch=64.
    train_kw = dict(epochs=20, warmup_steps=50, resample_every=250)
    configs = []
    for seed in seeds:
        for layer in layers:
            for sae_type in ["flat", "rank1", "bilinear"]:
                configs.append(dict(sae_type=sae_type, layer=layer, head=0, k=32,
                                    expansion_factor=1, batch_size=256, seed=seed, **train_kw))
            # rank1 and bilinear at higher expansion factors (cheap decoder+encoder)
            for sae_type in ["rank1", "bilinear"]:
                configs.append(dict(sae_type=sae_type, layer=layer, head=0, k=32,
                                    expansion_factor=2, batch_size=64, seed=seed, **train_kw))
    return configs


def _build_perhead_configs(
    layer: int = 9,
    heads: list[int] | None = None,
    sae_types: list[str] | None = None,
    n_features: int = 2048,
    k: int = 32,
    seeds: list[int] | None = None,
    n_heads: int = 16,
) -> list[dict]:
    """Generate training configs for per-head SAEs.

    Default: n_heads heads x SAE types x 1 seed.
    Each head gets its own SAE trained on that head's extracted states.
    """
    if heads is None:
        heads = list(range(n_heads))
    if sae_types is None:
        sae_types = ["flat", "rank1", "bilinear"]
    if seeds is None:
        seeds = [42]

    train_kw = dict(epochs=20, warmup_steps=50, resample_every=250)
    configs = []
    for seed in seeds:
        for head in heads:
            for sae_type in sae_types:
                configs.append(dict(
                    sae_type=sae_type, layer=layer, head=head, k=k,
                    n_features=n_features, batch_size=256, seed=seed,
                    **train_kw,
                ))
    return configs


def _build_nf_sweep_configs(layers: list[int], seeds: list[int] = [0, 1, 2]) -> list[dict]:
    """Grid over dictionary sizes (n_features) for feature utilization study."""
    train_kw = dict(epochs=20, warmup_steps=50, resample_every=250)
    configs = []
    for seed in seeds:
        for layer in layers:
            for sae_type in ["flat", "rank1", "bilinear"]:
                for nf in [1024, 2048, 4096, 8192]:
                    bs = 128 if nf >= 8192 else 256
                    configs.append(dict(
                        sae_type=sae_type, layer=layer, head=0, k=32,
                        expansion_factor=1, n_features=nf, batch_size=bs,
                        seed=seed, **train_kw,
                    ))
    return configs


def _build_batchtopk_sweep_configs(
    layers: list[int],
    n_features: int = 2048,
    k: int = 32,
    seeds: list[int] = [0, 1, 2],  # noqa: B006
) -> list[dict]:
    """Grid for BatchTopK ablation: flat/rank1/bilinear with use_batchtopk=True."""
    train_kw = dict(epochs=20, warmup_steps=50, resample_every=250)
    configs = []
    for seed in seeds:
        for layer in layers:
            for sae_type in ["flat", "rank1", "bilinear"]:
                configs.append(dict(
                    sae_type=sae_type, layer=layer, head=0, k=k,
                    n_features=n_features, batch_size=256,
                    seed=seed, use_batchtopk=True, **train_kw,
                ))
    return configs


# -- Stage 0: Tokenize corpus once, shared by all layer extractions --

@app.function(volumes={DATA: vol, MODELS: model_vol}, image=image, timeout=3600, memory=32768)
def extract_corpus(
    n_samples: int = 10000,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
):
    """Tokenize one corpus once and save it as /data/states*/corpus.npy.

    All extract_layer() calls read from this file so every layer sees
    the same text.
    """
    import json, os, sys
    from pathlib import Path
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import save_corpus_tokens, load_corpus_from_file, decode_batch_texts

    corpus_source = _normalize_corpus_source(corpus_source)
    output_dir = _states_dir(corpus_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    _guard_shared_state_namespace(output_dir, model_name, seq_len, n_samples, corpus_source)
    corpus_path = output_dir / "corpus.npy"

    texts_path = output_dir / "texts.json"

    print(f"Tokenizing with {model_name} on corpus_source={corpus_source}")

    # Reuse a cached corpus only when it is long enough. Downstream stages
    # read the first n_samples sequences so checkpoint tags stay aligned.
    if corpus_path.exists():
        import numpy as np
        existing = np.load(str(corpus_path), mmap_mode="r")
        if existing.shape[0] >= n_samples and existing.shape[1] == seq_len:
            print(f"Corpus already exists: {existing.shape[0]} seqs x {existing.shape[1]} tokens. Reusing first {n_samples}.")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            batches = load_corpus_from_file(str(corpus_path), 32, n_samples=n_samples)
            texts = decode_batch_texts(tokenizer, batches)
            texts_path.write_text(json.dumps(texts))
            vol.commit()
            return {"n_seqs": n_samples, "seq_len": seq_len, "corpus_source": corpus_source}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus_text_path = None
    if corpus_source == "ultrachat_200k":
        corpus_text_path, materialized_meta = _materialize_ultrachat_corpus(
            tokenizer, output_dir, n_samples, seq_len,
        )
        print(
            f"Using {materialized_meta['conversation_count']} UltraChat conversations "
            f"from {materialized_meta['split']}"
        )

    n_seqs = save_corpus_tokens(
        tokenizer,
        str(corpus_text_path) if corpus_text_path is not None else None,
        seq_len,
        n_samples,
        str(corpus_path),
    )

    # always regenerate texts.json from the current corpus.npy
    batches = load_corpus_from_file(str(corpus_path), 32)
    texts = decode_batch_texts(tokenizer, batches)
    texts_path.write_text(json.dumps(texts))
    print(f"Saved {len(texts)} text sequences to texts.json")

    vol.commit()
    return {"n_seqs": n_seqs, "seq_len": seq_len, "corpus_source": corpus_source}


# -- Stage 1a: Extract one layer on one GPU --

def _extract_layer_impl(
    layer: int,
    n_samples: int,
    seq_len: int,
    model_name: str,
    corpus_source: str = "openwebtext",
):
    """Core extraction logic shared by L4 and A10G variants."""
    import json, os, sys, time
    from pathlib import Path
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import (
        load_model_and_tokenizer, get_gdn_layer_indices,
        load_corpus_tokens, load_corpus_from_file,
        setup_memmaps, probe_state_dims, extract_states,
    )

    corpus_source = _normalize_corpus_source(corpus_source)
    batch_size = _extract_batch_size(model_name)
    print(
        f"Extracting layer {layer} from {model_name} "
        f"(batch_size={batch_size}, corpus_source={corpus_source})"
    )

    output_dir = _states_dir(corpus_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    _guard_shared_state_namespace(output_dir, model_name, seq_len, n_samples, corpus_source)
    layer_dir = output_dir / f"layer_{layer}"

    # skip if already extracted
    layer_meta_path = layer_dir / "layer_metadata.json"
    if layer_meta_path.exists():
        existing = json.loads(layer_meta_path.read_text())
        if (existing.get("n_samples", 0) >= n_samples
                and existing.get("seq_len") == seq_len
                and existing.get("model") == model_name
                and existing.get("corpus_source", "openwebtext") == corpus_source):
            print(f"Layer {layer} already extracted ({existing['n_samples']} samples). Skipping.")
            return existing

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")

    # Log GPU memory after model load
    import torch
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    all_gdn = get_gdn_layer_indices(config)
    if layer not in all_gdn:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {all_gdn}")

    n_heads, key_dim, val_dim = probe_state_dims(model, layer, tokenizer, "cuda")
    print(f"Layer {layer}: {n_heads} heads x ({key_dim}, {val_dim})")

    # load shared corpus (extract_corpus must have run first)
    corpus_path = output_dir / "corpus.npy"
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_samples)
        print(f"Loaded shared corpus from {corpus_path}")
    else:
        # fallback: tokenize independently (backward compat)
        print("WARNING: corpus.npy not found, tokenizing independently (layers may differ)")
        corpus_text_path = None
        if corpus_source == "ultrachat_200k":
            corpus_text_path, _ = _materialize_ultrachat_corpus(tokenizer, output_dir, n_samples, seq_len)
        batches = load_corpus_tokens(
            tokenizer,
            str(corpus_text_path) if corpus_text_path is not None else None,
            seq_len,
            n_samples,
            batch_size,
        )
    actual_samples = sum(b.shape[0] for b in batches)

    # check for partial progress from a crashed run
    progress_path = layer_dir / "progress.json"
    start_batch, start_offset, resume = 0, 0, False
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        saved_batch = prog.get("batches_done", 0)
        saved_offset = prog.get("samples_written", 0)
        expected_offset = sum(b.shape[0] for b in batches[:saved_batch]) if 0 <= saved_batch <= len(batches) else -1
        valid_progress = (
            prog.get("n_samples_total") == actual_samples
            and prog.get("batch_count") == len(batches)
            and prog.get("seq_len") == seq_len
            and 0 < saved_batch < len(batches)
            and saved_offset == expected_offset
        )
        if valid_progress:
            start_batch = saved_batch
            start_offset = saved_offset
            resume = True
            print(f"Resuming layer {layer} from batch {start_batch}/{len(batches)}")
        else:
            progress_path.unlink(missing_ok=True)
            print(f"Ignoring stale progress for layer {layer}; restarting extraction")

    memmaps = setup_memmaps(output_dir, [layer], n_heads, key_dim, val_dim, actual_samples, resume=resume)

    t0 = time.time()
    n_written = extract_states(model, config, batches, [layer], memmaps, "cuda",
                               start_batch=start_batch, start_offset=start_offset,
                               progress_path=progress_path)
    elapsed = time.time() - t0

    for mm in memmaps[layer]:
        mm.flush()

    layer_meta = {
        "model": model_name,
        "corpus": _corpus_label(corpus_source),
        "corpus_source": corpus_source,
        "layer": layer, "all_gdn_layers": all_gdn, "n_samples": n_written,
        "n_heads": n_heads, "key_head_dim": key_dim, "value_head_dim": val_dim,
        "state_shape_per_head": [n_written, key_dim, val_dim], "dtype": "float16",
        "seq_len": seq_len, "extraction_time_s": round(elapsed, 1),
    }
    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_meta_path.write_text(json.dumps(layer_meta, indent=2))
    if progress_path.exists():
        progress_path.unlink()
    vol.commit()

    print(f"Layer {layer}: {n_written} samples in {elapsed:.1f}s")
    return layer_meta


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def extract_layer(
    layer: int,
    n_samples: int = 10000,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
):
    """Extract GDN states on L4 (for <=2B models)."""
    return _extract_layer_impl(layer, n_samples, seq_len, model_name, corpus_source)


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def extract_layer_a10g(
    layer: int,
    n_samples: int = 10000,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-4B",
    corpus_source: str = "openwebtext",
):
    """Extract GDN states on A10G (for >2B models)."""
    return _extract_layer_impl(layer, n_samples, seq_len, model_name, corpus_source)


# -- Stage 1: Extract all layers on one GPU (backward compat for --stage extract) --
# NOTE: extract_layer() above is the primary path used by run_all() for parallel extraction.
# This single-GPU extract() is kept for backward compat with --stage extract CLI usage.

def _extract_all_impl(n_samples, layers, seq_len, model_name, corpus_source, allow_overwrite):
    """Shared extraction logic for L4 and A10G single-GPU extract functions."""
    import json, os, sys, time
    from pathlib import Path
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import (
        load_model_and_tokenizer, get_gdn_layer_indices,
        load_corpus_tokens, setup_memmaps, probe_state_dims, extract_states,
    )

    corpus_source = _normalize_corpus_source(corpus_source)
    batch_size = _extract_batch_size(model_name)
    print(
        f"Single-GPU extraction: {model_name} "
        f"(batch_size={batch_size}, corpus_source={corpus_source})"
    )

    output_dir = _states_dir(corpus_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not allow_overwrite:
        _guard_shared_state_namespace(output_dir, model_name, seq_len, n_samples, corpus_source)

    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        same_config = (existing.get("n_samples", 0) >= n_samples
                       and set(existing.get("layer_indices", [])) >= set(layers)
                       and existing.get("seq_len") == seq_len
                       and existing.get("model") == model_name
                       and existing.get("corpus_source", "openwebtext") == corpus_source)
        if same_config:
            print(f"Already extracted ({existing['n_samples']} samples, seq_len={existing.get('seq_len')}). Skipping.")
            return existing

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    all_gdn = get_gdn_layer_indices(config)
    invalid = [l for l in layers if l not in all_gdn]
    if invalid:
        raise ValueError(f"Layers {invalid} are not GDN layers. Valid: {all_gdn}")

    n_heads, key_dim, val_dim = probe_state_dims(model, layers[0], tokenizer, "cuda")
    print(f"State dims: {n_heads} heads x ({key_dim}, {val_dim})")

    corpus_text_path = None
    if corpus_source == "ultrachat_200k":
        corpus_text_path, _ = _materialize_ultrachat_corpus(tokenizer, output_dir, n_samples, seq_len)
    batches = load_corpus_tokens(
        tokenizer,
        str(corpus_text_path) if corpus_text_path is not None else None,
        seq_len,
        n_samples,
        batch_size,
    )
    actual_samples = sum(b.shape[0] for b in batches)

    # Save corpus.npy so downstream stages (temporal, eval) can reuse the exact tokens
    import torch
    import numpy as np
    corpus_path = output_dir / "corpus.npy"
    if not corpus_path.exists():
        all_tokens = torch.cat(batches, dim=0).numpy()
        np.save(str(corpus_path), all_tokens)
        print(f"Saved corpus.npy: {all_tokens.shape}")

    progress_path = output_dir / "progress.json"
    start_batch, start_offset, resume = 0, 0, False
    if progress_path.exists():
        prog = json.loads(progress_path.read_text())
        start_batch = prog.get("batches_done", 0)
        start_offset = prog.get("samples_written", 0)
        if 0 < start_batch < len(batches):
            resume = True
            print(f"Resuming from batch {start_batch}/{len(batches)} (sample {start_offset}/{actual_samples})")
        elif start_batch >= len(batches):
            resume = True
            print(f"Extraction data complete ({start_offset} samples), writing metadata")

    memmaps = setup_memmaps(output_dir, layers, n_heads, key_dim, val_dim, actual_samples, resume=resume)

    t0 = time.time()
    n_written = extract_states(model, config, batches, layers, memmaps, "cuda",
                               start_batch=start_batch, start_offset=start_offset,
                               progress_path=progress_path)
    elapsed = time.time() - t0

    for layer_idx in layers:
        for mm in memmaps[layer_idx]:
            mm.flush()

    metadata = {
        "model": model_name,
        "corpus": _corpus_label(corpus_source),
        "corpus_source": corpus_source,
        "layer_indices": layers, "all_gdn_layers": all_gdn, "n_samples": n_written,
        "n_heads": n_heads, "key_head_dim": key_dim, "value_head_dim": val_dim,
        "state_shape_per_head": [n_written, key_dim, val_dim], "dtype": "float16",
        "seq_len": seq_len, "extraction_time_s": round(elapsed, 1),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    if progress_path.exists():
        progress_path.unlink()
    vol.commit()

    print(f"Extracted {n_written} samples from {len(layers)} layers in {elapsed:.1f}s")
    return metadata


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def extract(
    n_samples: int = 10000,
    layers: list[int] = [1, 9, 17],  # noqa: B006
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
    allow_overwrite: bool = False,
):
    """Single-GPU extraction on L4 (for <=2B models)."""
    return _extract_all_impl(n_samples, layers, seq_len, model_name, corpus_source, allow_overwrite)


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def extract_a10g(
    n_samples: int = 10000,
    layers: list[int] = [9],  # noqa: B006
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-4B",
    corpus_source: str = "openwebtext",
    allow_overwrite: bool = False,
):
    """Single-GPU extraction on A10G (for >2B models)."""
    return _extract_all_impl(n_samples, layers, seq_len, model_name, corpus_source, allow_overwrite)


# -- Stage 2: Train a single SAE --

def _train_sae_impl(
    sae_type: str = "flat",
    layer: int = 12,
    head: int = 0,
    expansion_factor: int = 1,
    k: int = 32,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 256,
    warmup_steps: int = 50,
    resample_every: int = 250,
    seed: int = 42,
    rank: int = 1,
    n_features: int | None = None,
    use_batchtopk: bool = False,
    corpus_source: str = "openwebtext",
) -> dict:
    import json, sys
    from pathlib import Path
    sys.path.insert(0, "/root")
    from train import train

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)

    # ensure we see committed data from extract_layer (volume propagation delay)
    vol.reload()

    unified_meta = states_dir / "metadata.json"
    layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
    if unified_meta.exists():
        meta = json.loads(unified_meta.read_text())
    elif layer_meta_path.exists():
        meta = json.loads(layer_meta_path.read_text())
    else:
        raise FileNotFoundError(f"No metadata at {unified_meta} or {layer_meta_path}")

    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    if n_features is None:
        n_features = key_dim * val_dim * expansion_factor

    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    rank_tag = f"r{rank}_" if rank > 1 else ""
    btk_tag = "btk_" if use_batchtopk else ""
    nf_tag = f"nf{n_features}" if n_features != key_dim * val_dim * expansion_factor else f"ef{expansion_factor}"
    output_dir = f"{DATA}/checkpoints/{exp_tag}/{rank_tag}{btk_tag}{sae_type}_L{layer}_H{head}_{nf_tag}_k{k}_s{seed}"

    result = train(
        sae_type=sae_type, data_dir=str(states_dir), layer=layer, head=head,
        n_features=n_features, k=k, lr=lr, batch_size=batch_size, epochs=epochs,
        warmup_steps=warmup_steps, resample_every=resample_every,
        output_dir=output_dir, seed=seed, rank=rank,
        use_batchtopk=use_batchtopk,
    )

    vol.commit()
    return result

@app.function(volumes={DATA: vol}, **GPU_KWARGS)
def train_sae(
    sae_type: str = "flat",
    layer: int = 12,
    head: int = 0,
    expansion_factor: int = 1,
    k: int = 32,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 256,
    warmup_steps: int = 50,
    resample_every: int = 250,
    seed: int = 42,
    rank: int = 1,
    n_features: int | None = None,
    use_batchtopk: bool = False,
    corpus_source: str = "openwebtext",
) -> dict:  # type: ignore[type-arg]
    return _train_sae_impl(
        sae_type=sae_type, layer=layer, head=head,
        expansion_factor=expansion_factor, n_features=n_features, k=k, lr=lr, batch_size=batch_size, epochs=epochs,
        warmup_steps=warmup_steps, resample_every=resample_every,
        seed=seed, rank=rank, use_batchtopk=use_batchtopk, corpus_source=corpus_source,
    )


@app.function(volumes={DATA: vol}, **GPU_KWARGS_A100)
def train_sae_a100(
    sae_type: str = "flat",
    layer: int = 12,
    head: int = 0,
    expansion_factor: int = 1,
    k: int = 32,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 256,
    warmup_steps: int = 50,
    resample_every: int = 250,
    seed: int = 42,
    rank: int = 1,
    n_features: int | None = None,
    use_batchtopk: bool = False,
    corpus_source: str = "openwebtext",
) -> dict:  # type: ignore[type-arg]
    """Same as train_sae but on A100 for large rank configs."""
    return _train_sae_impl(
        sae_type=sae_type, layer=layer, head=head,
        expansion_factor=expansion_factor, k=k, epochs=epochs,
        lr=lr, batch_size=batch_size, warmup_steps=warmup_steps,
        resample_every=resample_every, seed=seed, rank=rank,
        n_features=n_features, use_batchtopk=use_batchtopk, corpus_source=corpus_source,
    )


# -- Stage 3: Train sweep (all configs via .map) --

@app.function(volumes={DATA: vol}, image=image, timeout=14400)
def train_sweep(
    layers: list[int] = [1, 9, 17],  # noqa: B006
    corpus_source: str = "openwebtext",
):
    import json
    import os
    from pathlib import Path
    configs = _build_sweep_configs(layers)
    corpus_source = _normalize_corpus_source(corpus_source)
    meta_path = _states_dir(corpus_source) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)

    # skip already-completed configs
    vol.reload()
    pending, completed = [], []
    for c in configs:
        tag = f"{c['sae_type']}_L{c['layer']}_H{c['head']}_ef{c['expansion_factor']}_k{c['k']}_s{c.get('seed', 42)}"
        ckpt_dir = f"{DATA}/checkpoints/{exp_tag}/{tag}"
        cfg_path = f"{ckpt_dir}/config.json"
        if os.path.exists(f"{ckpt_dir}/best.pt") and os.path.exists(cfg_path):
            saved_cfg = json.loads(open(cfg_path).read())
            saved_sha = saved_cfg.get("code_sha", "unknown")
            if CURRENT_CODE_SHA == "unknown" or saved_sha != CURRENT_CODE_SHA:
                print(f"  STALE {tag}: code_sha {saved_sha} != {CURRENT_CODE_SHA}, retraining")
                pending.append(c)
            else:
                completed.append(tag)
        else:
            pending.append(c)

    if completed:
        print(f"Skipping {len(completed)} completed: {completed}")

    print(f"Launching {len(pending)} training jobs (of {len(configs)} total):")
    for c in pending:
        print(f"  {c['sae_type']} layer={c['layer']} ef={c['expansion_factor']} k={c['k']} seed={c.get('seed', 42)}")

    results = []
    failures: list[str] = []
    handles = [(train_sae.spawn(**c, corpus_source=corpus_source), c) for c in pending]
    for h, c in handles:
        tag = f"{c['sae_type']} L{c['layer']} ef={c['expansion_factor']} k={c['k']}"
        try:
            result = h.get()
            print(f"  DONE {tag}: mse={result['best_mse']:.6f} dead={result['final_n_dead']} [{result['total_time_s']:.0f}s]")
            results.append(result)
        except Exception as e:
            print(f"  FAIL {tag}: {e}")
            failures.append(f"{tag}: {e}")
    if failures:
        raise RuntimeError("train_allheads failed: " + "; ".join(failures))
    return results


@app.function(volumes={DATA: vol}, image=image, timeout=14400)
def train_nf_sweep(
    layers: list[int] = [1, 9, 17],  # noqa: B006
    corpus_source: str = "openwebtext",
):
    """Sweep over dictionary sizes (n_features) to measure feature utilization."""
    import json
    import os
    from pathlib import Path
    configs = _build_nf_sweep_configs(layers)
    corpus_source = _normalize_corpus_source(corpus_source)
    meta_path = _states_dir(corpus_source) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)

    # skip already-completed configs
    vol.reload()
    pending, completed = [], []
    for c in configs:
        nf_tag = f"nf{c['n_features']}"
        tag = f"{c['sae_type']}_L{c['layer']}_H{c['head']}_{nf_tag}_k{c['k']}_s{c.get('seed', 42)}"
        ckpt_dir = f"{DATA}/checkpoints/{exp_tag}/{tag}"
        cfg_path = f"{ckpt_dir}/config.json"
        if os.path.exists(f"{ckpt_dir}/best.pt") and os.path.exists(cfg_path):
            saved_cfg = json.loads(open(cfg_path).read())
            saved_sha = saved_cfg.get("code_sha", "unknown")
            if CURRENT_CODE_SHA == "unknown" or saved_sha != CURRENT_CODE_SHA:
                print(f"  STALE {tag}: code_sha {saved_sha} != {CURRENT_CODE_SHA}, retraining")
                pending.append(c)
            else:
                completed.append(tag)
        else:
            pending.append(c)

    if completed:
        print(f"Skipping {len(completed)} completed: {completed}")

    print(f"Launching {len(pending)} nf-sweep jobs (of {len(configs)} total):")
    for c in pending:
        print(f"  {c['sae_type']} layer={c['layer']} nf={c['n_features']} k={c['k']} seed={c.get('seed', 42)}")

    results = []
    failures: list[str] = []
    handles = [(train_sae.spawn(**c, corpus_source=corpus_source), c) for c in pending]
    for h, c in handles:
        tag = f"{c['sae_type']} L{c['layer']} nf={c['n_features']} k={c['k']} s{c.get('seed', 42)}"
        try:
            result = h.get()
            print(f"  DONE {tag}: mse={result['best_mse']:.6f} dead={result['final_n_dead']} [{result['total_time_s']:.0f}s]")
            results.append(result)
        except Exception as e:
            print(f"  FAIL {tag}: {e}")
            failures.append(f"{tag}: {e}")
    if failures:
        raise RuntimeError("train_nf_sweep failed: " + "; ".join(failures))
    return results


@app.function(volumes={DATA: vol}, image=image, timeout=14400)
def train_batchtopk_sweep(
    layers: list[int] = [9],  # noqa: B006
    n_features: int = 2048,
    k: int = 32,
    corpus_source: str = "openwebtext",
):
    """Sweep BatchTopK activation across flat/rank1/bilinear SAEs."""
    import json
    import os
    from pathlib import Path
    configs = _build_batchtopk_sweep_configs(layers, n_features=n_features, k=k)
    corpus_source = _normalize_corpus_source(corpus_source)
    meta_path = _states_dir(corpus_source) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)

    # skip already-completed configs
    vol.reload()
    pending, completed = [], []
    for c in configs:
        nf_tag = f"nf{c['n_features']}"
        tag = f"btk_{c['sae_type']}_L{c['layer']}_H{c['head']}_{nf_tag}_k{c['k']}_s{c.get('seed', 42)}"
        ckpt_dir = f"{DATA}/checkpoints/{exp_tag}/{tag}"
        cfg_path = f"{ckpt_dir}/config.json"
        if os.path.exists(f"{ckpt_dir}/best.pt") and os.path.exists(cfg_path):
            saved_cfg = json.loads(open(cfg_path).read())
            saved_sha = saved_cfg.get("code_sha", "unknown")
            if CURRENT_CODE_SHA == "unknown" or saved_sha != CURRENT_CODE_SHA:
                print(f"  STALE {tag}: code_sha {saved_sha} != {CURRENT_CODE_SHA}, retraining")
                pending.append(c)
            else:
                completed.append(tag)
        else:
            pending.append(c)

    if completed:
        print(f"Skipping {len(completed)} completed: {completed}")

    print(f"Launching {len(pending)} batchtopk-sweep jobs (of {len(configs)} total):")
    for c in pending:
        print(f"  {c['sae_type']} layer={c['layer']} nf={c['n_features']} k={c['k']} seed={c.get('seed', 42)} batchtopk=True")

    results = []
    failures: list[str] = []
    handles = [(train_sae.spawn(**c, corpus_source=corpus_source), c) for c in pending]
    for h, c in handles:
        tag = f"btk_{c['sae_type']} L{c['layer']} nf={c['n_features']} k={c['k']} s{c.get('seed', 42)}"
        try:
            result = h.get()
            print(f"  DONE {tag}: mse={result['best_mse']:.6f} dead={result['final_n_dead']} [{result['total_time_s']:.0f}s]")
            results.append(result)
        except Exception as e:
            print(f"  FAIL {tag}: {e}")
            failures.append(f"{tag}: {e}")
    if failures:
        raise RuntimeError("train_batchtopk_sweep failed: " + "; ".join(failures))
    return results


# -- Stage 4: Analysis --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def analyze(corpus_source: str = "openwebtext"):
    import json, sys
    from pathlib import Path

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, "/root")
    from sae import build_sae_from_config
    from analyze import rank_analysis, feature_activation_stats

    vol.reload()  # ensure we see all committed checkpoints

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    fig_dir = _figures_dir(corpus_source)
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No unified metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    results = {}

    print("Rank analysis on raw GDN states")
    rank_stats = {}
    for layer in meta["layer_indices"]:
        head_path = states_dir / f"layer_{layer}" / "head_0.npy"
        if not head_path.exists():
            continue
        # prefer per-layer metadata for accurate sample count (handles partial extraction)
        layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
        if layer_meta_path.exists():
            n_samples = json.loads(layer_meta_path.read_text())["n_samples"]
        else:
            n_samples = meta["n_samples"]
        data = np.lib.format.open_memmap(str(head_path), mode="r", dtype=np.float16,
                                          shape=(n_samples, key_dim, val_dim))
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=min(1000, n_samples), replace=False)
        sample = torch.from_numpy(data[idx].astype(np.float32))

        rd = rank_analysis(sample)
        rank_stats[layer] = {
            "mean_effective_rank": float(rd["effective_rank"].mean()),
            "mean_singular_values": rd["singular_values"].mean(0)[:10].tolist(),
        }
        print(f"  Layer {layer}: effective rank = {rank_stats[layer]['mean_effective_rank']:.1f}")

    results["rank_stats"] = rank_stats

    if rank_stats:
        fig, axes = plt.subplots(1, len(rank_stats), figsize=(5 * len(rank_stats), 4))
        if len(rank_stats) == 1:
            axes = [axes]
        for ax, (layer, stats) in zip(axes, rank_stats.items()):
            svs = stats["mean_singular_values"]
            ax.bar(range(len(svs)), svs)
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Mean singular value")
            ax.set_title(f"Layer {layer}")
        plt.tight_layout()
        plt.savefig(fig_dir / "singular_value_spectra.png", dpi=150)
        plt.close()

    def iter_ckpts(layer_filter=None):
        if not ckpt_root.exists():
            return
        for d in sorted(ckpt_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            cp, bp = d / "config.json", d / "best.pt"
            if not cp.exists() or not bp.exists():
                continue
            try:
                cfg = json.loads(cp.read_text())
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARN: skipping {d.name}: bad config.json: {e}")
                continue
            if layer_filter is not None and cfg.get("layer") != layer_filter:
                continue
            # Keep the summary focused on the original paper sweep checkpoints.
            # Later per-head nf=2048 runs live under the same experiment tag but set
            # expansion_factor=0, which would otherwise balloon analysis time and
            # mix distinct experiment families into the main comparison tables.
            if cfg.get("head", 0) != 0 or cfg.get("expansion_factor", 0) <= 0:
                continue
            try:
                ckpt = torch.load(bp, map_location="cpu", weights_only=True)
            except (RuntimeError, EOFError) as e:
                print(f"  WARN: skipping corrupt checkpoint {bp}: {e}")
                bp.unlink(missing_ok=True)  # remove so skip-completed logic won't reuse it
                continue
            yield d.name, cfg, ckpt

    print("Collecting training results")
    comparison = {}
    for tag, cfg, ckpt in iter_ckpts():
        comparison[tag] = {
            "sae_type": cfg["sae_type"], "layer": cfg["layer"],
            "best_val_mse": ckpt.get("val_mse"),
            "n_features": cfg.get("n_features"),
            "k": cfg["k"],
        }
        print(f"  {tag}: val_mse={ckpt.get('val_mse', 'N/A')}")
    results["training_results"] = comparison

    print("Feature activity analysis")
    interp = {}
    for layer in meta["layer_indices"]:
        head_path = states_dir / f"layer_{layer}" / "head_0.npy"
        if not head_path.exists():
            continue
        # per-layer sample count for correct memmap shape
        layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
        if layer_meta_path.exists():
            layer_n = json.loads(layer_meta_path.read_text())["n_samples"]
        else:
            layer_n = meta["n_samples"]
        data = np.lib.format.open_memmap(str(head_path), mode="r", dtype=np.float16,
                                          shape=(layer_n, key_dim, val_dim))
        rng = np.random.default_rng(42)
        idx = rng.choice(layer_n, size=min(2000, layer_n), replace=False)
        sample = torch.from_numpy(data[idx].astype(np.float32))

        for tag, cfg, ckpt in iter_ckpts(layer_filter=layer):
            sae_type = cfg["sae_type"]
            sae = build_sae_from_config(
                cfg,
                state_dict=ckpt["model_state_dict"],
                default_d_k=key_dim,
                default_d_v=val_dim,
            )
            sae.load_state_dict(ckpt["model_state_dict"])
            sae = sae.cuda().eval()

            is_flat = sae_type == "flat"
            # process in small batches to avoid OOM on large SAEs
            n_analyze = min(500, len(sample))
            sub = sample[:n_analyze]
            inp = sub.reshape(n_analyze, -1).cuda() if is_flat else sub.cuda()
            with torch.no_grad():
                stats = feature_activation_stats(sae, inp)
            n_alive = int((stats["max"] > 1e-8).sum())
            interp[tag] = {
                "n_alive": n_alive, "n_dead": len(stats["max"]) - n_alive,
                "mean_freq": float(stats["frequency"].mean()),
            }
            print(f"  {tag}: {n_alive}/{len(stats['max'])} alive")
            del sae, inp, sub
            torch.cuda.empty_cache()

    results["feature_interpretability"] = interp

    (fig_dir / "analysis_results.json").write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"Analysis complete. Results at {fig_dir}")
    return results


# -- Stage 4b: Extract texts for existing states (retroactive) --
# If states were extracted without texts, this re-streams the corpus
# and decodes the same sequences to produce texts.json.

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def extract_texts(
    n_samples: int = 10000,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
):
    """Generate texts.json for existing extracted states.

    Streams the same OpenWebText corpus and decodes the same token sequences
    that were used for state extraction. No model forward pass needed.
    """
    import json, os, sys
    from pathlib import Path
    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import load_corpus_tokens, decode_batch_texts

    corpus_source = _normalize_corpus_source(corpus_source)
    output_dir = _states_dir(corpus_source)
    texts_path = output_dir / "texts.json"

    print(f"Generating texts.json with tokenizer from {model_name}")

    # always regenerate from corpus.npy if available (prevents stale text pairing)
    corpus_path = output_dir / "corpus.npy"
    if corpus_path.exists():
        from extract_states import load_corpus_from_file, decode_batch_texts as _decode
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        batches = load_corpus_from_file(str(corpus_path), 32, n_samples=n_samples)
        texts = _decode(tokenizer, batches)
        texts_path.write_text(json.dumps(texts))
        vol.commit()
        print(f"Regenerated texts.json from corpus.npy: {len(texts)} entries")
        return {"n_texts": len(texts)}

    # fallback: stream fresh corpus (no corpus.npy available)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus_text_path = None
    if corpus_source == "ultrachat_200k":
        corpus_text_path, _ = _materialize_ultrachat_corpus(tokenizer, output_dir, n_samples, seq_len)
    batches = load_corpus_tokens(
        tokenizer,
        str(corpus_text_path) if corpus_text_path is not None else None,
        seq_len,
        n_samples,
        32,
    )
    actual = sum(b.shape[0] for b in batches)
    texts = decode_batch_texts(tokenizer, batches)[:actual]

    output_dir.mkdir(parents=True, exist_ok=True)
    texts_path.write_text(json.dumps(texts))
    vol.commit()

    print(f"Saved {len(texts)} text sequences to {texts_path}")
    return {"n_texts": len(texts)}


# -- Stage 4c: Feature interpretability analysis --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def interpret(
    layers: list[int] = [9],  # noqa: B006
    sae_types: list[str] = ["flat", "rank1", "bilinear"],  # noqa: B006
    n_top_features: int = 20,
    n_top_contexts: int = 10,
    corpus_source: str = "openwebtext",
):
    """Run feature interpretability analysis on trained SAE checkpoints.

    For each checkpoint found, produces:
      - Per-feature activation stats, top activating text contexts
      - v_dec/w_dec vectors and outer product data for rank-1 SAEs
      - Cross-SAE comparison table
      - All figures via visualize.py
    """
    import json, sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    sys.path.insert(0, "/root")
    from interpret import run_interpretability, compare_saes
    from visualize import (
        plot_rank1_feature_grid, plot_rank1_feature_detail,
        plot_enc_dec_alignment, plot_comparison_table, plot_activation_histogram,
    )
    import numpy as np

    vol.reload()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    output_root = Path(f"{DATA}/interpret") / exp_tag

    # Check for texts
    texts_path = states_dir / "texts.json"
    has_texts = texts_path.exists()
    if not has_texts:
        print("WARNING: texts.json not found. Run --stage extract-texts first for text contexts.")
        print("Proceeding without text annotations.")

    all_results = []

    for layer in layers:
        for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
            cfg_path = d / "config.json"
            best_path = d / "best.pt"
            if not (cfg_path.exists() and best_path.exists()):
                continue

            cfg = json.loads(cfg_path.read_text())
            if cfg.get("layer") != layer:
                continue
            if cfg.get("sae_type") not in sae_types:
                continue

            tag = d.name
            out_dir = str(output_root / tag)
            print(f"\n{'='*60}\nAnalyzing: {tag}\n{'='*60}")

            try:
                result = run_interpretability(
                    sae_path=str(best_path),
                    data_dir=str(states_dir),
                    layer=layer,
                    head=cfg.get("head", 0),
                    n_top_features=n_top_features,
                    n_top_contexts=n_top_contexts,
                    output_dir=out_dir,
                    device="cuda",
                )
                all_results.append(result)

                fig_dir = Path(out_dir) / "figures"
                fig_dir.mkdir(exist_ok=True)

                features = result["features"]
                has_rank1 = any(f.get("v_dec") for f in features)

                if has_rank1:
                    plot_rank1_feature_grid(
                        features, n_features=min(20, len(features)),
                        save_path=str(fig_dir / "rank1_feature_grid.pdf"))
                    for feat in features[:6]:
                        fi = feat["feature_idx"]
                        plot_rank1_feature_detail(
                            feat, save_path=str(fig_dir / f"feature_{fi}_detail.pdf"))
                    if any("v_enc_dec_cosine" in f for f in features):
                        plot_enc_dec_alignment(
                            features, save_path=str(fig_dir / "enc_dec_alignment.pdf"))

                stats_path = Path(out_dir) / "activation_stats.npz"
                if stats_path.exists():
                    plot_activation_histogram(
                        dict(np.load(str(stats_path))),
                        save_path=str(fig_dir / "activation_hist.pdf"))

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

    # Cross-SAE comparison
    if len(all_results) >= 2:
        comp = compare_saes(all_results, output_dir=str(output_root / "comparison"))
        comp_fig_dir = output_root / "comparison" / "figures"
        comp_fig_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison_table(comp, save_path=str(comp_fig_dir / "comparison.pdf"))

    vol.commit()

    summary = {
        "n_checkpoints_analyzed": len(all_results),
        "layers": layers,
        "sae_types": sae_types,
        "has_text_contexts": has_texts,
        "output_dir": str(output_root),
    }
    print(f"\nInterpretability analysis complete: {len(all_results)} checkpoints")
    return summary


# -- Stage 4c-s0: Targeted interpretability on S0-aligned features --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def interpret_s0(
    task: str = "gsm8k",
    n_top_contexts: int = 10,
):
    """Run interpretability on features identified by S0 decoder decomposition.

    Reads S0 results from the volume, extracts the top feature indices per
    checkpoint, and runs run_interpretability(feature_indices=...) on each.
    """
    import json, sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    sys.path.insert(0, "/root")
    from interpret import run_interpretability
    from visualize import (
        plot_rank1_feature_grid, plot_rank1_feature_detail,
        plot_activation_histogram,
    )
    import numpy as np

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag

    s0_path = Path(f"{DATA}/s0_decomposition/{task}/results.json")
    if not s0_path.exists():
        raise FileNotFoundError(
            f"No S0 results at {s0_path}. Run --stage s0 first.")
    s0_results = json.loads(s0_path.read_text())
    decompositions = s0_results.get("decompositions", {})
    if not decompositions:
        raise ValueError("S0 results contain no decompositions")

    # Check for texts
    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError(
            f"No texts.json at {texts_path}. Run --stage extract-texts first.")

    output_root = Path(f"{DATA}/interpret_s0") / exp_tag / task
    all_summaries = {}

    for tag, dec in decompositions.items():
        dd = dec.get("decoder_decomposition")
        if dd is None:
            print(f"  {tag}: no decoder_decomposition, skipping")
            continue

        # Extract feature indices from projection_top_k
        proj_top_k = dd.get("projection_top_k", [])
        if not proj_top_k:
            print(f"  {tag}: empty projection_top_k, skipping")
            continue

        feature_indices = [entry["feature"] for entry in proj_top_k]
        layer = dec["layer"]
        head = dec.get("head", 0)

        ckpt_dir = ckpt_root / tag
        best_path = ckpt_dir / "best.pt"
        if not best_path.exists():
            print(f"  {tag}: no checkpoint at {best_path}, skipping")
            continue

        out_dir = str(output_root / tag)
        print(f"\n{'='*60}")
        print(f"S0-targeted interpret: {tag}")
        print(f"  Layer {layer}, {len(feature_indices)} features from decoder projection")
        print(f"  Top features: {feature_indices[:10]}...")
        print(f"{'='*60}")

        try:
            result = run_interpretability(
                sae_path=str(best_path),
                data_dir=str(states_dir),
                layer=layer,
                head=head,
                n_top_contexts=n_top_contexts,
                output_dir=out_dir,
                device="cuda",
                feature_indices=feature_indices,
            )

            # Save the S0 projection scores alongside interpret results
            s0_meta = {
                "task": task,
                "projection_top_k": proj_top_k,
                "s0_norm": dd.get("s0_norm"),
                "projection_scores_max": dd.get("projection_scores_max"),
            }
            with open(os.path.join(out_dir, "s0_metadata.json"), "w") as f:
                json.dump(s0_meta, f, indent=2)

            fig_dir = Path(out_dir) / "figures"
            fig_dir.mkdir(exist_ok=True)

            features = result["features"]
            has_rank1 = any(f.get("v_dec") for f in features)

            if has_rank1:
                plot_rank1_feature_grid(
                    features, n_features=min(32, len(features)),
                    save_path=str(fig_dir / "s0_feature_grid.pdf"))
                for feat in features[:8]:
                    fi = feat["feature_idx"]
                    plot_rank1_feature_detail(
                        feat, save_path=str(fig_dir / f"feature_{fi}_detail.pdf"))

            stats_path = Path(out_dir) / "activation_stats.npz"
            if stats_path.exists():
                plot_activation_histogram(
                    dict(np.load(str(stats_path))),
                    save_path=str(fig_dir / "activation_hist.pdf"))

            # Summarize: for each feature, report frequency + top context snippet
            feat_summary = []
            for feat in features:
                fi = feat["feature_idx"]
                ctxs = feat.get("top_contexts", [])
                top_text = ctxs[0]["text"][:200] if ctxs else "(no activations)"
                proj_score = next(
                    (e["score"] for e in proj_top_k if e["feature"] == fi), None)
                feat_summary.append({
                    "feature": fi,
                    "projection_score": proj_score,
                    "frequency": feat["frequency"],
                    "max_activation": feat["max_activation"],
                    "top_context_snippet": top_text,
                })

            all_summaries[tag] = feat_summary
            print(f"\n  Results for {tag}:")
            for fs in feat_summary:
                print(f"    Feature {fs['feature']:>6}: "
                      f"proj={fs['projection_score']:+.4f}  "
                      f"freq={fs['frequency']:.4f}  "
                      f"max={fs['max_activation']:.4f}")
                print(f"      Top text: {fs['top_context_snippet'][:120]}...")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    output_root.mkdir(parents=True, exist_ok=True)
    with open(str(output_root / "s0_interpret_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    vol.commit()

    print(f"\nS0-targeted interpretability complete: {len(all_summaries)} checkpoints")
    return all_summaries


# -- Stage 4d: Download analysis results and plot effective-rank curve --

@app.function(volumes={DATA: vol}, image=image, timeout=300)
def plot_curve():
    """Generate effective-rank curve from analysis_results.json on the volume.

    Returns the analysis_results dict so the local entrypoint can also plot.
    """
    import json, sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, "/root")
    from visualize import plot_effective_rank_curve

    vol.reload()

    fig_dir = Path(f"{DATA}/figures")
    results_path = fig_dir / "analysis_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No analysis_results.json at {results_path}. Run analyze first.")

    results = json.loads(results_path.read_text())
    plot_effective_rank_curve(results, save_path=str(fig_dir / "fig_effective_rank_curve.pdf"))
    plot_effective_rank_curve(results, save_path=str(fig_dir / "fig_effective_rank_curve.png"))

    vol.commit()
    print(f"Effective-rank curve saved to {fig_dir}")
    return results


# -- Stage 4e: Download all results from volume to local disk --

@app.function(volumes={DATA: vol}, image=image, timeout=600)
def download_results():
    """Collect all key results from the volume into a single JSON bundle.

    Returns a dict with: metadata, analysis_results, interpret summaries,
    s0 results, and checkpoint configs. This lets the local entrypoint
    save everything to disk without needing volume access.
    """
    import json
    from pathlib import Path

    vol.reload()

    bundle = {}
    states_dir = Path(f"{DATA}/states")
    fig_dir = Path(f"{DATA}/figures")

    # metadata
    meta_path = states_dir / "metadata.json"
    if meta_path.exists():
        bundle["metadata"] = json.loads(meta_path.read_text())

    # analysis results (includes rank_stats and training_results)
    ar_path = fig_dir / "analysis_results.json"
    if ar_path.exists():
        bundle["analysis_results"] = json.loads(ar_path.read_text())

    # interpret summaries (per-checkpoint)
    interpret_root = Path(f"{DATA}/interpret")
    interpret_summaries = {}
    if interpret_root.exists():
        for exp_dir in interpret_root.iterdir():
            for ckpt_dir in sorted(exp_dir.iterdir()):
                if ckpt_dir.name == "comparison":
                    comp_path = ckpt_dir / "comparison.json"
                    if comp_path.exists():
                        interpret_summaries["comparison"] = json.loads(comp_path.read_text())
                    continue
                summary_path = ckpt_dir / "summary.json"
                if summary_path.exists():
                    interpret_summaries[ckpt_dir.name] = json.loads(summary_path.read_text())
    bundle["interpret_summaries"] = interpret_summaries

    # s0 decomposition results
    s0_root = Path(f"{DATA}/s0_decomposition")
    s0_results = {}
    if s0_root.exists():
        for task_dir in s0_root.iterdir():
            results_path = task_dir / "results.json"
            if results_path.exists():
                s0_results[task_dir.name] = json.loads(results_path.read_text())
    bundle["s0_results"] = s0_results

    # checkpoint configs (sae_type, layer, mse, etc.)
    ckpt_root = Path(f"{DATA}/checkpoints")
    ckpt_configs = {}
    if ckpt_root.exists():
        for exp_dir in ckpt_root.iterdir():
            for ckpt_dir in sorted(exp_dir.iterdir()):
                cfg_path = ckpt_dir / "config.json"
                if cfg_path.exists():
                    ckpt_configs[f"{exp_dir.name}/{ckpt_dir.name}"] = json.loads(cfg_path.read_text())
    bundle["checkpoint_configs"] = ckpt_configs

    print(f"Bundle: {len(bundle.get('analysis_results', {}).get('training_results', {}))} training results, "
          f"{len(interpret_summaries)} interpret summaries, "
          f"{len(s0_results)} s0 tasks, "
          f"{len(ckpt_configs)} checkpoints")
    return bundle


# -- Helper: write unified metadata to volume --

@app.function(volumes={DATA: vol}, image=image, timeout=60)
def _write_unified_metadata(metadata: dict):
    import json
    corpus_source = metadata.get("corpus_source", "openwebtext")
    meta_path = _states_dir(corpus_source) / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2))
    vol.commit()


# -- Stage 5: Full pipeline (pipelined extraction + training) --
#
# Timeline for 3 layers, 16 training configs:
#
#   Sequential (old):
#     [--- extract all 3 layers: ~46 min ---][--- train 16 SAEs: ~5 min ---][analyze]
#     Total: ~53 min
#
#   Pipelined (new):
#     GPU 1: [--- extract L1: ~15 min ---]
#     GPU 2: [--- extract L12: ~15 min ---]
#     GPU 3: [--- extract L22: ~15 min ---]
#                                     |---> train L1 jobs (2 configs)
#                                     |---> train L12 jobs (10 configs)
#                                     |---> train L22 jobs (2 configs)
#                                                         |---> analyze
#     Total: ~20 min (extraction dominates, training overlaps)

@app.function(volumes={DATA: vol}, image=image, timeout=28800)
def run_all(
    n_samples: int = 10000,
    layers: list[int] = [1, 9, 17],  # noqa: B006
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
):
    import json
    import time
    from pathlib import Path
    import torch

    t0 = time.time()
    corpus_source = _normalize_corpus_source(corpus_source)
    large = _model_is_large(model_name)
    all_configs = _build_sweep_configs(layers)
    exp_tag = _experiment_tag(model_name, seq_len, n_samples, corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    print(f"Model: {model_name} ({'A10G' if large else 'L4'})")

    # Phase 0: tokenize corpus once so all layers see identical text
    print("STAGE 0: Tokenize shared corpus")
    corpus_result = extract_corpus.remote(
        n_samples=n_samples,
        seq_len=seq_len,
        model_name=model_name,
        corpus_source=corpus_source,
    )
    print(f"  Corpus ready: {corpus_result['n_seqs']} seqs x {corpus_result['seq_len']} tokens")

    # Phase 1: launch extraction for each layer on its own GPU
    extract_fn = extract_layer_a10g if large else extract_layer
    print(f"STAGE 1: Extract GDN states ({len(layers)} layers in parallel on {len(layers)} GPUs)")
    extract_handles = {}
    for layer in layers:
        extract_handles[layer] = extract_fn.spawn(
            layer=layer,
            n_samples=n_samples,
            seq_len=seq_len,
            model_name=model_name,
            corpus_source=corpus_source,
        )

    # Phase 2: as each layer finishes, launch its training jobs
    print("STAGE 2: Train SAEs (pipelined with extraction)")
    train_handles = []
    layer_metas = {}
    remaining = set(layers)

    while remaining:
        for layer in list(remaining):
            try:
                meta = extract_handles[layer].get(timeout=5)
            except TimeoutError:
                continue

            layer_metas[layer] = meta
            remaining.discard(layer)
            elapsed_layer = meta["extraction_time_s"]
            print(f"  Layer {layer} extracted in {elapsed_layer:.1f}s "
                  f"({(time.time() - t0) / 60:.1f} min elapsed), launching training")

            layer_configs = [c for c in all_configs if c["layer"] == layer]
            for cfg in layer_configs:
                # include seed in tag to match train_sae() output path
                tag = (
                    f"{cfg['sae_type']}_L{cfg['layer']}_H{cfg['head']}"
                    f"_ef{cfg['expansion_factor']}_k{cfg['k']}"
                    f"_s{cfg.get('seed', 42)}"
                )
                best_path = ckpt_root / tag / "best.pt"
                cfg_path = ckpt_root / tag / "config.json"
                if best_path.exists() and cfg_path.exists():
                    saved = json.loads(cfg_path.read_text())
                    saved_sha = saved.get("code_sha", "unknown")
                    if CURRENT_CODE_SHA == "unknown" or saved_sha != CURRENT_CODE_SHA:
                        print(f"    STALE {tag}: sha {saved_sha} != {CURRENT_CODE_SHA}, retraining")
                    else:
                        print(f"    Skipping completed {tag}")
                        continue
                train_handles.append(train_sae.spawn(**cfg, corpus_source=corpus_source))

    # write unified metadata.json (analyze needs it)
    first = next(iter(layer_metas.values()))
    # use min across layers so consumers never read past any layer's actual data
    min_samples = min(m["n_samples"] for m in layer_metas.values())
    unified_meta = {
        "model": model_name,
        "corpus": _corpus_label(corpus_source),
        "corpus_source": corpus_source,
        "layer_indices": layers, "all_gdn_layers": first["all_gdn_layers"],
        "n_samples": min_samples,
        "n_heads": first["n_heads"], "key_head_dim": first["key_head_dim"],
        "value_head_dim": first["value_head_dim"],
        "state_shape_per_head": [min_samples, first["key_head_dim"], first["value_head_dim"]],
        "dtype": "float16",
        "seq_len": seq_len,
        "extraction_time_s": round(time.time() - t0, 1),
        "per_layer_times": {str(l): m["extraction_time_s"] for l, m in layer_metas.items()},
    }
    _write_unified_metadata.remote(unified_meta)
    print(f"Extraction complete: {(time.time() - t0) / 60:.1f} min wall clock")

    # collect training results
    train_results = []
    for handle in train_handles:
        result = handle.get()
        tag = f"{result['sae_type']} L{result['layer']} ef={result['expansion_factor']}"
        print(f"  DONE {tag}: mse={result['best_mse']:.6f} dead={result['final_n_dead']} [{result['total_time_s']:.0f}s]")
        train_results.append(result)

    # add summaries for configs that were already complete when this run started
    for cfg in all_configs:
        # include seed in tag to match train_sae() output path
        tag = (
            f"{cfg['sae_type']}_L{cfg['layer']}_H{cfg['head']}"
            f"_ef{cfg['expansion_factor']}_k{cfg['k']}"
            f"_s{cfg.get('seed', 42)}"
        )
        if any(
            r["sae_type"] == cfg["sae_type"]
            and r["layer"] == cfg["layer"]
            and r["head"] == cfg["head"]
            and r["expansion_factor"] == cfg["expansion_factor"]
            and r["k"] == cfg["k"]
            and r.get("seed") == cfg.get("seed", 42)
            for r in train_results
        ):
            continue

        config_path = ckpt_root / tag / "config.json"
        best_path = ckpt_root / tag / "best.pt"
        if not (config_path.exists() and best_path.exists()):
            continue

        stored_cfg = json.loads(config_path.read_text())
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        train_results.append({
            "sae_type": stored_cfg["sae_type"],
            "layer": stored_cfg["layer"],
            "head": stored_cfg["head"],
            "expansion_factor": stored_cfg["expansion_factor"],
            "k": stored_cfg["k"],
            "seed": stored_cfg.get("seed", 42),
            "code_sha": stored_cfg.get("code_sha", "unknown"),
            "n_features": stored_cfg["n_features"],
            "n_samples": unified_meta["n_samples"],
            "best_mse": best_ckpt.get("val_mse"),
            "final_mse": best_ckpt.get("val_mse"),
            "final_n_dead": None,
            "total_time_s": 0.0,
        })
    print(f"Training complete: {len(train_results)} configs, {(time.time() - t0) / 60:.1f} min elapsed")

    # Phase 3: analysis
    print("STAGE 3: Analysis")
    analysis_result = analyze.remote()

    elapsed = (time.time() - t0) / 60
    print(f"Pipeline completed in {elapsed:.1f} minutes")
    return {"extract": unified_meta, "train": train_results, "analysis": analysis_result}


# -- Stage 6: S0 State Decomposition --
#
# Trains S0 on Qwen3.5-0.8B using GSM8K training examples, then encodes the
# trained states through every SAE checkpoint. Compares zero (baseline) state
# vs S0-trained state to identify which features task adaptation activates.
#
# The S0 trainer (from arxiv:2604.01168) learns an initial recurrent state
# (n_heads, 128, 128) per GDN layer. Training takes ~1 min on an L4.

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def s0_decompose(
    task: str = "gsm8k",
    n_train_examples: int = 4,
    s0_steps: int = 20,
    s0_lr: float = 1e-3,
    s0_alpha: float = 0.07,
    model_name: str = "Qwen/Qwen3.5-0.8B",
):
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from analyze import s0_decomposition, s0_compare, s0_decoder_decomposition

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    layers = meta["layer_indices"]

    out_dir = Path(f"{DATA}/s0_decomposition/{task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Train S0 ----
    print(f"Training S0 on {task} ({s0_steps} steps, {n_train_examples} examples, model={model_name})")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    # detect GDN layers and state shape
    config = getattr(model.config, "text_config", model.config)
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    n_heads = config.linear_num_value_heads

    # load training data
    train_data = _load_s0_train_data(task, tokenizer, n_train_examples)
    print(f"  {len(train_data)} training examples loaded")

    # train S0 states: one (n_heads, key_dim, val_dim) parameter per GDN layer
    states = {
        i: torch.nn.Parameter(torch.zeros(n_heads, key_dim, val_dim, device="cuda", dtype=torch.float32))
        for i in gdn_layers
    }

    # patch GDN layers to inject initial states
    model_layers = model.model.layers if hasattr(model, "model") else model.layers
    originals = {}
    for i, state in states.items():
        gdn = model_layers[i].linear_attn
        originals[i] = gdn.chunk_gated_delta_rule

        def _make_patched(fn, s):
            def patched(*a, **kw):
                bs = a[0].shape[0]
                kw["initial_state"] = s.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
                return fn(*a, **kw)
            return patched

        gdn.chunk_gated_delta_rule = _make_patched(originals[i], state)

    # freeze model, train states
    for p in model.parameters():
        p.requires_grad_(False)
    for p in states.values():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(list(states.values()), lr=s0_lr)
    for step in range(s0_steps):
        optimizer.zero_grad()
        loss_sum = 0.0
        for text, prompt_len in train_data:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].cuda()
            labels = ids.clone()
            labels[0, :prompt_len] = -100
            loss = model(input_ids=ids, labels=labels).loss
            (loss / len(train_data)).backward()
            loss_sum += loss.item()

        # L2 regularization
        l2 = sum(p.pow(2).sum() for p in states.values())
        (5e-4 * l2).backward()
        torch.nn.utils.clip_grad_norm_(list(states.values()), 1.0)
        optimizer.step()

        log_every = max(1, s0_steps // 10)
        if step % log_every == 0 or step == s0_steps - 1:
            avg_norm = sum(float(p.detach().norm()) for p in states.values()) / len(states)
            print(f"  step {step}/{s0_steps}: loss={loss_sum/len(train_data):.4f} avg_norm={avg_norm:.6f}")

    # unpatch
    for i, fn in originals.items():
        model_layers[i].linear_attn.chunk_gated_delta_rule = fn

    s0_time = time.time() - t0
    print(f"S0 training done in {s0_time:.1f}s")

    # scale states by alpha (same as S0Trainer._scaled_states)
    scaled_states = {}
    for idx, state in states.items():
        raw = state.detach().clone()
        norm = raw.norm()
        if float(norm) > 0:
            scaled_states[idx] = (s0_alpha * raw / norm).cpu()
        else:
            scaled_states[idx] = (raw * s0_alpha).cpu()

    # save raw and scaled states
    torch.save({str(k): v for k, v in states.items()}, str(out_dir / "s0_states_raw.pt"))
    torch.save({str(k): v for k, v in scaled_states.items()}, str(out_dir / "s0_states_scaled.pt"))

    # free model memory
    del model, optimizer
    torch.cuda.empty_cache()

    # ---- Step 2: Encode through SAE checkpoints ----
    print("Encoding S0 states through SAE checkpoints")
    results = {"task": task, "s0_training_time_s": round(s0_time, 1),
               "s0_steps": s0_steps, "s0_alpha": s0_alpha,
               "n_train_examples": n_train_examples, "decompositions": {}}

    if not ckpt_root.exists():
        print(f"  No checkpoints at {ckpt_root}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        vol.commit()
        return results

    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        try:
            cfg = json.loads(cp.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARN: skipping {d.name}: bad config.json: {e}")
            continue
        tag = d.name
        layer = cfg.get("layer")
        if layer not in gdn_layers:
            continue

        try:
            ckpt = torch.load(bp, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError) as e:
            print(f"  WARN: corrupt checkpoint {bp}: {e}")
            continue

        # build SAE
        sae_type = cfg["sae_type"]
        sae = build_sae_from_config(
            cfg,
            state_dict=ckpt["model_state_dict"],
            default_d_k=key_dim,
            default_d_v=val_dim,
        )
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval()

        # decompose head 0 (SAEs are trained on head 0)
        head = cfg.get("head", 0)
        if layer in scaled_states:
            trained_head = scaled_states[layer][head]  # (key_dim, val_dim)
            zero_head = torch.zeros(key_dim, val_dim)

            dec = s0_compare(sae, zero_head, trained_head)
            dec["layer"] = layer
            dec["sae_type"] = sae_type
            dec["head"] = head
            dec["state_frobenius_norm"] = float(trained_head.norm())

            # Decoder-side decomposition (bypasses encoder bias)
            try:
                # Use raw (un-scaled) state direction for decoder projection
                raw_head = states[layer][head].detach().float().cpu()
                raw_norm = raw_head.norm()
                if raw_norm > 0:
                    # Rescale to match training data magnitude for meaningful projection
                    dec_decomp = s0_decoder_decomposition(sae, raw_head)
                else:
                    dec_decomp = s0_decoder_decomposition(sae, trained_head)
                dec["decoder_decomposition"] = dec_decomp
                n_nnls = dec_decomp.get("nnls_n_nonzero", 0)
                proj_max = dec_decomp.get("projection_scores_max", 0)
                print(f"  {tag}: encoder: gained={dec['n_gained']} cos={dec['cosine_similarity']:.4f} | "
                      f"decoder: nnls_features={n_nnls} proj_max={proj_max:.4f}")
            except Exception as e:
                print(f"  {tag}: encoder: gained={dec['n_gained']} cos={dec['cosine_similarity']:.4f} | "
                      f"decoder FAILED: {e}")

            results["decompositions"][tag] = dec

        del sae

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # also save a summary table
    summary_lines = [f"S0 Decomposition: {task} (alpha={s0_alpha}, {s0_steps} steps)"]
    summary_lines.append(f"Training time: {s0_time:.1f}s")
    summary_lines.append("")
    summary_lines.append(f"{'SAE':<45} {'Layer':>5} {'Gained':>7} {'Str':>5} {'Sup':>5} {'Cos':>7}")
    summary_lines.append("-" * 80)
    for tag, dec in results["decompositions"].items():
        summary_lines.append(
            f"{tag:<45} {dec['layer']:>5} {dec['n_gained']:>7} "
            f"{dec['n_strengthened']:>5} {dec['n_suppressed']:>5} "
            f"{dec['cosine_similarity']:>7.4f}"
        )
    summary = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary)
    print(f"\n{summary}")

    vol.commit()
    return results


def _load_s0_train_data(task: str, tokenizer, n_examples: int) -> list[tuple[str, int]]:
    """Load training examples for S0 training. Returns (text, prompt_token_len) pairs."""
    from datasets import load_dataset

    def _apply_chat(messages):
        # apply chat template without thinking tokens
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # strip thinking wrapper if present
        text = text.replace("<think>\n\n</think>\n\n", "")
        return text

    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        data = []
        for ex in ds.select(range(min(n_examples, len(ds)))):
            prompt_msg = [{"role": "user", "content":
                           f"Solve this math problem step by step. "
                           f"End with the answer after ####.\n\n{ex['question']}"}]
            prompt_text = _apply_chat(prompt_msg + [{"role": "assistant", "content": ""}])
            # full text = prompt + answer
            full_msg = prompt_msg + [{"role": "assistant", "content": ex["answer"]}]
            full_text = _apply_chat(full_msg)
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=2048)["input_ids"])
            data.append((full_text, prompt_len))
        return data

    if task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        data = []
        for ex in ds.select(range(min(n_examples, len(ds)))):
            prompt_msg = [{"role": "user", "content":
                           f"Complete the following Python function:\n\n{ex['prompt']}"}]
            prompt_text = _apply_chat(prompt_msg + [{"role": "assistant", "content": ""}])
            full_msg = prompt_msg + [{"role": "assistant", "content": ex["canonical_solution"]}]
            full_text = _apply_chat(full_msg)
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=2048)["input_ids"])
            data.append((full_text, prompt_len))
        return data

    raise ValueError(f"Unknown task: {task}. Supported: gsm8k, humaneval")


# -- Stage 6b: S0 Activation Shift Experiment --
#
# Compares SAE feature activations on GDN states extracted WITH vs WITHOUT S0.
# Unlike s0_decompose (which encodes a single S0 matrix), this runs N text
# sequences through the model twice: once with zero initial state, once with
# trained S0. For each SAE, it encodes both N-sample state sets and measures
# per-feature activation shift: frequency change, magnitude change, and the
# number of features with statistically significant change.
#
# Hypothesis: structured SAEs (rank1, bilinear) with more alive features detect
# finer-grained activation shifts than flat SAEs.

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def s0_shift(
    task: str = "gsm8k",
    layer: int = 9,
    head: int = 0,
    n_sequences: int = 200,
    n_train_examples: int = 4,
    s0_steps: int = 20,
    s0_lr: float = 1e-3,
    s0_alpha: float = 0.07,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    nf_target: int = 2048,
):
    """Extract GDN states with and without S0, encode through SAEs, measure feature shift.

    Steps:
        1. Load model, extract states at target layer for N sequences (baseline).
        2. Train S0 on task data, patch model, extract states again (S0 condition).
        3. For each SAE checkpoint at target layer with n_features=nf_target,
           encode both state sets and compute per-feature activation differences.
        4. Report: number of significantly shifted features per SAE type.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    out_dir = Path(f"{DATA}/s0_shift/{task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load model and corpus ----
    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")
    n_heads = config.linear_num_value_heads

    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")
    batch_size = 8
    batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
    actual = sum(b.shape[0] for b in batches)
    print(f"Loaded {actual} sequences from corpus")

    # ---- Step 2: Extract baseline states (no S0) ----
    print(f"Extracting baseline states at layer {layer}, head {head}")
    t0 = time.time()

    @torch.no_grad()
    def _extract_head_states(mdl, target_layer, target_head, token_batches):
        all_states = []
        for batch in token_batches:
            input_ids = batch.to("cuda")
            out = mdl(input_ids=input_ids, use_cache=True)
            state = out.past_key_values.layers[target_layer].recurrent_states
            # state: (batch, n_heads, d_k, d_v)
            head_state = state[:, target_head].float().cpu()
            all_states.append(head_state)
            del out
            torch.cuda.empty_cache()
        return torch.cat(all_states, dim=0)  # (N, d_k, d_v)

    states_baseline = _extract_head_states(model, layer, head, batches)
    t_baseline = time.time() - t0
    print(f"Baseline extraction: {states_baseline.shape} in {t_baseline:.1f}s")

    # ---- Step 3: Train S0 ----
    print(f"Training S0 on {task} ({s0_steps} steps, {n_train_examples} examples)")
    t0 = time.time()

    train_data = _load_s0_train_data(task, tokenizer, n_train_examples)
    print(f"  {len(train_data)} training examples loaded")

    s0_states = {
        i: torch.nn.Parameter(torch.zeros(n_heads, key_dim, val_dim, device="cuda", dtype=torch.float32))
        for i in gdn_layers
    }

    model_layers = model.model.layers if hasattr(model, "model") else model.layers
    originals = {}
    for i, state in s0_states.items():
        gdn = model_layers[i].linear_attn
        originals[i] = gdn.chunk_gated_delta_rule

        def _make_patched(fn, s):
            def patched(*a, **kw):
                bs = a[0].shape[0]
                kw["initial_state"] = s.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
                return fn(*a, **kw)
            return patched

        gdn.chunk_gated_delta_rule = _make_patched(originals[i], state)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in s0_states.values():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(list(s0_states.values()), lr=s0_lr)
    for step in range(s0_steps):
        optimizer.zero_grad()
        loss_sum = 0.0
        for text, prompt_len in train_data:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].cuda()
            labels = ids.clone()
            labels[0, :prompt_len] = -100
            loss = model(input_ids=ids, labels=labels).loss
            (loss / len(train_data)).backward()
            loss_sum += loss.item()

        l2 = sum(p.pow(2).sum() for p in s0_states.values())
        (5e-4 * l2).backward()
        torch.nn.utils.clip_grad_norm_(list(s0_states.values()), 1.0)
        optimizer.step()

        log_every = max(1, s0_steps // 5)
        if step % log_every == 0 or step == s0_steps - 1:
            avg_norm = sum(float(p.detach().norm()) for p in s0_states.values()) / len(s0_states)
            print(f"  step {step}/{s0_steps}: loss={loss_sum/len(train_data):.4f} avg_norm={avg_norm:.6f}")

    # scale states by alpha
    scaled_states = {}
    for idx, state in s0_states.items():
        raw = state.detach().clone()
        norm = raw.norm()
        if float(norm) > 0:
            scaled_states[idx] = s0_alpha * raw / norm
        else:
            scaled_states[idx] = raw * s0_alpha

    s0_time = time.time() - t0
    print(f"S0 training done in {s0_time:.1f}s")

    # ---- Step 4: Extract S0-conditioned states ----
    # Re-patch with scaled states for extraction (training used unscaled)
    for i, fn in originals.items():
        model_layers[i].linear_attn.chunk_gated_delta_rule = fn

    s0_originals = {}
    for i, scaled in scaled_states.items():
        gdn = model_layers[i].linear_attn
        s0_originals[i] = gdn.chunk_gated_delta_rule

        def _make_patched_scaled(fn, s):
            def patched(*a, **kw):
                bs = a[0].shape[0]
                kw["initial_state"] = s.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
                return fn(*a, **kw)
            return patched

        gdn.chunk_gated_delta_rule = _make_patched_scaled(s0_originals[i], scaled)

    # need to disable grad for extraction
    for p in s0_states.values():
        p.requires_grad_(False)

    print(f"Extracting S0-conditioned states at layer {layer}, head {head}")
    t0 = time.time()
    states_s0 = _extract_head_states(model, layer, head, batches)
    t_s0 = time.time() - t0
    print(f"S0 extraction: {states_s0.shape} in {t_s0:.1f}s")

    # unpatch
    for i, fn in s0_originals.items():
        model_layers[i].linear_attn.chunk_gated_delta_rule = fn

    # free model memory
    del model, optimizer
    torch.cuda.empty_cache()

    # ---- Step 5: State-level statistics ----
    diff_states = states_s0 - states_baseline  # (N, d_k, d_v)
    per_sample_diff_norm = diff_states.reshape(actual, -1).norm(dim=1)
    per_sample_base_norm = states_baseline.reshape(actual, -1).norm(dim=1)
    relative_change = (per_sample_diff_norm / (per_sample_base_norm + 1e-12)).numpy()

    state_stats = {
        "n_sequences": actual,
        "mean_baseline_norm": float(per_sample_base_norm.mean()),
        "mean_diff_norm": float(per_sample_diff_norm.mean()),
        "mean_relative_change": float(relative_change.mean()),
        "median_relative_change": float(np.median(relative_change)),
        "s0_frobenius_norm_layer": float(scaled_states[layer][head].norm()),
    }
    print(f"State-level: mean_relative_change={state_stats['mean_relative_change']:.6f}")

    # ---- Step 6: Encode through SAE checkpoints and measure feature shift ----
    print(f"Encoding through SAE checkpoints (nf={nf_target}, layer={layer})")

    results = {
        "task": task, "layer": layer, "head": head,
        "n_sequences": actual, "nf_target": nf_target,
        "s0_steps": s0_steps, "s0_alpha": s0_alpha,
        "s0_training_time_s": round(s0_time, 1),
        "state_stats": state_stats,
        "sae_results": {},
    }

    if not ckpt_root.exists():
        print(f"  No checkpoints at {ckpt_root}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        vol.commit()
        return results

    # Collect SAE checkpoints matching layer and nf_target
    sae_entries = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        try:
            cfg = json.loads(cp.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if cfg.get("layer") != layer:
            continue
        if cfg.get("n_features") != nf_target:
            continue
        sae_entries.append((d.name, cfg, bp))

    print(f"  Found {len(sae_entries)} checkpoints at layer={layer}, nf={nf_target}")

    # Group by sae_type for aggregation across seeds
    from collections import defaultdict
    type_results = defaultdict(list)

    for tag, cfg, bp in sae_entries:
        try:
            ckpt = torch.load(bp, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError) as e:
            print(f"  WARN: corrupt checkpoint {bp}: {e}")
            continue

        sae_type = cfg["sae_type"]
        nf = cfg["n_features"]
        sae = build_sae_from_config(
            cfg,
            state_dict=ckpt["model_state_dict"],
            default_d_k=key_dim,
            default_d_v=val_dim,
        )
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval()

        # Encode both state sets
        with torch.no_grad():
            acts_base = sae.encode(states_baseline)   # (N, nf)
            acts_s0 = sae.encode(states_s0)           # (N, nf)

        acts_base_np = acts_base.cpu().numpy()
        acts_s0_np = acts_s0.cpu().numpy()

        # Per-feature statistics
        freq_base = (acts_base_np > 0).mean(axis=0)   # (nf,)
        freq_s0 = (acts_s0_np > 0).mean(axis=0)
        mean_base = acts_base_np.mean(axis=0)
        mean_s0 = acts_s0_np.mean(axis=0)

        freq_diff = freq_s0 - freq_base
        mean_diff = mean_s0 - mean_base

        # Per-feature L1 activation difference across samples
        per_feature_l1 = np.abs(acts_s0_np - acts_base_np).mean(axis=0)  # (nf,)

        # Significance threshold: > 2 std of baseline activation variation
        # Baseline variation per feature = std of activation across samples
        base_std = acts_base_np.std(axis=0)  # (nf,)
        # For features that are always zero, use a small epsilon
        threshold = 2.0 * np.maximum(base_std, 1e-8)

        n_significant = int((per_feature_l1 > threshold).sum())
        n_alive_base = int((freq_base > 0).sum())
        n_alive_s0 = int((freq_s0 > 0).sum())

        # Alternative: count features where frequency changes by >5%
        n_freq_shift_5pct = int((np.abs(freq_diff) > 0.05).sum())
        n_freq_shift_10pct = int((np.abs(freq_diff) > 0.10).sum())

        # Features gained (dead in base, alive in S0)
        n_gained = int(((freq_base == 0) & (freq_s0 > 0)).sum())
        n_lost = int(((freq_base > 0) & (freq_s0 == 0)).sum())

        entry = {
            "sae_type": sae_type,
            "tag": tag,
            "n_features": nf,
            "k": cfg.get("k"),
            "n_alive_base": n_alive_base,
            "n_alive_s0": n_alive_s0,
            "n_gained": n_gained,
            "n_lost": n_lost,
            "n_significant_shift": n_significant,
            "n_freq_shift_5pct": n_freq_shift_5pct,
            "n_freq_shift_10pct": n_freq_shift_10pct,
            "mean_per_feature_l1": float(per_feature_l1.mean()),
            "max_per_feature_l1": float(per_feature_l1.max()),
            "median_per_feature_l1": float(np.median(per_feature_l1)),
            "mean_freq_diff": float(np.abs(freq_diff).mean()),
            "max_freq_diff": float(np.abs(freq_diff).max()),
            "cosine_sim_mean_acts": float(
                np.dot(mean_base, mean_s0)
                / (np.linalg.norm(mean_base) * np.linalg.norm(mean_s0) + 1e-12)
            ),
        }

        print(f"  {tag}: alive={n_alive_base}->{n_alive_s0} "
              f"sig_shift={n_significant} freq_5%={n_freq_shift_5pct} "
              f"gained={n_gained} lost={n_lost}")

        results["sae_results"][tag] = entry
        type_results[sae_type].append(entry)
        del sae, ckpt

    # ---- Step 7: Aggregate by SAE type ----
    summary = {}
    for sae_type, entries in type_results.items():
        n_seeds = len(entries)
        summary[sae_type] = {
            "n_seeds": n_seeds,
            "mean_alive_base": float(np.mean([e["n_alive_base"] for e in entries])),
            "mean_alive_s0": float(np.mean([e["n_alive_s0"] for e in entries])),
            "mean_significant_shift": float(np.mean([e["n_significant_shift"] for e in entries])),
            "std_significant_shift": float(np.std([e["n_significant_shift"] for e in entries])),
            "mean_freq_shift_5pct": float(np.mean([e["n_freq_shift_5pct"] for e in entries])),
            "std_freq_shift_5pct": float(np.std([e["n_freq_shift_5pct"] for e in entries])),
            "mean_freq_shift_10pct": float(np.mean([e["n_freq_shift_10pct"] for e in entries])),
            "mean_gained": float(np.mean([e["n_gained"] for e in entries])),
            "mean_lost": float(np.mean([e["n_lost"] for e in entries])),
            "mean_per_feature_l1": float(np.mean([e["mean_per_feature_l1"] for e in entries])),
            "mean_cosine_sim": float(np.mean([e["cosine_sim_mean_acts"] for e in entries])),
        }

    results["summary_by_type"] = summary

    print(f"\n{'='*80}")
    print(f"S0 Activation Shift Summary: {task}, layer {layer}, nf={nf_target}")
    print(f"{'='*80}")
    print(f"{'SAE Type':<12} {'Alive(base)':>12} {'Alive(S0)':>10} {'Sig Shift':>10} "
          f"{'Freq>5%':>8} {'Freq>10%':>9} {'Gained':>7} {'Lost':>6}")
    print("-" * 80)
    for sae_type, s in summary.items():
        print(f"{sae_type:<12} {s['mean_alive_base']:>12.1f} {s['mean_alive_s0']:>10.1f} "
              f"{s['mean_significant_shift']:>10.1f} "
              f"{s['mean_freq_shift_5pct']:>8.1f} {s['mean_freq_shift_10pct']:>9.1f} "
              f"{s['mean_gained']:>7.1f} {s['mean_lost']:>6.1f}")
    print(f"\nState-level relative change: {state_stats['mean_relative_change']:.6f}")

    # Save
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    np.savez_compressed(
        str(out_dir / "raw_states.npz"),
        baseline=states_baseline.numpy().astype(np.float16),
        s0=states_s0.numpy().astype(np.float16),
    )

    vol.commit()
    print(f"\nResults saved to {out_dir}")
    return results


# -- Stage 7: PCA/NMF Baselines --

@app.function(volumes={DATA: vol}, image=image, timeout=3600, memory=32768)
def run_baselines(layers: list[int] = [9]):  # noqa: B006
    """Run PCA and NMF baselines on extracted GDN states.

    Fits PCA (and NMF where non-negative) on flattened (N, d_k*d_v) states.
    Reports reconstruction MSE at k=1,2,4,8,16,32,64,128 components.
    Runs on CPU only (sklearn), no GPU needed.
    """
    import json, sys
    from pathlib import Path
    sys.path.insert(0, "/root")
    from baselines import run_baselines as _run_baselines

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    output_dir = Path(f"{DATA}/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for layer in layers:
        print(f"Running baselines for layer {layer}")
        result = _run_baselines(states_dir=str(states_dir), layer=layer, head=0)
        all_results[f"layer_{layer}"] = result

        # save per-layer
        layer_path = output_dir / f"layer_{layer}.json"
        layer_path.write_text(json.dumps(result, indent=2))
        print(f"  PCA-32 MSE: {result['pca'].get('32', 'N/A')}")
        if result.get("nmf"):
            print(f"  NMF-32 MSE: {result['nmf'].get('32', 'N/A')}")

    # save combined results
    (output_dir / "baselines.json").write_text(json.dumps(all_results, indent=2))
    vol.commit()

    print(f"Baselines complete for {len(layers)} layers. Results at {output_dir}")
    return all_results


# -- Null distribution for S0 decoder projection --

@app.function(image=image, timeout=3600, memory=32768)
def null_distribution_test(n_features: int = 16384, n_trials: int = 1000, d_k: int = 128, d_v: int = 128) -> dict:  # type: ignore[type-arg]
    import numpy as np
    import time

    print(f"Null distribution: {n_features} random rank-1 atoms, {n_trials} trials, {d_k}x{d_v}")
    t0 = time.time()
    max_scores = []

    for trial in range(n_trials):
        u = np.random.randn(d_k)
        v = np.random.randn(d_v)
        S0 = np.outer(u, v)
        V = np.random.randn(n_features, d_k)
        W = np.random.randn(n_features, d_v)
        # Normalize rows to unit norm (matching SAE decoder normalization)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        scores = np.einsum("ik,kv,iv->i", V, S0, W)
        max_scores.append(float(np.max(np.abs(scores))))
        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial+1}/{n_trials} ({time.time()-t0:.1f}s)")

    max_scores_arr = np.array(max_scores)
    return {
        "n_features": n_features, "n_trials": n_trials, "d_k": d_k, "d_v": d_v,
        "null_max_abs_score": {
            "mean": float(np.mean(max_scores_arr)),
            "std": float(np.std(max_scores_arr)),
            "median": float(np.median(max_scores_arr)),
            "p95": float(np.percentile(max_scores_arr, 95)),
            "p99": float(np.percentile(max_scores_arr, 99)),
            "max": float(np.max(max_scores_arr)),
        },
    }


# -- Proper null distribution for S0 decoder projection --
#
# Tests whether trained SAE features align with the real S0 state better than
# random rank-1 perturbations of the same norm. The null hypothesis: any
# rank-1 matrix with ||P||_F = ||S0||_F would project equally well onto the
# trained decoder atoms. Rejection means the S0 direction is special.

@app.function(volumes={DATA: vol}, image=image, timeout=3600, memory=32768)
def s0_proper_null(
    task: str = "gsm8k",
    n_trials: int = 1000,
    head: int = 0,
) -> dict:  # type: ignore[type-arg]
    """Proper null test: project real S0 onto trained SAE atoms vs random perturbations."""
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from sae import build_sae_from_config, MatrixSAE, BilinearMatrixSAE, FlatSAE

    vol.reload()

    # ---- Load S0 states ----
    s0_dir = Path(f"{DATA}/s0_decomposition/{task}")
    raw_path = s0_dir / "s0_states_raw.pt"
    scaled_path = s0_dir / "s0_states_scaled.pt"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No raw S0 states at {raw_path}. Run --stage s0 first.")

    raw_states = torch.load(raw_path, map_location="cpu", weights_only=True)
    scaled_states = torch.load(scaled_path, map_location="cpu", weights_only=True)
    print(f"Loaded S0 states: {len(raw_states)} layers")
    for k_layer, v_state in raw_states.items():
        print(f"  layer {k_layer}: shape={v_state.shape} norm={float(v_state.norm()):.6f}")

    # ---- Load metadata for checkpoint paths ----
    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoints at {ckpt_root}")

    # ---- Process each SAE checkpoint ----
    all_results = {}
    t0_total = time.time()

    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        try:
            cfg = json.loads(cp.read_text())
        except (json.JSONDecodeError, ValueError):
            continue

        tag = d.name
        layer = cfg.get("layer")
        sae_type = cfg.get("sae_type", "")

        layer_key = str(layer)
        if layer_key not in raw_states:
            continue

        s0_raw = raw_states[layer_key].detach().float()  # (n_heads, d_k, d_v)
        s0_head = s0_raw[head]  # (d_k, d_v)
        s0_frob = float(s0_head.norm())

        if s0_frob < 1e-12:
            print(f"  {tag}: S0 norm is zero, skipping")
            continue

        try:
            ckpt = torch.load(bp, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError) as e:
            print(f"  WARN: corrupt checkpoint {bp}: {e}")
            continue

        sae = build_sae_from_config(
            cfg, state_dict=ckpt["model_state_dict"],
            default_d_k=key_dim, default_d_v=val_dim,
        )
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval()

        # ---- Compute real projection score and null distribution (vectorized) ----
        t0 = time.time()
        is_matrix = isinstance(sae, (MatrixSAE, BilinearMatrixSAE))

        with torch.no_grad():
            if is_matrix:
                bias = sae.bias.detach().float()
                s0_centered = s0_head.detach() - bias
                if isinstance(sae, BilinearMatrixSAE):
                    V = sae.V_dec.detach().float().numpy()  # (nf, rank, d_k)
                    W_dec = sae.W_dec.detach().float().numpy()   # (nf, rank, d_v)
                else:
                    V = sae.V.detach().float().numpy()
                    W_dec = sae.W.detach().float().numpy()
                S0_np = s0_centered.numpy()
                bias_np = bias.numpy()

                if V.ndim == 3:
                    real_scores = np.einsum("irk,kv,irv->i", V, S0_np, W_dec)
                    # Precompute bias contribution per feature
                    bias_scores = np.einsum("irk,kv,irv->i", V, bias_np, W_dec)
                else:
                    real_scores = np.einsum("ik,kv,iv->i", V, S0_np, W_dec)
                    bias_scores = np.einsum("ik,kv,iv->i", V, bias_np, W_dec)
            elif isinstance(sae, FlatSAE):
                flat_bias = sae.decoder.bias.detach().float()
                s0_flat = (s0_head.detach().reshape(-1).float() - flat_bias).numpy()
                flat_bias_np = flat_bias.numpy()
                decoder_cols = sae.decoder.weight.detach().float().T.numpy()  # (nf, d_in)
                real_scores = decoder_cols @ s0_flat
            else:
                print(f"  {tag}: unknown SAE type {type(sae)}, skipping")
                del sae
                continue

        real_max = float(np.max(np.abs(real_scores)))
        n_features = len(real_scores)

        # ---- Vectorized null distribution ----
        U_all = np.random.randn(n_trials, key_dim).astype(np.float32)
        V_all = np.random.randn(n_trials, val_dim).astype(np.float32)
        # Compute norms: ||outer(u,v)||_F = ||u|| * ||v||
        norms = np.linalg.norm(U_all, axis=1) * np.linalg.norm(V_all, axis=1)
        scales = np.where(norms > 0, s0_frob / norms, 0.0).astype(np.float32)

        if is_matrix:
            # For P_j = scale_j * outer(u_j, v_j):
            # score(i,j) = scale_j * sum_r (V[i,r,:] @ u_j)(W[i,r,:] @ v_j) - bias_scores[i]
            if V.ndim == 3:
                # V: (nf, rank, d_k), U_all: (n_trials, d_k) -> VU: (nf, rank, n_trials)
                VU = np.einsum("irk,jk->irj", V, U_all)
                WV = np.einsum("irv,jv->irj", W_dec, V_all)
                # Sum over rank, apply scale: (nf, n_trials)
                raw_scores = np.einsum("irj,irj->ij", VU, WV)
            else:
                VU = V @ U_all.T   # (nf, n_trials)
                WV = W_dec @ V_all.T  # (nf, n_trials)
                raw_scores = VU * WV  # (nf, n_trials)

            # raw_scores: (nf, n_trials), scales: (n_trials,), bias_scores: (nf,)
            null_all = raw_scores * scales[np.newaxis, :] - bias_scores[:, np.newaxis]
            null_max_scores = np.max(np.abs(null_all), axis=0)  # (n_trials,)
        else:
            # Flat: decoder_cols @ (scale * vec(outer(u,v)) - flat_bias)
            bias_proj = decoder_cols @ flat_bias_np  # (nf,) constant
            D = decoder_cols.reshape(n_features, key_dim, val_dim)
            # Batch to avoid OOM: (nf, batch, d_v) intermediate
            null_max_scores = np.empty(n_trials, dtype=np.float32)
            batch_sz = 100
            for b_start in range(0, n_trials, batch_sz):
                b_end = min(b_start + batch_sz, n_trials)
                U_batch = U_all[b_start:b_end]  # (bs, d_k)
                V_batch = V_all[b_start:b_end]   # (bs, d_v)
                s_batch = scales[b_start:b_end]   # (bs,)
                DU = np.einsum("ikv,jk->ijv", D, U_batch)   # (nf, bs, d_v)
                raw_proj = np.einsum("ijv,jv->ij", DU, V_batch)  # (nf, bs)
                null_batch = raw_proj * s_batch[np.newaxis, :] - bias_proj[:, np.newaxis]
                null_max_scores[b_start:b_end] = np.max(np.abs(null_batch), axis=0)

        elapsed = time.time() - t0

        # ---- Statistics ----
        p_value = float(np.mean(null_max_scores >= real_max))
        z_score = float((real_max - np.mean(null_max_scores)) / (np.std(null_max_scores) + 1e-12))
        null_p99 = float(np.percentile(null_max_scores, 99))
        null_p95 = float(np.percentile(null_max_scores, 95))

        result = {
            "tag": tag,
            "layer": layer,
            "sae_type": sae_type,
            "n_features": n_features,
            "s0_frob_norm": s0_frob,
            "real_max_projection": real_max,
            "null_mean": float(np.mean(null_max_scores)),
            "null_std": float(np.std(null_max_scores)),
            "null_median": float(np.median(null_max_scores)),
            "null_p95": null_p95,
            "null_p99": null_p99,
            "null_max": float(np.max(null_max_scores)),
            "p_value": p_value,
            "z_score": z_score,
            "significant_p01": real_max > null_p99,
            "significant_p05": real_max > null_p95,
            "n_trials": n_trials,
            "time_s": round(elapsed, 1),
        }
        all_results[tag] = result

        sig_marker = "***" if real_max > null_p99 else ("**" if real_max > null_p95 else "")
        print(f"  {tag}: real={real_max:.6f} null_mean={np.mean(null_max_scores):.6f} "
              f"null_p99={null_p99:.6f} z={z_score:.2f} p={p_value:.4f} {sig_marker} "
              f"[{elapsed:.1f}s]")

        del sae

    # ---- Summary ----
    total_time = time.time() - t0_total
    n_sig_01 = sum(1 for r in all_results.values() if r["significant_p01"])
    n_sig_05 = sum(1 for r in all_results.values() if r["significant_p05"])
    n_total = len(all_results)

    summary = {
        "task": task,
        "n_trials": n_trials,
        "head": head,
        "n_checkpoints": n_total,
        "n_significant_p01": n_sig_01,
        "n_significant_p05": n_sig_05,
        "total_time_s": round(total_time, 1),
    }

    # Group by sae_type
    by_type: dict[str, list[dict]] = {}
    for r in all_results.values():
        by_type.setdefault(r["sae_type"], []).append(r)

    type_summaries = {}
    for stype, entries in sorted(by_type.items()):
        reals = [e["real_max_projection"] for e in entries]
        nulls = [e["null_mean"] for e in entries]
        zs = [e["z_score"] for e in entries]
        n_sig = sum(1 for e in entries if e["significant_p01"])
        type_summaries[stype] = {
            "n_checkpoints": len(entries),
            "n_significant_p01": n_sig,
            "mean_real_max": float(np.mean(reals)),
            "mean_null_mean": float(np.mean(nulls)),
            "mean_z_score": float(np.mean(zs)),
        }

    output = {
        "summary": summary,
        "type_summaries": type_summaries,
        "checkpoints": all_results,
    }

    out_path = Path(f"{DATA}/s0_decomposition/{task}/proper_null_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    vol.commit()

    print(f"\n{'='*80}")
    print(f"S0 Proper Null Test: {task} ({n_trials} random rank-1 perturbations)")
    print(f"{'='*80}")
    print(f"{'SAE Type':<12} {'N':>3} {'Sig(p<.01)':>10} {'Mean Real':>10} {'Mean Null':>10} {'Mean Z':>8}")
    print("-" * 60)
    for stype, ts in sorted(type_summaries.items()):
        print(f"{stype:<12} {ts['n_checkpoints']:>3} {ts['n_significant_p01']:>10} "
              f"{ts['mean_real_max']:>10.6f} {ts['mean_null_mean']:>10.6f} "
              f"{ts['mean_z_score']:>8.2f}")
    print(f"\nTotal: {n_sig_01}/{n_total} significant at p<0.01, "
          f"{n_sig_05}/{n_total} at p<0.05")
    print(f"Wall clock: {total_time:.1f}s")

    return output


# -- Stage 8: Downstream causal evaluation --
#
# Measures how well SAE-reconstructed states preserve model behavior.
# Split each sequence into prefix (builds up recurrent state) and suffix (scored).
# Replace the recurrent state for one layer's head 0 with the SAE reconstruction.
# Compare suffix perplexity against unmodified baseline.

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def evaluate_downstream_modal(
    layer: int = 9,
    n_sequences: int = 500,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    head: int = 0,
    split_fraction: float = 0.5,
    batch_size: int = 8,
    corpus_source: str = "openwebtext",
):
    """Compute downstream perplexity impact of SAE state reconstruction.

    For each SAE checkpoint trained on the given layer, measures how
    reconstructing the recurrent state through the SAE affects suffix perplexity.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from evaluate_downstream import (
        load_sae_from_checkpoint, evaluate_downstream, format_results_table,
    )

    vol.reload()

    t0 = time.time()

    # Load metadata to find experiment tag and checkpoints
    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    elif layer_meta_path.exists():
        meta = json.loads(layer_meta_path.read_text())
    else:
        raise FileNotFoundError(f"No metadata at {meta_path} or {layer_meta_path}")

    exp_tag = _experiment_tag(meta["model"], meta.get("seq_len", seq_len), meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Validate layer is GDN
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    corpus_path = states_dir / "corpus.npy"
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        corpus_text_path = None
        if corpus_source == "ultrachat_200k":
            corpus_text_path, _ = _materialize_ultrachat_corpus(tokenizer, states_dir, n_sequences, seq_len)
        batches = load_corpus_tokens(
            tokenizer,
            str(corpus_text_path) if corpus_text_path is not None else None,
            seq_len,
            n_sequences,
            batch_size,
        )
        actual = sum(b.shape[0] for b in batches)
        print(f"Tokenized {actual} sequences from {corpus_source}")

    sae_configs = []
    if ckpt_root.exists():
        for d in sorted(ckpt_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            cfg_path = d / "config.json"
            best_path = d / "best.pt"
            if not (cfg_path.exists() and best_path.exists()):
                continue

            try:
                cfg = json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, ValueError):
                continue

            if cfg.get("layer") != layer:
                continue
            if cfg.get("head", 0) != head:
                continue

            try:
                sae, sae_cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
            except Exception as e:
                print(f"  WARN: failed to load {d.name}: {e}")
                continue

            sae_configs.append({
                "tag": d.name,
                "sae": sae,
                "sae_type": cfg["sae_type"],
                "train_mse": train_mse,
            })

    if not sae_configs:
        print(f"No SAE checkpoints found for layer {layer} at {ckpt_root}")
        return {"error": "no checkpoints found"}

    print(f"Found {len(sae_configs)} SAE checkpoints for layer {layer}")

    results = evaluate_downstream(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        layer_idx=layer,
        sae_configs=sae_configs,
        head_idx=head,
        split_fraction=split_fraction,
        device="cuda",
    )
    results["model"] = model_name
    results["total_time_s"] = round(time.time() - t0, 1)

    table = format_results_table(results)
    print(f"\n{table}")

    out_dir = Path(f"{DATA}/downstream_eval") / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"layer_{layer}.json"
    results_path.write_text(json.dumps(results, indent=2))
    (out_dir / f"layer_{layer}_table.txt").write_text(table)

    vol.commit()
    print(f"\nResults saved to {results_path}")
    return results


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def evaluate_downstream_allheads_modal(
    layer: int = 9,
    n_sequences: int = 500,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    head: int = 0,
    split_fraction: float = 0.5,
    batch_size: int = 8,
    corpus_source: str = "openwebtext",
):
    """Downstream perplexity with ALL heads reconstructed through each SAE.

    Loads head-0-trained SAE checkpoints and applies each to every head in the
    target layer. This amplifies the structured-vs-flat MSE gap by n_heads
    compared to single-head replacement.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from evaluate_downstream import (
        load_sae_from_checkpoint, evaluate_downstream_allheads, format_results_table,
    )

    vol.reload()

    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    elif layer_meta_path.exists():
        meta = json.loads(layer_meta_path.read_text())
    else:
        raise FileNotFoundError(f"No metadata at {meta_path} or {layer_meta_path}")

    exp_tag = _experiment_tag(meta["model"], meta.get("seq_len", seq_len), meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Validate layer is GDN
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    n_heads = config.linear_num_value_heads
    print(f"Model has {n_heads} GDN heads per layer")

    corpus_path = states_dir / "corpus.npy"
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        corpus_text_path = None
        if corpus_source == "ultrachat_200k":
            corpus_text_path, _ = _materialize_ultrachat_corpus(tokenizer, states_dir, n_sequences, seq_len)
        batches = load_corpus_tokens(
            tokenizer,
            str(corpus_text_path) if corpus_text_path is not None else None,
            seq_len,
            n_sequences,
            batch_size,
        )
        actual = sum(b.shape[0] for b in batches)
        print(f"Tokenized {actual} sequences from {corpus_source}")

    # Find all SAE checkpoints for this layer (head=0 only, applied to all heads)
    sae_configs = []
    if ckpt_root.exists():
        for d in sorted(ckpt_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            cfg_path = d / "config.json"
            best_path = d / "best.pt"
            if not (cfg_path.exists() and best_path.exists()):
                continue

            try:
                cfg = json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, ValueError):
                continue

            if cfg.get("layer") != layer:
                continue
            if cfg.get("head", 0) != head:
                continue

            try:
                sae, sae_cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
            except Exception as e:
                print(f"  WARN: failed to load {d.name}: {e}")
                continue

            sae_configs.append({
                "tag": d.name,
                "sae": sae,
                "sae_type": cfg["sae_type"],
                "train_mse": train_mse,
            })

    if not sae_configs:
        print(f"No SAE checkpoints found for layer {layer} at {ckpt_root}")
        return {"error": "no checkpoints found"}

    print(f"Found {len(sae_configs)} SAE checkpoints for layer {layer}")
    print(f"Each SAE (trained on head {head}) will be applied to all {n_heads} heads")

    results = evaluate_downstream_allheads(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        layer_idx=layer,
        sae_configs=sae_configs,
        n_heads=n_heads,
        split_fraction=split_fraction,
        device="cuda",
    )
    results["model"] = model_name
    results["total_time_s"] = round(time.time() - t0, 1)

    table = format_results_table(results)
    print(f"\n{table}")

    out_dir = Path(f"{DATA}/downstream_eval") / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"layer_{layer}_allheads.json"
    results_path.write_text(json.dumps(results, indent=2))
    (out_dir / f"layer_{layer}_allheads_table.txt").write_text(table)

    vol.commit()
    print(f"\nResults saved to {results_path}")
    return results


# -- Stage 9: Train per-head SAEs (all 16 heads) --

@app.function(volumes={DATA: vol}, image=image, timeout=14400)
def train_perhead_sweep(
    layer: int = 9,
    heads: list[int] | None = None,
    sae_types: list[str] | None = None,
    n_features: int = 2048,
    k: int = 32,
    seeds: list[int] | None = None,
    corpus_source: str = "openwebtext",
):
    """Train SAEs for each GDN head at a given layer.

    Generates configs via _build_perhead_configs and spawns parallel train_sae
    jobs. Skips already-completed checkpoints by checking best.pt + code SHA.
    """
    import json
    import os
    from pathlib import Path

    configs = _build_perhead_configs(
        layer=layer, heads=heads, sae_types=sae_types,
        n_features=n_features, k=k, seeds=seeds,
    )

    vol.reload()
    corpus_source = _normalize_corpus_source(corpus_source)
    states_root = _states_dir(corpus_source)
    meta_path = states_root / "metadata.json"
    layer_meta_path = states_root / f"layer_{layer}" / "layer_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    elif layer_meta_path.exists():
        meta = json.loads(layer_meta_path.read_text())
    else:
        raise FileNotFoundError(f"No metadata at {meta_path} or {layer_meta_path}. Run extraction first.")
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)

    # Verify extracted states exist for all requested heads
    states_dir = states_root / f"layer_{layer}"
    requested_heads = (heads if heads is not None else list(range(16)))
    missing = [h for h in requested_heads if not (states_dir / f"head_{h}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing extracted states for heads {missing} at {states_dir}. "
            f"Run extraction first (extract_states.py extracts all heads by default)."
        )

    # Skip already-completed configs
    pending, completed = [], []
    for c in configs:
        nf_tag = f"nf{c['n_features']}"
        tag = f"{c['sae_type']}_L{c['layer']}_H{c['head']}_{nf_tag}_k{c['k']}_s{c.get('seed', 42)}"
        ckpt_dir = f"{DATA}/checkpoints/{exp_tag}/{tag}"
        cfg_path = f"{ckpt_dir}/config.json"
        if os.path.exists(f"{ckpt_dir}/best.pt") and os.path.exists(cfg_path):
            saved_cfg = json.loads(open(cfg_path).read())
            saved_sha = saved_cfg.get("code_sha", "unknown")
            if CURRENT_CODE_SHA == "unknown" or saved_sha != CURRENT_CODE_SHA:
                print(f"  STALE {tag}: code_sha {saved_sha} != {CURRENT_CODE_SHA}, retraining")
                pending.append(c)
            else:
                completed.append(tag)
        else:
            pending.append(c)

    if completed:
        print(f"Skipping {len(completed)} completed configs")

    print(f"Launching {len(pending)} per-head training jobs (of {len(configs)} total):")
    for c in pending:
        print(f"  {c['sae_type']} layer={c['layer']} head={c['head']} nf={c['n_features']} k={c['k']} seed={c.get('seed', 42)}")

    results = []
    handles = [(train_sae.spawn(**c, corpus_source=corpus_source), c) for c in pending]
    for h, c in handles:
        tag = f"{c['sae_type']} L{c['layer']} H{c['head']} nf={c['n_features']} k={c['k']}"
        try:
            result = h.get()
            print(f"  DONE {tag}: mse={result['best_mse']:.6f} dead={result['final_n_dead']} [{result['total_time_s']:.0f}s]")
            results.append(result)
        except Exception as e:
            print(f"  FAIL {tag}: {e}")
    return results


# -- Stage 10: Per-head matched downstream evaluation --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def evaluate_downstream_perhead_matched_modal(
    layer: int = 9,
    n_sequences: int = 500,
    seq_len: int = 1024,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    split_fraction: float = 0.5,
    batch_size: int = 8,
    n_features: int = 2048,
    k: int = 32,
    seed: int = 42,
    corpus_source: str = "openwebtext",
):
    """Downstream perplexity with per-head MATCHED SAEs.

    For each SAE type (flat, rank1, bilinear), loads 16 SAEs (one per head)
    and reconstructs each head through its matched SAE. This removes the
    spectral mismatch from the all-heads eval that reused head-0's SAE.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from evaluate_downstream import (
        load_sae_from_checkpoint, evaluate_downstream_perhead_matched,
        format_results_table,
    )

    vol.reload()

    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    elif layer_meta_path.exists():
        meta = json.loads(layer_meta_path.read_text())
    else:
        raise FileNotFoundError(f"No metadata at {meta_path} or {layer_meta_path}")

    exp_tag = _experiment_tag(meta["model"], meta.get("seq_len", seq_len), meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Validate layer is GDN
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    n_heads = config.linear_num_value_heads
    print(f"Model has {n_heads} GDN heads per layer")

    corpus_path = states_dir / "corpus.npy"
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        corpus_text_path = None
        if corpus_source == "ultrachat_200k":
            corpus_text_path, _ = _materialize_ultrachat_corpus(tokenizer, states_dir, n_sequences, seq_len)
        batches = load_corpus_tokens(
            tokenizer,
            str(corpus_text_path) if corpus_text_path is not None else None,
            seq_len,
            n_sequences,
            batch_size,
        )
        actual = sum(b.shape[0] for b in batches)
        print(f"Tokenized {actual} sequences from {corpus_source}")

    # For each SAE type, load all 16 per-head SAEs
    sae_type_configs = {}  # {tag: {head_idx: (sae, sae_type_str)}}

    for sae_type in ["flat", "rank1", "bilinear"]:
        head_saes = {}
        nf_tag = f"nf{n_features}"
        missing_heads = []

        for h in range(n_heads):
            tag = f"{sae_type}_L{layer}_H{h}_{nf_tag}_k{k}_s{seed}"
            ckpt_dir = ckpt_root / tag
            cfg_path = ckpt_dir / "config.json"
            best_path = ckpt_dir / "best.pt"

            if not (cfg_path.exists() and best_path.exists()):
                missing_heads.append(h)
                continue

            try:
                sae, sae_cfg, train_mse = load_sae_from_checkpoint(
                    str(best_path), str(cfg_path), device="cuda",
                )
                head_saes[h] = (sae, sae_type)
            except Exception as e:
                print(f"  WARN: failed to load {tag}: {e}")
                missing_heads.append(h)

        if missing_heads:
            print(f"  {sae_type}: missing heads {missing_heads}")

        if head_saes:
            type_tag = f"{sae_type} (per-head matched, {len(head_saes)}/{n_heads} heads)"
            sae_type_configs[type_tag] = head_saes
            print(f"  Loaded {sae_type}: {len(head_saes)}/{n_heads} heads")
        else:
            print(f"  {sae_type}: no per-head checkpoints found, skipping")

    if not sae_type_configs:
        print(f"No per-head SAE checkpoints found at {ckpt_root}")
        return {"error": "no per-head checkpoints found"}

    results = evaluate_downstream_perhead_matched(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        layer_idx=layer,
        sae_type_configs=sae_type_configs,
        n_heads=n_heads,
        split_fraction=split_fraction,
        device="cuda",
    )
    results["model"] = model_name
    results["seed"] = seed
    results["total_time_s"] = round(time.time() - t0, 1)

    table = format_results_table(results)
    print(f"\n{table}")

    out_dir = Path(f"{DATA}/downstream_eval") / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_suffix = f"_s{seed}"
    results_path = out_dir / f"layer_{layer}_perhead_matched{seed_suffix}.json"
    table_path = out_dir / f"layer_{layer}_perhead_matched{seed_suffix}_table.txt"
    results_path.write_text(json.dumps(results, indent=2))
    table_path.write_text(table)

    vol.commit()
    print(f"\nResults saved to {results_path}")
    return results


# -- Stage 9: Temporal dynamics of GDN recurrent states --
#
# GDN states accumulate via S_t = alpha*S_{t-1} + beta*k*v^T. Different SAE
# features should have different temporal profiles: early-onset, persistent,
# transient, late-emerging. This stage extracts states at multiple sequence
# positions and encodes them through trained SAEs to reveal those dynamics.
#
# Approach (option a): for each target position, truncate sequences to that
# length and run a full forward pass. 7 positions x N sequences. Simple and
# correct, at the cost of ~7x extraction time.

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def extract_temporal(
    layer: int = 9,
    head: int = 0,
    n_samples: int = 500,
    positions: list[int] = [32, 64, 128, 256, 512, 768, 1024],  # noqa: B006
    model_name: str = "Qwen/Qwen3.5-0.8B",
):
    """Extract GDN recurrent states at multiple sequence positions.

    For each position p in `positions`, truncates sequences to length p,
    runs a forward pass, and captures the recurrent state at `layer`.
    Saves per-position per-head arrays: temporal_states/layer_{L}/pos_{p}/head_{h}.npy

    This is option (a) from the plan: simple truncation, 7 forward passes per
    sequence. Guaranteed correct because the recurrent state after position p
    is exactly what the model computes for a length-p input.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import (
        load_model_and_tokenizer, get_gdn_layer_indices,
        load_corpus_from_file,
    )

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    out_root = Path(f"{DATA}/temporal_states/layer_{layer}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Check if already done
    done_path = out_root / "temporal_metadata.json"
    if done_path.exists():
        existing = json.loads(done_path.read_text())
        if (existing.get("n_samples", 0) >= n_samples
                and set(existing.get("positions", [])) == set(positions)
                and existing.get("model") == model_name
                and existing.get("layer") == layer):
            print(f"Temporal extraction already done: {existing['n_samples']} samples, "
                  f"{len(existing['positions'])} positions. Skipping.")
            return existing

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    gdn_layers = get_gdn_layer_indices(config)
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    # Load corpus (need max position length)
    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")

    batch_size = _extract_batch_size(model_name)
    batches_full = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_samples)
    actual_samples = sum(b.shape[0] for b in batches_full)
    print(f"Loaded {actual_samples} sequences (will truncate to each position)")

    # Probe state dimensions
    from extract_states import probe_state_dims
    n_heads, key_dim, val_dim = probe_state_dims(model, layer, tokenizer, "cuda")
    print(f"Layer {layer}: {n_heads} heads x ({key_dim}, {val_dim})")

    t0 = time.time()
    position_times = {}

    for pos in positions:
        pos_dir = out_root / f"pos_{pos}"
        pos_dir.mkdir(parents=True, exist_ok=True)

        # Check if this position already extracted
        pos_done = pos_dir / "done.json"
        if pos_done.exists():
            pd = json.loads(pos_done.read_text())
            if pd.get("n_samples") == actual_samples:
                print(f"  pos={pos}: already extracted, skipping")
                continue

        print(f"  Extracting states at position {pos}...")
        tp0 = time.time()

        head_memmaps = []
        for h in range(n_heads):
            fpath = str(pos_dir / f"head_{h}.npy")
            mm = np.lib.format.open_memmap(
                fpath, mode="w+", dtype=np.float16,
                shape=(actual_samples, key_dim, val_dim),
            )
            head_memmaps.append(mm)

        sample_offset = 0
        with torch.no_grad():
            for batch in batches_full:
                # Truncate to position length
                truncated = batch[:, :pos].to("cuda")
                bs = truncated.shape[0]

                outputs = model(input_ids=truncated, use_cache=True)
                cache = outputs.past_key_values

                # state shape: (batch, num_value_heads, key_head_dim, value_head_dim)
                state_np = (cache.layers[layer].recurrent_states
                            .float().cpu().numpy().astype(np.float16))
                for h in range(n_heads):
                    head_memmaps[h][sample_offset:sample_offset + bs] = state_np[:, h]

                sample_offset += bs
                del outputs, cache
                torch.cuda.empty_cache()

        for mm in head_memmaps:
            mm.flush()

        tp_elapsed = time.time() - tp0
        position_times[pos] = round(tp_elapsed, 1)
        print(f"  pos={pos}: {sample_offset} samples in {tp_elapsed:.1f}s")

        # Mark position as done
        pos_done.write_text(json.dumps({
            "n_samples": sample_offset, "position": pos, "layer": layer,
        }))

    elapsed = time.time() - t0

    metadata = {
        "model": model_name,
        "layer": layer,
        "head": head,
        "positions": positions,
        "n_samples": actual_samples,
        "n_heads": n_heads,
        "key_head_dim": key_dim,
        "value_head_dim": val_dim,
        "dtype": "float16",
        "extraction_time_s": round(elapsed, 1),
        "per_position_time_s": position_times,
    }
    done_path.write_text(json.dumps(metadata, indent=2))
    vol.commit()

    print(f"\nTemporal extraction complete: {len(positions)} positions x {actual_samples} samples "
          f"in {elapsed:.1f}s")
    return metadata


@app.function(volumes={DATA: vol}, **GPU_KWARGS)
def analyze_temporal(
    layer: int = 9,
    head: int = 0,
    n_features_target: int = 2048,
    n_clusters: int = 6,
):
    """Encode temporal states through trained SAEs and analyze feature dynamics.

    For each SAE checkpoint (bilinear and rank1 at nf=n_features_target):
      1. Load states at each position
      2. Encode through SAE
      3. Compute per-feature mean activation at each position (retention curve)
      4. Classify features: early-onset, persistent, transient, late-emerging, etc.
      5. Cluster features by normalized retention curve (k-means)

    Saves temporal_analysis.json with per-SAE results.
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from sae import build_sae_from_config

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    temporal_root = Path(f"{DATA}/temporal_states/layer_{layer}")
    temporal_meta_path = temporal_root / "temporal_metadata.json"
    if not temporal_meta_path.exists():
        raise FileNotFoundError(
            f"No temporal metadata at {temporal_meta_path}. "
            f"Run extract_temporal first.")
    temporal_meta = json.loads(temporal_meta_path.read_text())
    positions = temporal_meta["positions"]
    n_samples = temporal_meta["n_samples"]
    n_heads = temporal_meta["n_heads"]
    print(f"Temporal data: {n_samples} samples, {len(positions)} positions: {positions}")

    # states_by_pos[p] shape: (n_samples, key_dim, val_dim) as float32 tensor
    print(f"Loading states for head {head} at {len(positions)} positions...")
    states_by_pos = {}
    for pos in positions:
        fpath = temporal_root / f"pos_{pos}" / f"head_{head}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing state file: {fpath}")
        arr = np.lib.format.open_memmap(
            str(fpath), mode="r", dtype=np.float16,
            shape=(n_samples, key_dim, val_dim),
        )
        states_by_pos[pos] = torch.from_numpy(arr[:].astype(np.float32))
        print(f"  pos={pos}: shape={states_by_pos[pos].shape}")

    # Find SAE checkpoints matching layer and n_features_target
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoints at {ckpt_root}")

    sae_entries = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        try:
            cfg = json.loads(cp.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if cfg.get("layer") != layer:
            continue
        if cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_entries.append((d.name, cfg, bp))

    print(
        f"Found {len(sae_entries)} SAE checkpoints at "
        f"layer={layer}, head={head}, nf={n_features_target}"
    )
    if not sae_entries:
        raise FileNotFoundError(
            f"No SAE checkpoints found for layer={layer}, head={head}, nf={n_features_target}"
        )

    t0 = time.time()
    all_results = {}

    for tag, cfg, bp in sae_entries:
        sae_type = cfg["sae_type"]
        print(f"\n{'='*60}")
        print(f"Analyzing: {tag} ({sae_type})")
        print(f"{'='*60}")

        try:
            ckpt = torch.load(bp, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError) as e:
            print(f"  WARN: corrupt checkpoint {bp}: {e}")
            continue

        sae = build_sae_from_config(
            cfg,
            state_dict=ckpt["model_state_dict"],
            default_d_k=key_dim,
            default_d_v=val_dim,
        )
        sae.load_state_dict(ckpt["model_state_dict"])
        sae = sae.cuda().eval()

        nf = cfg["n_features"]

        # Encode states at each position
        # acts_by_pos[p] shape: (n_samples, nf)
        acts_by_pos = {}
        with torch.no_grad():
            for pos in positions:
                acts = sae.encode(states_by_pos[pos].cuda())  # (n_samples, nf)
                acts_by_pos[pos] = acts.cpu().numpy()
                torch.cuda.empty_cache()

        # Build retention curves: (nf, n_positions) = mean activation at each position
        n_pos = len(positions)
        retention = np.zeros((nf, n_pos), dtype=np.float32)
        freq_curves = np.zeros((nf, n_pos), dtype=np.float32)
        for pi, pos in enumerate(positions):
            a = acts_by_pos[pos]  # (n_samples, nf)
            retention[:, pi] = a.mean(axis=0)
            freq_curves[:, pi] = (a > 0).mean(axis=0)

        # Identify alive features (nonzero activation at any position)
        max_act_per_feature = retention.max(axis=1)  # (nf,)
        alive_mask = max_act_per_feature > 1e-8
        n_alive = int(alive_mask.sum())
        alive_indices = np.where(alive_mask)[0]
        print(f"  Alive features: {n_alive}/{nf}")

        if n_alive == 0:
            print(f"  No alive features, skipping clustering")
            all_results[tag] = {
                "sae_type": sae_type, "tag": tag, "n_features": nf,
                "n_alive": 0, "positions": positions,
            }
            del sae, ckpt
            torch.cuda.empty_cache()
            continue

        # Normalize retention curves for alive features (each to max=1)
        alive_retention = retention[alive_mask]  # (n_alive, n_pos)
        row_max = alive_retention.max(axis=1, keepdims=True)
        row_max = np.maximum(row_max, 1e-12)
        normalized = alive_retention / row_max  # (n_alive, n_pos)

        # Classify features by temporal profile heuristics
        def _classify_feature(curve):
            """Classify a normalized retention curve into a temporal category."""
            first_quarter = curve[:n_pos // 4].mean() if n_pos >= 4 else curve[0]
            last_quarter = curve[-(n_pos // 4):].mean() if n_pos >= 4 else curve[-1]
            mid = curve[n_pos // 4: 3 * n_pos // 4].mean() if n_pos >= 4 else curve[n_pos // 2]
            peak_idx = int(np.argmax(curve))
            peak_frac = peak_idx / max(n_pos - 1, 1)

            # Persistent: high throughout (min > 0.5)
            if curve.min() > 0.5:
                return "persistent"
            # Early-onset: peaks in first third, decays
            if peak_frac < 0.33 and last_quarter < 0.3:
                return "transient_early"
            # Late-emerging: low early, high late
            if first_quarter < 0.3 and last_quarter > 0.7:
                return "late_emerging"
            # Monotonic growth: each position roughly >= previous
            diffs = np.diff(curve)
            if (diffs >= -0.05).all() and curve[-1] > 0.5:
                return "monotonic_growth"
            # Transient: spikes then drops
            if curve.max() > 0.8 and curve.min() < 0.2 and mid > last_quarter:
                return "transient"
            return "other"

        categories = [_classify_feature(normalized[i]) for i in range(n_alive)]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"  Feature categories: {category_counts}")

        # K-means clustering on normalized retention curves
        from sklearn.cluster import KMeans

        n_clusters_use = min(n_clusters, n_alive)
        if n_clusters_use >= 2:
            kmeans = KMeans(n_clusters=n_clusters_use, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized)
            centroids = kmeans.cluster_centers_.tolist()  # (n_clusters, n_pos)
            cluster_sizes = [int((cluster_labels == c).sum()) for c in range(n_clusters_use)]
        else:
            cluster_labels = np.zeros(n_alive, dtype=int)
            centroids = [normalized.mean(axis=0).tolist()]
            cluster_sizes = [n_alive]

        print(f"  Cluster sizes: {cluster_sizes}")

        # Build per-feature results for top features (by max activation)
        top_k = min(50, n_alive)
        top_alive_order = np.argsort(-max_act_per_feature[alive_mask])[:top_k]
        top_features = []
        for rank_i, ai in enumerate(top_alive_order):
            fi = int(alive_indices[ai])
            top_features.append({
                "feature_idx": fi,
                "rank": rank_i,
                "max_activation": float(max_act_per_feature[fi]),
                "retention_curve": retention[fi].tolist(),
                "freq_curve": freq_curves[fi].tolist(),
                "normalized_curve": normalized[ai].tolist(),
                "category": categories[ai],
                "cluster": int(cluster_labels[ai]),
            })

        entry = {
            "sae_type": sae_type,
            "tag": tag,
            "n_features": nf,
            "n_alive": n_alive,
            "positions": positions,
            "category_counts": category_counts,
            "n_clusters": n_clusters_use,
            "cluster_centroids": centroids,
            "cluster_sizes": cluster_sizes,
            "top_features": top_features,
            # Summary stats
            "mean_onset_position": float(np.mean([
                positions[int(np.argmax(normalized[i] > 0.5))]
                if (normalized[i] > 0.5).any() else positions[-1]
                for i in range(n_alive)
            ])),
            "mean_final_activation": float(alive_retention[:, -1].mean()),
            "mean_peak_position_frac": float(np.mean([
                np.argmax(normalized[i]) / max(n_pos - 1, 1)
                for i in range(n_alive)
            ])),
        }

        all_results[tag] = entry
        print(f"  Mean onset position: {entry['mean_onset_position']:.0f}")
        print(f"  Mean peak position fraction: {entry['mean_peak_position_frac']:.2f}")
        print(f"  Mean final activation: {entry['mean_final_activation']:.4f}")

        del sae, ckpt
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    out_dir = Path(f"{DATA}/temporal_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "layer": layer,
        "head": head,
        "positions": positions,
        "n_samples": n_samples,
        "n_features_target": n_features_target,
        "n_clusters": n_clusters,
        "analysis_time_s": round(elapsed, 1),
        "sae_results": all_results,
    }

    # Cross-SAE comparison: aggregate categories by sae_type
    from collections import defaultdict
    type_summary = defaultdict(lambda: defaultdict(list))
    for tag, entry in all_results.items():
        st = entry["sae_type"]
        type_summary[st]["n_alive"].append(entry["n_alive"])
        for cat, cnt in entry.get("category_counts", {}).items():
            type_summary[st][f"cat_{cat}"].append(cnt)
        type_summary[st]["mean_onset"].append(entry.get("mean_onset_position", 0))
        type_summary[st]["mean_peak_frac"].append(entry.get("mean_peak_position_frac", 0))
        type_summary[st]["mean_final_act"].append(entry.get("mean_final_activation", 0))

    summary_by_type = {}
    for st, vals in type_summary.items():
        summary_by_type[st] = {
            k: float(np.mean(v)) for k, v in vals.items()
        }
    output["summary_by_type"] = summary_by_type

    (out_dir / "temporal_analysis.json").write_text(json.dumps(output, indent=2))
    vol.commit()

    print(f"\n{'='*80}")
    print(f"Temporal Analysis Summary: layer {layer}, head {head}")
    print(f"{'='*80}")
    for st, s in summary_by_type.items():
        print(f"  {st}:")
        print(f"    alive={s.get('n_alive', 0):.0f}  "
              f"onset={s.get('mean_onset', 0):.0f}  "
              f"peak_frac={s.get('mean_peak_frac', 0):.2f}  "
              f"final_act={s.get('mean_final_act', 0):.4f}")
        cats = {k: v for k, v in s.items() if k.startswith("cat_")}
        if cats:
            cat_str = ", ".join(f"{k[4:]}={v:.0f}" for k, v in sorted(cats.items()))
            print(f"    categories: {cat_str}")

    print(f"\nTemporal analysis complete in {elapsed:.1f}s")
    return output


# -- Stage: Feature quality analysis --

@app.function(volumes={DATA: vol}, **GPU_KWARGS)
def analyze_feature_quality(
    layer: int = 9,
    n_features_target: int = 2048,
    k: int = 32,
    head: int = 0,
) -> dict:
    """Per-sample MSE distribution and per-feature specificity for matched SAEs.

    For each checkpoint that matches the requested layer, head, dictionary size,
    and TopK setting:
      1. Load validation split of GDN states
      2. Forward pass through SAE
      3. Compute per-sample MSE distribution (mean, p90, p95, p99, max, CV)
      4. Compute per-feature activation frequency and conditional MSE
      5. Compute Gini coefficient of feature usage
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from sae import build_sae_from_config
    from split_utils import make_train_val_indices
    from feature_quality import analyze_sae, format_summary_table

    vol.reload()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")

    layer_meta_path = states_dir / f"layer_{layer}" / "layer_metadata.json"
    if layer_meta_path.exists():
        n_total = json.loads(layer_meta_path.read_text())["n_samples"]
    else:
        n_total = meta["n_samples"]

    data = np.lib.format.open_memmap(
        str(head_path), mode="r", dtype=np.float16,
        shape=(n_total, key_dim, val_dim),
    )

    _, val_indices = make_train_val_indices(n_total)
    val_data = torch.from_numpy(data[val_indices].astype(np.float32))
    print(f"Validation set: {len(val_indices)} samples from {n_total} total")
    print(f"State shape: ({key_dim}, {val_dim}), d_in={key_dim * val_dim}")

    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoints at {ckpt_root}")

    sae_entries = []
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        try:
            cfg = json.loads(cp.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if cfg.get("layer") != layer:
            continue
        if cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        if cfg.get("k") != k:
            continue
        sae_entries.append((d.name, cfg, bp))

    print(
        f"Found {len(sae_entries)} checkpoints at "
        f"layer={layer}, head={head}, nf={n_features_target}, k={k}"
    )
    if not sae_entries:
        raise FileNotFoundError(
            f"No checkpoints for layer={layer}, head={head}, "
            f"nf={n_features_target}, k={k}"
        )

    t0 = time.time()
    all_results = {}

    for tag, cfg, bp in sae_entries:
        sae_type = cfg["sae_type"]
        print(f"\n{'='*60}")
        print(f"Analyzing: {tag} ({sae_type})")
        print(f"{'='*60}")

        try:
            ckpt = torch.load(bp, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError) as e:
            print(f"  WARN: corrupt checkpoint {bp}: {e}")
            continue

        sae = build_sae_from_config(
            cfg,
            state_dict=ckpt["model_state_dict"],
            default_d_k=key_dim,
            default_d_v=val_dim,
        )
        sae.load_state_dict(ckpt["model_state_dict"])
        sae = sae.cuda().eval()

        is_flat = sae_type == "flat"
        result = analyze_sae(sae, val_data, k=cfg.get("k", k), is_flat=is_flat)
        result["sae_type"] = sae_type
        result["tag"] = tag
        result["val_mse_from_training"] = ckpt.get("val_mse")
        all_results[tag] = result

        mse = result["mse_distribution"]
        freq = result["feature_frequency"]
        print(f"  MSE: mean={mse['mean']:.5f} p95={mse['p95']:.5f} "
              f"p99={mse['p99']:.5f} CV={mse['cv']:.3f}")
        print(f"  Features: {freq['n_alive']}/{result['n_features']} alive "
              f"({freq['alive_pct']:.1f}%), freq={freq.get('mean_freq', 0):.4f}")
        print(f"  Gini={result['gini_coefficient']:.3f}")

        del sae, ckpt
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Aggregate by SAE type (average over seeds)
    by_type: dict[str, list[dict]] = {}
    for tag, r in all_results.items():
        st = r["sae_type"]
        by_type.setdefault(st, []).append(r)

    summary_by_type = {}
    for st, runs in by_type.items():
        n = len(runs)
        summary = {
            "n_seeds": n,
            "mse_mean": float(np.mean([r["mse_distribution"]["mean"] for r in runs])),
            "mse_mean_std": float(np.std([r["mse_distribution"]["mean"] for r in runs])),
            "mse_p95": float(np.mean([r["mse_distribution"]["p95"] for r in runs])),
            "mse_p99": float(np.mean([r["mse_distribution"]["p99"] for r in runs])),
            "mse_cv": float(np.mean([r["mse_distribution"]["cv"] for r in runs])),
            "n_alive": float(np.mean([r["feature_frequency"]["n_alive"] for r in runs])),
            "n_dead": float(np.mean([r["feature_frequency"]["n_dead"] for r in runs])),
            "dead_pct": float(np.mean([r["feature_frequency"]["dead_pct"] for r in runs])),
            "mean_freq": float(np.mean([r["feature_frequency"].get("mean_freq", 0) for r in runs])),
            "gini": float(np.mean([r["gini_coefficient"] for r in runs])),
            "cond_mse_mean": float(np.mean([r["conditional_mse"].get("mean", 0) for r in runs])),
        }
        summary_by_type[st] = summary

    output = {
        "layer": layer,
        "n_features_target": n_features_target,
        "k": k,
        "head": head,
        "n_val_samples": len(val_indices),
        "elapsed_s": elapsed,
        "sae_results": all_results,
        "summary_by_type": summary_by_type,
    }

    out_dir = Path(f"{DATA}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_quality_L{layer}_nf{n_features_target}.json"
    out_path.write_text(json.dumps(output, indent=2))
    vol.commit()

    print(f"\n{'='*80}")
    print(f"Feature Quality Summary: layer {layer}, nf={n_features_target}, k={k}")
    print(f"{'='*80}")
    print(format_summary_table(all_results))

    print(f"\nAggregated by SAE type (mean over seeds):")
    for st, s in sorted(summary_by_type.items()):
        print(f"  {st} (n={s['n_seeds']}):")
        print(f"    MSE: mean={s['mse_mean']:.5f}+/-{s['mse_mean_std']:.5f}  "
              f"p95={s['mse_p95']:.5f}  p99={s['mse_p99']:.5f}  CV={s['mse_cv']:.3f}")
        print(f"    Features: alive={s['n_alive']:.0f}  dead={s['dead_pct']:.1f}%  "
              f"freq={s['mean_freq']:.4f}  gini={s['gini']:.3f}")
        print(f"    Conditional MSE (mean on served inputs): {s['cond_mse_mean']:.5f}")

    print(f"\nSaved to {out_path}")
    print(f"Analysis complete in {elapsed:.1f}s")
    return output


# -- Stage: Feature probing (statistical + vocabulary projection) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def probe_features_modal(
    layer: int = 9,
    head: int = 0,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
) -> dict:
    """Enhanced feature probing: Spearman + nonlinear + contrastive + vocab projection.

    1. Spearman correlations with 55 text properties (expanded from 15)
    2. Random forest nonlinear probing (catches interaction effects)
    3. Contrastive probing (top vs bottom activation groups)
    4. Vocabulary projection through GDN output and unembedding
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from sae import build_sae_from_config
    from probe_features import (
        probe_features, probe_features_enhanced,
        project_features_to_vocab,
        format_vocab_projection, decode_token_ids,
    )

    vol.reload()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    n_total = meta["n_samples"]

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    data = np.lib.format.open_memmap(
        str(head_path), mode="r", dtype=np.float16,
        shape=(n_total, key_dim, val_dim),
    )
    states = torch.from_numpy(data[:n_total].astype(np.float32))
    print(f"Loaded {states.shape[0]} states, shape ({key_dim}, {val_dim})")

    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError("texts.json not found. Run --stage extract-texts first.")
    texts = json.loads(texts_path.read_text())[:n_total]
    # Align counts: use min of states and texts
    n_use = min(states.shape[0], len(texts))
    if n_use < states.shape[0]:
        states = states[:n_use]
        texts = texts[:n_use]
        print(f"Aligned to {n_use} samples (states={states.shape[0]}, texts={len(texts)})")
    else:
        print(f"Loaded {len(texts)} texts")

    # Find the rank-1 SAE checkpoint (prefer bilinear, fall back to rank1)
    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            # Prefer seed 0
            if best_ckpt is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            # Prefer rank1 (simpler, easier to interpret)
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No rank1/bilinear checkpoint for layer={layer}, nf={n_features_target}")

    print(f"\nUsing SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    # ---- Part 1: Enhanced probing (Spearman + RF + Contrastive) ----
    print("\n" + "=" * 60)
    print("Part 1: Enhanced Feature Probing (3 methods)")
    print("=" * 60)
    t0 = time.time()
    enhanced_result = probe_features_enhanced(
        sae, states, texts,
        batch_size=512,
        min_frequency=0.01,
        correlation_threshold=0.15,
        p_threshold=0.01,
        rf_accuracy_threshold=0.60,
        contrastive_effect_size=0.50,
    )
    probe_time = time.time() - t0

    # Extract the 'probe' result that figures/gen_probing_*.py consume.
    probe_result = enhanced_result["probe"]
    probe_result["elapsed_s"] = probe_time
    probe_result["sae_tag"] = best_tag

    # ---- Part 2: Vocabulary projection ----
    print("\n" + "=" * 60)
    print("Part 2: Vocabulary Projection")
    print("=" * 60)

    # Free SAE from GPU before loading model
    sae_cpu = sae.cpu()
    del sae
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=f"{MODELS}", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=f"{MODELS}",
        torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    t0 = time.time()
    vocab_result = project_features_to_vocab(
        sae_cpu, model, layer_idx=layer, head_idx=head, top_k=20,
    )
    vocab_time = time.time() - t0
    vocab_result["elapsed_s"] = vocab_time
    vocab_result["sae_tag"] = best_tag

    # Decode token IDs to strings for readability
    for feat in vocab_result["features"]:
        feat["w_top_strings"] = decode_token_ids(tokenizer, feat["w_top_tokens"])
        feat["w_bottom_strings"] = decode_token_ids(tokenizer, feat["w_bottom_tokens"])
        feat["v_top_strings"] = decode_token_ids(tokenizer, feat["v_top_tokens"])
        feat["v_bottom_strings"] = decode_token_ids(tokenizer, feat["v_bottom_tokens"])

    print("\nVocabulary Projection Report:")
    print("=" * 60)
    print(format_vocab_projection(vocab_result, tokenizer, n_features=30))

    # Combine results
    output = {
        "layer": layer,
        "head": head,
        "n_features_target": n_features_target,
        "model_name": model_name,
        "sae_tag": best_tag,
        "sae_type": best_cfg["sae_type"],
        "probe": probe_result,
        "nonlinear": enhanced_result["nonlinear"],
        "contrastive": enhanced_result["contrastive"],
        "union": enhanced_result["union"],
        "vocab_projection": vocab_result,
    }

    out_dir = _analysis_dir(corpus_source)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_features_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(output, indent=2))
    vol.commit()

    print(f"\nProbe: {probe_time:.1f}s, Vocab: {vocab_time:.1f}s")
    print(f"Saved to {out_path}")

    return output


# -- Stage: Held-out feature probing (validation split only) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def probe_features_heldout_modal(
    layer: int = 9,
    head: int = 0,
    n_features_target: int = 2048,
    split: str = "val",
    sae_tag: str = "",
    corpus_source: str = "openwebtext",
) -> dict:
    """Feature probing on held-out validation data only.

    Uses the same 80/20 split (seed=42) as SAE training.
    split="val" probes on the 20% not seen during training.
    split="train" probes on the 80% for comparison.
    Skips vocabulary projection (split-independent).
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from sae import build_sae_from_config
    from probe_features import probe_features_enhanced
    from split_utils import make_train_val_indices

    vol.reload()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    n_total = meta["n_samples"]

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    data = np.lib.format.open_memmap(
        str(head_path), mode="r", dtype=np.float16,
        shape=(n_total, key_dim, val_dim),
    )
    all_states = torch.from_numpy(data[:n_total].astype(np.float32))

    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError("texts.json not found. Run --stage extract-texts first.")
    all_texts = json.loads(texts_path.read_text())[:n_total]

    # Align
    n_use = min(all_states.shape[0], len(all_texts))
    all_states = all_states[:n_use]
    all_texts = all_texts[:n_use]

    # Get train/val split indices (same seed=42 as training)
    train_indices, val_indices = make_train_val_indices(n_use, train_fraction=0.8, seed=42)
    if split == "val":
        indices = val_indices
    elif split == "train":
        indices = train_indices
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # Subset states and texts
    states = all_states[indices]
    texts = [all_texts[i] for i in indices]
    print(f"Split={split}: {len(indices)} samples from {n_use} total")
    print(f"State shape: {states.shape}")

    # Find SAE checkpoint: search all experiment tags (ckpt dirs may use different n_samples)
    ckpt_parent = ckpt_root.parent
    search_roots = []
    if ckpt_parent.exists():
        for et_dir in sorted(ckpt_parent.iterdir()):
            if et_dir.is_dir():
                search_roots.append(et_dir)
    if not search_roots and ckpt_root.exists():
        search_roots = [ckpt_root]

    best_ckpt = None
    best_cfg = None
    best_tag = None

    if sae_tag:
        # Direct lookup by tag name
        print(f"Looking for exact sae_tag={sae_tag}")
        for root in search_roots:
            d = root / sae_tag
            cp, bp = d / "config.json", d / "best.pt"
            if cp.exists() and bp.exists():
                best_ckpt = bp
                best_cfg = json.loads(cp.read_text())
                best_tag = sae_tag
                break
    else:
        # Search heuristically (prefer non-btk rank1)
        print(f"Searching {len(search_roots)} experiment tag(s) for L{layer} H{head} nf={n_features_target}")
        for root in search_roots:
            for d in sorted(root.iterdir()) if root.exists() else []:
                cp, bp = d / "config.json", d / "best.pt"
                if not cp.exists() or not bp.exists():
                    continue
                cfg = json.loads(cp.read_text())
                if cfg.get("layer") != layer or cfg.get("head", 0) != head:
                    continue
                if cfg.get("n_features") != n_features_target:
                    continue
                st = cfg.get("sae_type", "")
                if st in ("rank1", "bilinear"):
                    # Prefer non-btk (plain rank1) over btk variants
                    is_btk = d.name.startswith("btk_")
                    if best_ckpt is None:
                        best_ckpt, best_cfg, best_tag = bp, cfg, d.name
                    elif is_btk and not best_tag.startswith("btk_"):
                        pass  # keep the non-btk one
                    elif not is_btk and best_tag.startswith("btk_"):
                        best_ckpt, best_cfg, best_tag = bp, cfg, d.name
                    elif cfg.get("seed", 99) < best_cfg.get("seed", 99):
                        best_ckpt, best_cfg, best_tag = bp, cfg, d.name

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No rank1/bilinear checkpoint for layer={layer}, nf={n_features_target} "
            f"(sae_tag={sae_tag!r}) in {[r.name for r in search_roots]}")

    print(f"\nUsing SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    print("\n" + "=" * 60)
    print(f"Held-out probing (split={split}, n={len(texts)})")
    print("=" * 60)
    t0 = time.time()
    enhanced_result = probe_features_enhanced(
        sae, states, texts,
        batch_size=512,
        min_frequency=0.01,
        correlation_threshold=0.15,
        p_threshold=0.01,
        rf_accuracy_threshold=0.60,
        contrastive_effect_size=0.50,
    )
    probe_time = time.time() - t0

    probe_result = enhanced_result["probe"]
    probe_result["elapsed_s"] = probe_time
    probe_result["sae_tag"] = best_tag

    output = {
        "layer": layer,
        "head": head,
        "n_features_target": n_features_target,
        "split": split,
        "n_split_samples": len(texts),
        "n_total_samples": n_use,
        "sae_tag": best_tag,
        "sae_type": best_cfg["sae_type"],
        "probe": probe_result,
        "nonlinear": enhanced_result["nonlinear"],
        "contrastive": enhanced_result["contrastive"],
        "union": enhanced_result["union"],
    }

    out_dir = _analysis_dir(corpus_source)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_features_L{layer}_H{head}_{split}.json"
    out_path.write_text(json.dumps(output, indent=2))
    vol.commit()

    print(f"\nProbe time: {probe_time:.1f}s")
    print(f"Saved to {out_path}")

    return output


# -- Stage: Circuit-level ablation (causal feature analysis) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def circuit_ablation_modal(
    layer: int = 9,
    head: int = 0,
    n_sequences: int = 200,
    n_per_property: int = 2,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    seq_len: int = 1024,
    quartile_size: float = 0.25,
    corpus_source: str = "openwebtext",
) -> dict:
    """Circuit-level ablation: zero individual SAE features, measure property-selective PPL damage.

    Tests whether features CAUSE property-specific behavior (not just correlate).
    For each feature correlated with property P, zeroing it should hurt PPL more
    on HIGH-P texts than LOW-P texts.

    Runs three analyses:
      1. Targeted ablation: per-feature PPL on matched HIGH/LOW groups
      2. Cross-property matrix: each feature's ablation effect on ALL property groups
      3. Diagonal dominance: do features selectively affect their correlated property?

    Time budget: ~20 min on A10G with 200 sequences and 2 features per property.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from circuit_ablation import (
        run_circuit_ablation,
        select_target_features,
        format_circuit_ablation_report,
    )

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No probe results at {probe_path}. Run --stage probe-features first.")
    probe_results = json.loads(probe_path.read_text())

    target_features = select_target_features(
        probe_results, n_per_property=n_per_property, min_rho=0.15)
    if not target_features:
        raise ValueError("No interpretable features found in probe results.")
    print(f"Selected {len(target_features)} target features from probe results")
    for f in target_features:
        print(f"  Feature {f['feature_idx']}: {f['property']} (rho={f['rho']:.3f})")

    # Load SAE checkpoint (same logic as probe_features_modal)
    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            if best_ckpt is None or best_cfg is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name

    if best_ckpt is None or best_cfg is None:
        raise FileNotFoundError(
            f"No rank1/bilinear checkpoint for layer={layer}, nf={n_features_target}")
    print(f"Using SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Validate layer is GDN
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    # Load corpus (tokenized sequences)
    corpus_path = states_dir / "corpus.npy"
    batch_size = 1  # process one seq at a time for per-seq PPL
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        batches = load_corpus_tokens(tokenizer, None, seq_len, n_sequences, batch_size)
        actual = sum(b.shape[0] for b in batches)
        print(f"Streamed {actual} sequences from OpenWebText")

    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError("texts.json not found. Run --stage extract-texts first.")
    texts = json.loads(texts_path.read_text())[:actual]
    print(f"Loaded {len(texts)} texts for property computation")

    results = run_circuit_ablation(
        model=model,
        corpus_batches=batches,
        texts=texts,
        layer_idx=layer,
        sae=sae,
        sae_type=best_cfg["sae_type"],
        target_features=target_features,
        head_idx=head,
        split_fraction=0.5,
        quartile_size=quartile_size,
        device="cuda",
    )

    results["model_name"] = model_name
    results["sae_tag"] = best_tag
    results["total_time_s"] = round(time.time() - t0, 1)

    print("\n" + "=" * 70)
    print(format_circuit_ablation_report(results))

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"circuit_ablation_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {results['total_time_s']:.0f}s")

    return results


# -- Stage: Group ablation v2 (dose-response curves) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def circuit_ablation_v2_modal(
    layer: int = 9,
    head: int = 0,
    n_sequences: int = 200,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    seq_len: int = 1024,
    quartile_size: float = 0.25,
    min_rho: float = 0.10,
    max_properties: int = 8,
    corpus_source: str = "openwebtext",
) -> dict:
    """Group ablation with dose-response curves.

    Ablates GROUPS of property-correlated features simultaneously (1, 2, 4, 8, 16, all)
    and measures PPL on high-property vs low-property text at each dose.

    Produces the dose-response figure data: X=features ablated, Y=PPL change,
    two lines (HIGH/LOW) per property. The widening gap proves causal encoding.

    Time budget: ~15-20 min on A10G with 200 sequences and 8 properties.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from circuit_ablation_v2 import (
        build_property_feature_groups,
        run_group_ablation,
        format_group_ablation_report,
    )

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No probe results at {probe_path}. Run --stage probe-features first.")
    probe_results = json.loads(probe_path.read_text())

    property_groups = build_property_feature_groups(
        probe_results, min_rho=min_rho, min_features=4)
    if not property_groups:
        raise ValueError(f"No properties with >= 4 correlated features at |rho| >= {min_rho}")
    print(f"Found {len(property_groups)} properties with correlated feature groups:")
    for prop, feats in sorted(property_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {prop}: {len(feats)} features (top |rho|={feats[0]['abs_rho']:.3f})")

    # Load SAE checkpoint (same logic as circuit_ablation_modal)
    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            if best_ckpt is None or best_cfg is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No rank1/bilinear checkpoint for layer={layer}, nf={n_features_target}")
    print(f"Using SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Validate layer is GDN
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    corpus_path = states_dir / "corpus.npy"
    batch_size = 1
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        batches = load_corpus_tokens(tokenizer, None, seq_len, n_sequences, batch_size)
        actual = sum(b.shape[0] for b in batches)
        print(f"Streamed {actual} sequences from OpenWebText")

    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError("texts.json not found. Run --stage extract-texts first.")
    texts = json.loads(texts_path.read_text())[:actual]
    print(f"Loaded {len(texts)} texts for property computation")

    results = run_group_ablation(
        model=model,
        corpus_batches=batches,
        texts=texts,
        layer_idx=layer,
        sae=sae,
        sae_type=best_cfg["sae_type"],
        property_groups=property_groups,
        head_idx=head,
        split_fraction=0.5,
        quartile_size=quartile_size,
        dose_levels=[1, 2, 4, 8, 16],
        max_properties=max_properties,
        device="cuda",
    )

    results["model_name"] = model_name
    results["sae_tag"] = best_tag
    results["min_rho"] = min_rho
    results["total_time_s"] = round(time.time() - t0, 1)

    print("\n" + "=" * 70)
    print(format_group_ablation_report(results))

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"circuit_ablation_v2_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {results['total_time_s']:.0f}s")

    return results


# -- Stage: Feature ablation vs random directions --

@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    gpu="A10G",
    image=image,
    timeout=14400,
    memory=32768,
)
def feature_vs_random_ablation_modal(
    layer: int = 9,
    head: int = 0,
    n_sequences: int = 200,
    n_random: int = 50,
    max_alive_features: int = 0,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    seq_len: int = 1024,
) -> dict:
    """Compare PPL damage from ablating alive SAE features vs random directions.

    For each alive feature: zero it in SAE reconstruction, measure mean PPL delta.
    For N random unit vectors: project out from raw state, measure mean PPL delta.
    Compare |delta_loss| distributions with Mann-Whitney U test and Cohen's d.

    Time budget: ~30-60 min on A10G with 200 sequences, ~150 alive features, 50 random.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from circuit_ablation_random import (
        find_alive_feature_indices,
        run_feature_vs_random_ablation,
    )

    n_random = int(os.environ.get("MATRIX_SAE_RANDOM_DIRECTIONS", str(n_random)))
    if max_alive_features <= 0:
        max_alive_features = int(os.environ.get("MATRIX_SAE_MAX_ALIVE_FEATURES", "0"))

    vol.reload()
    t0 = time.time()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    # Load SAE checkpoint from any checkpoint root on the volume.
    best_ckpt = None
    best_cfg = None
    best_tag = None
    ckpt_parent = Path(f"{DATA}/checkpoints")
    ckpt_roots = []
    if ckpt_parent.exists():
        for exp_dir in ckpt_parent.iterdir():
            if exp_dir.is_dir():
                ckpt_roots.append(exp_dir)
    print(f"Scanning {len(ckpt_roots)} checkpoint roots: {[r.name for r in ckpt_roots]}")

    exact_tag = f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    for ckpt_root in ckpt_roots:
        exact_dir = ckpt_root / exact_tag
        cp, bp = exact_dir / "config.json", exact_dir / "best.pt"
        if cp.exists() and bp.exists():
            best_ckpt = bp
            best_cfg = json.loads(cp.read_text())
            best_tag = exact_tag
            print(f"Found exact checkpoint tag: {exact_tag}")
            break

    # Fallback: search all roots for rank1 match
    if best_ckpt is None:
        for ckpt_root in ckpt_roots:
            for d in sorted(ckpt_root.iterdir()):
                if not d.is_dir():
                    continue
                cp, bp = d / "config.json", d / "best.pt"
                if not cp.exists() or not bp.exists():
                    continue
                cfg = json.loads(cp.read_text())
                if cfg.get("layer") != layer or cfg.get("head", 0) != head:
                    continue
                if cfg.get("n_features") != n_features_target:
                    continue
                sae_type = cfg.get("sae_type", "")
                if sae_type == "rank1" and not cfg.get("use_batchtopk", False):
                    if best_ckpt is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                        best_ckpt = bp
                        best_cfg = cfg
                        best_tag = d.name

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No rank1 checkpoint for layer={layer}, nf={n_features_target}. "
            f"Searched roots: {[r.name for r in ckpt_roots]}")
    print(f"Using SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    # Load clean states for alive-feature discovery.
    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    n_total = int(meta["n_samples"])
    data = np.lib.format.open_memmap(
        str(head_path), mode="r", dtype=np.float16,
        shape=(n_total, key_dim, val_dim),
    )
    states = torch.from_numpy(data[:n_total].astype(np.float32))
    alive_indices = find_alive_feature_indices(
        sae=sae,
        states=states,
        sae_type=best_cfg["sae_type"],
        batch_size=128,
        device="cuda",
    )
    if not alive_indices:
        raise RuntimeError("Alive-feature scan found zero active SAE features.")
    if max_alive_features > 0:
        alive_indices = alive_indices[:max_alive_features]
    print(f"Found {len(alive_indices)} alive features from saved states")

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Load the exact evaluation corpus from volume assets, without falling back to streaming.
    corpus_path = states_dir / "corpus.npy"
    batch_size = 1
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_sequences)
    else:
        texts_path = states_dir / "texts.json"
        if not texts_path.exists():
            raise FileNotFoundError(
                f"No corpus.npy or texts.json under {states_dir}; cannot reconstruct the clean eval corpus."
            )
        texts = json.loads(texts_path.read_text())[:n_sequences]
        batches = []
        eos_id = tokenizer.eos_token_id
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < seq_len:
                token_ids = token_ids + [eos_id] * (seq_len - len(token_ids))
            else:
                token_ids = token_ids[:seq_len]
            batches.append(torch.tensor([token_ids], dtype=torch.long))

    # Flatten to individual sequences
    corpus_seqs = []
    for batch in batches:
        for i in range(batch.shape[0]):
            corpus_seqs.append(batch[i:i + 1])
    actual = len(corpus_seqs)
    print(f"Loaded {actual} sequences")

    results = run_feature_vs_random_ablation(
        model=model,
        corpus_seqs=corpus_seqs,
        layer_idx=layer,
        sae=sae,
        sae_type=best_cfg["sae_type"],
        alive_feature_indices=alive_indices,
        head_idx=head,
        split_fraction=0.5,
        n_random=n_random,
        d_k=key_dim,
        d_v=val_dim,
        device="cuda",
        seed=42,
    )

    results["model_name"] = model_name
    results["sae_tag"] = best_tag
    results["sae_type"] = best_cfg["sae_type"]
    results["layer"] = layer
    results["head"] = head
    results["total_time_s"] = round(time.time() - t0, 1)

    out_dir = Path(f"{DATA}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_vs_random_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {results['total_time_s']:.0f}s")

    return results


# -- Stage: Causal feature clamping (Q3) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def causal_clamp_modal(
    layer: int = 9,
    head: int = 0,
    n_prompts: int = 50,
    n_features_target: int = 2048,
    n_top_features: int = 5,
    skip_features: int = 0,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    prompt_len: int = 512,
    gen_len: int = 256,
    corpus_source: str = "openwebtext",
) -> dict:
    """Causal feature clamping: clamp feature HIGH, generate, measure property shift.

    For each of the top N features by |rho|:
      1. Compute 95th percentile activation from training states
      2. For each prompt, generate with/without clamping
      3. Measure text property shift, Cohen's d, paired t-test

    If clamping feature F shifts generated text toward property P,
    that is causal evidence that F encodes P in the recurrent state.

    Time budget: ~30 min on A10G with 50 prompts and 5 features.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from causal_clamp import (
        run_causal_clamp,
        select_top_features_by_rho,
        format_causal_clamp_report,
    )

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    n_total = meta["n_samples"]

    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No probe results at {probe_path}. Run --stage probe-features first.")
    probe_results = json.loads(probe_path.read_text())

    target_features = select_top_features_by_rho(probe_results, n_features=n_top_features)
    if not target_features:
        raise ValueError("No interpretable features found in probe results.")
    if skip_features > 0:
        print(f"Skipping first {skip_features} features (already completed)")
        target_features = target_features[skip_features:]
    print(f"Selected {len(target_features)} target features by |rho|:")
    for f in target_features:
        print(f"  Feature {f['feature_idx']}: {f['property']} (rho={f['rho']:.3f})")

    # Load SAE checkpoint (same logic as circuit_ablation_modal)
    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            if best_ckpt is None or best_cfg is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No rank1/bilinear checkpoint for layer={layer}, nf={n_features_target}")
    print(f"Using SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    # Load states for percentile computation.
    # Use original 5K training data (where features are alive), not 50K pool.
    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    all_states = np.load(str(head_path), mmap_mode="r")
    n_for_percentile = min(5000, all_states.shape[0])
    states = torch.from_numpy(np.array(all_states[:n_for_percentile], dtype=np.float32))
    print(f"Loaded {states.shape[0]} states for percentile computation (first {n_for_percentile})")

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # Validate layer is GDN
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    # Load corpus (tokenized sequences)
    corpus_path = states_dir / "corpus.npy"
    batch_size = 1
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_prompts)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        batches = load_corpus_tokens(tokenizer, None, 1024, n_prompts, batch_size)
        actual = sum(b.shape[0] for b in batches)
        print(f"Streamed {actual} sequences from OpenWebText")

    texts_path = states_dir / "texts.json"
    texts = []
    if texts_path.exists():
        texts = json.loads(texts_path.read_text())[:actual]
    print(f"Loaded {len(texts)} texts")

    results = run_causal_clamp(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        texts=texts,
        states=states,
        layer_idx=layer,
        sae=sae,
        sae_type=best_cfg["sae_type"],
        target_features=target_features,
        head_idx=head,
        prompt_len=prompt_len,
        gen_len=gen_len,
        n_prompts=n_prompts,
        device="cuda",
    )

    results["model_name"] = model_name
    results["sae_tag"] = best_tag
    results["total_time_s"] = round(time.time() - t0, 1)

    print("\n" + "=" * 70)
    print(format_causal_clamp_report(results))

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"causal_clamp_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {results['total_time_s']:.0f}s")

    return results


# -- Stage: Logit-based causal intervention --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def causal_logit_modal(
    layer: int = 9,
    head: int = 0,
    n_prompts: int = 50,
    n_features_target: int = 2048,
    n_top_features: int = 5,
    skip_features: int = 0,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    prompt_len: int = 512,
    positions_per_prompt: int = 1,
    corpus_source: str = "openwebtext",
) -> dict:
    """Direct next-token logit intervention for top interpretable features."""
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_model_and_tokenizer, load_corpus_from_file
    from causal_clamp import (
        LOGIT_TEST_PROPERTIES,
        format_logit_causal_report,
        run_logit_causal_intervention,
        select_top_features_by_rho,
    )

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No probe results at {probe_path}. Run --stage probe-features first.")
    probe_results = json.loads(probe_path.read_text())

    target_features = select_top_features_by_rho(
        probe_results,
        n_features=n_top_features,
        allowed_properties=LOGIT_TEST_PROPERTIES,
    )
    if skip_features > 0:
        print(f"Skipping first {skip_features} features (already completed)")
        target_features = target_features[skip_features:]
    if not target_features:
        raise ValueError(
            f"No supported interpretable features found. Allowed properties: {sorted(LOGIT_TEST_PROPERTIES)}"
        )
    print(f"Selected {len(target_features)} features for direct logit intervention:")
    for feat in target_features:
        print(
            f"  Feature {feat['feature_idx']}: {feat['property']} "
            f"(rho={feat['rho']:+.3f})"
        )

    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            if best_ckpt is None or best_cfg is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name

    if best_ckpt is None or best_cfg is None:
        raise FileNotFoundError(
            f"No rank1/bilinear checkpoint for layer={layer}, nf={n_features_target}"
        )
    print(f"Using SAE: {best_tag} (type={best_cfg['sae_type']})")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    all_states = np.load(str(head_path), mmap_mode="r")
    n_for_percentile = min(5000, all_states.shape[0])
    states = torch.from_numpy(np.array(all_states[:n_for_percentile], dtype=np.float32))
    print(f"Loaded {states.shape[0]} states for percentile computation")

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    corpus_path = states_dir / "corpus.npy"
    batch_size = 1
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_prompts)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        batches = load_corpus_tokens(tokenizer, None, 1024, n_prompts, batch_size)
        actual = sum(b.shape[0] for b in batches)
        print(f"Streamed {actual} sequences from OpenWebText")

    results = run_logit_causal_intervention(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        states=states,
        layer_idx=layer,
        sae=sae,
        sae_type=best_cfg["sae_type"],
        target_features=target_features,
        head_idx=head,
        prompt_len=prompt_len,
        positions_per_prompt=positions_per_prompt,
        n_prompts=n_prompts,
        device="cuda",
    )
    results["model_name"] = model_name
    results["sae_tag"] = best_tag
    results["total_time_s"] = round(time.time() - t0, 1)

    print("\n" + "=" * 70)
    print(format_logit_causal_report(results))

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"causal_logit_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {results['total_time_s']:.0f}s")

    return results


# -- Stage: Mechanistic profile and grouped format clamp --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def mechanistic_profile_modal(
    layer: int = 9,
    head: int = 12,
    n_sequences: int = 2000,
    n_features_target: int = 2048,
    family: str = "document_format",
    n_top_features: int = 16,
    n_quantiles: int = 5,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
) -> dict:
    """Compute one-head activation profiles for a chosen property family."""
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from causal_clamp import (
        PROPERTY_FAMILIES,
        compute_family_activation_profile,
        select_primary_family_property,
        select_top_family_features,
    )

    vol.reload()
    t0 = time.time()

    if family not in PROPERTY_FAMILIES:
        raise ValueError(f"Unknown family={family!r}. Use one of {sorted(PROPERTY_FAMILIES)}")
    family_properties = list(PROPERTY_FAMILIES[family])
    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)

    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(f"No probe results at {probe_path}")
    probe_results = json.loads(probe_path.read_text())

    feature_group = select_top_family_features(
        probe_results,
        family_properties=family_properties,
        n_features=n_top_features,
    )
    if not feature_group:
        raise ValueError(f"No family-aligned features found for {family_properties}")
    main_property = select_primary_family_property(probe_results, family_properties)

    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            if best_ckpt is None or best_cfg is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name
    if best_ckpt is None or best_cfg is None:
        raise FileNotFoundError(f"No rank1/bilinear checkpoint for layer={layer}, head={head}, nf={n_features_target}")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim,
        default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    texts_path = states_dir / "texts.json"
    if not head_path.exists() or not texts_path.exists():
        raise FileNotFoundError(f"Need both {head_path} and {texts_path}")

    all_states = np.load(str(head_path), mmap_mode="r")
    n_use = min(n_sequences, all_states.shape[0])
    states = torch.from_numpy(np.array(all_states[:n_use], dtype=np.float32))
    texts = json.loads(texts_path.read_text())[:n_use]

    result = compute_family_activation_profile(
        texts=texts,
        states=states,
        sae=sae,
        feature_group=feature_group,
        family_properties=family_properties,
        n_quantiles=n_quantiles,
    )
    result.update({
        "model_name": model_name,
        "corpus_source": corpus_source,
        "family": family,
        "main_property": main_property,
        "sae_tag": best_tag,
        "layer": layer,
        "head": head,
        "sae_type": best_cfg["sae_type"],
        "total_time_s": round(time.time() - t0, 1),
    })

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mechanistic_profile_{family}_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Main property: {main_property}")
    print(f"Top features: {[f['feature_idx'] for f in feature_group[:8]]}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def mechanistic_clamp_modal(
    layer: int = 9,
    head: int = 12,
    n_prompts: int = 24,
    n_features_target: int = 2048,
    family: str = "document_format",
    n_top_features: int = 16,
    n_random_groups: int = 4,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    prompt_len: int = 512,
    gen_len: int = 128,
    corpus_source: str = "openwebtext",
) -> dict:
    """Clamp a family-aligned feature group and measure family-specific output shifts."""
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from causal_clamp import (
        PROPERTY_FAMILIES,
        format_group_causal_report,
        run_group_causal_clamp,
        select_primary_family_property,
        select_top_family_features,
    )

    vol.reload()
    t0 = time.time()

    if family not in PROPERTY_FAMILIES:
        raise ValueError(f"Unknown family={family!r}. Use one of {sorted(PROPERTY_FAMILIES)}")
    family_properties = list(PROPERTY_FAMILIES[family])
    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)

    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"], corpus_source)
    ckpt_root = Path(f"{DATA}/checkpoints") / exp_tag
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(f"No probe results at {probe_path}")
    probe_results = json.loads(probe_path.read_text())

    feature_group = select_top_family_features(
        probe_results,
        family_properties=family_properties,
        n_features=n_top_features,
    )
    if not feature_group:
        raise ValueError(f"No family-aligned features found for {family_properties}")
    main_property = select_primary_family_property(probe_results, family_properties)

    best_ckpt = None
    best_cfg = None
    best_tag = None
    for d in sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []:
        cp, bp = d / "config.json", d / "best.pt"
        if not cp.exists() or not bp.exists():
            continue
        cfg = json.loads(cp.read_text())
        if cfg.get("layer") != layer or cfg.get("head", 0) != head:
            continue
        if cfg.get("n_features") != n_features_target:
            continue
        sae_type = cfg.get("sae_type", "")
        if sae_type in ("rank1", "bilinear"):
            if best_ckpt is None or best_cfg is None or cfg.get("seed", 99) < best_cfg.get("seed", 99):
                if best_cfg is None or sae_type == best_cfg.get("sae_type", ""):
                    best_ckpt = bp
                    best_cfg = cfg
                    best_tag = d.name
            if sae_type == "rank1" and (best_cfg is None or best_cfg.get("sae_type") != "rank1"):
                best_ckpt = bp
                best_cfg = cfg
                best_tag = d.name
    if best_ckpt is None or best_cfg is None:
        raise FileNotFoundError(f"No rank1/bilinear checkpoint for layer={layer}, head={head}, nf={n_features_target}")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        best_cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim,
        default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    all_states = np.load(str(head_path), mmap_mode="r")
    n_for_percentile = min(5000, all_states.shape[0])
    states = torch.from_numpy(np.array(all_states[:n_for_percentile], dtype=np.float32))

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    corpus_path = states_dir / "corpus.npy"
    batch_size = 1
    if corpus_path.exists():
        batches = load_corpus_from_file(str(corpus_path), batch_size, n_samples=n_prompts)
        actual = sum(b.shape[0] for b in batches)
        print(f"Loaded {actual} sequences from {corpus_path}")
    else:
        from extract_states import load_corpus_tokens
        batches = load_corpus_tokens(tokenizer, None, 1024, n_prompts, batch_size)
        actual = sum(b.shape[0] for b in batches)
        print(f"Streamed {actual} sequences from {corpus_source}")

    results = run_group_causal_clamp(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        states=states,
        layer_idx=layer,
        sae=sae,
        sae_type=best_cfg["sae_type"],
        feature_group=feature_group,
        family_properties=family_properties,
        main_property=main_property,
        head_idx=head,
        prompt_len=prompt_len,
        gen_len=gen_len,
        n_prompts=n_prompts,
        n_random_groups=n_random_groups,
        device="cuda",
    )
    results.update({
        "model_name": model_name,
        "corpus_source": corpus_source,
        "family": family,
        "sae_tag": best_tag,
        "total_time_s": round(time.time() - t0, 1),
    })

    print("\n" + "=" * 70)
    print(format_group_causal_report(results))

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mechanistic_clamp_{family}_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {results['total_time_s']:.0f}s")
    return results


# -- Stage: Write-to-use factor tracing --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_modal(
    layer: int = 9,
    head: int = 4,
    n_prompts: int = 32,
    n_features_target: int = 2048,
    feature_indices: list[int] | None = None,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    prompt_len: int = 512,
    max_offset: int = 64,
    corpus_source: str = "openwebtext",
) -> dict:
    """Trace selected SAE feature coefficients token-by-token on one corpus."""
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import (
        select_cross_corpus_trace_features,
        trace_feature_trajectories,
    )

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )

    openwebtext_root, openwebtext_meta = _checkpoint_root_for_corpus("openwebtext")
    key_dim = openwebtext_meta["key_head_dim"]
    val_dim = openwebtext_meta["value_head_dim"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim,
        default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    target_states_dir = _states_dir(corpus_source)
    target_analysis_dir = _analysis_dir(corpus_source)
    texts_path = target_states_dir / "texts.json"
    corpus_path = target_states_dir / "corpus.npy"
    if not texts_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"Need both {texts_path} and {corpus_path}. Run extraction and extract-texts first."
        )

    selection_features: list[dict]
    if feature_indices:
        selection_features = [{"feature_idx": int(idx)} for idx in feature_indices]
    else:
        owt_states_dir = _states_dir("openwebtext")
        uc_states_dir = _states_dir("ultrachat_200k")
        owt_texts_path = owt_states_dir / "texts.json"
        uc_texts_path = uc_states_dir / "texts.json"
        owt_head_path = owt_states_dir / f"layer_{layer}" / f"head_{head}.npy"
        uc_head_path = uc_states_dir / f"layer_{layer}" / f"head_{head}.npy"
        for needed in [owt_texts_path, uc_texts_path, owt_head_path, uc_head_path]:
            if not needed.exists():
                raise FileNotFoundError(
                    f"Missing cross-corpus selection input at {needed}. "
                    "Extract UltraChat states before running factor-trace."
                )
        n_select = 2000
        owt_states = torch.from_numpy(np.array(np.load(str(owt_head_path), mmap_mode="r")[:n_select], dtype=np.float32))
        uc_states = torch.from_numpy(np.array(np.load(str(uc_head_path), mmap_mode="r")[:n_select], dtype=np.float32))
        owt_texts = json.loads(owt_texts_path.read_text())[:n_select]
        uc_texts = json.loads(uc_texts_path.read_text())[:n_select]
        selection_features = select_cross_corpus_trace_features(
            sae=sae,
            sae_type=cfg["sae_type"],
            openwebtext_states=owt_states,
            openwebtext_texts=owt_texts,
            ultrachat_states=uc_states,
            ultrachat_texts=uc_texts,
            min_rho=0.20,
            max_features=3,
        )

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    batches = load_corpus_from_file(str(corpus_path), batch_size=4, n_samples=n_prompts)
    actual = sum(batch.shape[0] for batch in batches)
    result = trace_feature_trajectories(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=batches,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        feature_specs=selection_features,
        prompt_len=prompt_len,
        max_offset=max_offset,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "n_prompts": actual,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "prompt_len": prompt_len,
        "max_offset": max_offset,
        "total_time_s": round(time.time() - t0, 1),
    })

    out_dir = target_analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"factor_trace_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Selected features: {[item['feature_idx'] for item in result['selected_features']]}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_intervention_modal(
    layer: int = 9,
    head: int = 4,
    n_prompts: int = 16,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    prompt_len: int = 512,
    n_random_controls: int = 4,
    corpus_source: str = "openwebtext",
) -> dict:
    """Write-time feature zeroing with use-time logit readout."""
    import json, os, sys, time
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import run_factor_trace_intervention

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )

    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    trace_path = analysis_dir / f"factor_trace_L{layer}_H{head}.json"
    corpus_path = states_dir / "corpus.npy"
    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not trace_path.exists():
        raise FileNotFoundError(f"No factor trace at {trace_path}. Run factor-trace first.")
    if not corpus_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Need both {corpus_path} and {head_path}")

    trace_result = json.loads(trace_path.read_text())
    if not trace_result.get("selected_features"):
        raise ValueError("Factor trace has no selected features")
    scored_features = []
    for item in trace_result["selected_features"]:
        feature_idx = int(item["feature_idx"])
        summary = trace_result.get("event_aligned", {}).get(str(feature_idx), {})
        sent_score = float(summary.get("sentence_boundary", {}).get("jump_minus_control", 0.0))
        para_score = float(summary.get("paragraph_boundary", {}).get("jump_minus_control", 0.0))
        scored_features.append((max(sent_score, para_score), feature_idx))
    scored_features.sort(reverse=True)
    target_feature_idx = int(scored_features[0][1])

    batches = load_corpus_from_file(str(corpus_path), batch_size=1, n_samples=max(n_prompts, trace_result.get("n_prompts", n_prompts)))
    corpus_sequences = []
    for batch in batches:
        for row in range(batch.shape[0]):
            corpus_sequences.append(batch[row : row + 1])
    corpus_sequences = corpus_sequences[: trace_result.get("n_prompts", n_prompts)]

    states = torch.from_numpy(np.array(np.load(str(head_path), mmap_mode="r")[: min(2000, trace_result.get("n_prompts", n_prompts) * 8)], dtype=np.float32))
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = run_factor_trace_intervention(
        model=model,
        tokenizer=tokenizer,
        corpus_sequences=corpus_sequences,
        trace_result=trace_result,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        states=states,
        target_feature_idx=target_feature_idx,
        n_prompts=n_prompts,
        n_random=n_random_controls,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "prompt_len": prompt_len,
        "total_time_s": round(time.time() - t0, 1),
    })

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"factor_trace_intervention_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Target feature: {target_feature_idx}")
    print(
        "KL ratio vs controls: "
        f"{result.get('summary', {}).get('target_vs_random_kl_ratio', 0):.2f}x"
    )
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_transplant_modal(
    layer: int = 9,
    head: int = 4,
    n_pairs: int = 16,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "openwebtext",
) -> dict:
    """Transplant a traced write value into matched recipient prompts and score local readouts."""
    import json, os, sys, time
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import run_factor_trace_transplant

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )

    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    trace_path = analysis_dir / f"factor_trace_L{layer}_H{head}.json"
    corpus_path = states_dir / "corpus.npy"
    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not trace_path.exists():
        raise FileNotFoundError(f"No factor trace at {trace_path}. Run factor-trace first.")
    if not corpus_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Need both {corpus_path} and {head_path}")

    trace_result = json.loads(trace_path.read_text())
    if not trace_result.get("selected_features"):
        raise ValueError("Factor trace has no selected features")

    selected_feature_ids = {int(item["feature_idx"]) for item in trace_result["selected_features"]}
    if {62, 105}.issubset(selected_feature_ids):
        target_feature_indices = [62, 105]
    elif 62 in selected_feature_ids:
        target_feature_indices = [62]
    else:
        target_feature_indices = [int(trace_result["selected_features"][0]["feature_idx"])]

    batches = load_corpus_from_file(
        str(corpus_path),
        batch_size=1,
        n_samples=max(n_pairs * 4, trace_result.get("n_prompts", n_pairs * 4)),
    )
    corpus_sequences = []
    for batch in batches:
        for row in range(batch.shape[0]):
            corpus_sequences.append(batch[row : row + 1])
    corpus_sequences = corpus_sequences[: trace_result.get("n_prompts", len(corpus_sequences))]

    states = torch.from_numpy(
        np.array(
            np.load(str(head_path), mmap_mode="r")[: min(2000, trace_result.get("n_prompts", n_pairs * 8) * 8)],
            dtype=np.float32,
        )
    )
    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = run_factor_trace_transplant(
        model=model,
        tokenizer=tokenizer,
        corpus_sequences=corpus_sequences,
        trace_result=trace_result,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        states=states,
        target_feature_indices=target_feature_indices,
        n_pairs=n_pairs,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "total_time_s": round(time.time() - t0, 1),
    })

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"factor_trace_transplant_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Target features: {target_feature_indices}")
    print(
        "Boundary shift vs wrong-feature: "
        f"{result.get('summary', {}).get('boundary_ratio_vs_wrong_feature', 0):.2f}x"
    )
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def hierarchical_transplant_modal(
    layer: int = 9,
    head: int = 4,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "ultrachat_200k",
    top_k: int = 32,
) -> dict:
    """Hierarchical causal faithfulness: full-state vs SAE-recon vs feature-diff transplant."""
    import json, os, sys, time
    from pathlib import Path
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from hierarchical_transplant import (
        run_hierarchical_transplant,
        print_results_table,
        build_row_specs_from_benchmark,
    )

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)

    # Resolve SAE checkpoint (prefer bilinear)
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=checkpoint_tag.strip() or None,
        corpus_source="openwebtext",
        preferred_types=("bilinear", "bilinear_tied", "rank1"),
    )

    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()
    print(f"Loaded SAE: type={cfg.get('sae_type')}, tag={resolved_tag}")

    analysis_dir = _analysis_dir(corpus_source)
    benchmark_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H{head}_{corpus_source}.json"
    if not benchmark_path.exists():
        benchmark_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H{head}.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"No benchmark at {benchmark_path}. Run factor-trace-benchmark first."
        )
    print(f"Benchmark: {benchmark_path}")

    states_dir = _states_dir(corpus_source)
    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}")

    batches = load_corpus_from_file(str(corpus_path), batch_size=1, n_samples=512)
    corpus_sequences = [batch[0:1] for batch in batches]

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    row_specs = build_row_specs_from_benchmark(
        str(benchmark_path), corpus_sequences, tokenizer
    )

    if not row_specs:
        raise ValueError("No qualifying rows found in benchmark data")

    result = run_hierarchical_transplant(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer_idx=layer,
        head_idx=head,
        row_specs=row_specs,
        top_k=top_k,
        device="cuda",
    )

    result["experiment"] = "hierarchical_transplant"
    result["layer"] = layer
    result["head"] = head
    result["sae_type"] = cfg.get("sae_type", "bilinear")
    result["sae_tag"] = resolved_tag
    result["model_name"] = model_name
    result["corpus_source"] = corpus_source
    result["top_k"] = top_k
    result["total_time_s"] = round(time.time() - t0, 1)

    print_results_table(result)

    # Save
    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hierarchical_transplant_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_benchmark_modal(
    layer: int = 9,
    head: int = 4,
    n_rows: int = 64,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "ultrachat_200k",
) -> dict:
    """Mine high-confidence localized use sites for the signed H4 boundary code."""
    import json, os, sys, time
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import build_use_site_benchmark

    vol.reload()
    t0 = time.time()
    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )
    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    trace_path = analysis_dir / f"factor_trace_L{layer}_H{head}.json"
    corpus_path = states_dir / "corpus.npy"
    if not trace_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"Need {trace_path} and {corpus_path}")

    trace_result = json.loads(trace_path.read_text())
    batches = load_corpus_from_file(str(corpus_path), batch_size=1, n_samples=trace_result.get("n_prompts", n_rows * 4))
    corpus_sequences = []
    for batch in batches:
        for row in range(batch.shape[0]):
            corpus_sequences.append(batch[row : row + 1])
    corpus_sequences = corpus_sequences[: trace_result.get("n_prompts", len(corpus_sequences))]

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = build_use_site_benchmark(
        model=model,
        tokenizer=tokenizer,
        corpus_sequences=corpus_sequences,
        trace_result=trace_result,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        target_feature_indices=[62, 105],
        max_rows=n_rows,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "total_time_s": round(time.time() - t0, 1),
    })
    out_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Accepted rows: {result.get('n_rows_accepted', 0)}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_use_site_modal(
    layer: int = 9,
    head: int = 4,
    n_rows: int = 32,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "ultrachat_200k",
) -> dict:
    """Run use-site signed-direction causal edits on benchmarked rows."""
    import json, os, sys, time
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import run_use_site_signed_causal

    vol.reload()
    t0 = time.time()
    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )
    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    benchmark_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H{head}.json"
    corpus_path = states_dir / "corpus.npy"
    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not benchmark_path.exists() or not corpus_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Need {benchmark_path}, {corpus_path}, and {head_path}")

    benchmark_result = json.loads(benchmark_path.read_text())
    batches = load_corpus_from_file(str(corpus_path), batch_size=1, n_samples=benchmark_result.get("n_rows_accepted", n_rows) * 8)
    corpus_sequences = []
    for batch in batches:
        for row in range(batch.shape[0]):
            corpus_sequences.append(batch[row : row + 1])
    states = torch.from_numpy(
        np.array(
            np.load(str(head_path), mmap_mode="r")[: min(2000, len(corpus_sequences) * 8)],
            dtype=np.float32,
        )
    )

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = run_use_site_signed_causal(
        model=model,
        tokenizer=tokenizer,
        corpus_sequences=corpus_sequences,
        benchmark_result=benchmark_result,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        states=states,
        n_rows=n_rows,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "total_time_s": round(time.time() - t0, 1),
    })
    out_path = analysis_dir / f"factor_trace_use_site_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    s = result.get("summary", {})
    print(f"Rows evaluated: {result.get('n_rows_evaluated', 0)}")
    print(f"Target beats reverse prob: {100.0 * s.get('target_beats_reverse_prob_fraction', 0):.1f}%")
    print(f"Target - reverse prob shift mean: {s.get('target_minus_reverse_actual_token_prob_shift_mean', 0):+.4f}")
    print(f"Promotion gate pass: {s.get('promotion_gate_pass', False)}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_readout_map_modal(
    layer: int = 9,
    head: int = 4,
    n_rows: int = 32,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "ultrachat_200k",
) -> dict:
    """Map cheap downstream readouts for the signed H4 boundary direction."""
    import json, os, sys, time
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import run_use_site_readout_map

    vol.reload()
    t0 = time.time()
    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )
    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    benchmark_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H{head}.json"
    corpus_path = states_dir / "corpus.npy"
    if not benchmark_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"Need {benchmark_path} and {corpus_path}")

    benchmark_result = json.loads(benchmark_path.read_text())
    batches = load_corpus_from_file(
        str(corpus_path),
        batch_size=1,
        n_samples=max(benchmark_result.get("n_rows_accepted", n_rows) * 8, 256),
    )
    corpus_sequences = []
    for batch in batches:
        for row in range(batch.shape[0]):
            corpus_sequences.append(batch[row : row + 1])

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = run_use_site_readout_map(
        model=model,
        tokenizer=tokenizer,
        corpus_sequences=corpus_sequences,
        benchmark_result=benchmark_result,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        n_rows=n_rows,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "total_time_s": round(time.time() - t0, 1),
    })
    out_path = analysis_dir / f"factor_trace_readout_map_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    print(f"Rows evaluated: {result.get('n_rows_evaluated', 0)}")
    s = result.get("summary", {})
    print(f"Target - reverse prob shift mean: {s.get('actual_token_prob_delta_target_minus_reverse_mean', 0):+.4f}")
    print(f"Target - reverse logit shift mean: {s.get('actual_token_logit_delta_target_minus_reverse_mean', 0):+.4f}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def factor_trace_dot_direction_modal(
    layer: int = 9,
    head: int = 4,
    n_rows: int = 5,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "ultrachat_200k",
) -> dict:
    """Train a '.' logistic direction and run exact-dot use-site interventions."""
    import json, os, sys, time
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config
    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from factor_trace import run_dot_direction_use_site

    vol.reload()
    t0 = time.time()
    corpus_source = _normalize_corpus_source(corpus_source)
    canonical_tag = checkpoint_tag.strip() or f"rank1_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("rank1",),
    )
    _, meta = _checkpoint_root_for_corpus("openwebtext")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg,
        state_dict=ckpt["model_state_dict"],
        default_d_k=meta["key_head_dim"],
        default_d_v=meta["value_head_dim"],
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    corpus_path = states_dir / "corpus.npy"
    benchmark_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H{head}.json"
    use_site_path = analysis_dir / f"factor_trace_use_site_L{layer}_H{head}.json"
    trace_path = analysis_dir / f"factor_trace_L{layer}_H{head}.json"
    if not corpus_path.exists() or not benchmark_path.exists() or not use_site_path.exists() or not trace_path.exists():
        raise FileNotFoundError(f"Need {corpus_path}, {benchmark_path}, {use_site_path}, and {trace_path}")

    benchmark_result = json.loads(benchmark_path.read_text())
    use_site_result = json.loads(use_site_path.read_text())
    trace_result = json.loads(trace_path.read_text())
    n_prompts = int(trace_result.get("n_prompts", len(trace_result.get("prompt_records", [])) or 256))

    direction_batches = load_corpus_from_file(str(corpus_path), batch_size=4, n_samples=n_prompts)
    seq_batches = load_corpus_from_file(str(corpus_path), batch_size=1, n_samples=n_prompts)
    corpus_sequences = []
    for batch in seq_batches:
        for row in range(batch.shape[0]):
            corpus_sequences.append(batch[row : row + 1])

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = run_dot_direction_use_site(
        model=model,
        tokenizer=tokenizer,
        corpus_batches=direction_batches,
        corpus_sequences=corpus_sequences,
        benchmark_result=benchmark_result,
        use_site_result=use_site_result,
        layer_idx=layer,
        head_idx=head,
        sae=sae,
        sae_type=cfg["sae_type"],
        prompt_len=min(int(trace_result.get("prompt_len", 512)), 512),
        direction_prompt_count=n_prompts,
        n_rows=n_rows,
        device="cuda",
    )
    result.update({
        "layer": layer,
        "head": head,
        "model_name": model_name,
        "corpus_source": corpus_source,
        "sae_type": cfg["sae_type"],
        "sae_tag": resolved_tag,
        "total_time_s": round(time.time() - t0, 1),
    })
    out_path = analysis_dir / f"factor_trace_dot_direction_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"Saved to {out_path}")
    summary = result.get("summary", {})
    print(f"Rows evaluated: {summary.get('n_rows', 0)}")
    print(f"Logistic test accuracy: {result.get('logistic', {}).get('test_accuracy', 0):.3f}")
    print(f"Success rows: {summary.get('success_rows', 0)}")
    print(f"Success gate pass: {summary.get('success_gate_pass', False)}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


# -- Stage: Multi-head recurrent state transplant --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def multihead_transplant_modal(
    layer: int = 9,
    max_rows: int = 32,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    corpus_source: str = "ultrachat_200k",
) -> dict:
    """Transplant all 16 heads' recurrent states from boundary to non-boundary context."""
    import json, os, sys, time
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from extract_states import load_corpus_from_file, load_model_and_tokenizer
    from multihead_transplant import run_full_experiment

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)

    # Load benchmark data (stored without corpus suffix on Modal volume)
    benchmark_path = analysis_dir / f"factor_trace_benchmark_L{layer}_H4.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"No benchmark data at {benchmark_path}. Run factor-trace-benchmark first."
        )

    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")

    batches = load_corpus_from_file(str(corpus_path), batch_size=1, n_samples=300)
    corpus_sequences = []
    for batch in batches:
        for row_idx in range(batch.shape[0]):
            corpus_sequences.append(batch[row_idx:row_idx + 1])

    model, tokenizer, config = load_model_and_tokenizer(model_name, "cuda")
    gdn_layers = [i for i, t in enumerate(config.layer_types) if t == "linear_attention"]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. Valid: {gdn_layers}")

    result = run_full_experiment(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer,
        benchmark_path=str(benchmark_path),
        corpus_sequences=corpus_sequences,
        device="cuda",
        max_rows=max_rows,
    )
    result.update({
        "model_name": model_name,
        "corpus_source": corpus_source,
        "benchmark_path": str(benchmark_path),
        "total_time_s": round(time.time() - t0, 1),
    })

    out_dir = analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multihead_transplant_L{layer}.json"
    out_path.write_text(json.dumps(result, indent=2))
    vol.commit()

    # Also save locally
    local_dir = Path("/root/results/data")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / f"multihead_transplant_L{layer}.json").write_text(json.dumps(result, indent=2))

    p1 = result["phase1_all_heads"]
    p2 = result["phase2_leave_one_out"]
    p3 = result["phase3_minimal_set"]
    print(f"\nSaved to {out_path}")
    print(f"Phase 1: mean_delta={p1['mean_delta']:+.4f} n_positive={p1['n_positive']}/{p1['n_rows']}")
    print(f"Phase 2: top heads = {p2['top_heads_by_contribution']}")
    print(f"Phase 3: mean_delta={p3['mean_delta']:+.4f} fraction_of_full={p3['fraction_of_full']:.2f}")
    print(f"Total time: {result['total_time_s']:.0f}s")
    return result


# -- Stage: Generation-time SAE intervention --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def generation_intervention_modal(
    layer: int = 9,
    head: int = 4,
    n_prompts: int = 100,
    n_tokens: int = 200,
    boost_scale: float = 3.0,
    suppress_scale: float = 0.0,
    n_boundary_features: int = 10,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    corpus_source: str = "openwebtext",
) -> dict:
    """Continuous SAE feature intervention during autoregressive generation.

    At every step, intercept the GDN recurrent state, encode through the SAE,
    scale boundary-related feature activations, decode back, and patch the cache.
    Compare baseline / boost / suppress across n_prompts prompts.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    # Load probe results for feature selection (may be on openwebtext even if corpus is ultrachat)
    probe_path = analysis_dir / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        probe_path = _analysis_dir("openwebtext") / f"probe_features_L{layer}_H{head}.json"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No probe results found. Run --stage probe-features first.")
    print(f"Loading probe results from {probe_path}")

    from generation_intervention import (
        select_boundary_features,
        run_generation_experiment,
        print_summary,
    )

    boundary_info = select_boundary_features(
        str(probe_path),
        n_features=n_boundary_features,
    )
    boundary_feature_indices = [e["feature_idx"] for e in boundary_info["combined"]]
    print(f"Selected {len(boundary_feature_indices)} boundary features: {boundary_feature_indices}")
    print(f"Target properties: {boundary_info['target_properties']}")
    for entry in boundary_info["combined"]:
        print(f"  F{entry['feature_idx']}: {entry['property']} rho={entry['rho']:.3f}")

    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=checkpoint_tag.strip() or None,
        corpus_source="openwebtext",
        preferred_types=("bilinear", "bilinear_tied", "rank1"),
    )
    print(f"Using SAE: {resolved_tag} (type={cfg.get('sae_type', '?')})")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae = build_sae_from_config(
        cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()
    sae_type = infer_sae_type(cfg, ckpt["model_state_dict"])

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    # Build prompts from corpus (first n_prompts sequences, first 50 tokens as prompt)
    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")
    batches = load_corpus_from_file(str(corpus_path), batch_size=32, n_samples=n_prompts)
    prompts = []
    prompt_token_len = 50
    for batch in batches:
        for seq in batch:
            prefix_ids = seq[:prompt_token_len].tolist()
            prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            prompts.append(prompt_text)
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            break
    print(f"Built {len(prompts)} prompts (first {prompt_token_len} tokens each)")

    print(f"\nStarting generation experiment: {len(prompts)} prompts x 3 conditions x {n_tokens} tokens")
    result = run_generation_experiment(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_type=sae_type,
        layer_idx=layer,
        head_idx=head,
        prompts=prompts,
        boundary_features=boundary_feature_indices,
        boost_scale=boost_scale,
        suppress_scale=suppress_scale,
        n_tokens=n_tokens,
        temperature=temperature,
    )

    result["sae_tag"] = resolved_tag
    result["model_name"] = model_name
    result["corpus_source"] = corpus_source
    result["boundary_feature_details"] = boundary_info["combined"]

    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
    out_name = f"generation_intervention_L{layer}_H{head}{suffix}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print_summary(result)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    return result


# -- Stage: Multi-head generation intervention --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def generation_intervention_multihead_modal(
    layer: int = 9,
    n_heads: int = 16,
    n_prompts: int = 100,
    n_tokens: int = 200,
    boost_scale: float = 3.0,
    suppress_scale: float = 0.0,
    n_boundary_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    corpus_source: str = "openwebtext",
    period_token_id: int = 13,
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
) -> dict:
    """Multi-head SAE feature intervention during autoregressive generation.

    Loads per-head SAEs for all heads at the target layer, finds boundary
    features per head via activation differences at period positions, then
    intervenes on all heads simultaneously during generation.

    sae_types controls which SAE architecture to load. Pass ("flat",) to run
    the flat-SAE control experiment for the geometry comparison.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")
    corpus_arr = np.load(str(corpus_path), mmap_mode="r")  # (n_seqs, seq_len)
    n_corpus_seqs = corpus_arr.shape[0]
    seq_len = corpus_arr.shape[1]

    # Identify boundary vs non-boundary positions across corpus
    # Boundary = position where the NEXT token is a period
    corpus_full = np.array(corpus_arr)  # materialize from mmap for vectorized ops
    boundary_mask = np.zeros((n_corpus_seqs, seq_len), dtype=bool)
    boundary_mask[:, :-1] = (corpus_full[:, 1:] == period_token_id)

    n_boundary_total = int(boundary_mask.sum())
    n_nonboundary_total = int((~boundary_mask).sum())
    print(f"Corpus: {n_corpus_seqs} seqs x {seq_len} tokens")
    print(f"Boundary positions: {n_boundary_total}, non-boundary: {n_nonboundary_total}")

    from generation_intervention import (
        select_boundary_features_fast,
        run_generation_experiment,
        print_summary,
    )

    sae_per_head: dict[int, object] = {}
    boundary_features_per_head: dict[int, list[int]] = {}
    feature_details_per_head: dict[str, list[dict]] = {}
    sae_type = None
    n_loaded = 0

    for h in range(n_heads):
        try:
            ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
                layer=layer,
                head=h,
                n_features_target=n_features_target,
                corpus_source="openwebtext",
                preferred_types=sae_types,
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
        if not head_states_path.exists():
            print(f"  H{h}: no extracted states at {head_states_path}, skipping")
            continue

        head_states = np.load(str(head_states_path), mmap_mode="r")  # (n_samples, d_k, d_v)
        # head_states has one entry per (seq, pos) = n_seqs * seq_len total
        n_states = head_states.shape[0]
        expected = n_corpus_seqs * seq_len
        if n_states < expected:
            # States may cover fewer samples than corpus; adjust mask
            n_use_seqs = n_states // seq_len
            flat_mask = boundary_mask[:n_use_seqs].reshape(-1)
        else:
            n_use_seqs = n_corpus_seqs
            flat_mask = boundary_mask[:n_use_seqs].reshape(-1)

        states_flat = head_states[:n_use_seqs * seq_len]

        boundary_indices = np.where(flat_mask)[0]
        nonboundary_indices = np.where(~flat_mask)[0]

        n_sample = min(len(boundary_indices), len(nonboundary_indices), 10000)
        if n_sample == 0:
            print(f"  H{h}: no boundary positions found, skipping")
            continue

        rng = np.random.RandomState(42)
        b_idx = rng.choice(boundary_indices, size=n_sample, replace=False)
        nb_idx = rng.choice(nonboundary_indices, size=n_sample, replace=False)

        states_boundary = np.array(states_flat[b_idx])
        states_nonboundary = np.array(states_flat[nb_idx])

        top_features = select_boundary_features_fast(
            sae_model, this_sae_type,
            states_boundary, states_nonboundary,
            n_features=n_boundary_features,
        )

        feat_indices = [f["feature_idx"] for f in top_features]
        sae_per_head[h] = sae_model
        boundary_features_per_head[h] = feat_indices
        feature_details_per_head[str(h)] = top_features
        n_loaded += 1

        print(f"  H{h}: {resolved_tag} | top features: {feat_indices} | "
              f"max |diff|={top_features[0]['mean_diff']:.4f}")

    if n_loaded == 0:
        # Debug: show what the resolver sees
        ckpt_root, _ = _checkpoint_root_for_corpus("openwebtext")
        print(f"  DEBUG: checkpoint root = {ckpt_root}")
        print(f"  DEBUG: exists = {ckpt_root.exists()}")
        if ckpt_root.exists():
            dirs = sorted(d.name for d in ckpt_root.iterdir() if "flat" in d.name and f"L{layer}" in d.name)
            print(f"  DEBUG: flat L{layer} dirs = {dirs[:5]}")
        raise FileNotFoundError(
            f"No per-head SAE checkpoints found for layer {layer}. "
            "Run --stage train-allheads first."
        )

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")
    total_features = sum(len(v) for v in boundary_features_per_head.values())
    print(f"Total boundary features across all heads: {total_features}")

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    batches = load_corpus_from_file(str(corpus_path), batch_size=32, n_samples=n_prompts)
    prompts = []
    prompt_token_len = 50
    for batch in batches:
        for seq in batch:
            prefix_ids = seq[:prompt_token_len].tolist()
            prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            prompts.append(prompt_text)
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            break
    print(f"Built {len(prompts)} prompts (first {prompt_token_len} tokens each)")

    print(f"\nStarting multi-head generation experiment: "
          f"{len(prompts)} prompts x 3 conditions x {n_tokens} tokens x {n_loaded} heads")
    result = run_generation_experiment(
        model=model,
        tokenizer=tokenizer,
        sae_type=sae_type,
        layer_idx=layer,
        prompts=prompts,
        boost_scale=boost_scale,
        suppress_scale=suppress_scale,
        n_tokens=n_tokens,
        temperature=temperature,
        sae_per_head=sae_per_head,
        boundary_features_per_head=boundary_features_per_head,
    )

    result["model_name"] = model_name
    result["corpus_source"] = corpus_source
    result["feature_details_per_head"] = feature_details_per_head
    result["period_token_id"] = period_token_id

    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
    type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
    out_name = f"generation_intervention_multihead_L{layer}{suffix}{type_suffix}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print_summary(result)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    return result


# -- Stage: Multi-head generation intervention with RANDOM features (control) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def generation_intervention_multihead_random_modal(
    layer: int = 9,
    n_heads: int = 16,
    n_prompts: int = 100,
    n_tokens: int = 200,
    boost_scale: float = 3.0,
    suppress_scale: float = 0.0,
    n_random_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    corpus_source: str = "openwebtext",
    period_token_id: int = 13,
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
    random_seed: int = 123,
) -> dict:
    """Random-feature control for multi-head generation intervention.

    Same setup as the boundary-feature experiment, but selects n_random_features
    RANDOM alive features per head instead of boundary-correlated ones.  If random
    features produce d~0 on sentence/period/newline metrics while boundary features
    produce d>0, the specificity claim is confirmed: the effect is tied to
    boundary-correlated features, not an artifact of any bilinear boost.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
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

    from generation_intervention import run_generation_experiment, print_summary

    sae_per_head: dict[int, object] = {}
    random_features_per_head: dict[int, list[int]] = {}
    feature_details_per_head: dict[str, list[dict]] = {}
    sae_type = None
    n_loaded = 0
    rng = np.random.RandomState(random_seed)

    for h in range(n_heads):
        try:
            ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
                layer=layer,
                head=h,
                n_features_target=n_features_target,
                corpus_source="openwebtext",
                preferred_types=sae_types,
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

        # Load extracted states for this head to find alive features
        head_states_path = states_dir / f"layer_{layer}" / f"head_{h}.npy"
        if not head_states_path.exists():
            print(f"  H{h}: no extracted states at {head_states_path}, skipping")
            continue

        head_states = np.load(str(head_states_path), mmap_mode="r")
        n_states = head_states.shape[0]
        n_use = min(n_states, n_corpus_seqs * seq_len)

        # Encode a sample of states to find alive features
        sample_size = min(n_use, 5000)
        sample_indices = rng.choice(n_use, size=sample_size, replace=False)
        sample_states = np.array(head_states[sample_indices])

        sample_tensor = torch.tensor(sample_states, dtype=torch.float32, device="cuda")
        if this_sae_type == "flat":
            sample_tensor = sample_tensor.reshape(sample_tensor.shape[0], -1)

        # Encode in batches to find alive features
        alive_mask = None
        batch_size = 512
        for start in range(0, len(sample_tensor), batch_size):
            batch = sample_tensor[start:start + batch_size]
            coeffs = sae_model.encode(batch)
            batch_alive = (coeffs.abs() > 0).any(dim=0)
            if alive_mask is None:
                alive_mask = batch_alive
            else:
                alive_mask = alive_mask | batch_alive

        alive_indices = torch.nonzero(alive_mask, as_tuple=True)[0].cpu().numpy().tolist()
        n_alive = len(alive_indices)

        if n_alive < n_random_features:
            print(f"  H{h}: only {n_alive} alive features, need {n_random_features}, skipping")
            continue

        # Select random features from alive set
        random_feats = rng.choice(alive_indices, size=n_random_features, replace=False).tolist()

        sae_per_head[h] = sae_model
        random_features_per_head[h] = random_feats
        feature_details_per_head[str(h)] = [
            {"feature_idx": f, "selection": "random", "n_alive": n_alive}
            for f in random_feats
        ]
        n_loaded += 1

        print(f"  H{h}: {resolved_tag} | {n_alive} alive | random features: {random_feats}")

    if n_loaded == 0:
        raise FileNotFoundError(
            f"No per-head SAE checkpoints found for layer {layer}. "
            "Run --stage train-allheads first."
        )

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")
    total_features = sum(len(v) for v in random_features_per_head.values())
    print(f"Total random features across all heads: {total_features}")

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    # Build prompts (same logic as boundary experiment)
    batches = load_corpus_from_file(str(corpus_path), batch_size=32, n_samples=n_prompts)
    prompts = []
    prompt_token_len = 50
    for batch in batches:
        for seq in batch:
            prefix_ids = seq[:prompt_token_len].tolist()
            prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            prompts.append(prompt_text)
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            break
    print(f"Built {len(prompts)} prompts (first {prompt_token_len} tokens each)")

    print(f"\nStarting RANDOM-feature multi-head generation experiment: "
          f"{len(prompts)} prompts x 3 conditions x {n_tokens} tokens x {n_loaded} heads")
    result = run_generation_experiment(
        model=model,
        tokenizer=tokenizer,
        sae_type=sae_type,
        layer_idx=layer,
        prompts=prompts,
        boost_scale=boost_scale,
        suppress_scale=suppress_scale,
        n_tokens=n_tokens,
        temperature=temperature,
        sae_per_head=sae_per_head,
        boundary_features_per_head=random_features_per_head,
    )

    result["experiment"] = "generation_intervention_multihead_random"
    result["model_name"] = model_name
    result["corpus_source"] = corpus_source
    result["feature_details_per_head"] = feature_details_per_head
    result["feature_selection"] = "random_alive"
    result["random_seed"] = random_seed
    result["period_token_id"] = period_token_id

    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
    type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
    out_name = f"generation_intervention_multihead_random_L{layer}{suffix}{type_suffix}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print_summary(result)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    return result


# -- Stage: Multi-head ADDITIVE generation intervention (boundary features) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def generation_intervention_multihead_additive_modal(
    layer: int = 9,
    n_heads: int = 16,
    n_prompts: int = 100,
    n_tokens: int = 200,
    additive_boost_strength: float = 0.5,
    n_boundary_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    corpus_source: str = "openwebtext",
    period_token_id: int = 13,
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
) -> dict:
    """Additive-boost boundary feature intervention (no compounding).

    Instead of multiplicative scaling (coeffs *= scale) which compounds through
    the GDN recurrence, this uses additive push (coeffs += push) calibrated from
    each feature's mean_boundary activation. The push is constant per step,
    so total perturbation grows linearly (200 * push) instead of exponentially.

    additive_boost_strength controls the push magnitude:
      push = mean_boundary_activation * additive_boost_strength
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
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

    # Identify boundary vs non-boundary positions
    corpus_full = np.array(corpus_arr)
    boundary_mask = np.zeros((n_corpus_seqs, seq_len), dtype=bool)
    boundary_mask[:, :-1] = (corpus_full[:, 1:] == period_token_id)

    n_boundary_total = int(boundary_mask.sum())
    n_nonboundary_total = int((~boundary_mask).sum())
    print(f"Corpus: {n_corpus_seqs} seqs x {seq_len} tokens")
    print(f"Boundary positions: {n_boundary_total}, non-boundary: {n_nonboundary_total}")

    from generation_intervention import (
        select_boundary_features_fast,
        run_generation_experiment,
        print_summary,
    )

    sae_per_head: dict[int, object] = {}
    boundary_features_per_head: dict[int, list[int]] = {}
    additive_push_per_head: dict[int, dict[int, float]] = {}
    feature_details_per_head: dict[str, list[dict]] = {}
    sae_type = None
    n_loaded = 0

    for h in range(n_heads):
        try:
            ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
                layer=layer,
                head=h,
                n_features_target=n_features_target,
                corpus_source="openwebtext",
                preferred_types=sae_types,
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
        if not head_states_path.exists():
            print(f"  H{h}: no extracted states at {head_states_path}, skipping")
            continue

        head_states = np.load(str(head_states_path), mmap_mode="r")
        n_states = head_states.shape[0]
        expected = n_corpus_seqs * seq_len
        if n_states < expected:
            n_use_seqs = n_states // seq_len
            flat_mask = boundary_mask[:n_use_seqs].reshape(-1)
        else:
            n_use_seqs = n_corpus_seqs
            flat_mask = boundary_mask[:n_use_seqs].reshape(-1)

        states_flat = head_states[:n_use_seqs * seq_len]

        boundary_indices = np.where(flat_mask)[0]
        nonboundary_indices = np.where(~flat_mask)[0]

        n_sample = min(len(boundary_indices), len(nonboundary_indices), 10000)
        if n_sample == 0:
            print(f"  H{h}: no boundary positions found, skipping")
            continue

        rng = np.random.RandomState(42)
        b_idx = rng.choice(boundary_indices, size=n_sample, replace=False)
        nb_idx = rng.choice(nonboundary_indices, size=n_sample, replace=False)

        states_boundary = np.array(states_flat[b_idx])
        states_nonboundary = np.array(states_flat[nb_idx])

        top_features = select_boundary_features_fast(
            sae_model, this_sae_type,
            states_boundary, states_nonboundary,
            n_features=n_boundary_features,
        )

        feat_indices = [f["feature_idx"] for f in top_features]

        # Compute additive push from mean_boundary * boost_strength
        push_dict: dict[int, float] = {}
        for f in top_features:
            fi = f["feature_idx"]
            mean_b = f["mean_boundary"]
            # Push in the direction of the mean_diff sign:
            # positive mean_diff => feature activates MORE at boundaries => push up
            # negative mean_diff => feature activates LESS at boundaries => push down (negative)
            sign = 1.0 if f["mean_diff"] >= 0 else -1.0
            push_dict[fi] = sign * abs(mean_b) * additive_boost_strength

        sae_per_head[h] = sae_model
        boundary_features_per_head[h] = feat_indices
        additive_push_per_head[h] = push_dict
        feature_details_per_head[str(h)] = top_features
        n_loaded += 1

        print(f"  H{h}: {resolved_tag} | top features: {feat_indices} | "
              f"max |diff|={top_features[0]['mean_diff']:.4f} | "
              f"push range: [{min(push_dict.values()):.5f}, {max(push_dict.values()):.5f}]")

    if n_loaded == 0:
        raise FileNotFoundError(
            f"No per-head SAE checkpoints found for layer {layer}. "
            "Run --stage train-allheads first."
        )

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")
    total_features = sum(len(v) for v in boundary_features_per_head.values())
    print(f"Total boundary features across all heads: {total_features}")
    print(f"Additive boost strength: {additive_boost_strength}")

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    batches = load_corpus_from_file(str(corpus_path), batch_size=32, n_samples=n_prompts)
    prompts = []
    prompt_token_len = 50
    for batch in batches:
        for seq in batch:
            prefix_ids = seq[:prompt_token_len].tolist()
            prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            prompts.append(prompt_text)
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            break
    print(f"Built {len(prompts)} prompts (first {prompt_token_len} tokens each)")

    print(f"\nStarting ADDITIVE multi-head generation experiment: "
          f"{len(prompts)} prompts x 3 conditions x {n_tokens} tokens x {n_loaded} heads")
    result = run_generation_experiment(
        model=model,
        tokenizer=tokenizer,
        sae_type=sae_type,
        layer_idx=layer,
        prompts=prompts,
        n_tokens=n_tokens,
        temperature=temperature,
        sae_per_head=sae_per_head,
        boundary_features_per_head=boundary_features_per_head,
        additive=True,
        additive_push_per_head=additive_push_per_head,
        additive_boost_strength=additive_boost_strength,
    )

    result["model_name"] = model_name
    result["corpus_source"] = corpus_source
    result["feature_details_per_head"] = feature_details_per_head
    result["period_token_id"] = period_token_id
    result["intervention_mode"] = "additive"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
    type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
    out_name = f"generation_intervention_multihead_additive_L{layer}{suffix}{type_suffix}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print_summary(result)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    return result


# -- Stage: Multi-head ADDITIVE generation intervention with RANDOM features (control) --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def generation_intervention_multihead_additive_random_modal(
    layer: int = 9,
    n_heads: int = 16,
    n_prompts: int = 100,
    n_tokens: int = 200,
    additive_boost_strength: float = 0.5,
    n_random_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    corpus_source: str = "openwebtext",
    period_token_id: int = 13,
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
    random_seed: int = 123,
) -> dict:
    """Additive-boost random feature control (no compounding).

    Same as additive boundary experiment but with random alive features.
    The push for each random feature is calibrated from its mean activation
    across a sample of states, matching the magnitude calibration used for
    boundary features. If additive mode eliminates compounding artifacts,
    random features should produce near-zero effect on paragraph/newline
    metrics while boundary features produce positive effects.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()
    t0 = time.time()

    corpus_source = _normalize_corpus_source(corpus_source)
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
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

    from generation_intervention import run_generation_experiment, print_summary

    sae_per_head: dict[int, object] = {}
    random_features_per_head: dict[int, list[int]] = {}
    additive_push_per_head: dict[int, dict[int, float]] = {}
    feature_details_per_head: dict[str, list[dict]] = {}
    sae_type = None
    n_loaded = 0
    rng = np.random.RandomState(random_seed)

    for h in range(n_heads):
        try:
            ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
                layer=layer,
                head=h,
                n_features_target=n_features_target,
                corpus_source="openwebtext",
                preferred_types=sae_types,
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
        if not head_states_path.exists():
            print(f"  H{h}: no extracted states at {head_states_path}, skipping")
            continue

        head_states = np.load(str(head_states_path), mmap_mode="r")
        n_states = head_states.shape[0]
        n_use = min(n_states, n_corpus_seqs * seq_len)

        # Encode a sample to find alive features and their mean activations
        sample_size = min(n_use, 5000)
        sample_indices = rng.choice(n_use, size=sample_size, replace=False)
        sample_states = np.array(head_states[sample_indices])

        sample_tensor = torch.tensor(sample_states, dtype=torch.float32, device="cuda")
        if this_sae_type == "flat":
            sample_tensor = sample_tensor.reshape(sample_tensor.shape[0], -1)

        # Encode in batches to find alive features and compute mean activations
        all_coeffs = []
        batch_size = 512
        for start in range(0, len(sample_tensor), batch_size):
            batch = sample_tensor[start:start + batch_size]
            coeffs = sae_model.encode(batch)
            all_coeffs.append(coeffs.detach().cpu())

        all_coeffs_cat = torch.cat(all_coeffs, dim=0)  # (sample_size, n_features)
        alive_mask = (all_coeffs_cat.abs() > 0).any(dim=0)
        alive_indices = torch.nonzero(alive_mask, as_tuple=True)[0].numpy().tolist()
        n_alive = len(alive_indices)

        if n_alive < n_random_features:
            print(f"  H{h}: only {n_alive} alive features, need {n_random_features}, skipping")
            continue

        # Select random features from alive set
        random_feats = rng.choice(alive_indices, size=n_random_features, replace=False).tolist()

        # Compute mean activation per random feature (for calibrating push magnitude)
        mean_acts = all_coeffs_cat.mean(dim=0).numpy()
        push_dict: dict[int, float] = {}
        details_list = []
        for fi in random_feats:
            mean_act = float(mean_acts[fi])
            # Random features get push magnitude based on their own mean activation,
            # matching the calibration approach used for boundary features
            push = abs(mean_act) * additive_boost_strength
            # Random direction: use positive push (same as boost for boundary)
            push_dict[fi] = push
            details_list.append({
                "feature_idx": fi,
                "selection": "random",
                "n_alive": n_alive,
                "mean_activation": mean_act,
                "additive_push": push,
            })

        sae_per_head[h] = sae_model
        random_features_per_head[h] = random_feats
        additive_push_per_head[h] = push_dict
        feature_details_per_head[str(h)] = details_list
        n_loaded += 1

        push_vals = list(push_dict.values())
        print(f"  H{h}: {resolved_tag} | {n_alive} alive | random features: {random_feats} | "
              f"push range: [{min(push_vals):.5f}, {max(push_vals):.5f}]")

    if n_loaded == 0:
        raise FileNotFoundError(
            f"No per-head SAE checkpoints found for layer {layer}. "
            "Run --stage train-allheads first."
        )

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")
    total_features = sum(len(v) for v in random_features_per_head.values())
    print(f"Total random features across all heads: {total_features}")
    print(f"Additive boost strength: {additive_boost_strength}")

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    batches = load_corpus_from_file(str(corpus_path), batch_size=32, n_samples=n_prompts)
    prompts = []
    prompt_token_len = 50
    for batch in batches:
        for seq in batch:
            prefix_ids = seq[:prompt_token_len].tolist()
            prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            prompts.append(prompt_text)
            if len(prompts) >= n_prompts:
                break
        if len(prompts) >= n_prompts:
            break
    print(f"Built {len(prompts)} prompts (first {prompt_token_len} tokens each)")

    print(f"\nStarting ADDITIVE RANDOM-feature multi-head generation experiment: "
          f"{len(prompts)} prompts x 3 conditions x {n_tokens} tokens x {n_loaded} heads")
    result = run_generation_experiment(
        model=model,
        tokenizer=tokenizer,
        sae_type=sae_type,
        layer_idx=layer,
        prompts=prompts,
        n_tokens=n_tokens,
        temperature=temperature,
        sae_per_head=sae_per_head,
        boundary_features_per_head=random_features_per_head,
        additive=True,
        additive_push_per_head=additive_push_per_head,
        additive_boost_strength=additive_boost_strength,
    )

    result["experiment"] = "generation_intervention_multihead_additive_random"
    result["model_name"] = model_name
    result["corpus_source"] = corpus_source
    result["feature_details_per_head"] = feature_details_per_head
    result["feature_selection"] = "random_alive"
    result["random_seed"] = random_seed
    result["period_token_id"] = period_token_id
    result["intervention_mode"] = "additive"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
    type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
    out_name = f"generation_intervention_multihead_additive_random_L{layer}{suffix}{type_suffix}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print_summary(result)

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    return result


# -- Stage: Qualitative generation demo (instruction-formatted prompts) --

# 20 diverse instruction prompts that produce coherent multi-sentence prose.
# Topics span history, science, cooking, travel, technology, nature.
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


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS_A10G)
def generation_intervention_qualitative_modal(
    layer: int = 9,
    n_heads: int = 16,
    n_tokens: int = 400,
    boost_scale: float = 3.0,
    suppress_scale: float = 0.0,
    n_boundary_features: int = 10,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    period_token_id: int = 13,
    sae_types: tuple[str, ...] = ("bilinear", "bilinear_tied", "rank1"),
    random_seed: int = 123,
    n_random_features: int = 10,
    use_chat_template: bool = True,
    custom_prompts: tuple[str, ...] | None = None,
    additive: bool = False,
    additive_boost_strengths: tuple[float, ...] = (5.0,),
) -> dict:
    """Qualitative generation demo with instruction-formatted prompts.

    Instead of corpus prefix fragments, uses hand-crafted instruction prompts
    formatted through the model's chat template. The baseline produces coherent
    prose; boundary-boost adds paragraph structure; random-boost degrades quality.

    When additive=False (default, multiplicative mode):
      Runs 3 conditions per prompt:
        1. baseline: no intervention
        2. boundary_boost: boundary features *= boost_scale
        3. random_boost: random features *= boost_scale

    When additive=True:
      Uses constant additive push (no compounding through recurrence).
      push = mean_boundary_activation * boost_strength.
      Runs conditions per boost_strength:
        1. baseline: no intervention
        2. boundary_boost_Xs: boundary features += push * strength
        3. random_boost_Xs: random features += push * strength

    Saves full generated text for each prompt+condition for qualitative comparison.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    from sae import build_sae_from_config, infer_sae_type
    from extract_states import load_model_and_tokenizer, load_corpus_from_file

    vol.reload()
    t0 = time.time()

    corpus_source = "openwebtext"
    states_dir = _states_dir(corpus_source)
    analysis_dir = _analysis_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    # Load corpus for boundary detection (need states to find boundary features)
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

    from generation_intervention import (
        select_boundary_features_fast,
        run_generation_experiment,
        compute_generation_stats,
        generate_with_intervention,
    )

    # Load per-head SAEs, find boundary features AND random features
    sae_per_head: dict[int, object] = {}
    boundary_features_per_head: dict[int, list[int]] = {}
    random_features_per_head: dict[int, list[int]] = {}
    boundary_push_per_head: dict[int, dict[int, float]] = {}  # for additive mode
    random_push_per_head: dict[int, dict[int, float]] = {}    # for additive mode
    feature_details_per_head: dict[str, list[dict]] = {}
    sae_type = None
    n_loaded = 0
    rng = np.random.RandomState(random_seed)

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
        expected = n_corpus_seqs * seq_len
        n_use_seqs = min(n_states // seq_len, n_corpus_seqs)
        flat_mask = boundary_mask[:n_use_seqs].reshape(-1)
        states_flat = head_states[:n_use_seqs * seq_len]

        boundary_indices = np.where(flat_mask)[0]
        nonboundary_indices = np.where(~flat_mask)[0]
        n_sample = min(len(boundary_indices), len(nonboundary_indices), 10000)
        if n_sample == 0:
            continue

        b_idx = rng.choice(boundary_indices, size=n_sample, replace=False)
        nb_idx = rng.choice(nonboundary_indices, size=n_sample, replace=False)

        # Boundary features
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
            if additive:
                all_coeffs_list.append(coeffs.detach().cpu())

        alive_indices = torch.nonzero(alive_mask, as_tuple=True)[0].cpu().numpy().tolist()
        # Exclude boundary features from random selection
        alive_non_boundary = [i for i in alive_indices if i not in set(feat_indices)]
        if len(alive_non_boundary) >= n_random_features:
            random_feats = rng.choice(alive_non_boundary, size=n_random_features, replace=False).tolist()
        else:
            random_feats = rng.choice(alive_indices, size=n_random_features, replace=False).tolist()

        # Compute additive push values when in additive mode
        if additive:
            all_coeffs_cat = torch.cat(all_coeffs_list, dim=0)
            mean_acts = all_coeffs_cat.mean(dim=0).numpy()

            # Boundary features: push = |mean_boundary| (direction from mean_diff sign)
            # Actual push = push_base * boost_strength (applied at generation time)
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

        if additive and h in boundary_push_per_head:
            b_vals = list(boundary_push_per_head[h].values())
            r_vals = list(random_push_per_head[h].values())
            print(f"  H{h}: boundary={feat_indices[:3]}... random={random_feats[:3]}... "
                  f"boundary_push=[{min(b_vals):.4f},{max(b_vals):.4f}] "
                  f"random_push=[{min(r_vals):.4f},{max(r_vals):.4f}]")
        else:
            print(f"  H{h}: boundary={feat_indices[:3]}... random={random_feats[:3]}...")

    if n_loaded == 0:
        raise FileNotFoundError(f"No per-head SAE checkpoints for layer {layer}.")

    print(f"\nLoaded {n_loaded}/{n_heads} per-head SAEs")

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    raw_prompts = list(custom_prompts) if custom_prompts else list(QUALITATIVE_PROMPTS)

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        formatted_prompts = []
        for p in raw_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            # Strip thinking wrapper if present (Qwen3.5 non-thinking mode)
            text = text.replace("<think>\n\n</think>\n\n", "")
            formatted_prompts.append(text)
        print(f"Formatted {len(formatted_prompts)} prompts with chat template")
        print(f"Example prompt:\n{formatted_prompts[0][:200]}")
    else:
        formatted_prompts = raw_prompts
        print(f"Using {len(formatted_prompts)} raw prompts (no chat template)")

    if additive:
        # Additive mode: push = push_base * boost_strength per feature per head
        conditions: dict[str, dict] = {"baseline": {}}
        for bs in additive_boost_strengths:
            bs_label = f"{bs:.0f}x" if bs == int(bs) else f"{bs}x"
            # Boundary boost: push_base[h][fi] * bs
            boundary_cond = {}
            for h, push_dict in boundary_push_per_head.items():
                boundary_cond[h] = {fi: pv * bs for fi, pv in push_dict.items()}
            conditions[f"boundary_boost_{bs_label}"] = boundary_cond

            # Random boost: push_base[h][fi] * bs
            random_cond = {}
            for h, push_dict in random_push_per_head.items():
                random_cond[h] = {fi: pv * bs for fi, pv in push_dict.items()}
            conditions[f"random_boost_{bs_label}"] = random_cond

        print(f"Additive mode with boost strengths: {additive_boost_strengths}")
        print(f"Conditions: {list(conditions.keys())}")
    else:
        def _build_updates(features_dict, scale):
            if scale == 1.0:
                return {}
            return {h: {f: scale for f in feats} for h, feats in features_dict.items() if feats}

        conditions = {
            "baseline": {},
            "boundary_boost": _build_updates(boundary_features_per_head, boost_scale),
            "random_boost": _build_updates(random_features_per_head, boost_scale),
        }

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
                additive=additive,
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
        base_paras = entry["baseline"]["stats"]["n_paragraphs"]
        # Find the first boundary boost condition for logging
        first_boundary = next((c for c in cond_names if c.startswith("boundary_boost")), None)
        first_random = next((c for c in cond_names if c.startswith("random_boost")), None)
        boost_paras = entry[first_boundary]["stats"]["n_paragraphs"] if first_boundary else 0
        rand_paras = entry[first_random]["stats"]["n_paragraphs"] if first_random else 0
        print(
            f"  [{i+1}/{len(formatted_prompts)}] {elapsed:.0f}s elapsed, "
            f"{remaining:.0f}s remaining | "
            f"paragraphs: baseline={base_paras:.0f} boundary={boost_paras:.0f} random={rand_paras:.0f}"
        )
        all_results.append(entry)

    total_time = time.time() - t0

    stat_names = list(compute_generation_stats("test text.").keys())
    condition_means: dict[str, dict[str, float]] = {}
    for cond_name in conditions:
        means = {}
        for stat in stat_names:
            vals = [r[cond_name]["stats"][stat] for r in all_results]
            means[stat] = float(np.mean(vals))
        condition_means[cond_name] = means

    result = {
        "experiment": "generation_intervention_qualitative_additive" if additive else "generation_intervention_qualitative",
        "layer": layer,
        "n_heads": n_loaded,
        "n_prompts": len(formatted_prompts),
        "n_tokens": n_tokens,
        "temperature": temperature,
        "sae_type": sae_type,
        "model_name": model_name,
        "use_chat_template": use_chat_template,
        "additive": additive,
        "total_time_s": total_time,
        "conditions": condition_means,
        "condition_names": cond_names,
        "boundary_features_per_head": {
            str(h): feats for h, feats in boundary_features_per_head.items()
        },
        "random_features_per_head": {
            str(h): feats for h, feats in random_features_per_head.items()
        },
        "feature_details_per_head": feature_details_per_head,
        "per_prompt": all_results,
    }
    if additive:
        result["additive_boost_strengths"] = list(additive_boost_strengths)
        result["intervention_mode"] = "additive"
        result["boundary_push_per_head"] = {
            str(h): {str(fi): v for fi, v in pushes.items()}
            for h, pushes in boundary_push_per_head.items()
        }
        result["random_push_per_head"] = {
            str(h): {str(fi): v for fi, v in pushes.items()}
            for h, pushes in random_push_per_head.items()
        }
    else:
        result["boost_scale"] = boost_scale

    # Save
    analysis_dir.mkdir(parents=True, exist_ok=True)
    add_suffix = "_additive" if additive else ""
    out_name = f"generation_intervention_qualitative{add_suffix}_L{layer}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*80}")
    mode_label = "ADDITIVE" if additive else "MULTIPLICATIVE"
    print(f"QUALITATIVE GENERATION DEMO ({mode_label}): L{layer}, {n_loaded} heads, {len(formatted_prompts)} prompts")
    print(f"{'='*80}")
    for cond_name in cond_names:
        m = condition_means[cond_name]
        print(f"\n  {cond_name:30s}: paragraphs={m['n_paragraphs']:.1f}  "
              f"newlines={m['n_newlines']:.1f}  sentences={m['n_sentences']:.1f}  "
              f"words={m['n_words']:.0f}")

    # Print 5 example comparisons
    print(f"\n{'='*80}")
    print("EXAMPLE COMPARISONS (first 5 prompts)")
    print(f"{'='*80}")
    for entry in all_results[:5]:
        print(f"\n--- Prompt {entry['prompt_idx']}: {entry['prompt_text']} ---")
        for cond_name in cond_names:
            text = entry[cond_name]["text"]
            paras = entry[cond_name]["stats"]["n_paragraphs"]
            norms = entry[cond_name]["mean_intervention_norm"]
            # Show first 500 chars with visible newline markers
            display = text[:500].replace("\n\n", "\n\n[PARA]\n\n").replace("\n", "\\n\n")
            print(f"\n  [{cond_name}] (paragraphs={paras:.0f}, norm={norms:.2f}):")
            print(f"  {display[:500]}")

    print(f"\nTotal time: {total_time:.0f}s")
    return result


# -- Stage: Single-feature steering demo (additive intervention on one feature at a time) --

SINGLE_FEATURE_DEMO_PROMPTS = [
    "Write a paragraph about the history of bridges.",
    "Explain how photosynthesis works.",
    "Describe the process of making bread from scratch.",
    "Write about what a visitor would see in Tokyo.",
    "Explain how a computer processor works.",
    "Describe the water cycle from ocean to rainfall.",
    "Write a paragraph about coral reef ecosystems.",
    "Explain why we have seasons on Earth.",
    "Describe how human memory works.",
    "Write about the construction of the Great Wall of China.",
]


@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def single_feature_demo_modal(
    layer: int = 9,
    head: int = 12,
    n_tokens: int = 400,
    n_features_target: int = 2048,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    temperature: float = 0.7,
    boost_strength: float = 2.0,
    sae_types: tuple[str, ...] = ("rank1", "bilinear", "bilinear_tied"),
    checkpoint_tag: str = "",
    n_top_features: int = 5,
) -> dict:
    """Single-feature additive steering demo.

    Loads the SAE checkpoint (matching the probe results), finds the top
    alive features by running probe-style correlation on extracted states,
    then runs 3 conditions (baseline, boost, suppress) per feature on 10
    instruction prompts.

    The additive push is calibrated from each feature's mean nonzero
    activation: push = mean_nonzero * boost_strength.

    This is the "Golden Gate Bridge" experiment for recurrent state features.
    """
    import json, os, sys, time
    from pathlib import Path
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

    corpus_source = "openwebtext"
    states_dir = _states_dir(corpus_source)
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]

    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer, head=head, n_features_target=n_features_target,
        corpus_source="openwebtext", preferred_types=sae_types,
        checkpoint_tag=checkpoint_tag or None,
    )
    print(f"SAE checkpoint: {resolved_tag} ({ckpt_path})")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae_model = build_sae_from_config(
        cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae_model.load_state_dict(ckpt["model_state_dict"])
    sae_model = sae_model.cuda().eval()
    sae_type = infer_sae_type(cfg, ckpt["model_state_dict"])
    print(f"SAE type: {sae_type}")

    # Compute activations on extracted states for feature selection + calibration
    head_states_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_states_path.exists():
        raise FileNotFoundError(f"No extracted states at {head_states_path}")

    head_states = np.load(str(head_states_path), mmap_mode="r")
    n_total = head_states.shape[0]
    sample_size = min(n_total, 10000)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, size=sample_size, replace=False)
    sample_states = np.array(head_states[sample_idx])

    # Encode sample through SAE
    sample_tensor = torch.tensor(sample_states, dtype=torch.float32, device="cuda")
    if sae_type == "flat":
        sample_tensor = sample_tensor.reshape(sample_tensor.shape[0], -1)

    all_coeffs = []
    with torch.no_grad():
        for start in range(0, len(sample_tensor), 512):
            batch = sample_tensor[start:start + 512]
            coeffs = sae_model.encode(batch)
            all_coeffs.append(coeffs.cpu())
    all_coeffs_cat = torch.cat(all_coeffs, dim=0)  # (sample_size, n_features)

    # Find alive features (>1% activation rate)
    alive_mask = (all_coeffs_cat > 0).float().mean(dim=0) > 0.01
    alive_indices = torch.nonzero(alive_mask, as_tuple=True)[0].tolist()
    print(f"\nAlive features (>1% activation rate): {len(alive_indices)}")

    # Load text properties for correlation-based feature selection
    # Use the corpus tokens to compute per-position text stats
    corpus_path = states_dir / "corpus.npy"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus at {corpus_path}. Run extraction first.")
    corpus_arr = np.load(str(corpus_path), mmap_mode="r")
    n_corpus_seqs = corpus_arr.shape[0]
    seq_len_corpus = corpus_arr.shape[1]

    # Compute simple per-position properties from token IDs at sampled positions
    # Map sample_idx back to (seq, pos) in the corpus
    n_use_seqs = min(n_total // seq_len_corpus, n_corpus_seqs)
    sample_seq_ids = sample_idx // seq_len_corpus
    sample_pos_ids = sample_idx % seq_len_corpus

    # Properties: is_period (token==13), is_newline (token==198 or 10)
    corpus_full = np.array(corpus_arr[:n_use_seqs])
    valid_mask = sample_seq_ids < n_use_seqs
    sample_tokens = np.zeros(sample_size, dtype=np.int64)
    for i in range(sample_size):
        if valid_mask[i]:
            sample_tokens[i] = corpus_full[sample_seq_ids[i], sample_pos_ids[i]]

    # Compute correlations between each alive feature and token properties
    from scipy import stats as sp_stats
    is_period = (sample_tokens == 13).astype(float)
    is_upper_start = np.zeros(sample_size)
    # Approximate "word entropy" proxy: use token ID variance in local window
    # For a simpler approach, correlate with period density as a format proxy

    # Select features with highest activation variance (most informative for steering)
    feature_stats = []
    for fi in alive_indices:
        acts = all_coeffs_cat[:, fi].numpy()
        mean_act = float(acts.mean())
        nonzero = acts[acts > 0]
        if len(nonzero) < 10:
            continue
        mean_nonzero = float(nonzero.mean())
        alive_frac = len(nonzero) / sample_size
        std_act = float(acts.std())
        # Correlate with period positions
        if valid_mask.sum() > 100:
            rho_period, p_period = sp_stats.spearmanr(acts[valid_mask], is_period[valid_mask])
        else:
            rho_period, p_period = 0.0, 1.0

        feature_stats.append({
            "feature_idx": fi,
            "mean": mean_act,
            "mean_nonzero": mean_nonzero,
            "alive_frac": alive_frac,
            "std": std_act,
            "rho_period": float(rho_period) if not np.isnan(rho_period) else 0.0,
            "p_period": float(p_period) if not np.isnan(p_period) else 1.0,
        })

    # Sort by activation variance (features that vary a lot are most steerable)
    feature_stats.sort(key=lambda x: abs(x["rho_period"]), reverse=True)

    # Take top features, ensuring a mix of positive and negative period correlations
    target_features = []
    feature_labels = {}
    pos_count = neg_count = 0
    for fs in feature_stats:
        if len(target_features) >= n_top_features:
            break
        fi = fs["feature_idx"]
        rho = fs["rho_period"]
        if rho > 0 and pos_count < (n_top_features + 1) // 2:
            target_features.append(fi)
            feature_labels[fi] = f"period_rho={rho:+.3f}, alive={fs['alive_frac']:.2f}"
            pos_count += 1
        elif rho < 0 and neg_count < (n_top_features + 1) // 2:
            target_features.append(fi)
            feature_labels[fi] = f"period_rho={rho:+.3f}, alive={fs['alive_frac']:.2f}"
            neg_count += 1
    # Fill remaining slots if needed
    for fs in feature_stats:
        if len(target_features) >= n_top_features:
            break
        if fs["feature_idx"] not in target_features:
            fi = fs["feature_idx"]
            target_features.append(fi)
            feature_labels[fi] = f"period_rho={fs['rho_period']:+.3f}, alive={fs['alive_frac']:.2f}"

    print(f"\nSelected {len(target_features)} features for steering:")
    # Also build a lookup for mean activations
    feat_stat_map = {fs["feature_idx"]: fs for fs in feature_stats}
    for fi in target_features:
        fs = feat_stat_map[fi]
        print(f"  F{fi}: {feature_labels[fi]}, mean_nonzero={fs['mean_nonzero']:.4f}")

    print(f"\nLoading model: {model_name}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, device="cuda")

    raw_prompts = list(SINGLE_FEATURE_DEMO_PROMPTS)
    formatted_prompts = []
    for p in raw_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = text.replace("<think>\n\n</think>\n\n", "")
        formatted_prompts.append(text)
    print(f"Formatted {len(formatted_prompts)} prompts with chat template")
    print(f"Example prompt:\n{formatted_prompts[0][:200]}")

    # Run generation for each feature x condition x prompt
    device = next(model.parameters()).device
    all_results = []

    for fi in target_features:
        fs = feat_stat_map[fi]
        mean_act = fs["mean"]
        mean_nonzero = fs["mean_nonzero"]
        # Calibrate push from mean nonzero activation
        push_base = max(abs(mean_nonzero), 0.01)
        push_value = push_base * boost_strength

        print(f"\n{'='*80}")
        print(f"Feature {fi}: {feature_labels[fi]}")
        print(f"  push_base={push_base:.4f}, push_value={push_value:.4f}")
        print(f"{'='*80}")

        feature_result = {
            "feature_idx": fi,
            "feature_label": feature_labels[fi],
            "mean_activation": mean_act,
            "mean_nonzero_activation": mean_nonzero,
            "push_value": push_value,
            "prompts": [],
        }

        conditions = {
            "baseline": {},
            "boost": {head: {fi: push_value}},
            "suppress": {head: {fi: -push_value}},
        }

        t_feat = time.time()
        for i, (raw_prompt, fmt_prompt) in enumerate(zip(raw_prompts, formatted_prompts)):
            prompt_ids = tokenizer(fmt_prompt, return_tensors="pt")["input_ids"].to(device)
            prompt_entry = {
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
                    sae_per_head={head: sae_model} if updates_per_head else {},
                    feature_updates_per_head=updates_per_head,
                    additive=True,
                )
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                gen_stats = compute_generation_stats(gen_text)
                prompt_entry[cond_name] = {
                    "text": gen_text,
                    "stats": gen_stats,
                    "n_generated": meta["n_generated"],
                    "mean_intervention_norm": meta["mean_intervention_norm"],
                }

            elapsed_feat = time.time() - t_feat
            base_words = prompt_entry["baseline"]["stats"]["n_words"]
            boost_words = prompt_entry["boost"]["stats"]["n_words"]
            supp_words = prompt_entry["suppress"]["stats"]["n_words"]
            print(f"  [{i+1}/{len(formatted_prompts)}] {elapsed_feat:.0f}s | "
                  f"words: base={base_words:.0f} boost={boost_words:.0f} supp={supp_words:.0f}")

            feature_result["prompts"].append(prompt_entry)

        all_results.append(feature_result)

    total_time = time.time() - t0

    result = {
        "experiment": "single_feature_demo",
        "layer": layer,
        "head": head,
        "n_prompts": len(formatted_prompts),
        "n_tokens": n_tokens,
        "n_features_tested": len(target_features),
        "boost_strength": boost_strength,
        "temperature": temperature,
        "sae_type": sae_type,
        "sae_checkpoint": resolved_tag,
        "model_name": model_name,
        "total_time_s": total_time,
        "features": all_results,
    }

    analysis_dir = _analysis_dir(corpus_source)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"single_feature_demo_L{layer}_H{head}.json"
    out_path = analysis_dir / out_name
    out_path.write_text(json.dumps(result, indent=2, default=str))
    vol.commit()
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*80}")
    print(f"SINGLE-FEATURE STEERING DEMO: L{layer}H{head}")
    print(f"SAE: {resolved_tag} ({sae_type})")
    print(f"{len(target_features)} features x {len(formatted_prompts)} prompts x 3 conditions")
    print(f"Total time: {total_time:.0f}s")
    print(f"{'='*80}")

    for feat_result in all_results:
        fi = feat_result["feature_idx"]
        label = feat_result["feature_label"]
        push = feat_result["push_value"]
        print(f"\n{'='*80}")
        print(f"Feature {fi}: {label} (push={push:.4f})")
        print(f"{'='*80}")

        for prompt_entry in feat_result["prompts"][:3]:
            print(f"\n--- Prompt {prompt_entry['prompt_idx']}: {prompt_entry['prompt_text']} ---")
            for cond_name in ["baseline", "boost", "suppress"]:
                text = prompt_entry[cond_name]["text"]
                norm = prompt_entry[cond_name]["mean_intervention_norm"]
                words = prompt_entry[cond_name]["stats"]["n_words"]
                print(f"\n  [{cond_name}] (words={words:.0f}, norm={norm:.2f}):")
                print(f"  {text[:500]}")

    return result


# -- Stage: TopK-disabled rank-1 control (legacy stage name: nonsparse-baseline) --

@app.function(volumes={DATA: vol}, image=image, timeout=14400, memory=32768, gpu="L4")
def nonsparse_baseline_modal(
    layer: int = 9,
    head: int = 0,
    n_features: int = 2048,
    epochs: int = 20,
    seed: int = 42,
    model_name: str = "Qwen/Qwen3.5-0.8B",
) -> dict:
    """Train a TopK-disabled, ReLU-only rank-1 control and run probing.

    Setting k=nf removes TopK competition but does not produce a fully dense
    model, because ReLU still gates negative pre-activations. This stage is a
    control for the effect of TopK selection, not a claim that all features are
    active on every example.
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from train import train
    from sae import build_sae_from_config
    from probe_features import probe_features, PROPERTY_NAMES

    vol.reload()
    t0 = time.time()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    n_total = meta["n_samples"]
    exp_tag = _experiment_tag(meta["model"], meta["seq_len"], meta["n_samples"])

    # Backward-compatible stage name, but the actual control is "TopK-disabled"
    # rather than fully non-sparse because ReLU still gates negative features.
    print(f"Training TopK-disabled rank1 control: nf={n_features}, k={n_features}")
    output_dir = f"{DATA}/checkpoints/{exp_tag}/nonsparse_rank1_L{layer}_H{head}_nf{n_features}_k{n_features}_s{seed}"

    train_result = train(
        sae_type="rank1",
        data_dir=str(states_dir),
        layer=layer, head=head,
        n_features=n_features, k=n_features,  # k=nf disables TopK competition
        lr=3e-4, batch_size=256, epochs=epochs,
        warmup_steps=50, resample_every=250,
        output_dir=output_dir, seed=seed,
    )
    print(f"Training done: MSE={train_result['best_mse']:.6f}, "
          f"dead={train_result['final_n_dead']}, time={train_result['total_time_s']:.0f}s")

    vol.commit()

    best_path = Path(output_dir) / "best.pt"
    config_path = Path(output_dir) / "config.json"
    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    cfg = json.loads(config_path.read_text())
    sae = build_sae_from_config(
        cfg, state_dict=ckpt["model_state_dict"],
        default_d_k=key_dim, default_d_v=val_dim,
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    state_data = np.lib.format.open_memmap(
        str(head_path), mode="r", dtype=np.float16,
        shape=(n_total, key_dim, val_dim),
    )
    states = torch.from_numpy(state_data[:n_total].astype(np.float32))

    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError("texts.json not found. Run --stage extract-texts first.")
    texts = json.loads(texts_path.read_text())[:n_total]

    n_use = min(states.shape[0], len(texts))
    states = states[:n_use]
    texts = texts[:n_use]

    print(f"\nRunning Spearman probing on TopK-disabled control ({n_use} samples)...")
    probe_result = probe_features(
        sae, states, texts,
        batch_size=512,
        min_frequency=0.01,
        correlation_threshold=0.15,
        p_threshold=0.01,
    )

    # Also load the sparse SAE probe results for comparison
    sparse_probe_path = Path(f"{DATA}/analysis") / f"probe_features_L{layer}_H{head}.json"
    sparse_comparison = None
    if sparse_probe_path.exists():
        sparse_data = json.loads(sparse_probe_path.read_text())
        sparse_probe = sparse_data.get("probe", {})
        sparse_comparison = {
            "sparse_n_interpretable": sparse_probe.get("n_interpretable", 0),
            "sparse_n_alive": sparse_probe.get("n_alive", 0),
            "sparse_interpretable_fraction": sparse_probe.get("interpretable_fraction", 0),
            "sparse_sae_tag": sparse_data.get("sae_tag", ""),
        }

    output = {
        "experiment": "topk_disabled_rank1_control",
        "layer": layer,
        "head": head,
        "n_features": n_features,
        "k": n_features,
        "seed": seed,
        "train_mse": train_result["best_mse"],
        "train_dead": train_result["final_n_dead"],
        "train_time_s": train_result["total_time_s"],
        "probe": {
            "n_alive": probe_result["n_alive"],
            "n_interpretable": probe_result["n_interpretable"],
            "n_interpretable_bonferroni": probe_result["n_interpretable_bonferroni"],
            "interpretable_fraction": probe_result["interpretable_fraction"],
            "interpretable_fraction_bonferroni": probe_result["interpretable_fraction_bonferroni"],
            "property_summary": probe_result["property_summary"],
        },
        "sparse_comparison": sparse_comparison,
        "total_time_s": round(time.time() - t0, 1),
    }

    print(f"\n{'='*60}")
    print(f"TopK-Disabled ReLU-Only Control (k={n_features})")
    print(f"{'='*60}")
    print(f"  Control: {probe_result['n_interpretable']}/{probe_result['n_alive']} "
          f"interpretable ({probe_result['interpretable_fraction']*100:.1f}%)")
    if sparse_comparison:
        print(f"  Sparse (k=32): {sparse_comparison['sparse_n_interpretable']}/"
              f"{sparse_comparison['sparse_n_alive']} interpretable "
              f"({sparse_comparison['sparse_interpretable_fraction']*100:.1f}%)")
        if probe_result["interpretable_fraction"] < sparse_comparison["sparse_interpretable_fraction"]:
            print(f"  --> Sparsity improves interpretability by "
                  f"{(sparse_comparison['sparse_interpretable_fraction'] - probe_result['interpretable_fraction'])*100:.1f} "
                  f"percentage points")
        else:
            print(f"  --> The TopK-disabled control has equal or higher interpretability")

    # Save
    out_dir = Path(f"{DATA}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nonsparse_baseline_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(output, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")

    return output


# -- Stage: Probing stability across subsets (Q1) --

@app.function(volumes={DATA: vol}, image=image, timeout=21600, memory=32768, gpu="L4")
def probe_stability_modal(
    layer: int = 9,
    head: int = 0,
    n_features: int = 2048,
    k: int = 32,
    n_subsets: int = 3,
    subset_size: int = 5000,
) -> dict:
    """Test probing stability with explicit metadata about text overlap.

    Trains one rank1 SAE on states [0:5000] from the 50K pool.
    Then probes that SAME SAE on the overlapping decoded slice first, followed by
    held-out slices when decoded texts exist for them.
    """
    import json, sys, time
    from pathlib import Path
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from train import train
    from sae import build_sae_from_config
    from probe_features import probe_features

    vol.reload()
    t0 = time.time()

    states_dir = Path(f"{DATA}/states")
    meta_path = states_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Run extraction first.")
    meta = json.loads(meta_path.read_text())
    key_dim, val_dim = meta["key_head_dim"], meta["value_head_dim"]
    n_total = meta["n_samples"]

    texts_path = states_dir / "texts.json"
    if not texts_path.exists():
        raise FileNotFoundError("texts.json not found. Run --stage extract-texts first.")
    all_texts = json.loads(texts_path.read_text())
    n_texts = len(all_texts)
    print(f"Loaded {n_texts} texts, {n_total} states")

    head_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    if not head_path.exists():
        raise FileNotFoundError(f"No state data at {head_path}")
    all_states_mm = np.load(str(head_path), mmap_mode="r")

    # Step 1: Train SAE on first 5K (train split)
    train_size = subset_size
    print(f"Training rank1 SAE on first {train_size} states...")
    tmp_dir = Path(f"/tmp/probe_stability_train/layer_{layer}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_states = np.array(all_states_mm[:train_size], dtype=np.float32)
    np.save(str(tmp_dir / f"head_{head}.npy"), train_states)

    output_dir = "/tmp/probe_stability_ckpts/train_sae"
    train_result = train(
        sae_type="rank1",
        data_dir="/tmp/probe_stability_train",
        layer=layer, head=head,
        n_features=n_features, k=k,
        lr=3e-4, batch_size=256, epochs=20,
        warmup_steps=50, resample_every=250,
        output_dir=output_dir, seed=42,
    )
    print(f"  Train SAE: MSE={train_result['best_mse']:.6f}, dead={train_result['final_n_dead']}")

    best_path = Path(output_dir) / "best.pt"
    cfg_path = Path(output_dir) / "config.json"
    ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
    cfg = json.loads(cfg_path.read_text())
    sae = build_sae_from_config(cfg, state_dict=ckpt["model_state_dict"],
                                default_d_k=key_dim, default_d_v=val_dim)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.cuda().eval()

    # Step 2: Probe on train split first (should match original probing)
    # Then probe on held-out subsets
    subset_results = []
    probe_slices = [
        (0, "same-split-overlap", 0, min(train_size, n_texts)),
    ]
    for si in range(n_subsets):
        start = train_size + si * subset_size
        end = start + subset_size
        n_use = min(subset_size, n_texts - start, n_total - start)
        if n_use > 0:
            probe_slices.append((si + 1, f"held-out-{si}", start, start + n_use))

    for idx, label, start, end in probe_slices:
        n_use = end - start
        split_type = "same-split-overlap" if idx == 0 else "held-out"
        print(f"\n{'='*60}")
        print(f"Probe {label}: states [{start}:{end}] ({n_use} samples), SAME SAE")
        print(f"{'='*60}")

        subset_states = np.array(all_states_mm[start:end], dtype=np.float32)
        subset_texts = all_texts[start:end]

        states_tensor = torch.from_numpy(subset_states)
        print(f"Running probing on {n_use} samples...")
        probe_result = probe_features(
            sae, states_tensor, subset_texts,
            batch_size=512,
            min_frequency=0.01,
            correlation_threshold=0.15,
            p_threshold=0.01,
        )

        # Extract top properties by number of correlated features
        prop_summary = probe_result.get("property_summary", {})
        top_props = sorted(
            prop_summary.items(),
            key=lambda x: x[1].get("n_correlated_features", 0),
            reverse=True,
        )
        top_10_props = [p[0] for p in top_props[:10] if p[1]["n_correlated_features"] > 0]

        subset_results.append({
            "subset_idx": idx,
            "label": label,
            "split_type": split_type,
            "start": start,
            "end": end,
            "n_samples": n_use,
            "train_mse": train_result["best_mse"],
            "train_dead": train_result["final_n_dead"],
            "n_alive": probe_result["n_alive"],
            "n_interpretable": probe_result["n_interpretable"],
            "interpretable_fraction": probe_result["interpretable_fraction"],
            "top_10_properties": top_10_props,
            "property_feature_counts": {
                p: s["n_correlated_features"]
                for p, s in prop_summary.items()
                if s["n_correlated_features"] > 0
            },
        })

        print(f"  Interpretable: {probe_result['n_interpretable']}/{probe_result['n_alive']} "
              f"({probe_result['interpretable_fraction']*100:.1f}%)")
        print(f"  Top properties: {top_10_props[:5]}")

        torch.cuda.empty_cache()

    del sae, ckpt
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Cross-Subset Comparison")
    print(f"{'='*60}")

    # Jaccard similarity of top-10 properties between all pairs
    jaccard_scores = []
    for i in range(len(subset_results)):
        for j in range(i + 1, len(subset_results)):
            set_i = set(subset_results[i]["top_10_properties"])
            set_j = set(subset_results[j]["top_10_properties"])
            if set_i or set_j:
                jaccard = len(set_i & set_j) / len(set_i | set_j)
            else:
                jaccard = 0.0
            jaccard_scores.append({
                "subset_a": subset_results[i]["subset_idx"],
                "subset_b": subset_results[j]["subset_idx"],
                "jaccard": round(jaccard, 3),
                "intersection": sorted(set_i & set_j),
                "union_size": len(set_i | set_j),
            })
            print(f"  Subsets {i} vs {j}: Jaccard={jaccard:.3f}, "
                  f"shared={sorted(set_i & set_j)}")

    # Properties that appear in ALL subsets' top-10
    universal_props: list[str] = []
    if subset_results:
        all_top_sets = [set(sr["top_10_properties"]) for sr in subset_results]
        universal_props = sorted(set.intersection(*all_top_sets)) if all_top_sets else []
        print(f"\n  Properties in ALL subsets' top-10: {universal_props}")

    # Mean interpretable fraction across subsets
    fracs = [sr["interpretable_fraction"] for sr in subset_results]

    output = {
        "experiment": "probe_stability",
        "layer": layer,
        "head": head,
        "n_features": n_features,
        "k": k,
        "n_subsets": len(subset_results),
        "n_states_available": n_total,
        "n_texts_available": n_texts,
        "held_out_subsets_requested": n_subsets,
        "held_out_subsets_available": max(len(subset_results) - 1, 0),
        "train_slice": {"start": 0, "end": train_size},
        "decoded_train_overlap": n_texts > 0,
        "subset_size": subset_size,
        "subset_results": subset_results,
        "jaccard_scores": jaccard_scores,
        "mean_jaccard": round(float(np.mean([j["jaccard"] for j in jaccard_scores])), 3) if jaccard_scores else 0.0,
        "universal_top_properties": universal_props if subset_results else [],
        "mean_interpretable_fraction": round(float(np.mean(fracs)), 3) if fracs else 0.0,
        "std_interpretable_fraction": round(float(np.std(fracs)), 3) if fracs else 0.0,
        "interpretation_note": (
            "Held-out generalization is only testable for decoded text slices beyond the "
            "training window. When decoded texts cover only the training slice, this stage "
            "reduces to a same-split stability check."
        ),
        "total_time_s": round(time.time() - t0, 1),
    }

    # Save
    out_dir = Path(f"{DATA}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_stability_L{layer}_H{head}.json"
    out_path.write_text(json.dumps(output, indent=2))
    vol.commit()
    print(f"\nSaved to {out_path}")
    print(f"Total time: {output['total_time_s']:.0f}s")

    return output


# -- Scaling Diagnosis Experiments (E1, E2, E3) --

@app.function(volumes={DATA: vol}, image=image, timeout=7200, memory=32768)
def data_diversity_modal(layer: int = 9, head: int = 0):
    """E2: Compute data diversity metrics comparing 5K vs 50K state distributions.

    Returns pairwise Frobenius distance stats, effective rank of sample covariance,
    and singular value distributions for both data scales.
    """
    import json, time
    from pathlib import Path
    import numpy as np

    vol.reload()
    t0 = time.time()

    states_dir_50k = Path(f"{DATA}/states/layer_{layer}")
    data_path = states_dir_50k / f"head_{head}.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"No 50K states at {data_path}")

    all_states = np.load(str(data_path), mmap_mode="r")
    n_total = all_states.shape[0]
    d_k, d_v = all_states.shape[1], all_states.shape[2]
    print(f"Loaded states: {all_states.shape} from {data_path}")

    results = {}
    for label, n_use in [("5k", 5000), ("50k", min(50000, n_total))]:
        states = np.array(all_states[:n_use], dtype=np.float32)
        flat = states.reshape(n_use, -1)  # (n, d_k*d_v)

        # Effective rank of sample covariance
        cov = np.cov(flat, rowvar=False)  # (d, d)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        p = eigvals / eigvals.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

        # Pairwise Frobenius distance (subsample for speed)
        rng = np.random.RandomState(42)
        n_pairs = min(2000, n_use)
        idx_a = rng.choice(n_use, size=n_pairs, replace=True)
        idx_b = rng.choice(n_use, size=n_pairs, replace=True)
        diffs = flat[idx_a] - flat[idx_b]
        frob_dists = np.sqrt((diffs ** 2).sum(axis=1))

        # Singular values of the data matrix
        n_svd = min(1000, n_use)
        U, S, Vt = np.linalg.svd(flat[:n_svd], full_matrices=False)
        top_svs = S[:20].tolist()

        # Per-sample Frobenius norms
        norms = np.sqrt((flat ** 2).sum(axis=1))

        results[label] = {
            "n_samples": n_use,
            "eff_rank_covariance": round(eff_rank, 2),
            "mean_frob_distance": round(float(frob_dists.mean()), 6),
            "std_frob_distance": round(float(frob_dists.std()), 6),
            "median_frob_distance": round(float(np.median(frob_dists)), 6),
            "mean_sample_norm": round(float(norms.mean()), 6),
            "std_sample_norm": round(float(norms.std()), 6),
            "top_20_singular_values": top_svs,
            "sv1_sv2_ratio": round(top_svs[0] / max(top_svs[1], 1e-12), 3),
        }
        print(f"{label}: eff_rank={eff_rank:.2f}, mean_frob_dist={frob_dists.mean():.6f}, "
              f"sv1/sv2={top_svs[0]/max(top_svs[1],1e-12):.3f}")

    results["diversity_ratio"] = {
        "eff_rank_ratio": round(results["50k"]["eff_rank_covariance"] / max(results["5k"]["eff_rank_covariance"], 1e-12), 3),
        "frob_dist_ratio": round(results["50k"]["mean_frob_distance"] / max(results["5k"]["mean_frob_distance"], 1e-12), 3),
        "norm_ratio": round(results["50k"]["mean_sample_norm"] / max(results["5k"]["mean_sample_norm"], 1e-12), 3),
    }
    results["wall_time_s"] = round(time.time() - t0, 1)
    return results


@app.function(volumes={DATA: vol}, image=image, timeout=7200, memory=32768, gpu="L4")
def train_on_subset(
    layer: int,
    head: int,
    subset_idx: int,
    subset_start: int,
    subset_end: int,
    sae_type: str,
    seed: int,
    n_features: int = 2048,
    k: int = 32,
):
    """E1 worker: Train a single SAE on a subset slice of the 50K pool.

    Uses validation-based dead counts (features that never fire on any val sample)
    instead of training-based steps_since_active, which has metric artifacts.
    """
    import json, os, sys, time
    from pathlib import Path
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/root")
    from train import train, GDNStateDataset, evaluate
    from split_utils import make_train_val_subsets
    from sae import build_sae_from_config

    vol.reload()

    data_path = Path(f"{DATA}/states/layer_{layer}/head_{head}.npy")
    if not data_path.exists():
        raise FileNotFoundError(f"No states at {data_path}")

    all_states = np.load(str(data_path), mmap_mode="r")
    subset_data = np.array(all_states[subset_start:subset_end], dtype=np.float32)

    # Write subset to temp dir
    tmp_dir = Path(f"/tmp/subset_{subset_idx}/layer_{layer}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(tmp_dir / f"head_{head}.npy"), subset_data)

    output_dir = f"/tmp/subset_ckpts/s{subset_idx}_{sae_type}_s{seed}"
    result = train(
        sae_type=sae_type,
        data_dir=f"/tmp/subset_{subset_idx}",
        layer=layer, head=head,
        n_features=n_features, k=k,
        lr=3e-4, batch_size=256, epochs=20,
        warmup_steps=50, resample_every=250,
        output_dir=output_dir, seed=seed,
    )

    # Compute validation-based dead count from best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = os.path.join(output_dir, "best.pt")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = build_sae_from_config(ckpt.get("config"), state_dict=ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    dataset = GDNStateDataset(str(tmp_dir / f"head_{head}.npy"))
    _, val_set = make_train_val_subsets(dataset, train_fraction=0.8, seed=42)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    is_flat = sae_type == "flat"
    val_metrics = evaluate(model, val_loader, device, is_flat)
    val_dead = int(val_metrics["dead"])

    train_dead = int(cast(int | float, result["final_n_dead"]))

    return {
        "subset_idx": subset_idx, "start": subset_start, "end": subset_end,
        "sae_type": sae_type, "seed": seed, "n_features": n_features,
        "best_mse": result["best_mse"], "final_mse": result["final_mse"],
        "val_dead": val_dead,
        "val_dead_pct": round(100 * val_dead / n_features, 1),
        "train_dead": train_dead,
        "train_dead_pct": round(100 * train_dead / n_features, 1),
        "dead_pct": round(100 * val_dead / n_features, 1),
        "total_time_s": result["total_time_s"],
    }


@app.function(volumes={DATA: vol}, image=image, timeout=21600, memory=32768, gpu="L4")
def train_budget_single(
    layer: int,
    head: int,
    sae_type: str,
    seed: int,
    n_features: int = 2048,
    k: int = 32,
    epochs: int = 200,
):
    """E3 worker: Train a single SAE on full 50K with extended epochs.

    Uses validation-based dead counts for consistency with paper metrics.
    """
    import os, sys, time
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/root")
    from train import train, GDNStateDataset, evaluate
    from split_utils import make_train_val_subsets
    from sae import build_sae_from_config

    vol.reload()
    states_dir = Path(f"{DATA}/states")

    output_dir = f"/tmp/budget_control/{sae_type}_s{seed}"
    result = train(
        sae_type=sae_type,
        data_dir=str(states_dir),
        layer=layer, head=head,
        n_features=n_features, k=k,
        lr=3e-4, batch_size=256, epochs=epochs,
        warmup_steps=50, resample_every=250,
        output_dir=output_dir, seed=seed,
    )

    # Compute validation-based dead count from best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = os.path.join(output_dir, "best.pt")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model = build_sae_from_config(ckpt.get("config"), state_dict=ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    data_path = states_dir / f"layer_{layer}" / f"head_{head}.npy"
    dataset = GDNStateDataset(str(data_path))
    _, val_set = make_train_val_subsets(dataset, train_fraction=0.8, seed=42)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    is_flat = sae_type == "flat"
    val_metrics = evaluate(model, val_loader, device, is_flat)
    val_dead = int(val_metrics["dead"])

    return {
        "sae_type": sae_type, "seed": seed, "epochs": epochs,
        "n_features": n_features,
        "best_mse": result["best_mse"], "final_mse": result["final_mse"],
        "val_dead": val_dead,
        "val_dead_pct": round(100 * val_dead / n_features, 1),
        "train_dead": result["final_n_dead"],
        "dead_pct": round(100 * val_dead / n_features, 1),
        "n_samples": result["n_samples"],
        "total_time_s": result["total_time_s"],
    }


def _build_scaling_curve_jobs(
    layer: int = 9,
    head: int = 0,
    sizes: list[int] | None = None,
    n_features: int = 2048,
    k: int = 32,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Build job configs for the data-efficiency scaling curve.

    Trains bilinear/flat/rank1 at each data size from the 50K pool.
    Each job uses train_on_subset with subset_idx=0, start=0, end=size.
    """
    if sizes is None:
        sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    if seeds is None:
        seeds = [0, 1, 2]
    sae_types = ["flat", "rank1", "bilinear"]
    jobs = []
    for size in sizes:
        for sae_type in sae_types:
            for seed in seeds:
                jobs.append(dict(
                    layer=layer, head=head,
                    subset_idx=0, subset_start=0, subset_end=size,
                    sae_type=sae_type, seed=seed,
                    n_features=n_features, k=k,
                ))
    return jobs


# -- Stage: Memory Slot Surgery alignment check --

@app.function(volumes={DATA: vol, MODELS: model_vol}, **GPU_KWARGS)
def memory_alignment_modal(
    layer: int = 9,
    head: int = 4,
    n_seqs: int = 100,
    seq_len: int = 512,
    batch_size: int = 4,
    top_n: int = 50,
    n_features_target: int = 2048,
    checkpoint_tag: str = "",
    model_name: str = "Qwen/Qwen3.5-0.8B",
) -> dict:
    """Check alignment between SAE decoder atoms and GDN k*v^T writes."""
    import json, os, sys, time
    from pathlib import Path

    os.environ["HF_HOME"] = f"{MODELS}/hf_cache"
    sys.path.insert(0, "/root")

    import torch
    from memory_alignment import load_model, load_sae, compute_alignment, report

    vol.reload()

    canonical_tag = checkpoint_tag.strip() or f"bilinear_L{layer}_H{head}_nf{n_features_target}_k32_s42"
    ckpt_path, cfg, resolved_tag = _resolve_sae_checkpoint(
        layer=layer,
        head=head,
        n_features_target=n_features_target,
        checkpoint_tag=canonical_tag,
        corpus_source="openwebtext",
        preferred_types=("bilinear", "bilinear_tied"),
    )
    print(f"Using checkpoint: {resolved_tag} ({ckpt_path})")

    model, tokenizer = load_model(model_name, "cuda")
    sae = load_sae(str(ckpt_path), "cuda")

    result = compute_alignment(
        model, tokenizer, sae,
        layer_idx=layer, head_idx=head,
        n_seqs=n_seqs, seq_len=seq_len,
        batch_size=batch_size, top_n=top_n,
        device="cuda",
    )

    report(result)

    out_dir = Path(DATA) / "memory_alignment" / f"L{layer}_H{head}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "alignment_results.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    return result


@app.local_entrypoint()
def main(
    stage: str = "all",
    n_samples: int = 10000,
    n_random: int = 50,
    layers: str = "1,9,17",
    seq_len: int = 1024,
    model: str = "Qwen/Qwen3.5-0.8B",
    sae_type: str = "rank1",
    layer: int = 12,
    head: int = 0,
    expansion_factor: int = 1,
    k: int = 32,
    epochs: int = 20,
    n_train_examples: int = 4,
    s0_steps: int = 20,
    rank: int = 1,
    n_features: int = 0,
    max_alive_features: int = 0,
    seed: int = 42,
    corpus_source: str = "openwebtext",
    family: str = "document_format",
    feature_indices: str = "",
    checkpoint_tag: str = "",
):
    import time
    layer_list = [int(x.strip()) for x in layers.split(",")]
    corpus_source = _normalize_corpus_source(corpus_source)

    t0 = time.time()
    print(f"Matrix SAE pipeline: stage={stage}")
    print(
        f"  model={model}, n_samples={n_samples}, layers={layer_list}, "
        f"seq_len={seq_len}, corpus_source={corpus_source}"
    )
    print(
        "  modal:"
        f" app={_modal_name('MATRIX_SAE_MODAL_APP_NAME', 'matrix-sae')},"
        f" data_volume={_modal_name('MATRIX_SAE_MODAL_DATA_VOLUME', 'matrix-sae-data')},"
        f" model_volume={_modal_name('MATRIX_SAE_MODAL_MODEL_VOLUME', 'hf-model-cache')}"
    )

    if stage == "extract":
        allow_overwrite = os.environ.get("MATRIX_SAE_ALLOW_STATE_OVERWRITE", "").strip() == "1"
        extract_fn = extract_a10g if _model_is_large(model) else extract
        result = extract_fn.remote(
            n_samples=n_samples,
            layers=layer_list,
            seq_len=seq_len,
            model_name=model,
            corpus_source=corpus_source,
            allow_overwrite=allow_overwrite,
        )
        print(f"Result: {result}")
    elif stage == "train":
        train_fn = train_sae_a100 if rank >= 4 else train_sae
        train_batch_size = 64 if rank >= 4 else 256
        nf = n_features if n_features > 0 else None
        result = train_fn.remote(
            sae_type=sae_type, layer=layer, head=head,
            expansion_factor=expansion_factor, k=k, epochs=epochs,
            rank=rank, batch_size=train_batch_size, n_features=nf, seed=seed,
            corpus_source=corpus_source,
        )
        print(f"Result: mse={result['best_mse']:.6f} dead={result['final_n_dead']} [{result['total_time_s']:.0f}s]")
    elif stage == "sweep":
        results = train_sweep.remote(layers=layer_list, corpus_source=corpus_source)
        print(f"{len(results)} configs trained.")
        for r in results:
            print(f"  {r['sae_type']} L{r['layer']} ef={r['expansion_factor']}: mse={r['best_mse']:.6f}")
    elif stage == "nf-sweep":
        results = train_nf_sweep.remote(layers=layer_list, corpus_source=corpus_source)
        print(f"{len(results)} nf-sweep configs trained.")
        for r in results:
            print(f"  {r['sae_type']} L{r['layer']} nf={r['n_features']} k={r['k']}: mse={r['best_mse']:.6f} dead={r['final_n_dead']}")
    elif stage == "batchtopk-sweep":
        nf = n_features if n_features > 0 else 2048
        results = train_batchtopk_sweep.remote(
            layers=layer_list,
            n_features=nf,
            k=k,
            corpus_source=corpus_source,
        )
        print(f"{len(results)} batchtopk-sweep configs trained.")
        for r in results:
            print(f"  btk_{r['sae_type']} L{r['layer']} nf={r.get('n_features', '?')} k={r.get('k', '?')}: "
                  f"mse={r['best_mse']:.6f} dead={r['final_n_dead']}")
    elif stage == "analyze":
        analyze.remote(corpus_source=corpus_source)
        print("Analysis complete.")
    elif stage == "extract-texts":
        result = extract_texts.remote(
            n_samples=n_samples,
            seq_len=seq_len,
            model_name=model,
            corpus_source=corpus_source,
        )
        print(f"Texts extracted: {result}")
    elif stage == "interpret":
        result = interpret.remote(layers=layer_list, corpus_source=corpus_source)
        print(f"Interpretability analysis: {result}")
    elif stage == "interpret-s0":
        result = interpret_s0.remote()
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "s0_interpret_summary.json"), "w") as f:
            json.dump(result, f, indent=2)
        n_ckpts = len(result) if result else 0
        n_feats = sum(len(v) for v in result.values()) if result else 0
        print(f"S0-targeted interpret: {n_ckpts} checkpoints, {n_feats} features total")
        print(f"Results saved to results/data/s0_interpret_summary.json")
    elif stage == "s0":
        result = s0_decompose.remote(
            model_name=model, n_train_examples=n_train_examples, s0_steps=s0_steps)
        print(f"S0 decomposition complete: {len(result.get('decompositions', {}))} SAE checkpoints analyzed")
    elif stage == "s0-shift":
        result = s0_shift.remote(
            task="gsm8k", layer=layer, head=head,
            n_sequences=n_samples, n_train_examples=n_train_examples,
            s0_steps=s0_steps, model_name=model,
            nf_target=n_features if n_features > 0 else 2048,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "s0_shift_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        summary = result.get("summary_by_type", {})
        print(f"\nS0 Activation Shift: {len(result.get('sae_results', {}))} checkpoints analyzed")
        for sae_type, s in summary.items():
            print(f"  {sae_type}: sig_shift={s['mean_significant_shift']:.1f} "
                  f"freq>5%={s['mean_freq_shift_5pct']:.1f} "
                  f"alive={s['mean_alive_base']:.0f}->{s['mean_alive_s0']:.0f}")
        print(f"Results saved to results/data/s0_shift_results.json")
    elif stage == "plot-curve":
        results = plot_curve.remote()
        # also generate local copy
        import importlib.util as _importlib_util
        import matplotlib

        matplotlib.use("Agg")
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "visualize.py")
        spec = _importlib_util.spec_from_file_location("visualize", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plotting script at {script_path}")
        visualize_module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(visualize_module)
        plot_effective_rank_curve = visualize_module.plot_effective_rank_curve
        local_fig_out = os.path.join(os.path.dirname(__file__), "results", "legacy_figures")
        local_data_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_fig_out, exist_ok=True)
        os.makedirs(local_data_out, exist_ok=True)
        plot_effective_rank_curve(results, save_path=os.path.join(local_fig_out, "fig_effective_rank_curve.pdf"))
        plot_effective_rank_curve(results, save_path=os.path.join(local_fig_out, "fig_effective_rank_curve.png"))
        with open(os.path.join(local_data_out, "analysis_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Local results saved to results/data/ and results/legacy_figures/")
    elif stage == "download":
        bundle = download_results.remote()
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "full_bundle.json"), "w") as f:
            json.dump(bundle, f, indent=2)
        # also save individual files for convenience
        for key in ["metadata", "analysis_results"]:
            if key in bundle:
                with open(os.path.join(local_out, f"{key}.json"), "w") as f:
                    json.dump(bundle[key], f, indent=2)
        if bundle.get("interpret_summaries"):
            with open(os.path.join(local_out, "interpret_summaries.json"), "w") as f:
                json.dump(bundle["interpret_summaries"], f, indent=2)
        if bundle.get("s0_results"):
            with open(os.path.join(local_out, "s0_results.json"), "w") as f:
                json.dump(bundle["s0_results"], f, indent=2)
        print(f"All results saved to results/data/")
    elif stage == "null-distribution":
        result = null_distribution_test.remote()
        print(f"Null distribution of max |score| across {result['n_features']} atoms ({result['n_trials']} trials):")
        for k, v in result["null_max_abs_score"].items():
            print(f"  {k}: {v:.2f}")
        import json as _json
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "null_distribution.json"), "w") as f:
            _json.dump(result, f, indent=2)
        print(f"Saved to results/data/null_distribution.json")
    elif stage == "s0-proper-null":
        result = s0_proper_null.remote(task="gsm8k", n_trials=1000, head=head)
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "s0_proper_null.json"), "w") as f:
            json.dump(result, f, indent=2)
        summary = result.get("summary", {})
        print(f"\nS0 Proper Null Test: {summary.get('n_checkpoints', 0)} checkpoints, "
              f"{summary.get('n_significant_p01', 0)} significant at p<0.01")
        for stype, ts in sorted(result.get("type_summaries", {}).items()):
            print(f"  {stype}: {ts['n_significant_p01']}/{ts['n_checkpoints']} sig, "
                  f"mean_z={ts['mean_z_score']:.2f}")
        print(f"Results saved to results/data/s0_proper_null.json")
    elif stage == "baselines":
        result = run_baselines.remote(layers=layer_list)
        print(f"Baselines complete: {len(result)} layers")
        for layer_key, layer_result in result.items():
            pca_32 = layer_result["pca"].get("32", "N/A")
            print(f"  {layer_key}: PCA-32 MSE={pca_32}")
    elif stage == "evaluate":
        # Evaluate downstream perplexity impact for each layer
        for l in layer_list:
            result = evaluate_downstream_modal.remote(
                layer=l, n_sequences=n_samples, seq_len=seq_len,
                model_name=model, head=head,
                corpus_source=corpus_source,
            )
            if "error" in result:
                print(f"Layer {l}: {result['error']}")
            else:
                from evaluate_downstream import format_results_table
                print(format_results_table(result))
                print()
    elif stage == "evaluate-allheads":
        # Evaluate downstream perplexity with ALL heads reconstructed through each SAE
        for l in layer_list:
            result = evaluate_downstream_allheads_modal.remote(
                layer=l, n_sequences=n_samples, seq_len=seq_len,
                model_name=model, head=head,
                corpus_source=corpus_source,
            )
            if "error" in result:
                print(f"Layer {l}: {result['error']}")
            else:
                from evaluate_downstream import format_results_table
                print(format_results_table(result))
                print()
    elif stage == "train-allheads":
        nf = n_features if n_features > 0 else 2048
        # Pass sae_types if specified via --checkpoint-tag (overloaded: "flat", "rank1", "bilinear")
        sae_types_arg = [checkpoint_tag] if checkpoint_tag in ("flat", "rank1", "bilinear") else None
        # Detect n_heads from metadata to support 4B (32 heads) vs 0.8B (16 heads)
        from pathlib import Path
        _meta_path = Path(
            f"{DATA}/states{'_' + _corpus_slug(corpus_source) if _corpus_slug(corpus_source) != 'openwebtext' else ''}/metadata.json"
        )
        _n_heads_default = 16
        if model and "4B" in model:
            _n_heads_default = 32
        results = train_perhead_sweep.remote(
            layer=layer, n_features=nf, k=k, seeds=[seed],
            sae_types=sae_types_arg,
            heads=list(range(_n_heads_default)),
            corpus_source=corpus_source,
        )
        print(f"{len(results)} per-head SAEs trained.")
        for r in results:
            print(f"  {r['sae_type']} L{r['layer']} H{r['head']}: mse={r['best_mse']:.6f} dead={r['final_n_dead']}")
    elif stage == "evaluate-perhead":
        nf = n_features if n_features > 0 else 2048
        for l in layer_list:
            result = evaluate_downstream_perhead_matched_modal.remote(
                layer=l, n_sequences=n_samples, seq_len=seq_len,
                model_name=model, n_features=nf, k=k, seed=seed,
                corpus_source=corpus_source,
            )
            if "error" in result:
                print(f"Layer {l}: {result['error']}")
            else:
                from evaluate_downstream import format_results_table
                print(format_results_table(result))
                print()
    elif stage == "temporal":
        # Phase 1: extract states at multiple positions
        extract_result = extract_temporal.remote(
            layer=layer, head=head, n_samples=n_samples,
            model_name=model,
        )
        print(f"Temporal extraction: {extract_result['n_samples']} samples, "
              f"{len(extract_result['positions'])} positions in "
              f"{extract_result['extraction_time_s']:.1f}s")
        # Phase 2: encode through SAEs and analyze
        nf = n_features if n_features > 0 else 2048
        analysis_result = analyze_temporal.remote(
            layer=layer, head=head, n_features_target=nf,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "temporal_analysis.json"), "w") as f:
            json.dump(analysis_result, f, indent=2)
        n_saes = len(analysis_result.get("sae_results", {}))
        print(f"Temporal analysis: {n_saes} SAEs analyzed")
        for tag, entry in analysis_result.get("sae_results", {}).items():
            cats = entry.get("category_counts", {})
            print(f"  {tag}: alive={entry.get('n_alive', 0)} categories={cats}")
        print(f"Results saved to results/data/temporal_analysis.json")
    elif stage == "feature-quality":
        nf = n_features if n_features > 0 else 2048
        result = analyze_feature_quality.remote(
            layer=layer, n_features_target=nf, k=k, head=head,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "feature_quality.json"), "w") as f:
            json.dump(result, f, indent=2)
        summary = result.get("summary_by_type", {})
        print(f"\nFeature Quality: {len(result.get('sae_results', {}))} checkpoints analyzed")
        for sae_type, s in sorted(summary.items()):
            print(f"  {sae_type} (n={s['n_seeds']}): "
                  f"MSE={s['mse_mean']:.5f} p95={s['mse_p95']:.5f} "
                  f"alive={s['n_alive']:.0f} dead={s['dead_pct']:.1f}% "
                  f"freq={s['mean_freq']:.4f} gini={s['gini']:.3f}")
        print(f"Results saved to results/data/feature_quality.json")
    elif stage == "probe-features":
        result = probe_features_modal.remote(
            layer=layer, head=head, n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        probe_suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        probe_fname = f"probe_features_L{layer}_H{head}{probe_suffix}.json"
        with open(os.path.join(local_out, probe_fname), "w") as f:
            json.dump(result, f, indent=2)
        probe = result.get("probe", {})
        union = result.get("union", {})
        print(f"\nSpearman Probing: {probe.get('n_interpretable', 0)}/{probe.get('n_alive', 0)} "
              f"features ({probe.get('interpretable_fraction', 0) * 100:.1f}%)")
        print(f"Random Forest:    {union.get('n_interpretable_rf', 0)}/{union.get('n_alive', 0)} "
              f"features ({100 * union.get('n_interpretable_rf', 0) / max(union.get('n_alive', 1), 1):.1f}%)")
        print(f"Contrastive:      {union.get('n_interpretable_contrastive', 0)}/{union.get('n_alive', 0)} "
              f"features ({100 * union.get('n_interpretable_contrastive', 0) / max(union.get('n_alive', 1), 1):.1f}%)")
        print(f"UNION:            {union.get('n_interpretable_union', 0)}/{union.get('n_alive', 0)} "
              f"features ({100 * union.get('interpretable_fraction_union', 0):.1f}%)")
        for prop, s in sorted(
            probe.get("property_summary", {}).items(),
            key=lambda x: x[1].get("n_correlated_features", 0), reverse=True
        )[:8]:
            print(f"  {prop}: {s['n_correlated_features']} features, "
                  f"max |rho|={s['max_abs_rho']:.3f}")
        vp = result.get("vocab_projection", {})
        for feat in sorted(vp.get("features", []),
                          key=lambda x: x.get("w_residual_norm", 0), reverse=True)[:5]:
            w_top = feat.get("w_top_strings", [])[:5]
            v_top = feat.get("v_top_strings", [])[:5]
            print(f"  Feature {feat['feature_idx']}: retrieves=[{', '.join(repr(t) for t in w_top)}] "
                  f"queried_by=[{', '.join(repr(t) for t in v_top)}]")
        print(f"Results saved to results/data/{probe_fname}")
    elif stage == "probe-features-heldout":
        # Load same-split probe result to get the exact SAE tag
        probe_suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        probe_fname = f"probe_features_L{layer}_H{head}{probe_suffix}.json"
        same_split_path = os.path.join(os.path.dirname(__file__), "results", "data", probe_fname)
        target_sae_tag = ""
        if os.path.exists(same_split_path):
            same_split_data = json.load(open(same_split_path))
            target_sae_tag = same_split_data.get("sae_tag", "")
            print(f"Using SAE tag from same-split probe: {target_sae_tag}")

        # Run probing on both val (held-out) and train splits for comparison
        for split_name in ["val", "train"]:
            print(f"\n{'=' * 60}")
            print(f"Running held-out probing: split={split_name}")
            print(f"{'=' * 60}")
            result = probe_features_heldout_modal.remote(
                layer=layer, head=head,
                n_features_target=n_features if n_features > 0 else 2048,
                split=split_name,
                sae_tag=target_sae_tag,
                corpus_source=corpus_source,
            )
            local_out = os.path.join(os.path.dirname(__file__), "results", "data")
            os.makedirs(local_out, exist_ok=True)
            fname = f"probe_features_heldout_L{layer}_H{head}_{split_name}{probe_suffix}.json"
            with open(os.path.join(local_out, fname), "w") as f:
                json.dump(result, f, indent=2)
            probe = result.get("probe", {})
            union = result.get("union", {})
            print(f"\n[{split_name.upper()} split, n={result.get('n_split_samples', '?')}]")
            print(f"  Spearman: {probe.get('n_interpretable', 0)}/{probe.get('n_alive', 0)} "
                  f"({probe.get('interpretable_fraction', 0) * 100:.1f}%)")
            print(f"  RF:       {union.get('n_interpretable_rf', 0)}/{union.get('n_alive', 0)} "
                  f"({100 * union.get('n_interpretable_rf', 0) / max(union.get('n_alive', 1), 1):.1f}%)")
            print(f"  Contrastive: {union.get('n_interpretable_contrastive', 0)}/{union.get('n_alive', 0)} "
                  f"({100 * union.get('n_interpretable_contrastive', 0) / max(union.get('n_alive', 1), 1):.1f}%)")
            print(f"  UNION:    {union.get('n_interpretable_union', 0)}/{union.get('n_alive', 0)} "
                  f"({100 * union.get('interpretable_fraction_union', 0):.1f}%)")
            for prop, s in sorted(
                probe.get("property_summary", {}).items(),
                key=lambda x: x[1].get("n_correlated_features", 0), reverse=True
            )[:8]:
                print(f"    {prop}: {s['n_correlated_features']} features, "
                      f"max |rho|={s['max_abs_rho']:.3f}")
            print(f"  Saved to results/data/{fname}")
    elif stage == "circuit-ablation":
        result = circuit_ablation_modal.remote(
            layer=layer, head=head,
            n_sequences=n_samples if n_samples != 10000 else 200,
            n_per_property=2,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model, seq_len=seq_len,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"circuit_ablation_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print(f"\nCircuit Ablation: {s.get('n_causal_positive', 0)}/{s.get('n_total', 0)} "
              f"features show selective PPL damage ({s.get('causal_fraction', 0)*100:.1f}%)")
        print(f"  Mean selectivity: {s.get('mean_selectivity', 0):+.4f}")
        print(f"  Diagonal dominance: {s.get('diagonal_dominance', 'N/A')}")
        # Per-feature results
        for r in sorted(result.get("feature_results", []),
                       key=lambda x: x.get("selectivity", 0), reverse=True):
            print(f"  Feature {r['feature_idx']} ({r['property']}): "
                  f"dPPL_high={r['delta_ppl_high']:+.3f}, "
                  f"dPPL_low={r['delta_ppl_low']:+.3f}, "
                  f"selectivity={r['selectivity']:+.4f}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "circuit-ablation-v2":
        result = circuit_ablation_v2_modal.remote(
            layer=layer, head=head,
            n_sequences=n_samples if n_samples != 10000 else 200,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model, seq_len=seq_len,
            quartile_size=0.25,
            max_properties=8,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"circuit_ablation_v2_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print(f"\nGroup Ablation v2:")
        print(f"  Best property: {s.get('best_property', 'N/A')} "
              f"(selectivity={s.get('best_selectivity', 0):+.4f})")
        print(f"  Mean max-dose selectivity: {s.get('mean_max_dose_selectivity', 0):+.4f}")
        print(f"  Positive selectivity: {s.get('n_positive_selectivity', 0)}/{s.get('n_total', 0)}")
        # Dose-response summary per property
        for prop, data in result.get("property_results", {}).items():
            curve = data.get("dose_curve", [])
            if len(curve) > 1:
                last = curve[-1]
                print(f"  {prop} (n={data['n_features_total']}): "
                      f"max dose selectivity={last['selectivity']:+.4f}, "
                      f"dPPL_high={last['delta_loss_high']:+.5f}, "
                      f"dPPL_low={last['delta_loss_low']:+.5f}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "feature-vs-random":
        result = feature_vs_random_ablation_modal.remote(
            layer=layer, head=head,
            n_sequences=n_samples if n_samples != 10000 else 50,
            n_random=n_random,
            max_alive_features=max_alive_features,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model, seq_len=seq_len,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "feature_vs_random.json"), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        alive = s.get("alive_features", {})
        rand = s.get("random_directions", {})
        comp = s.get("comparison", {})
        print(f"\nFeature vs Random Ablation:")
        print(f"  Alive features ({s.get('n_alive_features', 0)}): "
              f"mean |dL|={alive.get('mean_abs_delta_loss', 0):.6f}")
        print(f"  Random directions ({s.get('n_random_directions', 0)}): "
              f"mean |dL|={rand.get('mean_abs_delta_loss', 0):.6f}")
        print(f"  Ratio: {comp.get('ratio_mean_abs', 0):.2f}x")
        print(f"  Cohen's d: {comp.get('cohens_d', 0):.3f}")
        print(f"  Mann-Whitney p: {comp.get('mann_whitney_p', 0):.2e}")
        print(f"\nTop-10 most impactful features:")
        for r in result.get("feature_results", [])[:10]:
            print(f"  F{r['feature_idx']:>5}: dL={r['delta_loss']:+.6f}")
        print(f"Results saved to results/data/feature_vs_random.json")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "causal-clamp":
        skip = expansion_factor if expansion_factor > 1 else 0  # repurpose --expansion-factor as --skip
        result = causal_clamp_modal.remote(
            layer=layer, head=head,
            n_prompts=n_samples if n_samples != 10000 else 50,
            n_features_target=n_features if n_features > 0 else 2048,
            n_top_features=5,
            skip_features=skip,
            model_name=model,
            prompt_len=512,
            gen_len=256,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"causal_clamp_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print(f"\nCausal Feature Clamping:")
        print(f"  Significant (p<0.05): {s.get('n_significant_p05', 0)}/{result.get('n_features', 0)}")
        print(f"  Direction-aligned: {s.get('n_direction_aligned', 0)}/{result.get('n_features', 0)}")
        print(f"  Causal (sig+aligned): {s.get('n_significant_and_aligned', 0)}/{result.get('n_features', 0)} "
              f"({s.get('fraction_causal', 0)*100:.0f}%)")
        print(f"  Mean |Cohen's d|: {s.get('mean_abs_cohens_d', 0):.3f}")
        for r in result.get("feature_results", []):
            print(f"  Feature {r['feature_idx']} ({r['property']}): "
                  f"shift={r['mean_shift']:+.4f}, d={r['cohens_d']:+.3f}, "
                  f"p={r['p_value']:.4f}, aligned={r['direction_aligned']}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "causal-logit":
        skip = expansion_factor if expansion_factor > 1 else 0  # repurpose --expansion-factor as --skip
        result = causal_logit_modal.remote(
            layer=layer, head=head,
            n_prompts=n_samples if n_samples != 10000 else 50,
            n_features_target=n_features if n_features > 0 else 2048,
            n_top_features=5,
            skip_features=skip,
            model_name=model,
            prompt_len=512,
            positions_per_prompt=4,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"causal_logit_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nDirect Logit Intervention:")
        print(f"  Significant (p<0.05): {s.get('n_significant_p05', 0)}/{result.get('n_features', 0)}")
        print(f"  Direction-aligned: {s.get('n_direction_aligned', 0)}/{result.get('n_features', 0)}")
        print(f"  Causal (sig+aligned): {s.get('n_significant_and_aligned', 0)}/{result.get('n_features', 0)} "
              f"({s.get('fraction_causal', 0)*100:.0f}%)")
        print(f"  Mean |Cohen's d|: {s.get('mean_abs_cohens_d', 0):.3f}")
        for r in result.get("feature_results", []):
            print(f"  Feature {r['feature_idx']} ({r['property']}, {r['metric']}): "
                  f"shift={r['mean_shift']:+.4f}, d={r['cohens_d']:+.3f}, "
                  f"p={r['p_value']:.4f}, aligned={r['direction_aligned']}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "mechanistic-profile":
        result = mechanistic_profile_modal.remote(
            layer=layer,
            head=head,
            n_sequences=n_samples if n_samples != 10000 else 2000,
            n_features_target=n_features if n_features > 0 else 2048,
            family=family,
            n_top_features=16,
            n_quantiles=5,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"mechanistic_profile_{family}_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        print("\nMechanistic profile:")
        print(f"  Main property: {result.get('main_property')}")
        print(f"  Top features: {result.get('feature_group_summary', {}).get('top_feature_indices', [])}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "mechanistic-clamp":
        result = mechanistic_clamp_modal.remote(
            layer=layer,
            head=head,
            n_prompts=n_samples if n_samples != 10000 else 24,
            n_features_target=n_features if n_features > 0 else 2048,
            family=family,
            n_top_features=16,
            n_random_groups=max(4, n_random),
            model_name=model,
            prompt_len=512,
            gen_len=128,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"mechanistic_clamp_{family}_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nMechanistic grouped clamp:")
        print(f"  Main property: {result.get('main_property')}")
        print(f"  Dose ordered: {s.get('dose_ordered_main_shift')}")
        print(f"  Best shift: {s.get('best_main_property_shift', 0):+.4f}")
        print(f"  Random mean shift: {s.get('random_main_property_mean_shift', 0):+.4f}")
        print(f"  Ratio vs random mean: {s.get('ratio_vs_random_mean', 0):.2f}x")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace":
        parsed_features = [int(x.strip()) for x in feature_indices.split(",") if x.strip()]
        result = factor_trace_modal.remote(
            layer=layer,
            head=head,
            n_prompts=n_samples if n_samples != 10000 else 32,
            n_features_target=n_features if n_features > 0 else 2048,
            feature_indices=parsed_features or None,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            prompt_len=min(seq_len, 512),
            max_offset=64,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        print("\nFactor trace:")
        print(f"  Selected features: {[item['feature_idx'] for item in result.get('selected_features', [])]}")
        print(f"  Prompts traced: {result.get('n_prompts', 0)}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace-intervention":
        result = factor_trace_intervention_modal.remote(
            layer=layer,
            head=head,
            n_prompts=n_samples if n_samples != 10000 else 16,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            prompt_len=min(seq_len, 512),
            n_random_controls=4,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_intervention_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nFactor trace intervention:")
        print(f"  Target feature: {result.get('target_feature_idx')}")
        print(f"  Use-time KL mean: {s.get('target_use_time_kl_mean', 0):.6f}")
        print(f"  Control KL mean: {s.get('control_use_time_kl_mean', 0):.6f}")
        print(f"  KL ratio vs controls: {s.get('target_vs_random_kl_ratio', 0):.2f}x")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace-transplant":
        result = factor_trace_transplant_modal.remote(
            layer=layer,
            head=head,
            n_pairs=n_samples if n_samples != 10000 else 16,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_transplant_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nFactor trace transplant:")
        target_label = result.get("target_feature_indices") or [result.get("target_feature_idx")]
        print(f"  Target features: {target_label}")
        print(
            f"  Boundary shift vs wrong-feature: "
            f"{s.get('boundary_ratio_vs_wrong_feature', 0):.2f}x"
        )
        print(
            f"  Boundary shift vs wrong-time: "
            f"{s.get('boundary_ratio_vs_wrong_time', 0):.2f}x"
        )
        print(
            f"  Target beats wrong-feature: "
            f"{100.0 * s.get('target_beats_wrong_feature_fraction', 0):.1f}%"
        )
        print(
            f"  Actual-token logit shift mean: "
            f"{s.get('target_actual_token_logit_shift_mean', 0):+.4f} "
            f"(wrong-feature {s.get('wrong_feature_actual_token_logit_shift_mean', 0):+.4f}, "
            f"sign-flip {s.get('sign_flip_actual_token_logit_shift_mean', 0):+.4f})"
        )
        if "target_minus_sign_flip_actual_token_logit_shift_mean" in s:
            print(
                f"  Target - sign-flip actual-token logit shift: "
                f"{s.get('target_minus_sign_flip_actual_token_logit_shift_mean', 0):+.4f}"
            )
        if "target_minus_sign_flip_actual_token_prob_shift_mean" in s:
            print(
                f"  Target - sign-flip actual-token prob shift: "
                f"{s.get('target_minus_sign_flip_actual_token_prob_shift_mean', 0):+.4f}"
            )
        print(
            f"  Actual-token beats sign-flip: "
            f"{100.0 * s.get('target_beats_sign_flip_fraction', 0):.1f}%"
        )
        if "target_coeff_beats_sign_flip_fraction" in s:
            print(
                f"  Use-coeff beats sign-flip: "
                f"{100.0 * s.get('target_coeff_beats_sign_flip_fraction', 0):.1f}%"
            )
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace-benchmark":
        result = factor_trace_benchmark_modal.remote(
            layer=layer,
            head=head,
            n_rows=n_samples if n_samples != 10000 else 64,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_benchmark_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        print("\nFactor trace benchmark:")
        print(f"  Target features: {result.get('target_feature_indices')}")
        print(f"  Candidate pairs: {result.get('n_candidate_pairs', 0)}")
        print(f"  Accepted rows: {result.get('n_rows_accepted', 0)}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "hierarchical-transplant":
        result = hierarchical_transplant_modal.remote(
            layer=layer,
            head=head,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"hierarchical_transplant_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nHierarchical causal faithfulness:")
        print(f"  SAE type: {result.get('sae_type')}")
        print(f"  Rows: {len(result.get('rows', []))}")
        print(f"  Mean full delta: {s.get('mean_full_delta', 0):+.4f}")
        print(f"  Mean SAE delta: {s.get('mean_sae_delta', 0):+.4f}")
        print(f"  Mean feature delta: {s.get('mean_feature_delta', 0):+.4f}")
        faith = s.get('mean_sae_faithfulness')
        print(f"  Mean SAE faithfulness: {faith:.3f}" if faith is not None else "  Mean SAE faithfulness: N/A")
        feat_r = s.get('mean_feature_ratio')
        print(f"  Mean feature ratio: {feat_r:.3f}" if feat_r is not None else "  Mean feature ratio: N/A")
        print(f"  Full positive: {s.get('n_rows_full_positive', 0)}/{len(result.get('rows', []))}")
        print(f"  SAE positive: {s.get('n_rows_sae_positive', 0)}/{len(result.get('rows', []))}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace-use-site":
        result = factor_trace_use_site_modal.remote(
            layer=layer,
            head=head,
            n_rows=n_samples if n_samples != 10000 else 32,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_use_site_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nFactor trace use-site causal:")
        print(f"  Target features: {result.get('target_feature_indices')}")
        print(f"  Rows evaluated: {result.get('n_rows_evaluated', 0)}")
        print(f"  Target beats reverse prob: {100.0 * s.get('target_beats_reverse_prob_fraction', 0):.1f}%")
        print(f"  Target - reverse prob shift mean: {s.get('target_minus_reverse_actual_token_prob_shift_mean', 0):+.4f}")
        print(f"  Promotion gate pass: {s.get('promotion_gate_pass', False)}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace-readout-map":
        result = factor_trace_readout_map_modal.remote(
            layer=layer,
            head=head,
            n_rows=n_samples if n_samples != 10000 else 32,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_readout_map_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nFactor trace readout map:")
        print(f"  Target features: {result.get('target_feature_indices')}")
        print(f"  Rows evaluated: {result.get('n_rows_evaluated', 0)}")
        print(f"  Target - reverse prob shift mean: {s.get('actual_token_prob_delta_target_minus_reverse_mean', 0):+.4f}")
        print(f"  Target - reverse logit shift mean: {s.get('actual_token_logit_delta_target_minus_reverse_mean', 0):+.4f}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "factor-trace-dot-direction":
        result = factor_trace_dot_direction_modal.remote(
            layer=layer,
            head=head,
            n_rows=n_samples if n_samples != 10000 else 5,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        release_out = os.path.join(os.path.dirname(__file__), "release", "write_to_use_2026-04-09")
        os.makedirs(local_out, exist_ok=True)
        os.makedirs(release_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"factor_trace_dot_direction_L{layer}_H{head}{suffix}.json"
        for out_dir in [local_out, release_out]:
            with open(os.path.join(out_dir, local_name), "w") as f:
                json.dump(result, f, indent=2)
        s = result.get("summary", {})
        print("\nFactor trace dot direction:")
        print(f"  Logistic test accuracy: {result.get('logistic', {}).get('test_accuracy', 0):.3f}")
        print(f"  Rows evaluated: {s.get('n_rows', 0)}")
        print(f"  Success rows: {s.get('success_rows', 0)}")
        print(f"  Success gate pass: {s.get('success_gate_pass', False)}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "nonsparse-baseline":
        result = nonsparse_baseline_modal.remote(
            layer=layer, head=head,
            n_features=n_features if n_features > 0 else 2048,
            epochs=epochs,
            seed=seed,
            model_name=model,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "nonsparse_baseline.json"), "w") as f:
            json.dump(result, f, indent=2)
        probe = result.get("probe", {})
        sparse = result.get("sparse_comparison", {})
        print(f"\nTopK-Disabled ReLU-Only Control (legacy stage: nonsparse-baseline):")
        print(f"  Control: {probe.get('n_interpretable', 0)}/{probe.get('n_alive', 0)} "
              f"interpretable ({probe.get('interpretable_fraction', 0)*100:.1f}%)")
        if sparse:
            print(f"  Sparse (k=32): {sparse.get('sparse_n_interpretable', 0)}/"
                  f"{sparse.get('sparse_n_alive', 0)} interpretable "
                  f"({sparse.get('sparse_interpretable_fraction', 0)*100:.1f}%)")
        print(f"  Train MSE: {result.get('train_mse', 0):.6f}")
        print(f"Results saved to results/data/nonsparse_baseline.json")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "probe-stability":
        result = probe_stability_modal.remote(
            layer=layer, head=head,
            n_features=n_features if n_features > 0 else 2048,
            k=k,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "probe_stability.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nProbing Stability:")
        print(f"  Decoded texts available: {result.get('n_texts_available', 0)}")
        print(f"  Held-out subsets available: {result.get('held_out_subsets_available', 0)}/"
              f"{result.get('held_out_subsets_requested', 0)} requested")
        print(f"  Mean Jaccard (top-10 properties): {result.get('mean_jaccard', 0):.3f}")
        print(f"  Universal properties: {result.get('universal_top_properties', [])}")
        print(f"  Mean interpretable fraction: {result.get('mean_interpretable_fraction', 0)*100:.1f}% "
              f"+/- {result.get('std_interpretable_fraction', 0)*100:.1f}%")
        for sr in result.get("subset_results", []):
            print(f"  Subset {sr['subset_idx']} ({sr.get('split_type', 'unknown')}): "
                  f"{sr['n_interpretable']}/{sr['n_alive']} interpretable "
                  f"({sr['interpretable_fraction']*100:.1f}%), top={sr['top_10_properties'][:3]}")
        for js in result.get("jaccard_scores", []):
            print(f"  Subsets {js['subset_a']} vs {js['subset_b']}: "
                  f"Jaccard={js['jaccard']:.3f}, shared={js['intersection']}")
        print(f"Results saved to results/data/probe_stability.json")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "scaling-diagnosis":
        # Run ALL Phase 1 experiments in maximum parallelism:
        # E1: 5 subsets x 3 types x 3 seeds = 45 parallel GPU jobs
        # E2: 1 CPU job (data diversity)
        # E3: 3 types x 3 seeds = 9 parallel GPU jobs (200 epochs on 50K)
        # Total: 55 parallel tasks
        import numpy as np

        budget_epochs = epochs if epochs != 20 else 200
        n_subsets, subset_size = 5, 5000
        sae_types = ["flat", "rank1", "bilinear"]
        seeds = [0, 1, 2]

        print(f"Launching Phase 1 scaling diagnosis: 55 parallel tasks")
        print(f"  E1: {n_subsets} subsets x {len(sae_types)} types x {len(seeds)} seeds = {n_subsets * len(sae_types) * len(seeds)} jobs")
        print(f"  E2: 1 data diversity job")
        print(f"  E3: {len(sae_types)} types x {len(seeds)} seeds = {len(sae_types) * len(seeds)} budget control jobs ({budget_epochs} epochs)")

        # Spawn E2 (fastest, no GPU needed)
        h_diversity = data_diversity_modal.spawn(layer=layer, head=head)

        # Spawn E1 (45 jobs)
        e1_handles = []
        for si in range(n_subsets):
            start = si * subset_size
            end = start + subset_size
            for sae_type in sae_types:
                for seed in seeds:
                    h = train_on_subset.spawn(
                        layer=layer, head=head,
                        subset_idx=si, subset_start=start, subset_end=end,
                        sae_type=sae_type, seed=seed,
                    )
                    e1_handles.append((h, si, sae_type, seed))

        # Spawn E3 (9 jobs)
        e3_handles = []
        for sae_type in sae_types:
            for seed in seeds:
                h = train_budget_single.spawn(
                    layer=layer, head=head,
                    sae_type=sae_type, seed=seed, epochs=budget_epochs,
                )
                e3_handles.append((h, sae_type, seed))

        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)

        # Collect E2 (should finish first)
        print("\nWaiting for E2 (data diversity)...")
        try:
            diversity_result = h_diversity.get()
            with open(os.path.join(local_out, "data_diversity.json"), "w") as f:
                json.dump(diversity_result, f, indent=2)
            dr = diversity_result.get("diversity_ratio", {})
            print(f"E2 done: eff_rank ratio={dr.get('eff_rank_ratio')}x, frob_dist ratio={dr.get('frob_dist_ratio')}x")
        except Exception as e:
            print(f"E2 FAILED: {e}")

        # Collect E1 (45 jobs)
        print(f"\nCollecting E1 ({len(e1_handles)} subset training jobs)...")
        e1_results = []
        for i, (h, si, sae_type, seed) in enumerate(e1_handles):
            tag = f"subset{si}_{sae_type}_s{seed}"
            try:
                r = h.get()
                e1_results.append(r)
                print(f"  [{i+1}/{len(e1_handles)}] {tag}: dead={r['dead_pct']}% mse={r['best_mse']:.4e}")
            except Exception as e:
                print(f"  [{i+1}/{len(e1_handles)}] {tag} FAILED: {e}")
                e1_results.append({"subset_idx": si, "sae_type": sae_type, "seed": seed, "error": str(e)})

        e1_summary = {}
        for sae_type in sae_types:
            type_results = [r for r in e1_results if r.get("sae_type") == sae_type and "dead_pct" in r]
            per_subset = {}
            for r in type_results:
                per_subset.setdefault(r["subset_idx"], []).append(r["dead_pct"])
            subset_means = [np.mean(v) for v in per_subset.values()]
            e1_summary[sae_type] = {
                "per_subset_dead_pct": {str(k): round(float(np.mean(v)), 1) for k, v in per_subset.items()},
                "mean_across_subsets": round(float(np.mean(subset_means)), 1) if subset_means else None,
                "std_across_subsets": round(float(np.std(subset_means)), 1) if subset_means else None,
                "min_subset": round(float(np.min(subset_means)), 1) if subset_means else None,
                "max_subset": round(float(np.max(subset_means)), 1) if subset_means else None,
            }
        # Count bilinear wins
        if "bilinear" in e1_summary and "flat" in e1_summary:
            bil = e1_summary["bilinear"]["per_subset_dead_pct"]
            flt = e1_summary["flat"]["per_subset_dead_pct"]
            wins = sum(1 for si in bil if float(bil[si]) < float(flt.get(si, 100)))
            e1_summary["bilinear_wins_vs_flat"] = f"{wins}/{len(bil)}"

        e1_output = {
            "experiment": "subset_reproducibility",
            "layer": layer, "head": head,
            "n_subsets": n_subsets, "subset_size": subset_size,
            "results": e1_results, "summary": e1_summary,
        }
        with open(os.path.join(local_out, "subset_reproducibility.json"), "w") as f:
            json.dump(e1_output, f, indent=2)

        print(f"\nE1 Subset Reproducibility Summary:")
        for stype, s in e1_summary.items():
            if isinstance(s, dict) and "mean_across_subsets" in s:
                print(f"  {stype}: dead={s['mean_across_subsets']}% +/- {s['std_across_subsets']}% "
                      f"[{s['min_subset']}% - {s['max_subset']}%]")
        if "bilinear_wins_vs_flat" in e1_summary:
            print(f"  bilinear wins vs flat: {e1_summary['bilinear_wins_vs_flat']}")

        # Collect E3 (9 jobs, longest running)
        print(f"\nCollecting E3 ({len(e3_handles)} budget control jobs, {budget_epochs} epochs)...")
        e3_results = []
        for i, (h, sae_type, seed) in enumerate(e3_handles):
            tag = f"{sae_type}_s{seed}"
            try:
                r = h.get()
                e3_results.append(r)
                print(f"  [{i+1}/{len(e3_handles)}] {tag}: dead={r['dead_pct']}% mse={r['best_mse']:.4e} [{r['total_time_s']:.0f}s]")
            except Exception as e:
                print(f"  [{i+1}/{len(e3_handles)}] {tag} FAILED: {e}")
                e3_results.append({"sae_type": sae_type, "seed": seed, "error": str(e)})

        e3_summary = {}
        for sae_type in sae_types:
            type_results = [r for r in e3_results if r.get("sae_type") == sae_type and "dead_pct" in r]
            deads = [r["dead_pct"] for r in type_results]
            mses = [r["best_mse"] for r in type_results]
            e3_summary[sae_type] = {
                "mean_dead_pct": round(float(np.mean(deads)), 1) if deads else None,
                "std_dead_pct": round(float(np.std(deads)), 1) if deads else None,
                "mean_mse": float(np.mean(mses)) if mses else None,
                "per_seed": {str(r["seed"]): r["dead_pct"] for r in type_results},
            }

        e3_output = {
            "experiment": "budget_control_50k",
            "layer": layer, "head": head,
            "epochs": budget_epochs,
            "results": e3_results, "summary": e3_summary,
        }
        with open(os.path.join(local_out, "budget_control.json"), "w") as f:
            json.dump(e3_output, f, indent=2)

        print(f"\nE3 Budget Control Summary ({budget_epochs} epochs on 50K):")
        for stype, s in e3_summary.items():
            if s.get("mean_dead_pct") is not None:
                print(f"  {stype}: dead={s['mean_dead_pct']}% +/- {s['std_dead_pct']}%")

        print(f"\nAll Phase 1 results saved to results/data/")

    elif stage == "scaling-curve":
        # 6 sizes x 3 types x 3 seeds = 54 parallel GPU jobs
        import numpy as np

        sizes = [1000, 2000, 5000, 10000, 20000, 50000]
        jobs = _build_scaling_curve_jobs(layer=layer, head=head, sizes=sizes)
        print(f"Launching scaling curve: {len(jobs)} parallel jobs")
        print(f"  Sizes: {sizes}")
        print(f"  Types: flat, rank1, bilinear x 3 seeds each")

        handles = []
        for j in jobs:
            h = train_on_subset.spawn(**j)
            handles.append((h, j))

        results = []
        for i, (h, j) in enumerate(handles):
            tag = f"n={j['subset_end']}_{j['sae_type']}_s{j['seed']}"
            try:
                r = h.get()
                results.append(r)
                print(f"  [{i+1}/{len(handles)}] {tag}: dead={r['dead_pct']}%")
            except Exception as e:
                print(f"  [{i+1}/{len(handles)}] {tag} FAILED: {e}")
                results.append({**j, "error": str(e)})

        # Aggregate: mean dead% per (size, type)
        summary = {}
        for size in sizes:
            for sae_type in ["flat", "rank1", "bilinear"]:
                key = f"n{size}_{sae_type}"
                type_results = [r for r in results
                                if r.get("end") == size and r.get("sae_type") == sae_type and "dead_pct" in r]
                deads = [r["dead_pct"] for r in type_results]
                mses = [r["best_mse"] for r in type_results]
                summary[key] = {
                    "n_samples": size, "sae_type": sae_type,
                    "mean_dead_pct": round(float(np.mean(deads)), 1) if deads else None,
                    "std_dead_pct": round(float(np.std(deads)), 1) if deads else None,
                    "mean_mse": float(np.mean(mses)) if mses else None,
                    "alive_pct": round(100 - float(np.mean(deads)), 1) if deads else None,
                }

        output = {
            "experiment": "scaling_curve",
            "layer": layer, "head": head,
            "sizes": sizes, "n_features": 2048, "k": 32,
            "results": results, "summary": summary,
        }
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, "scaling_curve.json"), "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nScaling Curve Summary (alive %):")
        print(f"{'Size':>8} {'flat':>8} {'rank1':>8} {'bilinear':>10}")
        for size in sizes:
            row = []
            for sae_type in ["flat", "rank1", "bilinear"]:
                s = summary.get(f"n{size}_{sae_type}", {})
                alive = s.get("alive_pct", "?")
                row.append(f"{alive}%")
            print(f"{size:>8} {row[0]:>8} {row[1]:>8} {row[2]:>10}")
        print(f"\nResults saved to results/data/scaling_curve.json")

    elif stage == "multihead-transplant":
        result = multihead_transplant_modal.remote(
            layer=layer,
            max_rows=n_samples if n_samples != 10000 else 32,
            model_name=model,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_name = f"multihead_transplant_L{layer}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)

        p1 = result["phase1_all_heads"]
        p2 = result["phase2_leave_one_out"]
        p3 = result["phase3_minimal_set"]
        print(f"\nMulti-head transplant results (layer {layer}):")
        print(f"  Phase 1 (all heads): mean_delta={p1['mean_delta']:+.4f} "
              f"n_positive={p1['n_positive']}/{p1['n_rows']}")
        print(f"  Phase 2 (LOO): top heads = {p2['top_heads_by_contribution']}")
        for h_str, hc in sorted(p2["head_contributions"].items(), key=lambda x: x[1]["contribution"], reverse=True)[:4]:
            print(f"    Head {h_str}: contribution={hc['contribution']:+.4f}")
        print(f"  Phase 3 (minimal set {p3['heads_used']}): "
              f"mean_delta={p3['mean_delta']:+.4f} "
              f"fraction_of_full={p3['fraction_of_full']:.2f}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "generation-intervention":
        result = generation_intervention_modal.remote(
            layer=layer,
            head=head,
            n_prompts=n_samples if n_samples != 10000 else 100,
            n_tokens=200,
            boost_scale=3.0,
            suppress_scale=0.0,
            n_boundary_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
            temperature=0.7,
            corpus_source=corpus_source,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        local_name = f"generation_intervention_L{layer}_H{head}{suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        from generation_intervention import print_summary
        print_summary(result)
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage in ("generation-intervention-multihead", "generation-intervention-multihead-flat"):
        sae_types = ("flat",) if stage.endswith("-flat") else ("bilinear", "bilinear_tied", "rank1")
        result = generation_intervention_multihead_modal.remote(
            layer=layer,
            n_heads=16,
            n_prompts=n_samples if n_samples != 10000 else 100,
            n_tokens=200,
            boost_scale=1.5,
            suppress_scale=0.0,
            n_boundary_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
            corpus_source=corpus_source,
            sae_types=sae_types,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        sae_type = result.get("sae_type", "bilinear")
        type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
        local_name = f"generation_intervention_multihead_L{layer}{suffix}{type_suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        from generation_intervention import print_summary
        print_summary(result)
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "generation-intervention-multihead-random":
        sae_types = ("bilinear", "bilinear_tied", "rank1")
        result = generation_intervention_multihead_random_modal.remote(
            layer=layer,
            n_heads=16,
            n_prompts=n_samples if n_samples != 10000 else 100,
            n_tokens=200,
            boost_scale=1.5,
            suppress_scale=0.0,
            n_random_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
            corpus_source=corpus_source,
            sae_types=sae_types,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        sae_type = result.get("sae_type", "bilinear")
        type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
        local_name = f"generation_intervention_multihead_random_L{layer}{suffix}{type_suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        from generation_intervention import print_summary
        print_summary(result)
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "generation-intervention-multihead-additive":
        sae_types = ("bilinear", "bilinear_tied", "rank1")
        result = generation_intervention_multihead_additive_modal.remote(
            layer=layer,
            n_heads=16,
            n_prompts=n_samples if n_samples != 10000 else 100,
            n_tokens=200,
            additive_boost_strength=0.5,
            n_boundary_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
            corpus_source=corpus_source,
            sae_types=sae_types,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        sae_type = result.get("sae_type", "bilinear")
        type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
        local_name = f"generation_intervention_multihead_additive_L{layer}{suffix}{type_suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        from generation_intervention import print_summary
        print_summary(result)
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "generation-intervention-multihead-additive-random":
        sae_types = ("bilinear", "bilinear_tied", "rank1")
        result = generation_intervention_multihead_additive_random_modal.remote(
            layer=layer,
            n_heads=16,
            n_prompts=n_samples if n_samples != 10000 else 100,
            n_tokens=200,
            additive_boost_strength=0.5,
            n_random_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
            corpus_source=corpus_source,
            sae_types=sae_types,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        suffix = "" if corpus_source == "openwebtext" else f"_{corpus_source}"
        sae_type = result.get("sae_type", "bilinear")
        type_suffix = f"_{sae_type}" if sae_type != "bilinear" else ""
        local_name = f"generation_intervention_multihead_additive_random_L{layer}{suffix}{type_suffix}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        from generation_intervention import print_summary
        print_summary(result)
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
    elif stage == "generation-intervention-qualitative":
        result = generation_intervention_qualitative_modal.remote(
            layer=layer,
            n_heads=16,
            n_tokens=400,
            boost_scale=3.0,
            suppress_scale=0.0,
            n_boundary_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_name = f"generation_intervention_qualitative_L{layer}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nQualitative demo saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
        n_prompts_done = len(result.get("per_prompt", []))
        print(f"Generated {n_prompts_done} prompts x 3 conditions")
        # Show aggregate stats
        conds = result.get("conditions", {})
        for cond_name in ["baseline", "boundary_boost", "random_boost"]:
            if cond_name in conds:
                m = conds[cond_name]
                print(f"  {cond_name:20s}: paragraphs={m.get('n_paragraphs', 0):.1f}  "
                      f"newlines={m.get('n_newlines', 0):.1f}  "
                      f"words={m.get('n_words', 0):.0f}")
    elif stage == "generation-intervention-qualitative-additive":
        _nh = 32 if model and "4B" in model else 16
        result = generation_intervention_qualitative_modal.remote(
            layer=layer,
            n_heads=_nh,
            n_tokens=400,
            n_boundary_features=10,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
            additive=True,
            additive_boost_strengths=(2.0, 5.0, 10.0),
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_name = f"generation_intervention_qualitative_additive_L{layer}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nQualitative additive demo saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
        n_prompts_done = len(result.get("per_prompt", []))
        cond_names = result.get("condition_names", [])
        print(f"Generated {n_prompts_done} prompts x {len(cond_names)} conditions")
        conds = result.get("conditions", {})
        for cond_name in cond_names:
            if cond_name in conds:
                m = conds[cond_name]
                print(f"  {cond_name:30s}: paragraphs={m.get('n_paragraphs', 0):.1f}  "
                      f"newlines={m.get('n_newlines', 0):.1f}  "
                      f"words={m.get('n_words', 0):.0f}")
    elif stage == "single-feature-demo":
        result = single_feature_demo_modal.remote(
            layer=layer,
            head=head if head != 0 else 12,
            n_tokens=400,
            n_features_target=n_features if n_features > 0 else 2048,
            model_name=model,
            temperature=0.7,
            boost_strength=2.0,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_name = f"single_feature_demo_L{layer}_H{result.get('head', head)}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSingle-feature demo saved to results/data/{local_name}")
        print(f"Total time: {result.get('total_time_s', 0):.0f}s")
        print(f"Features tested: {result.get('n_features_tested', 0)}")
        for feat in result.get("features", []):
            fi = feat["feature_idx"]
            label = feat["feature_label"]
            push = feat["push_value"]
            print(f"\n  Feature {fi} ({label}), push={push:.4f}:")
            # Show first prompt comparison
            if feat.get("prompts"):
                p = feat["prompts"][0]
                for cond in ["baseline", "boost", "suppress"]:
                    text = p[cond]["text"][:200]
                    norm = p[cond]["mean_intervention_norm"]
                    print(f"    {cond:10s} (norm={norm:.2f}): {text}")
    elif stage == "memory-alignment":
        result = memory_alignment_modal.remote(
            layer=layer,
            head=head,
            n_seqs=n_samples if n_samples != 10000 else 100,
            seq_len=min(seq_len, 512),
            batch_size=4,
            top_n=50,
            n_features_target=n_features if n_features > 0 else 2048,
            checkpoint_tag=checkpoint_tag,
            model_name=model,
        )
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_name = f"memory_alignment_L{layer}_H{head}.json"
        with open(os.path.join(local_out, local_name), "w") as f:
            json.dump(result, f, indent=2)
        alive = [r for r in result["results"] if r["alive"]]
        if alive:
            k_abs = [r["mean_abs_k_cos"] for r in alive]
            v_abs = [r["mean_abs_v_cos"] for r in alive]
            import statistics
            print(f"\nMemory alignment summary ({len(alive)} alive features):")
            print(f"  Key |cos|: mean={statistics.mean(k_abs):.4f} median={statistics.median(k_abs):.4f}")
            print(f"  Val |cos|: mean={statistics.mean(v_abs):.4f} median={statistics.median(v_abs):.4f}")
            above_03 = sum(1 for k, v in zip(k_abs, v_abs) if (k * v) ** 0.5 > 0.3)
            print(f"  Features with combined >0.3: {above_03}/{len(alive)}")
        print(f"Results saved to results/data/{local_name}")
        print(f"Total time: {result.get('elapsed_s', 0):.0f}s")
    elif stage == "all":
        run_all.remote(
            n_samples=n_samples,
            layers=layer_list,
            seq_len=seq_len,
            model_name=model,
            corpus_source=corpus_source,
        )
        print("Full pipeline complete.")
    else:
        print(f"Unknown stage: {stage}. Use: extract, extract-texts, train, sweep, nf-sweep, batchtopk-sweep, analyze, interpret, interpret-s0, s0, s0-shift, s0-proper-null, temporal, feature-quality, baselines, evaluate, evaluate-allheads, train-allheads, evaluate-perhead, probe-features, probe-features-heldout, circuit-ablation, circuit-ablation-v2, feature-vs-random, causal-clamp, causal-logit, mechanistic-profile, mechanistic-clamp, factor-trace, factor-trace-intervention, factor-trace-transplant, factor-trace-benchmark, factor-trace-use-site, factor-trace-readout-map, factor-trace-dot-direction, multihead-transplant, generation-intervention, generation-intervention-multihead, generation-intervention-multihead-flat, generation-intervention-multihead-random, generation-intervention-multihead-additive, generation-intervention-multihead-additive-random, generation-intervention-qualitative, generation-intervention-qualitative-additive, single-feature-demo, nonsparse-baseline, probe-stability, memory-alignment, plot-curve, download, subset-reproducibility, data-diversity, budget-control, scaling-diagnosis, all")
        return

    print(f"Wall clock: {(time.time() - t0) / 60:.1f} minutes")

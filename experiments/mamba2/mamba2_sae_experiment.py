#!/usr/bin/env python3
"""Mamba-2 multi-head SAE sweep driven by exact write-process metrics.

Stages:
1. scan: score all 48 heads on the target layers using sampled exact write factors
2. extract: save recurrent states for selected (layer, head) pairs and higher-fidelity write samples
3. train: train flat and rank-1 SAEs on each selected pair
4. analyze: correlate write metrics with rank-1 SAE advantage across selected pairs
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np


def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (FileNotFoundError, OSError):
        return "unknown"


CODE_SHA = _code_sha()
MODEL_NAME = "AntonV/mamba2-780m-hf"

# Keep the original layer set, but move to head-level slices.
LAYERS = [0, 6, 14, 31, 46, 47]

SEQ_LEN = 1024
EXTRACT_BATCH_SIZE = 2
TRAIN_BATCH_SIZE = 128
TRAIN_EPOCHS = 50
TRAIN_WARMUP_STEPS = 100
TRAIN_RESAMPLE_EVERY = 250
SEEDS = [0, 1, 42]
SAE_TYPES = ["flat", "rank1"]

# New head-level sweep knobs.
HEAD_SCAN_N_SAMPLES = 512
TRAIN_N_SAMPLES = 5000
HEAD_SCAN_SAMPLE_SIZE = 512
PAIR_WRITE_SAMPLE_SIZE = 2048
N_SWEEP_PAIRS = 16
MAX_PAIRS_PER_LAYER = 4

SPECTRAL_AUDIT_BUNDLE = "/root/spectral_audit_mamba2.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "datasets",
        "numpy",
        "tqdm",
        "accelerate",
        "scipy",
    )
    .add_local_file("sae.py", "/root/sae.py", copy=True)
    .add_local_file("split_utils.py", "/root/split_utils.py", copy=True)
    .add_local_file("train.py", "/root/train.py", copy=True)
    .add_local_file("results/data/spectral_audit_mamba2.json", SPECTRAL_AUDIT_BUNDLE, copy=True)
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("mamba2-sae-experiment")
vol = modal.Volume.from_name("mamba2-sae-data-v4", create_if_missing=True)
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
DATA = "/data"
MODELS = "/models"

HEAD_SCAN_PATH = Path(f"{DATA}/mamba2_head_scan_metrics.json")
SELECTED_PAIRS_PATH = Path(f"{DATA}/mamba2_head_sweep_pairs.json")
STATE_DIR = Path(f"{DATA}/mamba2_states_multihead")
WRITE_SAMPLE_DIR = Path(f"{DATA}/mamba2_write_samples_multihead")
CHECKPOINT_DIR = Path(f"{DATA}/mamba2_checkpoints_multihead")
RESULTS_PATH = Path(f"{DATA}/mamba2_sae_results_multihead.json")


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if abs(den) > 1e-12 else float(default)


def _pair_key(layer: int, head: int) -> str:
    return f"L{layer}_H{head}"


def _effective_rank_from_values(values: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(vals.sum())
    if total <= eps:
        return 0.0
    p = vals / total
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def _apply_padding_mask(hidden_states: Any, attention_mask: Any) -> Any:
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[0] > 1 and attention_mask.shape[1] > 1:
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


def _resolve_cache(outputs: Any) -> Any:
    for attr in ("cache_params", "past_key_values"):
        cache = getattr(outputs, attr, None)
        if cache is not None:
            return cache
    raise AttributeError("Cannot find Mamba-2 cache in model outputs")


def _layer_ssm_state(cache: Any, layer_idx: int) -> Any:
    ssm_states = getattr(cache, "ssm_states", None)
    if ssm_states is not None:
        return ssm_states[layer_idx]
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise AttributeError("Cache has no ssm_states or layers")
    layer_cache = layers[layer_idx]
    for attr in ("recurrent_states", "ssm_states"):
        state = getattr(layer_cache, attr, None)
        if state is not None:
            return state
    raise AttributeError(f"Layer {layer_idx} has no recurrent_states or ssm_states")


def _get_mamba2_layers(model: Any) -> list[Any]:
    if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        return list(model.backbone.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError("Could not find Mamba-2 layers on model")


def _load_selected_pairs_payload() -> dict[str, Any]:
    if not SELECTED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing selected pairs file: {SELECTED_PAIRS_PATH}. Run stage=scan first.")
    with open(SELECTED_PAIRS_PATH) as f:
        return json.load(f)


def _load_selected_pairs() -> list[dict[str, Any]]:
    payload = _load_selected_pairs_payload()
    return payload.get("selected_pairs", [])


def _write_metric_summary(k_vectors: np.ndarray, v_vectors: np.ndarray) -> dict[str, float]:
    if k_vectors.shape[0] == 0 or v_vectors.shape[0] == 0:
        return {}

    valid = (np.linalg.norm(k_vectors, axis=1) > 1e-8) & (np.linalg.norm(v_vectors, axis=1) > 1e-8)
    k = _normalize_rows(k_vectors[valid].astype(np.float64, copy=False))
    v = _normalize_rows(v_vectors[valid].astype(np.float64, copy=False))
    if k.shape[0] < 2:
        return {}

    n = k.shape[0]
    c_k = (k.T @ k) / n
    c_v = (v.T @ v) / n
    eigs_k = np.linalg.eigvalsh(c_k)
    eigs_v = np.linalg.eigvalsh(c_v)
    erank_k = _effective_rank_from_values(eigs_k)
    erank_v = _effective_rank_from_values(eigs_v)
    write_reuse = 1.0 / np.sqrt(max(erank_k, 1e-12) * max(erank_v, 1e-12))

    gram = (k @ k.T) * (v @ v.T)
    gram = (gram + gram.T) * 0.5 / n
    eigs = np.linalg.eigvalsh(gram)
    eigs = np.clip(eigs, 0.0, None)
    eig_sum = float(eigs.sum())
    write_vec_erank = _effective_rank_from_values(eigs)
    top1 = float(eigs[-1] / eig_sum) if eig_sum > 1e-12 else 0.0

    return {
        "write_reuse_score": float(write_reuse),
        "write_vec_erank": float(write_vec_erank),
        "write_vec_top1_energy_fraction": float(top1),
        "erank_k": float(erank_k),
        "erank_v": float(erank_v),
        "n_valid_write_samples": int(k.shape[0]),
    }


def _load_spectral_reference() -> dict[str, Any]:
    for path in [
        Path(f"{DATA}/mamba2_states/spectral_audit_mamba2.json"),
        Path(SPECTRAL_AUDIT_BUNDLE),
        Path(__file__).resolve().parent / "results" / "data" / "spectral_audit_mamba2.json",
    ]:
        if not path.exists():
            continue
        with open(path) as f:
            return json.load(f)
    raise FileNotFoundError("Missing spectral audit reference for Mamba-2")


def _select_pairs_from_metrics(
    metrics_by_pair: dict[str, dict[str, float]],
    *,
    predictor_name: str,
    n_pairs: int = N_SWEEP_PAIRS,
    max_per_layer: int = MAX_PAIRS_PER_LAYER,
) -> list[dict[str, Any]]:
    ranked = []
    for pair_key, metrics in metrics_by_pair.items():
        predictor = metrics.get(predictor_name)
        if predictor is None or not np.isfinite(predictor):
            continue
        ranked.append((float(predictor), pair_key, metrics))
    if len(ranked) < n_pairs:
        raise ValueError(
            f"Only found {len(ranked)} valid pair metrics for {predictor_name}, cannot select {n_pairs} pairs"
        )

    ranked.sort(key=lambda item: item[0])
    selected: list[tuple[float, str, dict[str, float]]] = []
    used_keys: set[str] = set()
    layer_counts: dict[int, int] = {}

    def parse_pair(pair_key: str) -> tuple[int, int]:
        layer_str, head_str = pair_key.split("_")
        return int(layer_str[1:]), int(head_str[1:])

    for slot in range(n_pairs):
        if n_pairs == 1:
            target_idx = len(ranked) // 2
        else:
            target_idx = int(round(slot * (len(ranked) - 1) / (n_pairs - 1)))

        best_candidate = None
        best_distance = None
        for idx, item in enumerate(ranked):
            _, pair_key, _ = item
            if pair_key in used_keys:
                continue
            layer, _ = parse_pair(pair_key)
            if layer_counts.get(layer, 0) >= max_per_layer:
                continue
            distance = abs(idx - target_idx)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_candidate = item

        if best_candidate is None:
            break

        value, pair_key, metrics = best_candidate
        layer, head = parse_pair(pair_key)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        used_keys.add(pair_key)
        selected.append((value, pair_key, metrics))

    if len(selected) < n_pairs:
        for value, pair_key, metrics in ranked:
            if pair_key in used_keys:
                continue
            layer, head = parse_pair(pair_key)
            if layer_counts.get(layer, 0) >= max_per_layer:
                continue
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            used_keys.add(pair_key)
            selected.append((value, pair_key, metrics))
            if len(selected) >= n_pairs:
                break

    selected_pairs = []
    for rank_idx, (value, pair_key, metrics) in enumerate(selected, start=1):
        layer, head = parse_pair(pair_key)
        payload = dict(metrics)
        payload["layer"] = layer
        payload["head"] = head
        payload["rank_order"] = rank_idx
        selected_pairs.append(payload)

    return selected_pairs


def _append_head_samples(
    sample_store: dict[tuple[int, int], dict[str, np.ndarray]],
    counts: dict[tuple[int, int], int],
    layer: int,
    heads: list[int],
    k_batch: np.ndarray,
    v_batch: np.ndarray,
) -> None:
    for local_idx, head in enumerate(heads):
        pair = (layer, head)
        limit = sample_store[pair]["k"].shape[0]
        count = counts[pair]
        if count >= limit:
            continue
        flat_k = k_batch[:, :, local_idx, :].reshape(-1, k_batch.shape[-1])
        flat_v = v_batch[:, :, local_idx, :].reshape(-1, v_batch.shape[-1])
        take = min(limit - count, flat_k.shape[0])
        if take <= 0:
            continue
        sample_store[pair]["k"][count : count + take] = flat_k[:take]
        sample_store[pair]["v"][count : count + take] = flat_v[:take]
        counts[pair] = count + take


class Mamba2HeadFactorCapture:
    def __init__(self, model: Any, layer_idx: int):
        self.layer_idx = layer_idx
        self.mixer = _get_mamba2_layers(model)[layer_idx].mixer
        self.hidden_states = None
        self.attention_mask = None
        self.handle = self.mixer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"Could not capture hidden_states for layer {self.layer_idx}")
        self.hidden_states = hidden_states.detach()
        self.attention_mask = kwargs.get("attention_mask")

    def pop_heads(self, heads: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as F

        if self.hidden_states is None:
            raise RuntimeError(f"No captured activations for layer {self.layer_idx}")

        hidden_states = _apply_padding_mask(self.hidden_states, self.attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        mixer = self.mixer

        projected_states = mixer.in_proj(hidden_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * mixer.intermediate_size
            - 2 * mixer.n_groups * mixer.ssm_state_size
            - mixer.num_heads
        ) // 2
        _, _, _, hidden_states_b_c, dt = projected_states.split(
            [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads],
            dim=-1,
        )

        hidden_states_b_c = hidden_states_b_c.transpose(1, 2)
        hidden_states_b_c = mixer.act(mixer.conv1d(hidden_states_b_c)[..., :seq_len].transpose(1, 2))
        hidden_states_b_c = _apply_padding_mask(hidden_states_b_c, self.attention_mask)

        hidden_ssm, b_factor, _ = torch.split(
            hidden_states_b_c,
            [
                mixer.intermediate_size,
                mixer.n_groups * mixer.ssm_state_size,
                mixer.n_groups * mixer.ssm_state_size,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + mixer.dt_bias)
        dt = torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])

        v_factor = hidden_ssm.reshape(batch_size, seq_len, -1, mixer.head_dim).float()
        k_factor = b_factor.reshape(batch_size, seq_len, -1, mixer.ssm_state_size).float()
        k_factor = k_factor.repeat_interleave(mixer.num_heads // mixer.n_groups, dim=2, output_size=mixer.num_heads)
        v_factor = v_factor * dt[..., None]

        if heads is not None:
            index = torch.tensor(heads, device=v_factor.device, dtype=torch.long)
            k_factor = k_factor.index_select(2, index)
            v_factor = v_factor.index_select(2, index)

        k_np = k_factor.detach().cpu().numpy()
        v_np = v_factor.detach().cpu().numpy()

        self.hidden_states = None
        self.attention_mask = None
        return k_np, v_np

    def remove(self) -> None:
        self.handle.remove()


@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    image=image,
    gpu="A100",
    timeout=14400,
    memory=32768,
)
def scan_heads(
    layers: list[int] = LAYERS,
    n_samples: int = HEAD_SCAN_N_SAMPLES,
    seq_len: int = SEQ_LEN,
    batch_size: int = EXTRACT_BATCH_SIZE,
    sample_size_per_head: int = HEAD_SCAN_SAMPLE_SIZE,
    n_pairs: int = N_SWEEP_PAIRS,
    max_per_layer: int = MAX_PAIRS_PER_LAYER,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    os.environ["HF_HOME"] = MODELS
    device = "cuda"
    print(f"Scanning heads on {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model_vol.commit()

    config = model.config
    n_layers = int(config.num_hidden_layers)
    n_heads = int(getattr(config, "num_heads", getattr(config, "n_head", 48)))
    state_size = int(getattr(config, "state_size", getattr(config, "ssm_state_size", 128)))
    head_dim = int(getattr(config, "head_dim", 64))
    bad_layers = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad_layers:
        raise ValueError(f"Requested invalid layers {bad_layers}; model has layers 0..{n_layers - 1}")
    print(f"Model loaded: {n_layers} layers, {n_heads} heads, state = ({head_dim}, {state_size})")

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    tokens: list[int] = []
    target = n_samples * seq_len * 2
    for row in ds:
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        if len(tokens) >= target:
            break
    token_ids = tokens[:target]

    n_seqs = min(n_samples, len(token_ids) // seq_len)
    batches = []
    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        seqs = [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(start, end)]
        batches.append(torch.tensor(seqs, dtype=torch.long))
    actual_samples = sum(b.shape[0] for b in batches)
    print(f"Prepared {len(batches)} batches, {actual_samples} sequences")

    sample_store: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    counts: dict[tuple[int, int], int] = {}
    for layer in layers:
        for head in range(n_heads):
            pair = (layer, head)
            sample_store[pair] = {
                "k": np.empty((sample_size_per_head, state_size), dtype=np.float32),
                "v": np.empty((sample_size_per_head, head_dim), dtype=np.float32),
            }
            counts[pair] = 0

    captures = {layer: Mamba2HeadFactorCapture(model, layer) for layer in layers}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Head scan")):
            input_ids = batch.to(device)
            _ = model(input_ids=input_ids, use_cache=False)

            for layer, capture in captures.items():
                k_batch, v_batch = capture.pop_heads()
                _append_head_samples(
                    sample_store,
                    counts,
                    layer,
                    list(range(n_heads)),
                    k_batch,
                    v_batch,
                )

            if all(count >= sample_size_per_head for count in counts.values()):
                print(f"Filled all per-head samples by batch {batch_idx + 1}")
                break

    for capture in captures.values():
        capture.remove()

    spectral = _load_spectral_reference()
    per_layer_probe = spectral.get("multihead_probe", {})
    metrics_by_pair: dict[str, dict[str, float]] = {}
    for layer in layers:
        for head in range(n_heads):
            pair = (layer, head)
            count = counts[pair]
            metrics = _write_metric_summary(
                sample_store[pair]["k"][:count],
                sample_store[pair]["v"][:count],
            )
            metrics["layer"] = float(layer)
            metrics["head"] = float(head)
            probe_entry = per_layer_probe.get(str(layer), {})
            per_head_sv = probe_entry.get("per_head_sv1_sv2", [])
            per_head_er = probe_entry.get("per_head_eff_rank", [])
            if head < len(per_head_sv):
                metrics["probe_sv1_sv2"] = float(per_head_sv[head])
            if head < len(per_head_er):
                metrics["probe_effective_rank"] = float(per_head_er[head])
            metrics_by_pair[_pair_key(layer, head)] = metrics

    selected_pairs = _select_pairs_from_metrics(
        metrics_by_pair,
        predictor_name="write_vec_top1_energy_fraction",
        n_pairs=n_pairs,
        max_per_layer=max_per_layer,
    )

    payload = {
        "model": MODEL_NAME,
        "layers": layers,
        "n_heads": n_heads,
        "n_samples": actual_samples,
        "seq_len": seq_len,
        "sample_size_per_head": sample_size_per_head,
        "predictor": "write_vec_top1_energy_fraction",
        "metrics_by_pair": metrics_by_pair,
        "selected_pairs": selected_pairs,
        "code_sha": CODE_SHA,
    }
    with open(HEAD_SCAN_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    selected_payload = {
        "model": MODEL_NAME,
        "predictor": "write_vec_top1_energy_fraction",
        "selection_source": str(HEAD_SCAN_PATH),
        "selected_pairs": selected_pairs,
        "n_samples": actual_samples,
        "seq_len": seq_len,
        "sample_size_per_head": sample_size_per_head,
        "code_sha": CODE_SHA,
    }
    with open(SELECTED_PAIRS_PATH, "w") as f:
        json.dump(selected_payload, f, indent=2)

    vol.commit()
    print("\nSelected head-level sweep pairs:")
    for pair in selected_pairs:
        print(
            f"  L{pair['layer']:>2} H{pair['head']:>2}: "
            f"write_vec_top1={pair['write_vec_top1_energy_fraction']:.4f}, "
            f"write_reuse={pair['write_reuse_score']:.4f}"
        )
    return selected_payload


@app.function(
    volumes={DATA: vol, MODELS: model_vol},
    image=image,
    gpu="A100",
    timeout=14400,
    memory=32768,
)
def extract_states(
    n_samples: int = TRAIN_N_SAMPLES,
    seq_len: int = SEQ_LEN,
    batch_size: int = EXTRACT_BATCH_SIZE,
    write_sample_size: int = PAIR_WRITE_SAMPLE_SIZE,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    selected_pairs = _load_selected_pairs()
    if not selected_pairs:
        raise ValueError("No selected pairs found. Run stage=scan first.")

    pair_list = [(int(pair["layer"]), int(pair["head"])) for pair in selected_pairs]
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in pair_list:
        heads_by_layer.setdefault(layer, []).append(head)
    for layer in heads_by_layer:
        heads_by_layer[layer] = sorted(set(heads_by_layer[layer]))

    os.environ["HF_HOME"] = MODELS
    device = "cuda"
    print(f"Extracting selected states from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model_vol.commit()

    config = model.config
    head_dim = int(getattr(config, "head_dim", 64))
    state_size = int(getattr(config, "state_size", getattr(config, "ssm_state_size", 128)))

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    tokens: list[int] = []
    target = n_samples * seq_len * 2
    for row in ds:
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        if len(tokens) >= target:
            break
    token_ids = tokens[:target]

    n_seqs = min(n_samples, len(token_ids) // seq_len)
    batches = []
    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        seqs = [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(start, end)]
        batches.append(torch.tensor(seqs, dtype=torch.long))
    actual_samples = sum(b.shape[0] for b in batches)
    print(f"Prepared {len(batches)} batches, {actual_samples} sequences")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    WRITE_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    memmaps: dict[tuple[int, int], np.memmap] = {}
    for layer, head in pair_list:
        layer_dir = STATE_DIR / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = layer_dir / f"head_{head}_tmp.dat"
        memmaps[(layer, head)] = np.memmap(
            str(tmp_path),
            dtype=np.float32,
            mode="w+",
            shape=(actual_samples, head_dim, state_size),
        )

    sample_store: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    counts: dict[tuple[int, int], int] = {}
    for layer, head in pair_list:
        pair = (layer, head)
        sample_store[pair] = {
            "k": np.empty((write_sample_size, state_size), dtype=np.float32),
            "v": np.empty((write_sample_size, head_dim), dtype=np.float32),
        }
        counts[pair] = 0

    captures = {layer: Mamba2HeadFactorCapture(model, layer) for layer in heads_by_layer}

    sample_offset = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Extract selected heads")):
            input_ids = batch.to(device)
            bs = input_ids.shape[0]
            outputs = model(input_ids=input_ids, use_cache=True)
            cache = _resolve_cache(outputs)

            for layer, heads in heads_by_layer.items():
                state = _layer_ssm_state(cache, layer).float().cpu().numpy()
                for head in heads:
                    memmaps[(layer, head)][sample_offset : sample_offset + bs] = state[:, head]

            for layer, heads in heads_by_layer.items():
                capture = captures[layer]
                k_batch, v_batch = capture.pop_heads(heads)
                _append_head_samples(sample_store, counts, layer, heads, k_batch, v_batch)

            sample_offset += bs
            del outputs, cache
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(batches)}, {sample_offset}/{actual_samples} samples")

    for capture in captures.values():
        capture.remove()

    elapsed = time.time() - t0
    for (layer, head), mm in memmaps.items():
        mm.flush()
        arr = np.array(mm, dtype=np.float32)
        layer_dir = STATE_DIR / f"layer_{layer}"
        np.save(str(layer_dir / f"head_{head}.npy"), arr)
        tmp_path = layer_dir / f"head_{head}_tmp.dat"
        del mm, arr
        if tmp_path.exists():
            tmp_path.unlink()

    write_metrics_by_pair: dict[str, dict[str, float]] = {}
    for layer, head in pair_list:
        pair = (layer, head)
        layer_dir = WRITE_SAMPLE_DIR / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        count = counts[pair]
        k_arr = np.array(sample_store[pair]["k"][:count], dtype=np.float32)
        v_arr = np.array(sample_store[pair]["v"][:count], dtype=np.float32)
        np.save(str(layer_dir / f"head_{head}_k.npy"), k_arr)
        np.save(str(layer_dir / f"head_{head}_v.npy"), v_arr)
        write_metrics_by_pair[_pair_key(layer, head)] = _write_metric_summary(k_arr, v_arr)

    metadata = {
        "model": MODEL_NAME,
        "pairs": [{"layer": layer, "head": head} for layer, head in pair_list],
        "n_samples": actual_samples,
        "seq_len": seq_len,
        "extract_batch_size": batch_size,
        "head_dim": head_dim,
        "state_size": state_size,
        "dtype": "float32",
        "model_dtype": "bfloat16",
        "trust_remote_code": True,
        "write_sample_size": write_sample_size,
        "write_metrics_by_pair": write_metrics_by_pair,
        "extraction_time_s": elapsed,
        "code_sha": CODE_SHA,
    }
    with open(STATE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    vol.commit()
    print(f"Extracted {actual_samples} states for {len(pair_list)} (layer, head) pairs in {elapsed:.0f}s")
    return metadata


@app.function(
    volumes={DATA: vol},
    image=image,
    gpu="A10G",
    timeout=1200,
    memory=16384,
)
def train_sae(layer: int, head: int, sae_type: str, seed: int, n_features: int = 2048, k: int = 32):
    import sys

    sys.path.insert(0, "/root")
    from train import train

    metadata_path = STATE_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing extraction metadata: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("seq_len") != SEQ_LEN:
        raise ValueError(f"Expected seq_len={SEQ_LEN}, found {metadata.get('seq_len')}")
    if metadata.get("dtype") != "float32":
        raise ValueError(f"Expected float32 states, found {metadata.get('dtype')}")

    pair_set = {(int(pair["layer"]), int(pair["head"])) for pair in metadata.get("pairs", [])}
    if (layer, head) not in pair_set:
        raise ValueError(f"Requested missing pair (L{layer}, H{head})")

    ckpt_dir = CHECKPOINT_DIR / f"layer_{layer}" / f"head_{head}" / f"{sae_type}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        sae_type=sae_type,
        data_dir=str(STATE_DIR),
        layer=layer,
        head=head,
        n_features=n_features,
        k=k,
        seed=seed,
        output_dir=str(ckpt_dir),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=3e-4,
        lr_min=3e-5,
        warmup_steps=TRAIN_WARMUP_STEPS,
        resample_every=TRAIN_RESAMPLE_EVERY,
        log_every=25,
    )

    vol.commit()
    print(
        f"L{layer} H{head} {sae_type} s{seed}: "
        f"val_mse={result.get('best_mse', result.get('best_val_mse', '?'))}"
    )
    return {"layer": layer, "head": head, "sae_type": sae_type, "seed": seed, **result}


def _compute_advantage_pct(results: dict[str, dict[str, object]], better: str, baseline: str) -> float | None:
    if better not in results or baseline not in results:
        return None
    better_mse = float(results[better]["mean"])
    baseline_mse = float(results[baseline]["mean"])
    return _safe_div(baseline_mse - better_mse, baseline_mse) * 100.0


@app.function(
    volumes={DATA: vol},
    image=image,
    timeout=1800,
    memory=16384,
)
def analyze_results():
    import torch
    from scipy.stats import spearmanr

    selected_pairs = _load_selected_pairs()
    if not selected_pairs:
        raise ValueError("No selected pairs found. Run stage=scan first.")

    state_meta_path = STATE_DIR / "metadata.json"
    if not state_meta_path.exists():
        raise FileNotFoundError(f"Missing extraction metadata: {state_meta_path}")
    with open(state_meta_path) as f:
        state_meta = json.load(f)

    results_by_pair: list[dict[str, Any]] = []
    predictor_values: list[float] = []
    advantages: list[float] = []
    write_reuse_values: list[float] = []
    write_vec_eranks: list[float] = []
    paired_rows: list[dict[str, float | int]] = []

    for pair in selected_pairs:
        layer = int(pair["layer"])
        head = int(pair["head"])
        pair_key = _pair_key(layer, head)

        k_path = WRITE_SAMPLE_DIR / f"layer_{layer}" / f"head_{head}_k.npy"
        v_path = WRITE_SAMPLE_DIR / f"layer_{layer}" / f"head_{head}_v.npy"
        if not k_path.exists() or not v_path.exists():
            raise FileNotFoundError(f"Missing write samples for {pair_key}")
        write_metrics = _write_metric_summary(np.load(k_path), np.load(v_path))

        ckpt_base = CHECKPOINT_DIR / f"layer_{layer}" / f"head_{head}"
        mse_by_type: dict[str, dict[str, object]] = {}
        missing_seeds: dict[str, list[int]] = {}
        for sae_type in SAE_TYPES:
            mses = []
            present = []
            absent = []
            for seed in SEEDS:
                ckpt_path = ckpt_base / f"{sae_type}_s{seed}" / "best.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    val_mse = ckpt.get("val_mse", ckpt.get("best_val_mse"))
                    if val_mse is not None:
                        mses.append(float(val_mse))
                        present.append(seed)
                else:
                    absent.append(seed)
            if mses:
                mse_by_type[sae_type] = {
                    "mean": float(np.mean(mses)),
                    "std": float(np.std(mses)),
                    "seed_mses": dict(zip(present, mses, strict=False)),
                }
            if absent:
                missing_seeds[sae_type] = absent

        rank1_adv = _compute_advantage_pct(mse_by_type, better="rank1", baseline="flat")
        entry = {
            "layer": layer,
            "head": head,
            "pair_key": pair_key,
            "selection_predictor": float(pair.get("write_vec_top1_energy_fraction", np.nan)),
            "write_metrics": write_metrics,
            "mse_by_type": mse_by_type,
            "missing_seeds": missing_seeds,
            "rank1_advantage_pct": rank1_adv,
        }
        results_by_pair.append(entry)

        if rank1_adv is not None:
            predictor = write_metrics.get("write_vec_top1_energy_fraction")
            reuse = write_metrics.get("write_reuse_score")
            erank = write_metrics.get("write_vec_erank")
            if predictor is not None and np.isfinite(predictor):
                predictor_values.append(float(predictor))
                advantages.append(float(rank1_adv))
            if reuse is not None and np.isfinite(reuse):
                write_reuse_values.append(float(reuse))
            if erank is not None and np.isfinite(erank):
                write_vec_eranks.append(float(erank))
            if (
                predictor is not None
                and np.isfinite(predictor)
                and reuse is not None
                and np.isfinite(reuse)
                and erank is not None
                and np.isfinite(erank)
            ):
                paired_rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "predictor": float(predictor),
                        "selection_predictor": float(pair.get("write_vec_top1_energy_fraction", np.nan)),
                        "write_reuse": float(reuse),
                        "write_vec_erank": float(erank),
                        "advantage": float(rank1_adv),
                    }
                )

    def corr(xs: list[float], ys: list[float]) -> dict[str, float]:
        if len(xs) < 3 or len(xs) != len(ys):
            return {"spearman_rho": 0.0, "p_value": 1.0, "n": len(xs)}
        rho, p = spearmanr(xs, ys)
        return {"spearman_rho": float(rho), "p_value": float(p), "n": len(xs)}

    def corr_rows(rows: list[dict[str, float | int]], x_key: str, y_key: str) -> dict[str, float]:
        xs = [float(row[x_key]) for row in rows if np.isfinite(float(row[x_key])) and np.isfinite(float(row[y_key]))]
        ys = [float(row[y_key]) for row in rows if np.isfinite(float(row[x_key])) and np.isfinite(float(row[y_key]))]
        return corr(xs, ys)

    def layer_centered_corr(rows: list[dict[str, float | int]], x_key: str, y_key: str) -> dict[str, float]:
        grouped: dict[int, list[dict[str, float | int]]] = {}
        for row in rows:
            grouped.setdefault(int(row["layer"]), []).append(row)

        xs: list[float] = []
        ys: list[float] = []
        for layer_rows in grouped.values():
            if len(layer_rows) < 2:
                continue
            x_mean = float(np.mean([float(row[x_key]) for row in layer_rows]))
            y_mean = float(np.mean([float(row[y_key]) for row in layer_rows]))
            for row in layer_rows:
                xs.append(float(row[x_key]) - x_mean)
                ys.append(float(row[y_key]) - y_mean)
        return corr(xs, ys)

    def leave_one_layer_out(rows: list[dict[str, float | int]], x_key: str, y_key: str) -> dict[str, dict[str, float]]:
        outputs: dict[str, dict[str, float]] = {}
        layers = sorted({int(row["layer"]) for row in rows})
        for held_out in layers:
            subset = [row for row in rows if int(row["layer"]) != held_out]
            outputs[str(held_out)] = corr_rows(subset, x_key, y_key)
        return outputs

    def per_layer_corr(rows: list[dict[str, float | int]], x_key: str, y_key: str) -> dict[str, dict[str, float]]:
        outputs: dict[str, dict[str, float]] = {}
        layers = sorted({int(row["layer"]) for row in rows})
        for layer in layers:
            subset = [row for row in rows if int(row["layer"]) == layer]
            outputs[str(layer)] = corr_rows(subset, x_key, y_key)
        return outputs

    paired_predictor_values = []
    paired_reuse_values = []
    paired_eranks = []
    paired_advantages = []
    for entry in results_by_pair:
        adv = entry["rank1_advantage_pct"]
        wm = entry["write_metrics"]
        predictor = wm.get("write_vec_top1_energy_fraction")
        reuse = wm.get("write_reuse_score")
        erank = wm.get("write_vec_erank")
        if adv is None or predictor is None or reuse is None or erank is None:
            continue
        paired_predictor_values.append(float(predictor))
        paired_reuse_values.append(float(reuse))
        paired_eranks.append(float(erank))
        paired_advantages.append(float(adv))

    output = {
        "model": MODEL_NAME,
        "predictor_name": "write_vec_top1_energy_fraction",
        "selected_pairs": selected_pairs,
        "train_config": {
            "n_samples": state_meta["n_samples"],
            "seq_len": state_meta["seq_len"],
            "batch_size": TRAIN_BATCH_SIZE,
            "epochs": TRAIN_EPOCHS,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "resample_every": TRAIN_RESAMPLE_EVERY,
            "seeds": SEEDS,
            "sae_types": SAE_TYPES,
        },
        "per_pair": results_by_pair,
        "correlations": {
            "write_vec_top1_energy_fraction_vs_rank1_adv": corr(paired_predictor_values, paired_advantages),
            "write_reuse_score_vs_rank1_adv": corr(paired_reuse_values, paired_advantages),
            "write_vec_erank_vs_rank1_adv": corr(paired_eranks, paired_advantages),
            "selection_predictor_vs_recomputed_predictor": corr_rows(
                paired_rows,
                "selection_predictor",
                "predictor",
            ),
            "layer_centered_write_vec_top1_vs_rank1_adv": layer_centered_corr(
                paired_rows,
                "predictor",
                "advantage",
            ),
        },
        "stability": {
            "per_layer_write_vec_top1_vs_rank1_adv": per_layer_corr(
                paired_rows,
                "predictor",
                "advantage",
            ),
            "leave_one_layer_out_write_vec_top1_vs_rank1_adv": leave_one_layer_out(
                paired_rows,
                "predictor",
                "advantage",
            ),
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    vol.commit()

    main_corr = output["correlations"]["write_vec_top1_energy_fraction_vs_rank1_adv"]
    print(f"\n{'=' * 72}")
    print("MAMBA-2 MULTI-HEAD SAE RESULTS")
    print(f"{'=' * 72}")
    print(f"{'Pair':>9}  {'write_vec_top1':>14}  {'write_reuse':>12}  {'rank1_adv%':>11}")
    print("-" * 72)
    for entry in sorted(results_by_pair, key=lambda item: (item["layer"], item["head"])):
        wm = entry["write_metrics"]
        adv = entry["rank1_advantage_pct"]
        adv_str = f"{adv:+.1f}" if adv is not None else "n/a"
        print(
            f"L{entry['layer']:>2}H{entry['head']:>2}  "
            f"{wm.get('write_vec_top1_energy_fraction', float('nan')):>14.4f}  "
            f"{wm.get('write_reuse_score', float('nan')):>12.4f}  "
            f"{adv_str:>11}"
        )

    print("\nCorrelations:")
    for name, stats in output["correlations"].items():
        print(f"  {name}: rho={stats['spearman_rho']:+.3f}, p={stats['p_value']:.4f}, N={stats['n']}")

    print("\nPer-layer correlations:")
    for layer, stats in output["stability"]["per_layer_write_vec_top1_vs_rank1_adv"].items():
        print(f"  layer {layer}: rho={stats['spearman_rho']:+.3f}, p={stats['p_value']:.4f}, N={stats['n']}")

    print("\nLeave-one-layer-out correlations:")
    for layer, stats in output["stability"]["leave_one_layer_out_write_vec_top1_vs_rank1_adv"].items():
        print(f"  without layer {layer}: rho={stats['spearman_rho']:+.3f}, p={stats['p_value']:.4f}, N={stats['n']}")

    if main_corr["spearman_rho"] > 0.4 and main_corr["p_value"] < 0.05:
        print("RESULT: Head-level predictor transfer looks real.")
    else:
        print("RESULT: Head-level predictor transfer is still not established.")

    return output


@app.function(
    volumes={DATA: vol},
    image=image,
    timeout=300,
)
def get_selected_pairs():
    return _load_selected_pairs()


@app.local_entrypoint()
def main(
    stage: str = "all",
    n_pairs: int = N_SWEEP_PAIRS,
    scan_n_samples: int = HEAD_SCAN_N_SAMPLES,
    train_n_samples: int = TRAIN_N_SAMPLES,
):
    if stage in ("scan", "all"):
        print("=== Stage 1: Scan heads ===")
        payload = scan_heads.remote(n_samples=scan_n_samples, n_pairs=n_pairs)
        print(f"Selected {len(payload['selected_pairs'])} layer/head pairs")

    if stage in ("extract", "all"):
        print("=== Stage 2: Extract selected states ===")
        meta = extract_states.remote(n_samples=train_n_samples)
        print(f"Extraction complete: {meta['n_samples']} sequences")

    if stage in ("train", "all"):
        print("=== Stage 3: Train SAEs ===")
        selected_pairs = get_selected_pairs.remote()
        jobs = []
        for pair in selected_pairs:
            layer = int(pair["layer"])
            head = int(pair["head"])
            for sae_type in SAE_TYPES:
                for seed in SEEDS:
                    jobs.append(train_sae.spawn(layer, head, sae_type, seed))
        print(f"Spawned {len(jobs)} training jobs")
        results = [job.get() for job in jobs]
        print(f"Training complete: {len(results)} checkpoints")

    if stage in ("analyze", "all"):
        print("=== Stage 4: Analyze ===")
        output = analyze_results.remote()
        corr = output["correlations"]["write_vec_top1_energy_fraction_vs_rank1_adv"]
        print(f"Analysis complete: rho={corr['spearman_rho']:.3f}, p={corr['p_value']:.4f}")

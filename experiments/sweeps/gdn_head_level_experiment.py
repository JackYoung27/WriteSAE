#!/usr/bin/env python3
"""GDN head-level write geometry + SAE advantage correlation.

Extracts exact GDN write vectors for all 16 heads at layer 9 of Qwen3.5-0.8B,
computes write_vec_top1_energy_fraction per head, and correlates with rank-1
SAE advantage from existing per-head checkpoints.

This produces a head-level GDN result (N=16) that matches the Mamba-2
head-level experiment (N=16, rho=-0.588), enabling an apples-to-apples
cross-architecture comparison.

Usage:
    modal run gdn_head_level_experiment.py --stage extract
    modal run gdn_head_level_experiment.py --stage analyze
    modal run gdn_head_level_experiment.py --stage all
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal
import numpy as np
import torch


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

GDN_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
GDN_LAYER = 9
N_HEADS = 16
N_SEQUENCES = 500
SEQ_LEN = 1024
BATCH_SIZE = 2
WRITE_SAMPLE_SIZE = 2048

# Per-head SAE MSEs from existing checkpoints (modal_layer9_perhead_checkpoint_summary_clean_08b.json).
# Loaded at analyze time from local file or from volume metadata.

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
    .add_local_file(
        "results/data/modal_layer9_perhead_checkpoint_summary_clean_08b.json",
        "/root/perhead_checkpoint_summary.json",
        copy=True,
    )
    .env({"MATRIX_SAE_CODE_SHA": CODE_SHA})
)

app = modal.App("gdn-head-level")
gdn_vol = modal.Volume.from_name("matrix-sae-data-08b-clean", create_if_missing=True)
model_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

GDN_DATA = "/gdn_data"
MODELS = "/models"
WRITE_DIR = Path(f"{GDN_DATA}/write_vectors_multihead")


def _get_qwen_layers(model) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return list(model.language_model.layers)
    raise AttributeError("Could not find Qwen layers on model")


def _l2norm(x, dim: int = -1, eps: float = 1e-6):
    import torch
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _apply_padding_mask(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[0] > 1 and attention_mask.shape[1] > 1:
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


class GDNMultiHeadWriteCapture:
    """Capture exact write factors for multiple GDN heads in a single forward pass."""

    def __init__(self, model, layer_idx: int, heads: list[int]):
        self.layer_idx = layer_idx
        self.heads = heads
        self.gdn = _get_qwen_layers(model)[layer_idx].linear_attn
        self.hidden_states = None
        self.attention_mask = None
        self.handle = self.gdn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(f"Could not capture hidden_states for GDN layer {self.layer_idx}")
        self.hidden_states = hidden_states.detach()
        self.attention_mask = kwargs.get("attention_mask")

    def pop_write_factors(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return {head_idx: (k_writes, v_writes)} where each is (batch*seq_len, dim)."""
        import torch
        import torch.nn.functional as F

        if self.hidden_states is None:
            raise RuntimeError(f"No GDN activations captured for layer {self.layer_idx}")

        hidden_states = _apply_padding_mask(self.hidden_states, self.attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        gdn = self.gdn

        # Replay the GDN projections (same as GDNWriteCapture but for all heads).
        mixed_qkv = gdn.in_proj_qkv(hidden_states).transpose(1, 2)
        b = gdn.in_proj_b(hidden_states)
        a = gdn.in_proj_a(hidden_states)

        if gdn.causal_conv1d_fn is not None:
            mixed_qkv = gdn.causal_conv1d_fn(
                x=mixed_qkv,
                weight=gdn.conv1d.weight.squeeze(1),
                bias=gdn.conv1d.bias,
                activation=gdn.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(gdn.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        _, key, value = torch.split(
            mixed_qkv,
            [gdn.key_dim, gdn.key_dim, gdn.value_dim],
            dim=-1,
        )

        key = key.reshape(batch_size, seq_len, -1, gdn.head_k_dim).float()
        value = value.reshape(batch_size, seq_len, -1, gdn.head_v_dim).float()
        beta = b.sigmoid().float()
        g = (-gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)).float()

        if gdn.num_v_heads // gdn.num_k_heads > 1:
            key = key.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)

        key = _l2norm(key, dim=-1)

        # Vectorize the recurrence across all requested heads at once.
        # Shapes: key/value are (batch, seq_len, n_all_heads, dim).
        # Select only the requested heads for efficiency.
        head_indices = torch.tensor(self.heads, device=key.device, dtype=torch.long)
        n_h = len(self.heads)
        key_sel = key.index_select(2, head_indices)      # (B, T, n_h, k_dim)
        value_sel = value.index_select(2, head_indices)   # (B, T, n_h, v_dim)
        beta_sel = beta.index_select(2, head_indices)     # (B, T, n_h)
        g_sel = g.index_select(2, head_indices)           # (B, T, n_h)

        # Recurrent state: (B, n_h, k_dim, v_dim)
        recurrent = torch.zeros(
            batch_size, n_h, gdn.head_k_dim, gdn.head_v_dim,
            dtype=torch.float32, device=key.device,
        )
        k_writes = torch.empty(batch_size, seq_len, n_h, gdn.head_k_dim, dtype=torch.float32, device=key.device)
        v_writes = torch.empty(batch_size, seq_len, n_h, gdn.head_v_dim, dtype=torch.float32, device=key.device)

        for t in range(seq_len):
            # g_sel[:, t] is (B, n_h), need (B, n_h, 1, 1) for broadcast
            recurrent = recurrent * g_sel[:, t].exp().unsqueeze(-1).unsqueeze(-1)
            # key_sel[:, t] is (B, n_h, k_dim)
            # recurrent * key_sel[:, t, :, :, None] -> (B, n_h, k_dim, v_dim) * (B, n_h, k_dim, 1)
            # sum over k_dim -> (B, n_h, v_dim)
            kv_mem = (recurrent * key_sel[:, t].unsqueeze(-1)).sum(dim=-2)
            delta = (value_sel[:, t] - kv_mem) * beta_sel[:, t].unsqueeze(-1)
            # key_sel[:, t, :, :, None] outer delta[:, :, None, :] -> (B, n_h, k_dim, v_dim)
            recurrent = recurrent + key_sel[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            k_writes[:, t] = key_sel[:, t]
            v_writes[:, t] = delta

        # Split back into per-head results.
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        k_all = k_writes.detach().cpu().numpy()  # (B, T, n_h, k_dim)
        v_all = v_writes.detach().cpu().numpy()   # (B, T, n_h, v_dim)
        for local_idx, head_idx in enumerate(self.heads):
            k_np = k_all[:, :, local_idx, :].reshape(-1, gdn.head_k_dim)
            v_np = v_all[:, :, local_idx, :].reshape(-1, gdn.head_v_dim)
            results[head_idx] = (k_np, v_np)

        self.hidden_states = None
        self.attention_mask = None
        return results

    def remove(self) -> None:
        self.handle.remove()


def _stream_openwebtext_batches(tokenizer, n_sequences: int, seq_len: int, batch_size: int):
    import torch
    from datasets import load_dataset

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    token_ids: list[int] = []
    target = n_sequences * seq_len * 2
    for row in ds:
        token_ids.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        if len(token_ids) >= target:
            break

    actual_sequences = min(n_sequences, len(token_ids) // seq_len)
    batches = []
    for start in range(0, actual_sequences, batch_size):
        end = min(start + batch_size, actual_sequences)
        seqs = [token_ids[i * seq_len : (i + 1) * seq_len] for i in range(start, end)]
        batches.append(torch.tensor(seqs, dtype=torch.long))
    return batches, actual_sequences, len(token_ids)


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def _effective_rank(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = float(vals.sum())
    if total <= eps:
        return 0.0
    p = vals / total
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))


def _write_process_metrics(
    k_vectors: np.ndarray,
    v_vectors: np.ndarray,
    sample_size: int = WRITE_SAMPLE_SIZE,
) -> dict[str, float | int] | None:
    """Compute write_vec_top1_energy_fraction and related metrics for one head."""
    valid = (np.linalg.norm(k_vectors, axis=1) > 1e-8) & (np.linalg.norm(v_vectors, axis=1) > 1e-8)
    k_valid = k_vectors[valid].astype(np.float64, copy=False)
    v_valid = v_vectors[valid].astype(np.float64, copy=False)
    if k_valid.shape[0] < 2:
        return None

    rng = np.random.default_rng(0)
    sample_n = min(sample_size, k_valid.shape[0])
    if sample_n < k_valid.shape[0]:
        sample_idx = np.sort(rng.choice(k_valid.shape[0], size=sample_n, replace=False))
    else:
        sample_idx = np.arange(sample_n)

    k_sample = _normalize_rows(k_valid[sample_idx])
    v_sample = _normalize_rows(v_valid[sample_idx])

    gram_k = k_sample @ k_sample.T
    gram_v = v_sample @ v_sample.T
    gram = (gram_k * gram_v) / sample_n
    eigs = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    eigs = np.clip(eigs, 0.0, None)
    eig_sum = float(eigs.sum())
    write_vec_erank = _effective_rank(eigs)
    write_vec_top1 = float(eigs[-1] / eig_sum) if eig_sum > 1e-12 else 0.0

    # Also compute individual key/value reuse metrics.
    n = k_sample.shape[0]
    c_k = (k_sample.T @ k_sample) / n
    c_v = (v_sample.T @ v_sample) / n
    eigs_k = np.linalg.eigvalsh(c_k)
    eigs_v = np.linalg.eigvalsh(c_v)
    erank_k = _effective_rank(eigs_k)
    erank_v = _effective_rank(eigs_v)
    write_reuse = 1.0 / np.sqrt(max(erank_k, 1e-12) * max(erank_v, 1e-12))

    return {
        "write_vec_top1_energy_fraction": write_vec_top1,
        "write_vec_erank": write_vec_erank,
        "write_reuse_score": float(write_reuse),
        "erank_k": erank_k,
        "erank_v": erank_v,
        "n_valid_writes": int(k_valid.shape[0]),
        "sample_size": int(sample_n),
    }


@app.function(
    volumes={GDN_DATA: gdn_vol, MODELS: model_vol},
    image=image,
    gpu="A100",
    timeout=7200,
    memory=32768,
)
def extract_gdn_multihead_writes(
    layer: int = GDN_LAYER,
    heads: list[int] | None = None,
    n_sequences: int = N_SEQUENCES,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
):
    """Extract exact GDN write vectors for all heads at a single layer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    if heads is None:
        heads = list(range(N_HEADS))

    os.environ["HF_HOME"] = MODELS
    device = "cuda"

    print(f"Loading {GDN_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(GDN_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GDN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model_vol.commit()

    model_layers = _get_qwen_layers(model)
    gdn_layers = [idx for idx, layer_mod in enumerate(model_layers) if hasattr(layer_mod, "linear_attn")]
    if layer not in gdn_layers:
        raise ValueError(f"Layer {layer} is not a GDN layer. GDN layers: {gdn_layers}")

    sample_gdn = model_layers[layer].linear_attn
    head_k_dim = int(sample_gdn.head_k_dim)
    head_v_dim = int(sample_gdn.head_v_dim)
    n_value_heads = int(sample_gdn.num_v_heads)
    print(f"GDN layer {layer}: {n_value_heads} heads, k_dim={head_k_dim}, v_dim={head_v_dim}")
    print(f"Extracting write vectors for heads: {heads}")

    batches, actual_sequences, n_tokens = _stream_openwebtext_batches(tokenizer, n_sequences, seq_len, batch_size)
    total_tokens = actual_sequences * seq_len
    print(f"Collected {n_tokens} tokens, processing {actual_sequences} sequences in {len(batches)} batches")

    write_dir = WRITE_DIR / f"layer_{layer}"
    write_dir.mkdir(parents=True, exist_ok=True)

    # Allocate memmaps per head.
    k_memmaps: dict[int, np.memmap] = {}
    v_memmaps: dict[int, np.memmap] = {}
    for h in heads:
        k_memmaps[h] = np.memmap(
            str(write_dir / f"head_{h}_k_tmp.dat"),
            dtype=np.float32, mode="w+", shape=(total_tokens, head_k_dim),
        )
        v_memmaps[h] = np.memmap(
            str(write_dir / f"head_{h}_v_tmp.dat"),
            dtype=np.float32, mode="w+", shape=(total_tokens, head_v_dim),
        )

    capture = GDNMultiHeadWriteCapture(model, layer, heads)

    token_offset = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="GDN multihead extraction")):
            input_ids = batch.to(device)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]
            _ = model(input_ids=input_ids, use_cache=False)

            head_results = capture.pop_write_factors()
            for h in heads:
                k_batch, v_batch = head_results[h]
                k_memmaps[h][token_offset : token_offset + batch_tokens] = k_batch
                v_memmaps[h][token_offset : token_offset + batch_tokens] = v_batch

            token_offset += batch_tokens
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = token_offset / max(elapsed, 1e-6)
                eta = (total_tokens - token_offset) / max(rate, 1e-6)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}, "
                    f"{token_offset}/{total_tokens} tokens, "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

    capture.remove()
    elapsed = time.time() - t0

    # Save as .npy and clean up memmaps.
    for h in heads:
        k_arr = np.array(k_memmaps[h][:token_offset], dtype=np.float32)
        v_arr = np.array(v_memmaps[h][:token_offset], dtype=np.float32)
        np.save(str(write_dir / f"head_{h}_k.npy"), k_arr)
        np.save(str(write_dir / f"head_{h}_v.npy"), v_arr)
        del k_memmaps[h], v_memmaps[h]
        for tmp in write_dir.glob(f"head_{h}_*_tmp.dat"):
            tmp.unlink()

    metadata = {
        "model": GDN_MODEL_NAME,
        "layer": layer,
        "heads": heads,
        "n_sequences": actual_sequences,
        "seq_len": seq_len,
        "n_write_vectors": token_offset,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "extraction_time_s": elapsed,
        "method": "exact_gdn_multihead_factors_sequential_recurrence",
        "code_sha": CODE_SHA,
    }
    with open(WRITE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    gdn_vol.commit()
    print(
        f"Extracted {token_offset} write vectors for {len(heads)} heads at layer {layer} "
        f"in {elapsed:.0f}s ({elapsed / max(actual_sequences, 1):.1f}s/seq)"
    )
    return metadata


@app.function(
    volumes={GDN_DATA: gdn_vol},
    image=image,
    timeout=1200,
    memory=16384,
)
def analyze_gdn_head_level():
    """Correlate per-head write geometry with rank-1 SAE advantage."""
    from scipy.stats import spearmanr, pearsonr

    # Load extraction metadata.
    meta_path = WRITE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing write vector metadata: {meta_path}. Run --stage extract first.")
    with open(meta_path) as f:
        meta = json.load(f)

    layer = meta["layer"]
    heads = meta["heads"]
    print(f"Analyzing write geometry for {len(heads)} heads at GDN layer {layer}")

    # Load per-head SAE checkpoint summary.
    summary_path = Path("/root/perhead_checkpoint_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError("Missing per-head checkpoint summary")
    with open(summary_path) as f:
        ckpt_summary = json.load(f)

    # Build per-head MSE lookup: {(sae_type, head): val_mse}.
    mse_lookup: dict[tuple[str, int], float] = {}
    for row in ckpt_summary["rows"]:
        mse_lookup[(row["sae_type"], row["head"])] = row["val_mse"]

    # Compute write metrics and SAE advantage per head.
    per_head: list[dict[str, float | int]] = []
    write_vec_top1_values: list[float] = []
    advantages: list[float] = []
    write_reuse_values: list[float] = []

    for h in heads:
        layer_dir = WRITE_DIR / f"layer_{layer}"
        k_path = layer_dir / f"head_{h}_k.npy"
        v_path = layer_dir / f"head_{h}_v.npy"
        if not k_path.exists() or not v_path.exists():
            print(f"  Head {h}: missing write vectors, skipping")
            continue

        k = np.load(str(k_path))
        v = np.load(str(v_path))
        metrics = _write_process_metrics(k, v)
        if metrics is None:
            print(f"  Head {h}: insufficient valid writes, skipping")
            continue

        flat_mse = mse_lookup.get(("flat", h))
        rank1_mse = mse_lookup.get(("rank1", h))
        if flat_mse is None or rank1_mse is None:
            print(f"  Head {h}: missing SAE checkpoint MSE, skipping")
            continue

        rank1_adv = (flat_mse - rank1_mse) / flat_mse * 100.0

        entry = {
            "head": h,
            "flat_mse": flat_mse,
            "rank1_mse": rank1_mse,
            "rank1_advantage_pct": rank1_adv,
            **metrics,
        }
        per_head.append(entry)
        write_vec_top1_values.append(metrics["write_vec_top1_energy_fraction"])
        advantages.append(rank1_adv)
        write_reuse_values.append(metrics["write_reuse_score"])

        print(
            f"  Head {h:>2}: top1={metrics['write_vec_top1_energy_fraction']:.4f}  "
            f"erank={metrics['write_vec_erank']:.1f}  "
            f"reuse={metrics['write_reuse_score']:.4f}  "
            f"rank1_adv={rank1_adv:+.1f}%"
        )

    n = len(per_head)
    if n < 3:
        print(f"Only {n} valid heads, cannot compute correlations")
        return {"error": "insufficient_heads", "n_valid": n}

    # Correlations.
    def corr_pair(xs, ys, label):
        rho, p = spearmanr(xs, ys)
        r, rp = pearsonr(xs, ys)
        print(f"  {label}: Spearman rho={rho:+.3f} (p={p:.4f}), Pearson r={r:+.3f} (p={rp:.4f}), N={len(xs)}")
        return {"spearman_rho": float(rho), "spearman_p": float(p), "pearson_r": float(r), "pearson_p": float(rp), "n": len(xs)}

    print(f"\n{'=' * 72}")
    print(f"GDN HEAD-LEVEL RESULTS (layer {layer}, {n} heads)")
    print(f"{'=' * 72}")
    print(f"{'Head':>4}  {'top1':>8}  {'erank':>8}  {'reuse':>8}  {'rank1_adv%':>11}")
    print("-" * 55)
    for entry in sorted(per_head, key=lambda e: e["head"]):
        print(
            f"H{entry['head']:>3}  "
            f"{entry['write_vec_top1_energy_fraction']:>8.4f}  "
            f"{entry['write_vec_erank']:>8.1f}  "
            f"{entry['write_reuse_score']:>8.4f}  "
            f"{entry['rank1_advantage_pct']:>+11.1f}"
        )

    print("\nCorrelations:")
    corr_top1 = corr_pair(write_vec_top1_values, advantages, "write_vec_top1 vs rank1_adv")
    corr_reuse = corr_pair(write_reuse_values, advantages, "write_reuse vs rank1_adv")

    eranks = [e["write_vec_erank"] for e in per_head]
    corr_erank = corr_pair(eranks, advantages, "write_vec_erank vs rank1_adv")

    output = {
        "model": GDN_MODEL_NAME,
        "architecture": "gdn",
        "layer": layer,
        "n_heads": n,
        "per_head": per_head,
        "correlations": {
            "write_vec_top1_vs_rank1_adv": corr_top1,
            "write_reuse_vs_rank1_adv": corr_reuse,
            "write_vec_erank_vs_rank1_adv": corr_erank,
        },
        "extraction_metadata": meta,
        "code_sha": CODE_SHA,
    }

    out_path = Path(f"{GDN_DATA}/gdn_head_level_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    gdn_vol.commit()

    print(f"\nSaved results to {out_path}")
    return output


@app.local_entrypoint()
def main(
    stage: str = "all",
    layer: int = GDN_LAYER,
    n_sequences: int = N_SEQUENCES,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
):
    if stage in ("extract", "all"):
        print("=== Stage 1: Extract GDN write vectors for all heads ===")
        meta = extract_gdn_multihead_writes.remote(
            layer=layer,
            n_sequences=n_sequences,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        print(
            f"Extraction done: {meta['n_write_vectors']} vectors from "
            f"{meta['n_sequences']} sequences, {len(meta['heads'])} heads, "
            f"{meta['extraction_time_s']:.0f}s"
        )

    if stage in ("analyze", "all"):
        print("=== Stage 2: Analyze head-level write geometry vs SAE advantage ===")
        output = analyze_gdn_head_level.remote()
        if "error" in output:
            print(f"Analysis failed: {output['error']}")
            return

        # Save results locally.
        local_out = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(local_out, exist_ok=True)
        local_path = os.path.join(local_out, "gdn_head_level_results.json")
        with open(local_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Saved local copy to {local_path}")

        corr = output["correlations"]["write_vec_top1_vs_rank1_adv"]
        print(
            f"\nGDN head-level: rho={corr['spearman_rho']:+.3f} "
            f"(p={corr['spearman_p']:.4f}), N={corr['n']}"
        )

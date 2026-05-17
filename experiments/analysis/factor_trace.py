#!/usr/bin/env python3
"""Write-to-use tracing helpers for recurrent-state SAE features."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from experiments.steering.causal_clamp import compute_activation_matrix, reconstruct_with_feature_updates
from experiments.analysis.probe_features import compute_text_properties


PRIMARY_LAYOUT_PROPERTIES = ("last_sent_boundary_pos",)
SECONDARY_LAYOUT_PROPERTIES = ("newline_density", "paragraph_count")
LAYOUT_PROPERTY_SET = set(PRIMARY_LAYOUT_PROPERTIES + SECONDARY_LAYOUT_PROPERTIES)
EVENT_TYPES = ("sentence_boundary", "paragraph_boundary", "non_event_control")


def _encode_states(
    sae,
    sae_type: str,
    state_batch: torch.Tensor,
) -> torch.Tensor:
    if sae_type == "flat":
        return sae.encode(state_batch.reshape(state_batch.shape[0], -1))
    return sae.encode(state_batch)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 4:
        return 0.0
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    xr = x_rank.astype(np.float64)
    yr = y_rank.astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = math.sqrt(float(np.dot(xr, xr) * np.dot(yr, yr)))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(xr, yr) / denom)


def _best_layout_property(
    act_values: np.ndarray,
    property_arrays: dict[str, np.ndarray],
) -> tuple[str, float]:
    best_property = ""
    best_rho = 0.0
    for prop in PRIMARY_LAYOUT_PROPERTIES + SECONDARY_LAYOUT_PROPERTIES:
        rho = _safe_spearman(act_values, property_arrays[prop])
        if abs(rho) > abs(best_rho):
            best_property = prop
            best_rho = rho
    return best_property, best_rho


def decode_token_piece(
    tokenizer,
    token_id: int,
    cache: dict[int, str],
) -> str:
    if token_id not in cache:
        cache[token_id] = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return cache[token_id]


def classify_prefix_event(prefix_text: str) -> str:
    if prefix_text.endswith("\n\n") or prefix_text.endswith("\n"):
        return "paragraph_boundary"

    stripped = prefix_text.rstrip()
    if stripped and stripped[-1] in ".?!":
        return "sentence_boundary"
    return "non_event_control"


def select_cross_corpus_trace_features(
    sae,
    sae_type: str,
    openwebtext_states: torch.Tensor,
    openwebtext_texts: Sequence[str],
    ultrachat_states: torch.Tensor,
    ultrachat_texts: Sequence[str],
    min_rho: float = 0.20,
    max_features: int = 3,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """Choose stable H4 tracing features with one checkpoint across both corpora."""
    n_owt = min(len(openwebtext_texts), int(openwebtext_states.shape[0]))
    n_uc = min(len(ultrachat_texts), int(ultrachat_states.shape[0]))
    if n_owt == 0 or n_uc == 0:
        raise ValueError("Need non-empty states/texts for both corpora")

    act_owt = compute_activation_matrix(sae, openwebtext_states[:n_owt], batch_size=batch_size)
    act_uc = compute_activation_matrix(sae, ultrachat_states[:n_uc], batch_size=batch_size)

    alive_owt = (act_owt > 0).any(axis=0)
    alive_uc = (act_uc > 0).any(axis=0)
    shared_alive = np.nonzero(alive_owt & alive_uc)[0].astype(int).tolist()
    if not shared_alive:
        raise ValueError("No shared alive features across corpora")

    owt_props = [compute_text_properties(text) for text in openwebtext_texts[:n_owt]]
    uc_props = [compute_text_properties(text) for text in ultrachat_texts[:n_uc]]
    owt_arrays = {
        prop: np.array([row[prop] for row in owt_props], dtype=np.float64)
        for prop in PRIMARY_LAYOUT_PROPERTIES + SECONDARY_LAYOUT_PROPERTIES
    }
    uc_arrays = {
        prop: np.array([row[prop] for row in uc_props], dtype=np.float64)
        for prop in PRIMARY_LAYOUT_PROPERTIES + SECONDARY_LAYOUT_PROPERTIES
    }

    strict: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []
    for feature_idx in shared_alive:
        owt_best_prop, owt_best_rho = _best_layout_property(act_owt[:, feature_idx], owt_arrays)
        uc_best_prop, uc_best_rho = _best_layout_property(act_uc[:, feature_idx], uc_arrays)
        if not owt_best_prop or not uc_best_prop:
            continue
        if abs(owt_best_rho) < min_rho or abs(uc_best_rho) < min_rho:
            continue

        entry = {
            "feature_idx": int(feature_idx),
            "openwebtext_property": owt_best_prop,
            "openwebtext_rho": float(owt_best_rho),
            "ultrachat_property": uc_best_prop,
            "ultrachat_rho": float(uc_best_rho),
            "score": float(min(abs(owt_best_rho), abs(uc_best_rho))),
            "mean_score": float((abs(owt_best_rho) + abs(uc_best_rho)) / 2.0),
        }

        strict_ok = (
            (owt_best_prop == "last_sent_boundary_pos" and uc_best_prop in LAYOUT_PROPERTY_SET)
            or (uc_best_prop == "last_sent_boundary_pos" and owt_best_prop in LAYOUT_PROPERTY_SET)
        )
        if strict_ok:
            strict.append(entry)
        if owt_best_prop in LAYOUT_PROPERTY_SET and uc_best_prop in LAYOUT_PROPERTY_SET:
            relaxed.append(entry)

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, int]:
        primary_hits = int(item["openwebtext_property"] == "last_sent_boundary_pos") + int(
            item["ultrachat_property"] == "last_sent_boundary_pos"
        )
        return (primary_hits, item["score"], item["mean_score"])

    strict.sort(key=_sort_key, reverse=True)
    relaxed.sort(key=_sort_key, reverse=True)
    selected = strict[:max_features]
    if len(selected) < max_features:
        seen = {item["feature_idx"] for item in selected}
        for item in relaxed:
            if item["feature_idx"] in seen:
                continue
            selected.append(item)
            seen.add(item["feature_idx"])
            if len(selected) >= max_features:
                break
    if not selected:
        raise ValueError("No cross-corpus layout features met the tracing threshold")
    return selected


def trace_feature_trajectories(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    feature_specs: list[dict[str, Any]],
    prompt_len: int = 512,
    max_offset: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    """Capture per-token coefficients and event-aligned summaries for selected features."""
    t0 = time.perf_counter()
    feature_indices = [int(item["feature_idx"]) for item in feature_specs]
    feature_to_slot = {feature_idx: slot for slot, feature_idx in enumerate(feature_indices)}
    piece_cache: dict[int, str] = {}
    prompt_records: list[dict[str, Any]] = []
    aligned_values = {
        feature_idx: {
            event: {"coeff_curves": [], "delta_curves": [], "normalized_curves": []}
            for event in EVENT_TYPES
        }
        for feature_idx in feature_indices
    }

    print(
        "[factor-trace] "
        f"layer={layer_idx} head={head_idx} features={feature_indices} "
        f"batches={len(corpus_batches)} prompt_len={prompt_len}",
        flush=True,
    )

    prompt_index = 0
    for batch_idx_global, batch in enumerate(corpus_batches, start=1):
        if batch.numel() == 0:
            continue
        token_batch = batch[:, :prompt_len].to(device)
        batch_size, seq_len = token_batch.shape
        if seq_len < 2:
            continue

        coeffs = np.zeros((batch_size, seq_len, len(feature_indices)), dtype=np.float32)
        event_labels = [["non_event_control"] * seq_len for _ in range(batch_size)]
        prefixes = [""] * batch_size

        cache = None
        for pos in range(seq_len):
            step_ids = token_batch[:, pos : pos + 1]
            if cache is None:
                out = model(input_ids=step_ids, use_cache=True)
            else:
                out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            state_batch = cache.layers[layer_idx].recurrent_states[:, head_idx].float()
            act_batch = _encode_states(sae, sae_type, state_batch)
            coeffs[:, pos, :] = act_batch[:, feature_indices].detach().cpu().numpy()

            for batch_idx in range(batch_size):
                token_id = int(token_batch[batch_idx, pos].item())
                prefixes[batch_idx] += decode_token_piece(tokenizer, token_id, piece_cache)
                event_labels[batch_idx][pos] = classify_prefix_event(prefixes[batch_idx])

        deltas = np.empty_like(coeffs)
        deltas[:, 0, :] = coeffs[:, 0, :]
        deltas[:, 1:, :] = coeffs[:, 1:, :] - coeffs[:, :-1, :]

        rng = np.random.default_rng(0)
        offsets = list(range(max_offset + 1))
        for batch_idx in range(batch_size):
            labels = event_labels[batch_idx]
            prompt_record = {
                "prompt_index": prompt_index,
                "token_ids": [int(x) for x in token_batch[batch_idx].detach().cpu().tolist()],
                "event_labels": labels,
                "feature_coefficients": {
                    str(feature_idx): coeffs[batch_idx, :, slot].tolist()
                    for feature_idx, slot in feature_to_slot.items()
                },
                "feature_deltas": {
                    str(feature_idx): deltas[batch_idx, :, slot].tolist()
                    for feature_idx, slot in feature_to_slot.items()
                },
            }
            prompt_records.append(prompt_record)

            eligible_controls = [pos for pos, label in enumerate(labels) if label == "non_event_control" and pos + max_offset < seq_len]
            event_positions = [
                pos for pos, label in enumerate(labels)
                if label in ("sentence_boundary", "paragraph_boundary") and pos + max_offset < seq_len
            ]
            if event_positions and eligible_controls:
                control_sample = rng.choice(
                    np.array(eligible_controls),
                    size=min(len(event_positions), len(eligible_controls)),
                    replace=False,
                ).tolist()
            else:
                control_sample = []

            for feature_idx, slot in feature_to_slot.items():
                for event in ("sentence_boundary", "paragraph_boundary"):
                    positions = [pos for pos in event_positions if labels[pos] == event]
                    for pos in positions:
                        coeff_curve = coeffs[batch_idx, pos : pos + max_offset + 1, slot]
                        delta_curve = deltas[batch_idx, pos : pos + max_offset + 1, slot]
                        denom = max(abs(float(coeff_curve[0])), 1e-8)
                        aligned_values[feature_idx][event]["coeff_curves"].append(coeff_curve.astype(np.float64))
                        aligned_values[feature_idx][event]["delta_curves"].append(delta_curve.astype(np.float64))
                        aligned_values[feature_idx][event]["normalized_curves"].append(
                            (coeff_curve / denom).astype(np.float64)
                        )
                for pos in control_sample:
                    coeff_curve = coeffs[batch_idx, pos : pos + max_offset + 1, slot]
                    delta_curve = deltas[batch_idx, pos : pos + max_offset + 1, slot]
                    denom = max(abs(float(coeff_curve[0])), 1e-8)
                    aligned_values[feature_idx]["non_event_control"]["coeff_curves"].append(coeff_curve.astype(np.float64))
                    aligned_values[feature_idx]["non_event_control"]["delta_curves"].append(delta_curve.astype(np.float64))
                    aligned_values[feature_idx]["non_event_control"]["normalized_curves"].append(
                        (coeff_curve / denom).astype(np.float64)
                    )

            prompt_index += 1
            if prompt_index % 16 == 0:
                print(
                    "[factor-trace] "
                    f"prompts={prompt_index} batch={batch_idx_global}/{len(corpus_batches)} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

        print(
            "[factor-trace] "
            f"finished_batch={batch_idx_global}/{len(corpus_batches)} "
            f"prompts={prompt_index} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    aligned_summary: dict[str, dict[str, Any]] = {}
    for feature_idx in feature_indices:
        feature_summary: dict[str, Any] = {}
        for event in EVENT_TYPES:
            coeff_curves = aligned_values[feature_idx][event]["coeff_curves"]
            delta_curves = aligned_values[feature_idx][event]["delta_curves"]
            norm_curves = aligned_values[feature_idx][event]["normalized_curves"]
            if coeff_curves:
                coeff_stack = np.stack(coeff_curves, axis=0)
                delta_stack = np.stack(delta_curves, axis=0)
                norm_stack = np.stack(norm_curves, axis=0)
                feature_summary[event] = {
                    "count": int(coeff_stack.shape[0]),
                    "offsets": offsets,
                    "mean_coeff_curve": coeff_stack.mean(axis=0).tolist(),
                    "mean_delta_curve": delta_stack.mean(axis=0).tolist(),
                    "median_normalized_curve": np.median(norm_stack, axis=0).tolist(),
                    "mean_event_jump": float(delta_stack[:, 0].mean()),
                }
            else:
                feature_summary[event] = {
                    "count": 0,
                    "offsets": offsets,
                    "mean_coeff_curve": [0.0] * len(offsets),
                    "mean_delta_curve": [0.0] * len(offsets),
                    "median_normalized_curve": [0.0] * len(offsets),
                    "mean_event_jump": 0.0,
                }
        control_jump = feature_summary["non_event_control"]["mean_event_jump"]
        for event in ("sentence_boundary", "paragraph_boundary"):
            feature_summary[event]["jump_minus_control"] = (
                feature_summary[event]["mean_event_jump"] - control_jump
            )
        aligned_summary[str(feature_idx)] = feature_summary

    return {
        "prompt_len": prompt_len,
        "max_offset": max_offset,
        "selected_features": feature_specs,
        "prompt_records": prompt_records,
        "event_aligned": aligned_summary,
    }


def select_trace_events(
    trace_result: dict[str, Any],
    target_feature_idx: int,
    preferred_events: Sequence[str] = ("sentence_boundary", "paragraph_boundary"),
    offsets: Sequence[int] = (16, 32, 64),
    min_retention: float = 0.5,
) -> list[dict[str, Any]]:
    """Choose write/use event pairs from traced trajectories."""
    candidates: list[dict[str, Any]] = []
    key = str(target_feature_idx)
    for prompt in trace_result.get("prompt_records", []):
        labels = prompt["event_labels"]
        coeffs = prompt["feature_coefficients"][key]
        deltas = prompt["feature_deltas"][key]
        best: dict[str, Any] | None = None
        for pos, label in enumerate(labels):
            if label not in preferred_events:
                continue
            if pos >= len(coeffs):
                continue
            coeff = float(coeffs[pos])
            jump = float(deltas[pos])
            if jump <= 0 or coeff <= 0:
                continue
            use_offset = None
            for offset in offsets:
                if pos + offset >= len(coeffs):
                    continue
                if float(coeffs[pos + offset]) >= min_retention * coeff:
                    use_offset = int(offset)
                    break
            if use_offset is None:
                continue
            entry = {
                "prompt_index": int(prompt["prompt_index"]),
                "write_pos": int(pos),
                "event_type": label,
                "write_coeff": coeff,
                "write_jump": jump,
                "use_offset": use_offset,
                "use_pos": int(pos + use_offset),
                "baseline_use_coeff": float(coeffs[pos + use_offset]),
            }
            if best is None or entry["write_jump"] > best["write_jump"]:
                best = entry
        if best is not None:
            candidates.append(best)
    candidates.sort(key=lambda item: item["write_jump"], reverse=True)
    return candidates


def _context_tokens(
    prompt_record: dict[str, Any],
    pos: int,
    window: int = 24,
) -> set[int]:
    token_ids = prompt_record["token_ids"]
    start = max(0, pos - window)
    end = min(len(token_ids), pos + 1)
    return {int(tok) for tok in token_ids[start:end]}


def _context_jaccard(
    prompt_a: dict[str, Any],
    pos_a: int,
    prompt_b: dict[str, Any],
    pos_b: int,
    window: int = 24,
) -> float:
    ctx_a = _context_tokens(prompt_a, pos_a, window=window)
    ctx_b = _context_tokens(prompt_b, pos_b, window=window)
    if not ctx_a and not ctx_b:
        return 1.0
    if not ctx_a or not ctx_b:
        return 0.0
    return float(len(ctx_a & ctx_b) / max(len(ctx_a | ctx_b), 1))


def _nearest_non_event_position(
    prompt_record: dict[str, Any],
    write_pos: int,
    use_pos: int,
    min_gap: int = 8,
) -> int | None:
    labels = prompt_record["event_labels"]
    candidates = []
    for pos, label in enumerate(labels):
        if label != "non_event_control":
            continue
        if pos >= use_pos:
            continue
        if abs(pos - write_pos) < min_gap:
            continue
        candidates.append((abs(pos - write_pos), pos))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][1])


def select_transplant_pairs(
    trace_result: dict[str, Any],
    target_feature_idx: int,
    max_pairs: int = 16,
    preferred_events: Sequence[str] = ("paragraph_boundary", "sentence_boundary"),
    preferred_offsets: Sequence[int] = (16,),
    min_write_gain: float = 1.10,
    context_window: int = 24,
    min_context_similarity: float = 0.10,
) -> list[dict[str, Any]]:
    """Pair low-write recipients with stronger matched donors from the trace set."""
    prompt_map = {
        int(prompt["prompt_index"]): prompt
        for prompt in trace_result.get("prompt_records", [])
    }
    candidates = [
        item for item in select_trace_events(trace_result, target_feature_idx)
        if item["event_type"] in preferred_events
    ]
    if preferred_offsets:
        filtered = [item for item in candidates if int(item["use_offset"]) in preferred_offsets]
        if len(filtered) >= 2:
            candidates = filtered
    if len(candidates) < 2:
        candidates = select_trace_events(trace_result, target_feature_idx)
    if len(candidates) < 2:
        return []

    donors = sorted(candidates, key=lambda item: (item["write_jump"], item["write_coeff"]), reverse=True)
    recipients = sorted(candidates, key=lambda item: (item["write_jump"], item["write_coeff"]))
    donor_pool = donors[: max(4, min(len(donors), max_pairs))]
    pairs: list[dict[str, Any]] = []
    used_recipient_prompts: set[int] = set()

    for recipient in recipients:
        if len(pairs) >= max_pairs:
            break
        recipient_prompt = prompt_map.get(int(recipient["prompt_index"]))
        if recipient_prompt is None:
            continue
        if int(recipient["prompt_index"]) in used_recipient_prompts:
            continue
        best_match: tuple[tuple[int, int, float, float], dict[str, Any]] | None = None
        for donor in donor_pool:
            donor_prompt_index = int(donor["prompt_index"])
            recipient_prompt_index = int(recipient["prompt_index"])
            if donor_prompt_index == recipient_prompt_index:
                continue
            if float(donor["write_coeff"]) <= float(recipient["write_coeff"]) * min_write_gain:
                continue
            donor_prompt = prompt_map.get(donor_prompt_index)
            if donor_prompt is None:
                continue
            same_event = int(donor["event_type"] == recipient["event_type"])
            same_offset = int(donor["use_offset"] == recipient["use_offset"])
            context_sim = _context_jaccard(
                donor_prompt,
                int(donor["write_pos"]),
                recipient_prompt,
                int(recipient["write_pos"]),
                window=context_window,
            )
            if context_sim < min_context_similarity:
                continue
            score = (
                same_event,
                same_offset,
                context_sim,
                float(donor["write_jump"]) - float(recipient["write_jump"]),
            )
            candidate = {
                "donor_prompt_index": donor_prompt_index,
                "donor_write_pos": int(donor["write_pos"]),
                "donor_use_pos": int(donor["use_pos"]),
                "donor_use_offset": int(donor["use_offset"]),
                "donor_event_type": donor["event_type"],
                "donor_write_coeff": float(donor["write_coeff"]),
                "donor_write_jump": float(donor["write_jump"]),
                "recipient_prompt_index": recipient_prompt_index,
                "recipient_write_pos": int(recipient["write_pos"]),
                "recipient_use_pos": int(recipient["use_pos"]),
                "recipient_use_offset": int(recipient["use_offset"]),
                "recipient_event_type": recipient["event_type"],
                "recipient_write_coeff": float(recipient["write_coeff"]),
                "recipient_write_jump": float(recipient["write_jump"]),
                "context_similarity": float(context_sim),
            }
            if best_match is None or score > best_match[0]:
                best_match = (score, candidate)
        if best_match is None:
            continue
        pair = best_match[1]
        wrong_time_pos = _nearest_non_event_position(
            recipient_prompt,
            write_pos=pair["recipient_write_pos"],
            use_pos=pair["recipient_use_pos"],
        )
        pair["wrong_time_pos"] = wrong_time_pos
        pairs.append(pair)
        used_recipient_prompts.add(int(recipient["prompt_index"]))

    return pairs


def choose_similar_frequency_feature(
    target_feature_idx: int,
    frequencies: np.ndarray,
    exclude: set[int],
) -> int:
    target_freq = float(frequencies[target_feature_idx])
    best_idx = -1
    best_gap = float("inf")
    for idx, freq in enumerate(frequencies):
        if idx in exclude:
            continue
        gap = abs(float(freq) - target_freq)
        if gap < best_gap:
            best_gap = gap
            best_idx = idx
    if best_idx < 0:
        raise ValueError("Could not find a similar-frequency control feature")
    return int(best_idx)


def build_use_site_benchmark(
    model,
    tokenizer,
    corpus_sequences: list[torch.Tensor],
    trace_result: dict[str, Any],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    target_feature_indices: Sequence[int],
    max_rows: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    """Mine high-confidence downstream use sites for the signed H4 boundary code."""
    if len(target_feature_indices) != 2:
        raise ValueError("build_use_site_benchmark expects the signed pair [62, 105]")
    pos_feat, neg_feat = [int(x) for x in target_feature_indices]
    prompt_map = {
        int(prompt["prompt_index"]): prompt
        for prompt in trace_result.get("prompt_records", [])
    }
    sequence_map = {
        int(prompt["prompt_index"]): seq
        for prompt, seq in zip(trace_result.get("prompt_records", []), corpus_sequences)
    }
    readout_token_ids = _readout_class_token_ids(tokenizer)
    candidate_pairs = select_transplant_pairs(
        trace_result=trace_result,
        target_feature_idx=pos_feat,
        max_pairs=max(max_rows * 8, 128),
        preferred_events=("paragraph_boundary",),
        preferred_offsets=(8, 16, 32),
        min_write_gain=1.20,
        min_context_similarity=0.20,
    )
    if not candidate_pairs:
        raise ValueError("No candidate paragraph-boundary pairs for use-site benchmark")

    def _signed_coeff(prompt_record: dict[str, Any], pos: int) -> float:
        pos_val = float(prompt_record["feature_coefficients"][str(pos_feat)][pos])
        neg_val = float(prompt_record["feature_coefficients"][str(neg_feat)][pos])
        return pos_val - neg_val

    def _signed_delta(prompt_record: dict[str, Any], pos: int) -> tuple[float, float]:
        pos_delta = float(prompt_record["feature_deltas"][str(pos_feat)][pos])
        neg_delta = float(prompt_record["feature_deltas"][str(neg_feat)][pos])
        return pos_delta, neg_delta

    all_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    drop_counts: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()
    print(
        "[factor-trace-benchmark] "
        f"target={[pos_feat, neg_feat]} candidate_pairs={len(candidate_pairs)} max_rows={max_rows}",
        flush=True,
    )

    for pair_idx, pair in enumerate(candidate_pairs, start=1):
        recipient_prompt_index = int(pair["recipient_prompt_index"])
        donor_prompt_index = int(pair["donor_prompt_index"])
        if recipient_prompt_index not in prompt_map or recipient_prompt_index not in sequence_map:
            continue
        recipient_prompt = prompt_map[recipient_prompt_index]
        donor_prompt = prompt_map[donor_prompt_index]
        use_pos = int(pair["recipient_use_pos"])
        seq = sequence_map[recipient_prompt_index].to(device)
        seq_end = min(int(seq.shape[1]), use_pos + 10)
        if seq_end <= use_pos + 1:
            continue
        seq = seq[:, :seq_end]
        baseline = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=[pos_feat, neg_feat],
        )
        donor_write_signed = _signed_coeff(donor_prompt, int(pair["donor_write_pos"]))
        recipient_write_signed = _signed_coeff(recipient_prompt, int(pair["recipient_write_pos"]))
        donor_delta_pos, donor_delta_neg = _signed_delta(donor_prompt, int(pair["donor_write_pos"]))
        signed_write_gap = float(donor_write_signed - recipient_write_signed)

        start = max(int(pair["recipient_write_pos"]) + 1, use_pos - 2)
        end = min(use_pos + 8, int(seq.shape[1]) - 2)
        for decision_pos in range(start, end + 1):
            actual_next_token = int(seq[0, decision_pos + 1].item())
            readout_class, readout_text = _classify_readout_token(tokenizer, actual_next_token)
            row = {
                "donor_prompt_index": donor_prompt_index,
                "recipient_prompt_index": recipient_prompt_index,
                "write_pos": int(pair["recipient_write_pos"]),
                "use_pos": use_pos,
                "use_offset": int(pair["recipient_use_offset"]),
                "decision_pos": int(decision_pos),
                "intervention_pos": int(max(0, decision_pos - 1)),
                "wrong_time_pos": _nearest_non_event_position(
                    recipient_prompt,
                    write_pos=int(max(0, decision_pos - 1)),
                    use_pos=int(decision_pos),
                    min_gap=4,
                ),
                "context_similarity": float(pair["context_similarity"]),
                "signed_write_gap": signed_write_gap,
                "donor_delta_62": float(donor_delta_pos),
                "donor_delta_105": float(donor_delta_neg),
                "actual_next_token": actual_next_token,
                "readout_text": readout_text,
                "readout_class": readout_class,
                "accepted": False,
                "drop_reason": "",
            }
            if readout_class is None:
                row["drop_reason"] = "readout_class_excluded"
                drop_counts[row["drop_reason"]] += 1
                all_rows.append(row)
                continue
            baseline_logits = baseline["logits_by_pos"][decision_pos]
            baseline_prob = _token_prob_from_logits(baseline_logits, actual_next_token)
            row["baseline_actual_token_prob"] = baseline_prob
            row["baseline_headroom"] = float(baseline_prob * (1.0 - baseline_prob))
            row["baseline_readout_class_mass"] = _mass_from_logits(
                baseline_logits,
                readout_token_ids.get(readout_class, [actual_next_token]) or [actual_next_token],
            )
            signed_use_coeff = float(
                baseline["feature_coefficients"][str(pos_feat)][decision_pos]
                - baseline["feature_coefficients"][str(neg_feat)][decision_pos]
            )
            row["signed_use_coeff"] = signed_use_coeff
            if abs(signed_write_gap) < 0.02:
                row["drop_reason"] = "weak_signed_write_gap"
            elif abs(signed_use_coeff) < 0.01:
                row["drop_reason"] = "weak_signed_use_coeff"
            elif baseline_prob < 0.05 or baseline_prob > 0.80:
                row["drop_reason"] = "low_token_headroom"
            elif row["wrong_time_pos"] is None:
                row["drop_reason"] = "no_wrong_time"
            else:
                row["accepted"] = True
                row["drop_reason"] = "accepted"
                row["score"] = (
                    row["baseline_headroom"],
                    abs(signed_use_coeff),
                    float(pair["context_similarity"]),
                    abs(signed_write_gap),
                )
                accepted_rows.append(row)
                print(
                    f"[factor-trace-benchmark] accept {len(accepted_rows)} "
                    f"pair={pair_idx}/{len(candidate_pairs)} "
                    f"recipient={recipient_prompt_index}@{decision_pos} "
                    f"class={readout_class} tok={readout_text!r} "
                    f"p={baseline_prob:.3f} headroom={row['baseline_headroom']:.3f} "
                    f"signed_gap={signed_write_gap:+.4f} signed_use={signed_use_coeff:+.4f}",
                    flush=True,
                )
            drop_counts[row["drop_reason"]] += 1
            all_rows.append(row)

        if pair_idx % 5 == 0 or pair_idx == len(candidate_pairs):
            print(
                f"[factor-trace-benchmark] progress pair={pair_idx}/{len(candidate_pairs)} "
                f"rows={len(all_rows)} accepted={len(accepted_rows)} "
                f"drops={dict(drop_counts)} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    accepted_rows.sort(key=lambda r: r.get("score", (0, 0, 0, 0)), reverse=True)
    selected_rows = accepted_rows[:max_rows]
    print(
        "[factor-trace-benchmark] "
        f"accepted={len(accepted_rows)} selected={len(selected_rows)} "
        f"elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return {
        "target_feature_indices": [pos_feat, neg_feat],
        "n_candidate_pairs": len(candidate_pairs),
        "n_rows_total": len(all_rows),
        "n_rows_accepted": len(accepted_rows),
        "drop_counts": dict(drop_counts),
        "accepted_rows": selected_rows,
        "all_rows": all_rows,
        "total_time_s": float(time.perf_counter() - t0),
    }


def sample_matched_random_features(
    target_feature_idx: int,
    frequencies: np.ndarray,
    alive_features: list[int],
    exclude: set[int],
    n_random: int = 4,
    seed: int = 0,
) -> list[int]:
    rng = np.random.default_rng(seed)
    target_freq = float(frequencies[target_feature_idx])
    candidates = [idx for idx in alive_features if idx not in exclude]
    if not candidates:
        raise ValueError("No alive features available for random controls")
    candidates.sort(key=lambda idx: abs(float(frequencies[idx]) - target_freq))
    pool = candidates[: max(n_random * 8, 32)]
    if len(pool) <= n_random:
        return [int(idx) for idx in pool[:n_random]]
    picked = rng.choice(np.array(pool), size=n_random, replace=False)
    return [int(idx) for idx in picked.tolist()]


def boundary_token_ids(tokenizer) -> list[int]:
    token_ids: set[int] = set()
    for text in [".", "?", "!", "\n", ":"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(int(ids[0]))
    return sorted(token_ids)


def boundary_token_groups(tokenizer) -> dict[str, list[int]]:
    groups = {
        "newline": set(),
        "sentence_punct": set(),
        "colon": set(),
    }
    for text, key in [("\n", "newline"), (".", "sentence_punct"), ("?", "sentence_punct"), ("!", "sentence_punct"), (":", "colon")]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            groups[key].add(int(ids[0]))
    all_boundary = set().union(*groups.values())
    return {
        "boundary": sorted(all_boundary),
        "newline": sorted(groups["newline"]),
        "sentence_punct": sorted(groups["sentence_punct"]),
        "colon": sorted(groups["colon"]),
    }


@torch.no_grad()
def teacher_forced_feature_rollout(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    tracked_features: Sequence[int],
    intervention_updates: dict[int, float] | None = None,
    intervention_feature: int | None = None,
    intervention_pos: int | None = None,
    capture_hidden_state_positions: Sequence[int] | None = None,
    intervention_state_delta: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run one sequence token-by-token and optionally ablate one feature at one step."""
    seq = input_ids
    if seq.ndim != 2 or seq.shape[0] != 1:
        raise ValueError("teacher_forced_feature_rollout expects shape (1, seq_len)")

    tracked = [int(idx) for idx in tracked_features]
    cache = None
    coeff_records = {str(idx): [] for idx in tracked}
    logits_by_pos: dict[int, torch.Tensor] = {}
    hidden_state_positions = {int(pos) for pos in (capture_hidden_state_positions or [])}
    hidden_state_by_pos: dict[int, torch.Tensor] = {}

    for pos in range(int(seq.shape[1])):
        step_ids = seq[:, pos : pos + 1]
        if cache is None:
            out = model(
                input_ids=step_ids,
                use_cache=True,
                output_hidden_states=bool(hidden_state_positions),
            )
        else:
            out = model(
                input_ids=step_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=bool(hidden_state_positions),
            )
        cache = out.past_key_values
        state = cache.layers[layer_idx].recurrent_states[:, head_idx].float()
        acts = _encode_states(sae, sae_type, state)
        for idx in tracked:
            coeff_records[str(idx)].append(float(acts[0, idx].item()))
        logits_by_pos[pos] = out.logits[:, -1, :].float().detach().cpu()
        if pos in hidden_state_positions and out.hidden_states is not None:
            hidden_state_by_pos[pos] = out.hidden_states[-1][:, -1, :].float().detach().cpu()

        if intervention_pos == pos and (
            intervention_updates is not None
            or intervention_feature is not None
            or intervention_state_delta is not None
        ):
            layer_cache = cache.layers[layer_idx]
            original_head = layer_cache.recurrent_states[0, head_idx].float()
            if intervention_state_delta is not None:
                edited = original_head + intervention_state_delta.to(
                    device=original_head.device,
                    dtype=original_head.dtype,
                )
            else:
                ablate_list = [int(intervention_feature)] if intervention_feature is not None else None
                edited = reconstruct_with_feature_updates(
                    sae=sae,
                    state_head=original_head,
                    sae_type=sae_type,
                    set_updates=intervention_updates,
                    ablate_features=ablate_list,
                )
            layer_cache.recurrent_states[0, head_idx] = edited.to(layer_cache.recurrent_states.dtype)

    return {
        "feature_coefficients": coeff_records,
        "logits_by_pos": logits_by_pos,
        "hidden_state_by_pos": hidden_state_by_pos,
    }


def _kl_divergence_from_logits(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
) -> float:
    baseline_log_probs = torch.log_softmax(baseline_logits, dim=-1)
    edited_log_probs = torch.log_softmax(edited_logits, dim=-1)
    baseline_probs = baseline_log_probs.exp()
    kl = torch.sum(baseline_probs * (baseline_log_probs - edited_log_probs), dim=-1)
    return float(kl.item())


def _boundary_mass_shift(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    token_ids: Sequence[int],
) -> float:
    if not token_ids:
        return 0.0
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    edited_probs = torch.softmax(edited_logits, dim=-1)
    baseline_mass = baseline_probs[:, list(token_ids)].sum(dim=-1)
    edited_mass = edited_probs[:, list(token_ids)].sum(dim=-1)
    return float((edited_mass - baseline_mass).item())


def _mass_from_logits(
    logits: torch.Tensor,
    token_ids: Sequence[int],
) -> float:
    if not token_ids:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    return float(probs[:, list(token_ids)].sum(dim=-1).item())


def _token_logit_shift(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    token_id: int | None,
) -> float:
    if token_id is None or token_id < 0:
        return 0.0
    return float((edited_logits.float()[:, token_id] - baseline_logits.float()[:, token_id]).item())


def _decoded_boundary_group(tokenizer, token_id: int | None) -> tuple[str, str]:
    if token_id is None or token_id < 0:
        return "other", ""
    raw = tokenizer.decode([int(token_id)])
    if raw in {"\n", "\n\n"}:
        return "newline", raw
    stripped = raw.strip()
    if stripped in {".", "?", "!"} and len(stripped) == 1:
        return "sentence_punct", stripped
    if stripped == ":":
        return "colon", stripped
    return "other", raw


def _token_prob_from_logits(logits: torch.Tensor, token_id: int | None) -> float:
    if token_id is None or token_id < 0:
        return 0.0
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs[:, int(token_id)].item())


def _token_prob_shift(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    token_id: int | None,
) -> float:
    if token_id is None or token_id < 0:
        return 0.0
    base = torch.softmax(baseline_logits.float(), dim=-1)[:, int(token_id)]
    edit = torch.softmax(edited_logits.float(), dim=-1)[:, int(token_id)]
    return float((edit - base).item())


def _classify_readout_token(tokenizer, token_id: int | None) -> tuple[str | None, str]:
    if token_id is None or token_id < 0:
        return None, ""
    raw = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if raw in {"\n", "\n\n"}:
        return "paragraph_break", raw
    stripped = raw.strip()
    if stripped in {".", "?", "!"} and len(stripped) == 1:
        return "sentence_reset", stripped
    if stripped in {"The", "However", "In", "This"}:
        return "discourse_opener", stripped
    if stripped and stripped[0].isupper() and stripped.replace("'", "").isalpha():
        return "sentence_capitalized", stripped
    return None, raw


def _readout_class_token_ids(tokenizer) -> dict[str, list[int]]:
    buckets = {
        "paragraph_break": set(),
        "sentence_reset": set(),
        "discourse_opener": set(),
    }
    for text, bucket in [
        ("\n", "paragraph_break"),
        ("\n\n", "paragraph_break"),
        (".", "sentence_reset"),
        ("?", "sentence_reset"),
        ("!", "sentence_reset"),
        ("The", "discourse_opener"),
        ("However", "discourse_opener"),
        ("In", "discourse_opener"),
        ("This", "discourse_opener"),
    ]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            buckets[bucket].add(int(ids[0]))
    return {k: sorted(v) for k, v in buckets.items()}


def _preview_eval_site_from_prompt(
    tokenizer,
    prompt_record: dict[str, Any],
    anchor_use_pos: int,
    back_window: int = 4,
    forward_window: int = 8,
) -> dict[str, Any]:
    token_ids = prompt_record["token_ids"]
    best: tuple[tuple[int, int], dict[str, Any]] | None = None
    start = max(0, anchor_use_pos - back_window)
    end = min(len(token_ids) - 1, anchor_use_pos + forward_window)
    for pos in range(start, end + 1):
        if pos + 1 >= len(token_ids):
            continue
        actual_next_token = int(token_ids[pos + 1])
        actual_group, actual_text = _decoded_boundary_group(tokenizer, actual_next_token)
        if actual_group == "other":
            continue
        # Prefer the closest true boundary token, with a slight bias toward later use-sites.
        score = (-abs(pos - anchor_use_pos), int(pos >= anchor_use_pos))
        candidate = {
            "eval_pos": int(pos),
            "eval_offset_from_anchor": int(pos - anchor_use_pos),
            "eval_actual_next_token": actual_next_token,
            "eval_actual_group": actual_group,
            "eval_actual_text": actual_text,
        }
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else {}


def _select_eval_site(
    baseline_logits_by_pos: Sequence[torch.Tensor],
    input_ids: torch.Tensor,
    anchor_use_pos: int,
    token_groups: dict[str, list[int]],
    tokenizer,
    back_window: int = 4,
    forward_window: int = 8,
    preferred_groups: Sequence[str] = ("newline", "sentence_punct", "colon"),
) -> dict[str, Any]:
    seq_len = int(input_ids.shape[1])
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    start = max(0, anchor_use_pos - back_window)
    end = min(len(baseline_logits_by_pos) - 1, anchor_use_pos + forward_window)
    preferred_rank = {group: len(preferred_groups) - i for i, group in enumerate(preferred_groups)}

    for pos in range(start, end + 1):
        if pos + 1 >= seq_len:
            continue
        actual_next_token = int(input_ids[0, pos + 1].item())
        actual_group, actual_text = _decoded_boundary_group(tokenizer, actual_next_token)
        if actual_group == "other":
            continue

        logits = baseline_logits_by_pos[pos]
        boundary_mass = _mass_from_logits(logits, token_groups["boundary"])
        actual_group_mass = (
            _mass_from_logits(logits, token_groups[actual_group])
            if actual_group in token_groups
            else 0.0
        )
        score = (
            preferred_rank.get(actual_group, 0),
            actual_group_mass,
            boundary_mass,
            -abs(pos - anchor_use_pos),
        )
        candidate = {
            "eval_pos": int(pos),
            "eval_offset_from_anchor": int(pos - anchor_use_pos),
            "eval_actual_next_token": actual_next_token,
            "eval_actual_group": actual_group,
            "eval_actual_text": actual_text,
            "baseline_boundary_mass": boundary_mass,
            "baseline_actual_group_mass": actual_group_mass,
        }
        if best is None or score > best[0]:
            best = (score, candidate)

    if best is None:
        return {}
    return best[1]


def run_factor_trace_intervention(
    model,
    tokenizer,
    corpus_sequences: list[torch.Tensor],
    trace_result: dict[str, Any],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    states: torch.Tensor,
    target_feature_idx: int,
    n_prompts: int = 16,
    n_random: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """Measure write-to-use effects for one traced feature."""
    candidates = select_trace_events(trace_result, target_feature_idx)
    if not candidates:
        raise ValueError(f"No valid trace events found for feature {target_feature_idx}")

    act_matrix = compute_activation_matrix(sae, states)
    frequencies = (act_matrix > 0).mean(axis=0)
    alive_features = np.nonzero((act_matrix > 0).any(axis=0))[0].astype(int).tolist()

    selected = candidates[:n_prompts]
    exclude = {int(target_feature_idx)}
    similar_feature = choose_similar_frequency_feature(target_feature_idx, frequencies, exclude)
    exclude.add(similar_feature)
    random_features = sample_matched_random_features(
        target_feature_idx=target_feature_idx,
        frequencies=frequencies,
        alive_features=alive_features,
        exclude=exclude,
        n_random=n_random,
        seed=0,
    )
    control_features = [similar_feature] + random_features
    tracked_features = [target_feature_idx] + control_features
    boundary_ids = boundary_token_ids(tokenizer)
    t0 = time.perf_counter()

    print(
        "[factor-trace-intervention] target="
        f"{target_feature_idx} similar_control={similar_feature} "
        f"random_controls={random_features} candidates={len(candidates)} "
        f"evaluating={min(len(candidates), n_prompts)}",
        flush=True,
    )

    per_prompt = []
    target_kls: list[float] = []
    control_kls: list[float] = []
    target_boundary_shifts: list[float] = []
    control_boundary_shifts: list[float] = []
    target_attenuations: list[float] = []
    control_attenuations: list[float] = []

    sequence_map = {
        int(prompt["prompt_index"]): seq
        for prompt, seq in zip(trace_result.get("prompt_records", []), corpus_sequences)
    }

    for i, candidate in enumerate(selected, start=1):
        prompt_index = int(candidate["prompt_index"])
        if prompt_index not in sequence_map:
            print(
                f"[factor-trace-intervention] skip {i}/{len(selected)} "
                f"prompt={prompt_index} missing_sequence",
                flush=True,
            )
            continue
        seq = sequence_map[prompt_index][:, : candidate["use_pos"] + 1].to(device)
        prompt_t0 = time.perf_counter()

        print(
            f"[factor-trace-intervention] prompt {i}/{len(selected)} "
            f"prompt={prompt_index} event={candidate['event_type']} "
            f"write={candidate['write_pos']} use={candidate['use_pos']} "
            f"offset={candidate['use_offset']} seq_len={seq.shape[1]}",
            flush=True,
        )

        baseline = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_feature=None,
            intervention_pos=None,
        )
        target_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_feature=target_feature_idx,
            intervention_pos=int(candidate["write_pos"]),
        )
        baseline_logits = baseline["logits_by_pos"][candidate["use_pos"]]
        target_logits = target_rollout["logits_by_pos"][candidate["use_pos"]]
        target_kl = _kl_divergence_from_logits(baseline_logits, target_logits)
        target_boundary = _boundary_mass_shift(baseline_logits, target_logits, boundary_ids)
        baseline_use_coeff = baseline["feature_coefficients"][str(target_feature_idx)][candidate["use_pos"]]
        target_use_coeff = target_rollout["feature_coefficients"][str(target_feature_idx)][candidate["use_pos"]]
        target_attenuation = float(baseline_use_coeff - target_use_coeff)

        control_rows = []
        prompt_control_kls = []
        prompt_control_boundary = []
        prompt_control_attenuation = []
        for control_feature in control_features:
            control_rollout = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=tracked_features,
                intervention_feature=int(control_feature),
                intervention_pos=int(candidate["write_pos"]),
            )
            control_logits = control_rollout["logits_by_pos"][candidate["use_pos"]]
            control_kl = _kl_divergence_from_logits(baseline_logits, control_logits)
            control_boundary = _boundary_mass_shift(baseline_logits, control_logits, boundary_ids)
            control_use_coeff = control_rollout["feature_coefficients"][str(target_feature_idx)][candidate["use_pos"]]
            control_attenuation = float(baseline_use_coeff - control_use_coeff)
            prompt_control_kls.append(control_kl)
            prompt_control_boundary.append(control_boundary)
            prompt_control_attenuation.append(control_attenuation)
            control_rows.append({
                "feature_idx": int(control_feature),
                "use_time_kl": control_kl,
                "boundary_mass_shift": control_boundary,
                "target_feature_attenuation": control_attenuation,
            })

        target_kls.append(target_kl)
        control_kls.append(float(np.mean(prompt_control_kls)) if prompt_control_kls else 0.0)
        target_boundary_shifts.append(target_boundary)
        control_boundary_shifts.append(float(np.mean(prompt_control_boundary)) if prompt_control_boundary else 0.0)
        target_attenuations.append(target_attenuation)
        control_attenuations.append(float(np.mean(prompt_control_attenuation)) if prompt_control_attenuation else 0.0)
        per_prompt.append({
            **candidate,
            "baseline_use_coeff": float(baseline_use_coeff),
            "target_use_coeff": float(target_use_coeff),
            "target_use_time_kl": target_kl,
            "target_boundary_mass_shift": target_boundary,
            "target_feature_attenuation": target_attenuation,
            "controls": control_rows,
        })

        running_target = float(np.mean(target_kls)) if target_kls else 0.0
        running_control = float(np.mean(control_kls)) if control_kls else 0.0
        ratio = (
            running_target / running_control
            if abs(running_control) > 1e-8
            else float("inf")
        )
        print(
            f"[factor-trace-intervention] done {i}/{len(selected)} "
            f"target_kl={target_kl:.6f} control_kl={control_kls[-1]:.6f} "
            f"running_ratio={ratio:.2f}x "
            f"target_att={target_attenuation:.6f} control_att={control_attenuations[-1]:.6f} "
            f"elapsed={time.perf_counter() - prompt_t0:.1f}s",
            flush=True,
        )

    target_kl_mean = float(np.mean(target_kls)) if target_kls else 0.0
    control_kl_mean = float(np.mean(control_kls)) if control_kls else 0.0
    target_boundary_mean = float(np.mean(target_boundary_shifts)) if target_boundary_shifts else 0.0
    control_boundary_mean = float(np.mean(control_boundary_shifts)) if control_boundary_shifts else 0.0
    target_att_mean = float(np.mean(target_attenuations)) if target_attenuations else 0.0
    control_att_mean = float(np.mean(control_attenuations)) if control_attenuations else 0.0

    total_elapsed = time.perf_counter() - t0
    print(
        "[factor-trace-intervention] summary "
        f"prompts={len(per_prompt)} target_kl_mean={target_kl_mean:.6f} "
        f"control_kl_mean={control_kl_mean:.6f} "
        f"ratio={(target_kl_mean / control_kl_mean if abs(control_kl_mean) > 1e-8 else float('inf')):.2f}x "
        f"target_att_mean={target_att_mean:.6f} control_att_mean={control_att_mean:.6f} "
        f"elapsed={total_elapsed:.1f}s",
        flush=True,
    )

    return {
        "target_feature_idx": int(target_feature_idx),
        "similar_frequency_control": int(similar_feature),
        "random_controls": [int(idx) for idx in random_features],
        "n_prompts_evaluated": len(per_prompt),
        "total_time_s": total_elapsed,
        "per_prompt": per_prompt,
        "summary": {
            "target_use_time_kl_mean": target_kl_mean,
            "control_use_time_kl_mean": control_kl_mean,
            "target_vs_random_kl_ratio": (
                target_kl_mean / control_kl_mean if abs(control_kl_mean) > 1e-8 else float("inf")
            ),
            "target_boundary_mass_shift_mean": target_boundary_mean,
            "control_boundary_mass_shift_mean": control_boundary_mean,
            "target_feature_attenuation_mean": target_att_mean,
            "control_feature_attenuation_mean": control_att_mean,
        },
    }


def run_factor_trace_transplant(
    model,
    tokenizer,
    corpus_sequences: list[torch.Tensor],
    trace_result: dict[str, Any],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    states: torch.Tensor,
    target_feature_indices: Sequence[int] | int,
    n_pairs: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    """Transplant a donor write value into matched recipient prompts and score local readouts."""
    if isinstance(target_feature_indices, (int, np.integer)):
        target_features = [int(target_feature_indices)]
    else:
        target_features = [int(idx) for idx in target_feature_indices]
    if not target_features:
        raise ValueError("Need at least one target feature for transplant")
    primary_target_feature = int(target_features[0])

    prompt_map = {
        int(prompt["prompt_index"]): prompt
        for prompt in trace_result.get("prompt_records", [])
    }
    sequence_map = {
        int(prompt["prompt_index"]): seq
        for prompt, seq in zip(trace_result.get("prompt_records", []), corpus_sequences)
    }
    candidate_pairs = select_transplant_pairs(
        trace_result=trace_result,
        target_feature_idx=primary_target_feature,
        max_pairs=max(n_pairs * 12, 64),
        preferred_events=("paragraph_boundary",),
        preferred_offsets=(16,),
        min_write_gain=1.20,
        min_context_similarity=0.20,
    )
    if not candidate_pairs:
        raise ValueError(f"No transplant pairs found for features {target_features}")

    act_matrix = compute_activation_matrix(sae, states)
    frequencies = (act_matrix > 0).mean(axis=0)
    similar_features: list[int] = []
    excluded = set(target_features)
    for feat in target_features:
        similar_feat = choose_similar_frequency_feature(
            target_feature_idx=int(feat),
            frequencies=frequencies,
            exclude=excluded | set(similar_features),
        )
        similar_features.append(int(similar_feat))
    token_groups = boundary_token_groups(tokenizer)
    t0 = time.perf_counter()

    per_pair = []
    target_boundary = []
    wrong_feature_boundary = []
    wrong_time_boundary = []
    target_newline = []
    wrong_feature_newline = []
    wrong_time_newline = []
    target_sentence = []
    wrong_feature_sentence = []
    wrong_time_sentence = []
    target_kl = []
    wrong_feature_kl = []
    wrong_time_kl = []
    target_coeff_delta = []
    wrong_feature_coeff_delta = []
    wrong_time_coeff_delta = []

    print(
        "[factor-trace-transplant] "
        f"target={target_features} similar_control={similar_features} "
        f"candidate_pairs={len(candidate_pairs)}",
        flush=True,
    )

    baseline_candidates: list[dict[str, Any]] = []
    tracked_features = [*target_features, *similar_features]
    for pair in candidate_pairs:
        recipient_prompt_index = int(pair["recipient_prompt_index"])
        donor_prompt_index = int(pair["donor_prompt_index"])
        donor_prompt = prompt_map[donor_prompt_index]
        recipient_full = sequence_map[recipient_prompt_index].to(device)
        donor_coeffs = {
            int(feat): float(donor_prompt["feature_coefficients"][str(int(feat))][pair["donor_write_pos"]])
            for feat in target_features
        }
        use_pos = int(pair["recipient_use_pos"])
        eval_forward_window = 8
        seq_end = min(int(recipient_full.shape[1]), use_pos + eval_forward_window + 2)
        if seq_end <= use_pos + 1:
            continue
        seq = recipient_full[:, :seq_end]
        actual_next_token = int(recipient_full[0, use_pos + 1].item())
        baseline = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
        )
        eval_meta = _select_eval_site(
            baseline_logits_by_pos=baseline["logits_by_pos"],
            input_ids=seq,
            anchor_use_pos=use_pos,
            token_groups=token_groups,
            tokenizer=tokenizer,
            preferred_groups=("newline", "sentence_punct", "colon"),
        )
        if not eval_meta:
            continue
        eval_pos = int(eval_meta["eval_pos"])
        baseline_logits = baseline["logits_by_pos"][eval_pos]
        eval_meta["baseline_boundary_mass"] = _mass_from_logits(baseline_logits, token_groups["boundary"])
        eval_meta["baseline_actual_group_mass"] = _mass_from_logits(
            baseline_logits,
            token_groups[eval_meta["eval_actual_group"]],
        )
        eval_meta["baseline_actual_token_prob"] = _token_prob_from_logits(
            baseline_logits,
            int(eval_meta["eval_actual_next_token"]),
        )
        if (
            eval_meta["baseline_actual_token_prob"] < 0.03
            or eval_meta["baseline_actual_token_prob"] > 0.75
        ):
            continue
        target_updates = {}
        wrong_feature_updates = {}
        sign_flip_updates = {}
        for feat, donor_coeff in donor_coeffs.items():
            recipient_write_coeff = float(
                baseline["feature_coefficients"][str(int(feat))][int(pair["recipient_write_pos"])]
            )
            donor_prev_coeff = 0.0
            if int(pair["donor_write_pos"]) > 0:
                donor_prev_coeff = float(
                    donor_prompt["feature_coefficients"][str(int(feat))][int(pair["donor_write_pos"]) - 1]
                )
            donor_delta = float(donor_coeff - donor_prev_coeff)
            target_updates[int(feat)] = float(recipient_write_coeff + donor_delta)
            sign_flip_updates[int(feat)] = float(recipient_write_coeff - donor_delta)
        for feat, ctrl in zip(target_features, similar_features):
            ctrl_write_coeff = float(
                baseline["feature_coefficients"][str(int(ctrl))][int(pair["recipient_write_pos"])]
            )
            donor_prev_coeff = 0.0
            if int(pair["donor_write_pos"]) > 0:
                donor_prev_coeff = float(
                    donor_prompt["feature_coefficients"][str(int(feat))][int(pair["donor_write_pos"]) - 1]
                )
            donor_delta = float(donor_coeffs[int(feat)] - donor_prev_coeff)
            wrong_feature_updates[int(ctrl)] = float(ctrl_write_coeff + donor_delta)
        headroom = eval_meta["baseline_actual_token_prob"] * (1.0 - eval_meta["baseline_actual_token_prob"])
        donor_gap = float(pair["donor_write_coeff"]) - float(pair["recipient_write_coeff"])
        baseline_score = (
            headroom,
            float(eval_meta["baseline_actual_group_mass"]),
            float(pair["context_similarity"]),
            donor_gap,
        )
        baseline_candidates.append(
            {
                "pair": pair,
                "seq": seq,
                "actual_next_token": actual_next_token,
                "baseline": baseline,
                "eval_meta": eval_meta,
                "use_pos": use_pos,
                "target_updates": {int(k): float(v) for k, v in target_updates.items()},
                "wrong_feature_updates": {int(k): float(v) for k, v in wrong_feature_updates.items()},
                "sign_flip_updates": {int(k): float(v) for k, v in sign_flip_updates.items()},
                "score": baseline_score,
            }
        )

    baseline_candidates.sort(key=lambda item: item["score"], reverse=True)
    if not baseline_candidates:
        raise ValueError("No transplant candidates survived exact-token headroom filtering")

    print(
        "[factor-trace-transplant] "
        f"baseline-screened={len(baseline_candidates)} selected={min(len(baseline_candidates), n_pairs)}",
        flush=True,
    )

    for item in baseline_candidates[:n_pairs]:
        pair = item["pair"]
        seq = item["seq"]
        actual_next_token = int(item["actual_next_token"])
        baseline = item["baseline"]
        eval_meta = dict(item["eval_meta"])
        use_pos = int(item["use_pos"])
        eval_pos = int(eval_meta["eval_pos"])
        target_updates = {int(k): float(v) for k, v in item["target_updates"].items()}
        wrong_feature_updates = {int(k): float(v) for k, v in item["wrong_feature_updates"].items()}
        sign_flip_updates = {int(k): float(v) for k, v in item["sign_flip_updates"].items()}
        wrong_time_pos = pair.get("wrong_time_pos")
        prompt_t0 = time.perf_counter()

        recipient_prompt_index = int(pair["recipient_prompt_index"])
        donor_prompt_index = int(pair["donor_prompt_index"])

        print(
            f"[factor-trace-transplant] pair {len(per_pair)+1}/{min(len(baseline_candidates), n_pairs)} "
            f"donor={donor_prompt_index}@{pair['donor_write_pos']} "
            f"recipient={recipient_prompt_index}@{pair['recipient_write_pos']} "
            f"use={pair['recipient_use_pos']} offset={pair['recipient_use_offset']} "
            f"ctx={pair['context_similarity']:.2f} "
            f"tok={eval_meta['eval_actual_text']!r} p={eval_meta['baseline_actual_token_prob']:.3f}",
            flush=True,
        )

        target_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_updates=target_updates,
            intervention_pos=int(pair["recipient_write_pos"]),
        )
        wrong_feature_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_updates=wrong_feature_updates,
            intervention_pos=int(pair["recipient_write_pos"]),
        )
        sign_flip_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_updates=sign_flip_updates,
            intervention_pos=int(pair["recipient_write_pos"]),
        )

        target_logits = target_rollout["logits_by_pos"][eval_pos]
        wrong_feature_logits = wrong_feature_rollout["logits_by_pos"][eval_pos]
        sign_flip_logits = sign_flip_rollout["logits_by_pos"][eval_pos]

        baseline_use_coeff = float(
            sum(float(baseline["feature_coefficients"][str(int(feat))][use_pos]) for feat in target_features)
        )
        target_use_coeff = float(
            sum(float(target_rollout["feature_coefficients"][str(int(feat))][use_pos]) for feat in target_features)
        )
        wrong_feature_use_coeff = float(
            sum(float(wrong_feature_rollout["feature_coefficients"][str(int(feat))][use_pos]) for feat in target_features)
        )
        sign_flip_use_coeff = float(
            sum(float(sign_flip_rollout["feature_coefficients"][str(int(feat))][use_pos]) for feat in target_features)
        )

        target_boundary_shift = _boundary_mass_shift(baseline_logits, target_logits, token_groups["boundary"])
        wrong_feature_boundary_shift = _boundary_mass_shift(
            baseline_logits, wrong_feature_logits, token_groups["boundary"]
        )
        sign_flip_boundary_shift = _boundary_mass_shift(
            baseline_logits, sign_flip_logits, token_groups["boundary"]
        )
        target_newline_shift = _boundary_mass_shift(baseline_logits, target_logits, token_groups["newline"])
        wrong_feature_newline_shift = _boundary_mass_shift(
            baseline_logits, wrong_feature_logits, token_groups["newline"]
        )
        sign_flip_newline_shift = _boundary_mass_shift(
            baseline_logits, sign_flip_logits, token_groups["newline"]
        )
        target_sentence_shift = _boundary_mass_shift(
            baseline_logits, target_logits, token_groups["sentence_punct"]
        )
        wrong_feature_sentence_shift = _boundary_mass_shift(
            baseline_logits, wrong_feature_logits, token_groups["sentence_punct"]
        )
        sign_flip_sentence_shift = _boundary_mass_shift(
            baseline_logits, sign_flip_logits, token_groups["sentence_punct"]
        )

        wrong_time_boundary_shift = 0.0
        wrong_time_newline_shift = 0.0
        wrong_time_sentence_shift = 0.0
        wrong_time_kl_value = 0.0
        wrong_time_coeff_value = baseline_use_coeff
        wrong_time_actual_logit_shift = 0.0

        sign_flip_kl_value = _kl_divergence_from_logits(baseline_logits, sign_flip_logits)
        sign_flip_actual_logit_shift = _token_logit_shift(
            baseline_logits,
            sign_flip_logits,
            int(eval_meta["eval_actual_next_token"])
            if eval_meta["eval_actual_next_token"] is not None
            else actual_next_token,
        )
        if wrong_time_pos is not None:
            wrong_time_rollout = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=tracked_features,
                intervention_updates=target_updates,
                intervention_pos=int(wrong_time_pos),
            )
            wrong_time_logits = wrong_time_rollout["logits_by_pos"][eval_pos]
            wrong_time_boundary_shift = _boundary_mass_shift(
                baseline_logits, wrong_time_logits, token_groups["boundary"]
            )
            wrong_time_newline_shift = _boundary_mass_shift(
                baseline_logits, wrong_time_logits, token_groups["newline"]
            )
            wrong_time_sentence_shift = _boundary_mass_shift(
                baseline_logits, wrong_time_logits, token_groups["sentence_punct"]
            )
            wrong_time_kl_value = _kl_divergence_from_logits(baseline_logits, wrong_time_logits)
            wrong_time_actual_logit_shift = _token_logit_shift(
                baseline_logits,
                wrong_time_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            )
            wrong_time_coeff_value = float(
                sum(
                    float(wrong_time_rollout["feature_coefficients"][str(int(feat))][use_pos])
                    for feat in target_features
                )
            )

        row = {
            **pair,
            **eval_meta,
            "target_feature_indices": [int(feat) for feat in target_features],
            "similar_frequency_controls": [int(feat) for feat in similar_features],
            "primary_target_feature_idx": primary_target_feature,
            "donor_values": {str(k): float(v) for k, v in donor_coeffs.items()},
            "actual_next_token": int(eval_meta["eval_actual_next_token"])
            if eval_meta["eval_actual_next_token"] is not None
            else actual_next_token,
            "actual_next_token_logit_shift_target": _token_logit_shift(
                baseline_logits,
                target_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            ),
            "actual_next_token_prob_shift_target": _token_prob_shift(
                baseline_logits,
                target_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            ),
            "actual_next_token_logit_shift_wrong_feature": _token_logit_shift(
                baseline_logits,
                wrong_feature_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            ),
            "actual_next_token_prob_shift_wrong_feature": _token_prob_shift(
                baseline_logits,
                wrong_feature_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            ),
            "actual_next_token_logit_shift_sign_flip": sign_flip_actual_logit_shift,
            "actual_next_token_prob_shift_sign_flip": _token_prob_shift(
                baseline_logits,
                sign_flip_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            ),
            "actual_next_token_logit_shift_wrong_time": wrong_time_actual_logit_shift,
            "actual_next_token_prob_shift_wrong_time": _token_prob_shift(
                baseline_logits,
                wrong_time_logits,
                int(eval_meta["eval_actual_next_token"])
                if eval_meta["eval_actual_next_token"] is not None
                else actual_next_token,
            ) if wrong_time_pos is not None else 0.0,
            "target_use_time_kl": _kl_divergence_from_logits(baseline_logits, target_logits),
            "wrong_feature_use_time_kl": _kl_divergence_from_logits(baseline_logits, wrong_feature_logits),
            "sign_flip_use_time_kl": sign_flip_kl_value,
            "wrong_time_use_time_kl": wrong_time_kl_value,
            "target_boundary_mass_shift": target_boundary_shift,
            "wrong_feature_boundary_mass_shift": wrong_feature_boundary_shift,
            "sign_flip_boundary_mass_shift": sign_flip_boundary_shift,
            "wrong_time_boundary_mass_shift": wrong_time_boundary_shift,
            "target_newline_mass_shift": target_newline_shift,
            "wrong_feature_newline_mass_shift": wrong_feature_newline_shift,
            "sign_flip_newline_mass_shift": sign_flip_newline_shift,
            "wrong_time_newline_mass_shift": wrong_time_newline_shift,
            "target_sentence_mass_shift": target_sentence_shift,
            "wrong_feature_sentence_mass_shift": wrong_feature_sentence_shift,
            "sign_flip_sentence_mass_shift": sign_flip_sentence_shift,
            "wrong_time_sentence_mass_shift": wrong_time_sentence_shift,
            "baseline_use_coeff": baseline_use_coeff,
            "target_use_coeff": target_use_coeff,
            "wrong_feature_use_coeff": wrong_feature_use_coeff,
            "sign_flip_use_coeff": sign_flip_use_coeff,
            "wrong_time_use_coeff": wrong_time_coeff_value,
            "target_use_coeff_delta": float(target_use_coeff - baseline_use_coeff),
            "wrong_feature_use_coeff_delta": float(wrong_feature_use_coeff - baseline_use_coeff),
            "sign_flip_use_coeff_delta": float(sign_flip_use_coeff - baseline_use_coeff),
            "wrong_time_use_coeff_delta": float(wrong_time_coeff_value - baseline_use_coeff),
        }
        per_pair.append(row)

        target_boundary.append(row["target_boundary_mass_shift"])
        wrong_feature_boundary.append(row["wrong_feature_boundary_mass_shift"])
        wrong_time_boundary.append(row["wrong_time_boundary_mass_shift"])
        target_newline.append(row["target_newline_mass_shift"])
        wrong_feature_newline.append(row["wrong_feature_newline_mass_shift"])
        wrong_time_newline.append(row["wrong_time_newline_mass_shift"])
        target_sentence.append(row["target_sentence_mass_shift"])
        wrong_feature_sentence.append(row["wrong_feature_sentence_mass_shift"])
        wrong_time_sentence.append(row["wrong_time_sentence_mass_shift"])
        target_kl.append(row["target_use_time_kl"])
        wrong_feature_kl.append(row["wrong_feature_use_time_kl"])
        wrong_time_kl.append(row["wrong_time_use_time_kl"])
        target_coeff_delta.append(row["target_use_coeff_delta"])
        wrong_feature_coeff_delta.append(row["wrong_feature_use_coeff_delta"])
        wrong_time_coeff_delta.append(row["wrong_time_use_coeff_delta"])

        print(
            f"[factor-trace-transplant] done {len(per_pair)}/{n_pairs} "
            f"boundary={row['target_boundary_mass_shift']:+.6f} "
            f"wrong_feature={row['wrong_feature_boundary_mass_shift']:+.6f} "
            f"wrong_time={row['wrong_time_boundary_mass_shift']:+.6f} "
            f"coeff_delta={row['target_use_coeff_delta']:+.6f} "
            f"elapsed={time.perf_counter() - prompt_t0:.1f}s",
            flush=True,
        )

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "target_boundary_mass_shift_mean": _mean(target_boundary),
        "wrong_feature_boundary_mass_shift_mean": _mean(wrong_feature_boundary),
        "wrong_time_boundary_mass_shift_mean": _mean(wrong_time_boundary),
        "target_newline_mass_shift_mean": _mean(target_newline),
        "wrong_feature_newline_mass_shift_mean": _mean(wrong_feature_newline),
        "wrong_time_newline_mass_shift_mean": _mean(wrong_time_newline),
        "target_sentence_mass_shift_mean": _mean(target_sentence),
        "wrong_feature_sentence_mass_shift_mean": _mean(wrong_feature_sentence),
        "wrong_time_sentence_mass_shift_mean": _mean(wrong_time_sentence),
        "target_use_time_kl_mean": _mean(target_kl),
        "wrong_feature_use_time_kl_mean": _mean(wrong_feature_kl),
        "wrong_time_use_time_kl_mean": _mean(wrong_time_kl),
        "target_use_coeff_delta_mean": _mean(target_coeff_delta),
        "wrong_feature_use_coeff_delta_mean": _mean(wrong_feature_coeff_delta),
        "wrong_time_use_coeff_delta_mean": _mean(wrong_time_coeff_delta),
        "target_actual_token_logit_shift_mean": _mean(
            [float(row["actual_next_token_logit_shift_target"]) for row in per_pair]
        ),
        "wrong_feature_actual_token_logit_shift_mean": _mean(
            [float(row["actual_next_token_logit_shift_wrong_feature"]) for row in per_pair]
        ),
        "sign_flip_actual_token_logit_shift_mean": _mean(
            [float(row["actual_next_token_logit_shift_sign_flip"]) for row in per_pair]
        ),
        "target_actual_token_prob_shift_mean": _mean(
            [float(row["actual_next_token_prob_shift_target"]) for row in per_pair]
        ),
        "wrong_feature_actual_token_prob_shift_mean": _mean(
            [float(row["actual_next_token_prob_shift_wrong_feature"]) for row in per_pair]
        ),
        "sign_flip_actual_token_prob_shift_mean": _mean(
            [float(row["actual_next_token_prob_shift_sign_flip"]) for row in per_pair]
        ),
        "wrong_time_actual_token_prob_shift_mean": _mean(
            [float(row["actual_next_token_prob_shift_wrong_time"]) for row in per_pair]
        ),
        "wrong_time_actual_token_logit_shift_mean": _mean(
            [float(row["actual_next_token_logit_shift_wrong_time"]) for row in per_pair]
        ),
        "target_minus_sign_flip_actual_token_logit_shift_mean": _mean(
            [
                float(row["actual_next_token_logit_shift_target"]) - float(row["actual_next_token_logit_shift_sign_flip"])
                for row in per_pair
            ]
        ),
        "target_minus_sign_flip_actual_token_prob_shift_mean": _mean(
            [
                float(row["actual_next_token_prob_shift_target"]) - float(row["actual_next_token_prob_shift_sign_flip"])
                for row in per_pair
            ]
        ),
        "target_beats_wrong_feature_fraction": float(
            np.mean([a > b for a, b in zip(target_boundary, wrong_feature_boundary)])
        ) if target_boundary else 0.0,
        "target_beats_sign_flip_fraction": float(
            np.mean([row["actual_next_token_logit_shift_target"] > row["actual_next_token_logit_shift_sign_flip"] for row in per_pair])
        ) if per_pair else 0.0,
        "target_prob_beats_sign_flip_fraction": float(
            np.mean([row["actual_next_token_prob_shift_target"] > row["actual_next_token_prob_shift_sign_flip"] for row in per_pair])
        ) if per_pair else 0.0,
        "target_beats_wrong_time_fraction": float(
            np.mean([a > b for a, b in zip(target_boundary, wrong_time_boundary)])
        ) if target_boundary else 0.0,
        "target_actual_token_beats_wrong_feature_fraction": float(
            np.mean([row["actual_next_token_logit_shift_target"] > row["actual_next_token_logit_shift_wrong_feature"] for row in per_pair])
        ) if per_pair else 0.0,
        "target_actual_token_beats_wrong_time_fraction": float(
            np.mean([row["actual_next_token_logit_shift_target"] > row["actual_next_token_logit_shift_wrong_time"] for row in per_pair])
        ) if per_pair else 0.0,
        "target_coeff_beats_sign_flip_fraction": float(
            np.mean([row["target_use_coeff_delta"] > row["sign_flip_use_coeff_delta"] for row in per_pair])
        ) if per_pair else 0.0,
        "target_minus_sign_flip_use_coeff_delta_mean": _mean(
            [float(row["target_use_coeff_delta"]) - float(row["sign_flip_use_coeff_delta"]) for row in per_pair]
        ),
    }
    summary["boundary_ratio_vs_wrong_feature"] = (
        summary["target_boundary_mass_shift_mean"] / summary["wrong_feature_boundary_mass_shift_mean"]
        if abs(summary["wrong_feature_boundary_mass_shift_mean"]) > 1e-8
        else float("inf")
    )
    summary["boundary_ratio_vs_wrong_time"] = (
        summary["target_boundary_mass_shift_mean"] / summary["wrong_time_boundary_mass_shift_mean"]
        if abs(summary["wrong_time_boundary_mass_shift_mean"]) > 1e-8
        else float("inf")
    )

    return {
        "target_feature_idx": primary_target_feature,
        "target_feature_indices": [int(feat) for feat in target_features],
        "similar_frequency_control": int(similar_features[0]),
        "similar_frequency_controls": [int(feat) for feat in similar_features],
        "n_pairs_evaluated": len(per_pair),
        "total_time_s": float(time.perf_counter() - t0),
        "per_pair": per_pair,
        "summary": summary,
    }


def run_use_site_signed_causal(
    model,
    tokenizer,
    corpus_sequences: list[torch.Tensor],
    benchmark_result: dict[str, Any],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    states: torch.Tensor,
    n_rows: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Edit the signed H4 direction at the benchmarked use site and score local decisions."""
    target_features = [int(x) for x in benchmark_result.get("target_feature_indices", [])]
    if target_features != [62, 105]:
        raise ValueError(f"Expected signed pair [62, 105], got {target_features}")
    accepted_rows = list(benchmark_result.get("accepted_rows", []))
    if not accepted_rows:
        raise ValueError("No accepted benchmark rows for use-site causal stage")

    act_matrix = compute_activation_matrix(sae, states)
    frequencies = (act_matrix > 0).mean(axis=0)
    wrong_feature_pair = [
        choose_similar_frequency_feature(target_features[0], frequencies, exclude=set(target_features)),
        choose_similar_frequency_feature(
            target_features[1],
            frequencies,
            exclude=set(target_features) | {int(choose_similar_frequency_feature(target_features[0], frequencies, exclude=set(target_features)))},
        ),
    ]
    wrong_feature_pair = [int(x) for x in wrong_feature_pair]
    readout_token_ids = _readout_class_token_ids(tokenizer)
    t0 = time.perf_counter()

    sequence_map = {idx: seq.to(device) for idx, seq in enumerate(corpus_sequences)}
    tracked_features = [*target_features, *wrong_feature_pair]

    accepted_rows.sort(key=lambda r: r.get("score", (0, 0, 0, 0)), reverse=True)
    rows = accepted_rows[:n_rows]
    per_row = []

    print(
        "[factor-trace-use-site] "
        f"target={target_features} reverse_sign_control=True wrong_time=True rows={len(rows)}",
        flush=True,
    )

    for i, row in enumerate(rows, start=1):
        recipient_idx = int(row["recipient_prompt_index"])
        seq_full = sequence_map[recipient_idx]
        decision_pos = int(row["decision_pos"])
        intervention_pos = int(row["intervention_pos"])
        wrong_time_pos = row.get("wrong_time_pos")
        actual_next_token = int(row["actual_next_token"])
        seq_end = min(int(seq_full.shape[1]), decision_pos + 3)
        if seq_end <= decision_pos + 1:
            continue
        seq = seq_full[:, :seq_end]
        prompt_t0 = time.perf_counter()

        baseline = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
        )

        target_updates = {}
        reverse_updates = {}
        wrong_feature_updates = {}
        for feat, delta in zip(target_features, [float(row["donor_delta_62"]), float(row["donor_delta_105"])]):
            base_coeff = float(baseline["feature_coefficients"][str(int(feat))][intervention_pos])
            target_updates[int(feat)] = float(base_coeff + delta)
            reverse_updates[int(feat)] = float(base_coeff - delta)
        for ctrl, delta in zip(wrong_feature_pair, [float(row["donor_delta_62"]), float(row["donor_delta_105"])]):
            base_ctrl = float(baseline["feature_coefficients"][str(int(ctrl))][intervention_pos])
            wrong_feature_updates[int(ctrl)] = float(base_ctrl + delta)

        target_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_updates=target_updates,
            intervention_pos=intervention_pos,
        )
        reverse_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_updates=reverse_updates,
            intervention_pos=intervention_pos,
        )
        wrong_feature_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
            intervention_updates=wrong_feature_updates,
            intervention_pos=intervention_pos,
        )

        wrong_time_rollout = None
        if wrong_time_pos is not None:
            wrong_time_rollout = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=tracked_features,
                intervention_updates=target_updates,
                intervention_pos=int(wrong_time_pos),
            )

        baseline_logits = baseline["logits_by_pos"][decision_pos]
        target_logits = target_rollout["logits_by_pos"][decision_pos]
        reverse_logits = reverse_rollout["logits_by_pos"][decision_pos]
        wrong_feature_logits = wrong_feature_rollout["logits_by_pos"][decision_pos]
        wrong_time_logits = wrong_time_rollout["logits_by_pos"][decision_pos] if wrong_time_rollout is not None else baseline_logits

        readout_class = str(row["readout_class"])
        class_ids = readout_token_ids.get(readout_class, [actual_next_token]) or [actual_next_token]
        next_positions = [pos for pos in [decision_pos, min(decision_pos + 1, int(seq.shape[1]) - 1)] if pos in baseline["logits_by_pos"]]

        def _signed_coeff_change(rollout: dict[str, Any]) -> float:
            diffs = []
            for pos in next_positions:
                base = float(baseline["feature_coefficients"]["62"][pos] - baseline["feature_coefficients"]["105"][pos])
                edit = float(rollout["feature_coefficients"]["62"][pos] - rollout["feature_coefficients"]["105"][pos])
                diffs.append(edit - base)
            return float(np.mean(diffs)) if diffs else 0.0

        row_out = {
            **row,
            "actual_token_logit_shift_target": _token_logit_shift(baseline_logits, target_logits, actual_next_token),
            "actual_token_logit_shift_reverse": _token_logit_shift(baseline_logits, reverse_logits, actual_next_token),
            "actual_token_logit_shift_wrong_feature": _token_logit_shift(baseline_logits, wrong_feature_logits, actual_next_token),
            "actual_token_logit_shift_wrong_time": _token_logit_shift(baseline_logits, wrong_time_logits, actual_next_token),
            "actual_token_prob_shift_target": _token_prob_shift(baseline_logits, target_logits, actual_next_token),
            "actual_token_prob_shift_reverse": _token_prob_shift(baseline_logits, reverse_logits, actual_next_token),
            "actual_token_prob_shift_wrong_feature": _token_prob_shift(baseline_logits, wrong_feature_logits, actual_next_token),
            "actual_token_prob_shift_wrong_time": _token_prob_shift(baseline_logits, wrong_time_logits, actual_next_token),
            "readout_class_prob_shift_target": _boundary_mass_shift(baseline_logits, target_logits, class_ids),
            "readout_class_prob_shift_reverse": _boundary_mass_shift(baseline_logits, reverse_logits, class_ids),
            "readout_class_prob_shift_wrong_feature": _boundary_mass_shift(baseline_logits, wrong_feature_logits, class_ids),
            "readout_class_prob_shift_wrong_time": _boundary_mass_shift(baseline_logits, wrong_time_logits, class_ids),
            "signed_direction_coeff_shift_target": _signed_coeff_change(target_rollout),
            "signed_direction_coeff_shift_reverse": _signed_coeff_change(reverse_rollout),
            "signed_direction_coeff_shift_wrong_feature": _signed_coeff_change(wrong_feature_rollout),
            "signed_direction_coeff_shift_wrong_time": _signed_coeff_change(wrong_time_rollout) if wrong_time_rollout is not None else 0.0,
            "local_kl_target": _kl_divergence_from_logits(baseline_logits, target_logits),
            "local_kl_reverse": _kl_divergence_from_logits(baseline_logits, reverse_logits),
            "local_kl_wrong_time": _kl_divergence_from_logits(baseline_logits, wrong_time_logits) if wrong_time_rollout is not None else 0.0,
        }
        per_row.append(row_out)
        print(
            f"[factor-trace-use-site] row {len(per_row)}/{len(rows)} "
            f"tok={row['readout_text']!r} p={row['baseline_actual_token_prob']:.3f} "
            f"target_prob={row_out['actual_token_prob_shift_target']:+.4f} "
            f"reverse_prob={row_out['actual_token_prob_shift_reverse']:+.4f} "
            f"wrong_time_prob={row_out['actual_token_prob_shift_wrong_time']:+.4f} "
            f"signed_shift={row_out['signed_direction_coeff_shift_target']:+.4f} "
            f"elapsed={time.perf_counter() - prompt_t0:.1f}s",
            flush=True,
        )

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "target_actual_token_prob_shift_mean": _mean([r["actual_token_prob_shift_target"] for r in per_row]),
        "reverse_actual_token_prob_shift_mean": _mean([r["actual_token_prob_shift_reverse"] for r in per_row]),
        "wrong_time_actual_token_prob_shift_mean": _mean([r["actual_token_prob_shift_wrong_time"] for r in per_row]),
        "target_minus_reverse_actual_token_prob_shift_mean": _mean(
            [r["actual_token_prob_shift_target"] - r["actual_token_prob_shift_reverse"] for r in per_row]
        ),
        "target_actual_token_logit_shift_mean": _mean([r["actual_token_logit_shift_target"] for r in per_row]),
        "reverse_actual_token_logit_shift_mean": _mean([r["actual_token_logit_shift_reverse"] for r in per_row]),
        "target_minus_reverse_actual_token_logit_shift_mean": _mean(
            [r["actual_token_logit_shift_target"] - r["actual_token_logit_shift_reverse"] for r in per_row]
        ),
        "target_readout_class_prob_shift_mean": _mean([r["readout_class_prob_shift_target"] for r in per_row]),
        "reverse_readout_class_prob_shift_mean": _mean([r["readout_class_prob_shift_reverse"] for r in per_row]),
        "target_signed_direction_coeff_shift_mean": _mean([r["signed_direction_coeff_shift_target"] for r in per_row]),
        "reverse_signed_direction_coeff_shift_mean": _mean([r["signed_direction_coeff_shift_reverse"] for r in per_row]),
        "wrong_time_signed_direction_coeff_shift_mean": _mean([r["signed_direction_coeff_shift_wrong_time"] for r in per_row]),
        "target_beats_reverse_prob_fraction": float(np.mean([
            r["actual_token_prob_shift_target"] > r["actual_token_prob_shift_reverse"] for r in per_row
        ])) if per_row else 0.0,
        "target_beats_wrong_time_prob_fraction": float(np.mean([
            r["actual_token_prob_shift_target"] > r["actual_token_prob_shift_wrong_time"] for r in per_row
        ])) if per_row else 0.0,
        "target_localized_coeff_beats_wrong_time_fraction": float(np.mean([
            r["signed_direction_coeff_shift_target"] > r["signed_direction_coeff_shift_wrong_time"] for r in per_row
        ])) if per_row else 0.0,
    }
    summary["promotion_gate_pass"] = bool(
        summary["target_beats_reverse_prob_fraction"] >= 0.65
        and summary["target_minus_reverse_actual_token_prob_shift_mean"] > 0.0
        and summary["target_beats_wrong_time_prob_fraction"] >= 0.60
        and summary["target_signed_direction_coeff_shift_mean"] > summary["wrong_time_signed_direction_coeff_shift_mean"]
    )
    return {
        "target_feature_indices": target_features,
        "wrong_feature_pair": wrong_feature_pair,
        "n_rows_evaluated": len(per_row),
        "per_row": per_row,
        "summary": summary,
        "total_time_s": float(time.perf_counter() - t0),
    }


def run_use_site_readout_map(
    model,
    tokenizer,
    corpus_sequences: list[torch.Tensor],
    benchmark_result: dict[str, Any],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    n_rows: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Map localized downstream readouts for the signed H4 boundary direction."""
    del device  # kept for interface symmetry
    target_features = [int(x) for x in benchmark_result.get("target_feature_indices", [])]
    if target_features != [62, 105]:
        raise ValueError(f"Expected signed pair [62, 105], got {target_features}")
    accepted_rows = list(benchmark_result.get("accepted_rows", []))
    if not accepted_rows:
        raise ValueError("No accepted benchmark rows for readout-map stage")

    t0 = time.perf_counter()
    sequence_map = {idx: seq for idx, seq in enumerate(corpus_sequences)}
    token_class_ids = _readout_class_token_ids(tokenizer)
    accepted_rows.sort(key=lambda r: r.get("score", (0, 0, 0, 0)), reverse=True)
    rows = accepted_rows[:n_rows]
    per_row = []

    hidden_sum: np.ndarray | None = None
    hidden_abs_sum: np.ndarray | None = None
    hidden_pos_counts: np.ndarray | None = None
    logit_sum: np.ndarray | None = None
    logit_abs_sum: np.ndarray | None = None
    logit_pos_counts: np.ndarray | None = None

    print(
        "[factor-trace-readout-map] "
        f"target={target_features} reverse_sign_control=True rows={len(rows)}",
        flush=True,
    )

    for i, row in enumerate(rows, start=1):
        recipient_idx = int(row["recipient_prompt_index"])
        seq_full = sequence_map[recipient_idx]
        decision_pos = int(row["decision_pos"])
        intervention_pos = int(row["intervention_pos"])
        wrong_time_pos = row.get("wrong_time_pos")
        actual_next_token = int(row["actual_next_token"])
        seq_end = min(int(seq_full.shape[1]), decision_pos + 3)
        if seq_end <= decision_pos + 1:
            continue
        seq = seq_full[:, :seq_end]
        prompt_t0 = time.perf_counter()
        capture_positions = [decision_pos, min(decision_pos + 1, seq_end - 1)]

        baseline = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=target_features,
            capture_hidden_state_positions=capture_positions,
        )
        target_updates = {}
        reverse_updates = {}
        for feat, delta in zip(target_features, [float(row["donor_delta_62"]), float(row["donor_delta_105"])]):
            base_coeff = float(baseline["feature_coefficients"][str(int(feat))][intervention_pos])
            target_updates[int(feat)] = float(base_coeff + delta)
            reverse_updates[int(feat)] = float(base_coeff - delta)

        target_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=target_features,
            intervention_updates=target_updates,
            intervention_pos=intervention_pos,
            capture_hidden_state_positions=capture_positions,
        )
        reverse_rollout = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=target_features,
            intervention_updates=reverse_updates,
            intervention_pos=intervention_pos,
            capture_hidden_state_positions=capture_positions,
        )
        wrong_time_rollout = None
        if wrong_time_pos is not None:
            wrong_time_rollout = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=target_features,
                intervention_updates=target_updates,
                intervention_pos=int(wrong_time_pos),
                capture_hidden_state_positions=capture_positions,
            )

        baseline_logits = baseline["logits_by_pos"][decision_pos]
        target_logits = target_rollout["logits_by_pos"][decision_pos]
        reverse_logits = reverse_rollout["logits_by_pos"][decision_pos]
        wrong_time_logits = wrong_time_rollout["logits_by_pos"][decision_pos] if wrong_time_rollout is not None else baseline_logits
        readout_class = str(row["readout_class"])
        class_ids = token_class_ids.get(readout_class, [actual_next_token]) or [actual_next_token]

        hidden_delta_vecs = []
        for pos in capture_positions:
            base_hidden = baseline["hidden_state_by_pos"].get(pos)
            target_hidden = target_rollout["hidden_state_by_pos"].get(pos)
            reverse_hidden = reverse_rollout["hidden_state_by_pos"].get(pos)
            if base_hidden is None or target_hidden is None or reverse_hidden is None:
                continue
            delta_vec = (target_hidden - reverse_hidden).squeeze(0).numpy().astype(np.float32)
            hidden_delta_vecs.append(delta_vec)
        if hidden_delta_vecs:
            hidden_delta = np.mean(np.stack(hidden_delta_vecs, axis=0), axis=0)
            if hidden_sum is None:
                hidden_sum = np.zeros_like(hidden_delta)
                hidden_abs_sum = np.zeros_like(hidden_delta)
                hidden_pos_counts = np.zeros_like(hidden_delta)
            hidden_sum += hidden_delta
            hidden_abs_sum += np.abs(hidden_delta)
            hidden_pos_counts += (hidden_delta > 0).astype(np.float32)
        else:
            hidden_delta = np.zeros(0, dtype=np.float32)

        logit_delta = (target_logits - reverse_logits).squeeze(0).numpy().astype(np.float32)
        if logit_sum is None:
            logit_sum = np.zeros_like(logit_delta)
            logit_abs_sum = np.zeros_like(logit_delta)
            logit_pos_counts = np.zeros_like(logit_delta)
        logit_sum += logit_delta
        logit_abs_sum += np.abs(logit_delta)
        logit_pos_counts += (logit_delta > 0).astype(np.float32)

        per_row.append(
            {
                **row,
                "actual_token_logit_delta_target_minus_reverse": _token_logit_shift(reverse_logits, target_logits, actual_next_token),
                "actual_token_prob_delta_target_minus_reverse": _token_prob_shift(reverse_logits, target_logits, actual_next_token),
                "readout_class_prob_delta_target_minus_reverse": _boundary_mass_shift(reverse_logits, target_logits, class_ids),
                "actual_token_logit_delta_target_minus_wrong_time": _token_logit_shift(wrong_time_logits, target_logits, actual_next_token),
                "actual_token_prob_delta_target_minus_wrong_time": _token_prob_shift(wrong_time_logits, target_logits, actual_next_token),
                "hidden_state_delta_l2_target_minus_reverse": float(np.linalg.norm(hidden_delta)) if hidden_delta.size else 0.0,
                "elapsed_s": float(time.perf_counter() - prompt_t0),
            }
        )
        print(
            f"[factor-trace-readout-map] row {len(per_row)}/{len(rows)} "
            f"tok={row['readout_text']!r} class={readout_class} "
            f"delta_prob={per_row[-1]['actual_token_prob_delta_target_minus_reverse']:+.4f} "
            f"delta_logit={per_row[-1]['actual_token_logit_delta_target_minus_reverse']:+.4f} "
            f"hidden_l2={per_row[-1]['hidden_state_delta_l2_target_minus_reverse']:.4f} "
            f"elapsed={per_row[-1]['elapsed_s']:.1f}s",
            flush=True,
        )

    n_eval = max(len(per_row), 1)

    def _rank_dims(sum_arr: np.ndarray | None, abs_arr: np.ndarray | None, pos_arr: np.ndarray | None, top_k: int = 16) -> list[dict[str, float]]:
        if sum_arr is None or abs_arr is None or pos_arr is None:
            return []
        mean = sum_arr / n_eval
        mean_abs = abs_arr / n_eval
        pos_frac = pos_arr / n_eval
        sign_consistency = np.abs(2.0 * pos_frac - 1.0)
        score = np.abs(mean) * sign_consistency
        order = np.argsort(-score)[:top_k]
        ranked = []
        for idx in order:
            ranked.append(
                {
                    "index": int(idx),
                    "mean_delta": float(mean[idx]),
                    "mean_abs_delta": float(mean_abs[idx]),
                    "sign_consistency": float(sign_consistency[idx]),
                    "positive_fraction": float(pos_frac[idx]),
                    "score": float(score[idx]),
                }
            )
        return ranked

    def _rank_tokens(sum_arr: np.ndarray | None, abs_arr: np.ndarray | None, pos_arr: np.ndarray | None, top_k: int = 16) -> dict[str, list[dict[str, float | int | str]]]:
        if sum_arr is None or abs_arr is None or pos_arr is None:
            return {"positive": [], "negative": []}
        mean = sum_arr / n_eval
        mean_abs = abs_arr / n_eval
        pos_frac = pos_arr / n_eval
        pos_order = np.argsort(-mean)[:top_k]
        neg_order = np.argsort(mean)[:top_k]
        ranked: dict[str, list[dict[str, float | int | str]]] = {"positive": [], "negative": []}
        for label, order in [("positive", pos_order), ("negative", neg_order)]:
            for idx in order:
                token_text = tokenizer.decode(
                    [int(idx)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                ranked[label].append(
                    {
                        "token_id": int(idx),
                        "token_text": token_text,
                        "mean_delta": float(mean[idx]),
                        "mean_abs_delta": float(mean_abs[idx]),
                        "positive_fraction": float(pos_frac[idx]),
                    }
                )
        return ranked

    summary = {
        "actual_token_prob_delta_target_minus_reverse_mean": float(np.mean([r["actual_token_prob_delta_target_minus_reverse"] for r in per_row])) if per_row else 0.0,
        "actual_token_logit_delta_target_minus_reverse_mean": float(np.mean([r["actual_token_logit_delta_target_minus_reverse"] for r in per_row])) if per_row else 0.0,
        "readout_class_prob_delta_target_minus_reverse_mean": float(np.mean([r["readout_class_prob_delta_target_minus_reverse"] for r in per_row])) if per_row else 0.0,
        "actual_token_prob_delta_target_minus_wrong_time_mean": float(np.mean([r["actual_token_prob_delta_target_minus_wrong_time"] for r in per_row])) if per_row else 0.0,
        "stable_hidden_dims": _rank_dims(hidden_sum, hidden_abs_sum, hidden_pos_counts, top_k=16),
        "top_logit_tokens": _rank_tokens(logit_sum, logit_abs_sum, logit_pos_counts, top_k=12),
    }
    return {
        "target_feature_indices": target_features,
        "n_rows_evaluated": len(per_row),
        "per_row": per_row,
        "summary": summary,
        "total_time_s": float(time.perf_counter() - t0),
    }


def _period_token_id(tokenizer) -> int:
    ids = tokenizer.encode(".", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected '.' to map to a single token, got ids={ids}")
    return int(ids[0])


def _decoder_direction_from_weights(
    sae,
    weights: np.ndarray,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    if hasattr(sae, "V_dec") and hasattr(sae, "W_dec"):
        v = sae.V_dec.detach().float().cpu()
        w = sae.W_dec.detach().float().cpu()
    elif hasattr(sae, "V") and hasattr(sae, "W"):
        v = sae.V.detach().float().cpu()
        w = sae.W.detach().float().cpu()
    else:
        raise ValueError("SAE does not expose rank-1 decoder factors")
    if v.ndim == 2:
        v = v[:, None, :]
    if w.ndim == 2:
        w = w[:, None, :]
    weight_t = torch.from_numpy(weights.astype(np.float32))
    direction = torch.einsum("i,irk,irv->kv", weight_t, v, w)
    fro = float(direction.norm().item())
    if fro > 0:
        direction = direction / fro
    top = np.argsort(-np.abs(weights))[:16]
    top_weights = [
        {"feature_idx": int(idx), "weight": float(weights[idx])}
        for idx in top
        if abs(float(weights[idx])) > 1e-6
    ]
    return direction, top_weights


def collect_exact_token_direction_dataset(
    model,
    corpus_batches: list[torch.Tensor],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    target_token_id: int,
    prompt_len: int = 512,
    negative_keep_prob: float = 0.03,
    max_negative_multiplier: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """Collect SAE activations for exact '.' vs sampled non-'.' next-token decisions."""
    rng = np.random.default_rng(0)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict[str, int]] = []
    positives = 0
    negatives = 0
    prompt_index = 0
    t0 = time.perf_counter()

    print(
        "[factor-trace-dot-direction] "
        f"collect_dataset batches={len(corpus_batches)} prompt_len={prompt_len} target_token={target_token_id}",
        flush=True,
    )

    for batch_idx, batch in enumerate(corpus_batches, start=1):
        token_batch = batch[:, :prompt_len].to(device)
        batch_size, seq_len = token_batch.shape
        if seq_len < 2:
            continue
        cache = None
        for pos in range(seq_len - 1):
            step_ids = token_batch[:, pos : pos + 1]
            if cache is None:
                out = model(input_ids=step_ids, use_cache=True)
            else:
                out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            state_batch = cache.layers[layer_idx].recurrent_states[:, head_idx].float()
            acts = _encode_states(sae, sae_type, state_batch).detach().cpu().numpy().astype(np.float32)
            next_tokens = token_batch[:, pos + 1].detach().cpu().numpy()
            for row_idx in range(batch_size):
                label = int(next_tokens[row_idx] == target_token_id)
                keep = label == 1 or rng.random() < negative_keep_prob
                if not keep:
                    continue
                xs.append(acts[row_idx])
                ys.append(label)
                meta.append({"prompt_index": prompt_index + row_idx, "decision_pos": int(pos)})
                if label == 1:
                    positives += 1
                else:
                    negatives += 1
        prompt_index += batch_size
        if batch_idx % 4 == 0 or batch_idx == len(corpus_batches):
            print(
                "[factor-trace-dot-direction] "
                f"dataset_progress batch={batch_idx}/{len(corpus_batches)} "
                f"positives={positives} negatives={negatives} elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    if positives == 0:
        raise ValueError("No positive '.' examples collected for direction training")
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    meta_arr = np.array(meta, dtype=object)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    max_neg = max(len(pos_idx) * max_negative_multiplier, len(pos_idx))
    if len(neg_idx) > max_neg:
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    selected = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(selected)
    return {
        "x": x[selected],
        "y": y[selected],
        "meta": meta_arr[selected].tolist(),
        "n_positive": int(len(pos_idx)),
        "n_negative_sampled": int(len(neg_idx)),
        "total_time_s": float(time.perf_counter() - t0),
    }


def run_dot_direction_use_site(
    model,
    tokenizer,
    corpus_batches: list[torch.Tensor],
    corpus_sequences: list[torch.Tensor],
    benchmark_result: dict[str, Any],
    use_site_result: dict[str, Any],
    layer_idx: int,
    head_idx: int,
    sae,
    sae_type: str,
    prompt_len: int = 512,
    direction_prompt_count: int = 256,
    n_rows: int = 5,
    alphas: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    min_accuracy: float = 0.65,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train a '.' logistic direction in H4 feature space and test it on exact '.' use-sites."""
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

    target_token_id = _period_token_id(tokenizer)
    dataset = collect_exact_token_direction_dataset(
        model=model,
        corpus_batches=corpus_batches,
        layer_idx=layer_idx,
        head_idx=head_idx,
        sae=sae,
        sae_type=sae_type,
        target_token_id=target_token_id,
        prompt_len=prompt_len,
        device=device,
    )
    x = dataset["x"]
    y = dataset["y"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=0,
    )
    clf.fit(x_train, y_train)
    test_accuracy = float(clf.score(x_test, y_test))
    weights = clf.coef_[0].astype(np.float32)
    direction, top_weights = _decoder_direction_from_weights(sae, weights)
    print(
        "[factor-trace-dot-direction] "
        f"logistic accuracy={test_accuracy:.3f} positives={dataset['n_positive']} "
        f"negatives={dataset['n_negative_sampled']}",
        flush=True,
    )
    print(
        "[factor-trace-dot-direction] "
        f"top_weights={top_weights[:8]}",
        flush=True,
    )
    result: dict[str, Any] = {
        "target_feature_indices": [62, 105],
        "target_token_id": target_token_id,
        "target_token_text": ".",
        "logistic": {
            "test_accuracy": test_accuracy,
            "n_positive": dataset["n_positive"],
            "n_negative_sampled": dataset["n_negative_sampled"],
            "top_weights": top_weights,
        },
    }
    if test_accuracy < min_accuracy:
        result["status"] = "stopped_low_accuracy"
        result["rows"] = []
        result["summary"] = {"success_rows": 0, "success_gate_pass": False}
        return result

    benchmark_map = {
        (int(r["donor_prompt_index"]), int(r["recipient_prompt_index"]), int(r["decision_pos"])): r
        for r in benchmark_result.get("accepted_rows", [])
    }
    candidate_rows = []
    for row in use_site_result.get("per_row", []):
        if row.get("readout_class") != "sentence_reset" or row.get("readout_text") != ".":
            continue
        p = float(row["baseline_actual_token_prob"])
        if p < 0.3 or p > 0.8:
            continue
        delta = float(row["actual_token_prob_shift_target"] - row["actual_token_prob_shift_reverse"])
        coeff = float(row["signed_direction_coeff_shift_target"])
        score = (
            int(delta > 0),
            delta,
            p * (1.0 - p),
            coeff,
        )
        key = (int(row["donor_prompt_index"]), int(row["recipient_prompt_index"]), int(row["decision_pos"]))
        merged = dict(row)
        merged["benchmark_row"] = benchmark_map.get(key, {})
        merged["_score"] = score
        candidate_rows.append(merged)
    candidate_rows.sort(key=lambda r: r["_score"], reverse=True)
    selected_rows = candidate_rows[:n_rows]
    result["selected_rows"] = [
        {
            "donor_prompt_index": int(r["donor_prompt_index"]),
            "recipient_prompt_index": int(r["recipient_prompt_index"]),
            "decision_pos": int(r["decision_pos"]),
            "baseline_actual_token_prob": float(r["baseline_actual_token_prob"]),
            "prior_target_minus_reverse_prob": float(r["actual_token_prob_shift_target"] - r["actual_token_prob_shift_reverse"]),
            "prior_signed_direction_coeff_shift": float(r["signed_direction_coeff_shift_target"]),
        }
        for r in selected_rows
    ]
    print(
        "[factor-trace-dot-direction] "
        f"selected_rows={len(selected_rows)} exact_dot_only=True",
        flush=True,
    )

    sequence_map = {idx: seq.to(device) for idx, seq in enumerate(corpus_sequences)}
    tracked_features = [62, 105, 121, 9, 137]
    per_row = []
    direction = direction.to(device=device, dtype=torch.float32)
    t0 = time.perf_counter()

    for row_idx, row in enumerate(selected_rows, start=1):
        recipient_idx = int(row["recipient_prompt_index"])
        seq_full = sequence_map[recipient_idx]
        decision_pos = int(row["decision_pos"])
        intervention_pos = int(row.get("intervention_pos", max(0, decision_pos - 1)))
        wrong_time_pos = max(0, intervention_pos - 10)
        seq_end = min(int(seq_full.shape[1]), decision_pos + 3)
        seq = seq_full[:, :seq_end]
        baseline = teacher_forced_feature_rollout(
            model=model,
            input_ids=seq,
            layer_idx=layer_idx,
            head_idx=head_idx,
            sae=sae,
            sae_type=sae_type,
            tracked_features=tracked_features,
        )
        baseline_logits = baseline["logits_by_pos"][decision_pos]
        baseline_prob = _token_prob_from_logits(baseline_logits, target_token_id)
        token_ids = seq_full[0].detach().cpu().tolist()
        ctx_start = max(0, decision_pos - 24)
        ctx_end = min(len(token_ids), decision_pos + 2)
        context_text = tokenizer.decode(
            token_ids[ctx_start:ctx_end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        alpha_rows = []
        for alpha in alphas:
            scaled = direction * float(alpha)
            inject = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=tracked_features,
                intervention_state_delta=scaled,
                intervention_pos=intervention_pos,
            )
            reverse = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=tracked_features,
                intervention_state_delta=-scaled,
                intervention_pos=intervention_pos,
            )
            wrong_time = teacher_forced_feature_rollout(
                model=model,
                input_ids=seq,
                layer_idx=layer_idx,
                head_idx=head_idx,
                sae=sae,
                sae_type=sae_type,
                tracked_features=tracked_features,
                intervention_state_delta=scaled,
                intervention_pos=wrong_time_pos,
            )
            inject_logits = inject["logits_by_pos"][decision_pos]
            reverse_logits = reverse["logits_by_pos"][decision_pos]
            wrong_time_logits = wrong_time["logits_by_pos"][decision_pos]
            inject_prob = _token_prob_from_logits(inject_logits, target_token_id)
            reverse_prob = _token_prob_from_logits(reverse_logits, target_token_id)
            wrong_time_prob = _token_prob_from_logits(wrong_time_logits, target_token_id)
            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "baseline_prob": baseline_prob,
                    "inject_prob": inject_prob,
                    "reverse_prob": reverse_prob,
                    "wrong_time_prob": wrong_time_prob,
                    "inject_minus_reverse": float(inject_prob - reverse_prob),
                    "inject_minus_baseline": float(inject_prob - baseline_prob),
                    "reverse_minus_baseline": float(reverse_prob - baseline_prob),
                    "wrong_time_minus_baseline": float(wrong_time_prob - baseline_prob),
                }
            )
        best = max(alpha_rows, key=lambda item: item["inject_minus_reverse"])
        success = bool(
            best["inject_prob"] > best["baseline_prob"]
            and best["reverse_prob"] < best["baseline_prob"]
            and best["inject_minus_reverse"] > 0.05
        )
        per_row.append(
            {
                "donor_prompt_index": int(row["donor_prompt_index"]),
                "recipient_prompt_index": recipient_idx,
                "decision_pos": decision_pos,
                "intervention_pos": intervention_pos,
                "wrong_time_pos": wrong_time_pos,
                "context_text": context_text,
                "baseline_prob": baseline_prob,
                "prior_target_minus_reverse_prob": float(row["actual_token_prob_shift_target"] - row["actual_token_prob_shift_reverse"]),
                "prior_signed_direction_coeff_shift": float(row["signed_direction_coeff_shift_target"]),
                "alphas": alpha_rows,
                "best_alpha": float(best["alpha"]),
                "best_inject_prob": float(best["inject_prob"]),
                "best_reverse_prob": float(best["reverse_prob"]),
                "best_wrong_time_prob": float(best["wrong_time_prob"]),
                "best_inject_minus_reverse": float(best["inject_minus_reverse"]),
                "success": success,
            }
        )
        print(
            "[factor-trace-dot-direction] "
            f"row {row_idx}/{len(selected_rows)} donor={row['donor_prompt_index']}→recipient={recipient_idx}@{decision_pos} "
            f"base={baseline_prob:.3f} best_alpha={best['alpha']:.1f} "
            f"inject={best['inject_prob']:.3f} reverse={best['reverse_prob']:.3f} "
            f"wrong_time={best['wrong_time_prob']:.3f} delta={best['inject_minus_reverse']:+.4f} "
            f"success={success}",
            flush=True,
        )

    success_rows = sum(int(r["success"]) for r in per_row)
    result["status"] = "completed"
    result["rows"] = per_row
    result["summary"] = {
        "success_rows": int(success_rows),
        "n_rows": int(len(per_row)),
        "success_gate_pass": bool(success_rows >= 3),
        "max_inject_minus_reverse": float(max((r["best_inject_minus_reverse"] for r in per_row), default=0.0)),
        "mean_best_inject_minus_reverse": float(np.mean([r["best_inject_minus_reverse"] for r in per_row])) if per_row else 0.0,
        "total_time_s": float(time.perf_counter() - t0),
    }
    return result

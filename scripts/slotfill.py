"""slotfill.py

Fill named SLOT_* placeholders across paper + rebuttal blocks from a single
experiment-summary JSON.

Usage:
    # auto-fetch from HF (preferred)
    python scripts/slotfill.py --experiment rank2_downstream_pilot
    python scripts/slotfill.py --experiment writesae_mamba2
    python scripts/slotfill.py --experiment writesae_rwkv7
    python scripts/slotfill.py --experiment rank2_l9h7_pilot

    # or from a local JSON
    python scripts/slotfill.py --json /path/to/partition_summary.json --experiment writesae_mamba2

    # dry run: print substitutions without writing
    python scripts/slotfill.py --experiment rank2_downstream_pilot --dry-run

The script knows which slots map to which experiment and only writes those.
Repeated runs are idempotent: slots already filled (no [SLOT_X] bracket left
in the file) are silently skipped.

Schema mismatch?
    The EXPERIMENT_TO_SLOTS table at the top of this file defines the dotted
    JSON path for each slot. If the runner emits a different shape, either
    pre-process the JSON into the shape this script expects (recipes in
    SLOTFILL.md) or edit the dotted paths here. Both are first-class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# slot_name -> (dotted JSON path, str.format spec). "{:s}" passes verdict words verbatim.
EXPERIMENT_TO_SLOTS: dict[str, dict[str, tuple[str, str]]] = {
    "rank2_downstream_pilot": {
        "SLOT_R2_REGISTERS": ("rank2.partition.register_count", "{:d}"),
        "SLOT_R2_COS":       ("rank2.partition.median_register_cosine", "{:.3f}"),
        "SLOT_R2_WIN":       ("rank2.firing_level.atom_vs_ablate_pct", "{:.1f}"),
        "SLOT_R2_BLOCK":     ("rank2.factorization.block_rank1_fraction_pct", "{:.0f}"),
        "SLOT_R2_VERDICT":   ("rank2.verdict_word", "{:s}"),
    },
    "rank2_l9h7_pilot": {
        "SLOT_L9H7_WIN":     ("firing_level.atom_vs_ablate_pct", "{:.1f}"),
        "SLOT_L9H7_N":       ("firing_level.n_firings", "{:,}"),
        "SLOT_L9H7_VERDICT": ("verdict_word", "{:s}"),
    },
    "writesae_mamba2": {
        "SLOT_M2_REGISTERS": ("partition.register_count", "{:d}"),
        "SLOT_M2_NULL":      ("partition.null_count", "{:d}"),
        "SLOT_M2_BIC":       ("partition.delta_bic", "{:.0f}"),
        "SLOT_M2_HEAD":      ("firing_level.head_idx", "{:d}"),
        "SLOT_M2_KL_RATIO":  ("firing_level.kl_atom_over_kl_ablate", "{:.2f}"),
        "SLOT_M2_N":         ("firing_level.n_firings", "{:,}"),
    },
    "writesae_rwkv7": {
        "SLOT_R7_LAYER":     ("layer", "{:d}"),
        "SLOT_R7_REGISTERS": ("partition.register_count", "{:d}"),
        "SLOT_R7_COS":       ("partition.median_register_cosine", "{:.3f}"),
        "SLOT_R7_WIN":       ("firing_level.atom_vs_ablate_pct", "{:.1f}"),
        "SLOT_R7_N":         ("firing_level.n_firings", "{:,}"),
    },
}

VERDICT_WHITELIST = {
    "SLOT_R2_VERDICT":   {"matches", "degrades", "splits"},
    "SLOT_L9H7_VERDICT": {"matching", "replicating", "extending"},
}

TARGET_FILES = [
    REPO / "paper-9pager/source/sections/04_experiments.tex",
    REPO / "paper-9pager/drafts/rebuttal_drafts/03_rank_2_attack_response.md",
    REPO / "paper-9pager/drafts/rebuttal_drafts/06_cross_arch_writesae_response.md",
]

# Missing slots stay None so the figure's hatched-TODO renderer still triggers.
FIGURE_FILE = REPO / "paper-9pager/src/figures/build_fig_cross_arch_writesae.py"

HF_REPO = os.environ.get("MATRIX_SAE_HF_REPO", "<anon-handle>/matrix-sae-akash-ckpts")
HF_REVISION = os.environ.get("MATRIX_SAE_HF_REVISION", "main")


def load_summary_from_hf(experiment: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required for --experiment fetching; "
            "install with `pip install huggingface_hub` or pass --json instead."
        ) from e

    candidates = [
        f"{experiment}/partition_summary.json",
        f"{experiment}/summary.json",
        f"{experiment}/results.json",
        f"{experiment}/{experiment}_summary.json",
    ]
    last_err: Exception | None = None
    for cand in candidates:
        try:
            local = hf_hub_download(HF_REPO, cand, revision=HF_REVISION)
            return json.loads(Path(local).read_text())
        except Exception as e:  # noqa: BLE001 -- HF SDK throws many shapes
            last_err = e
            continue
    raise FileNotFoundError(
        f"no summary file for experiment '{experiment}' under {HF_REPO}; "
        f"tried {candidates}. Last error: {last_err!r}"
    )


def get_dotted(d: dict, path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"path '{path}' missing key '{part}'")
        cur = cur[part]
    return cur


def format_value(slot: str, raw: Any, fmt: str) -> str:
    if slot in VERDICT_WHITELIST:
        if not isinstance(raw, str):
            raise ValueError(
                f"{slot} expected a string verdict, got {type(raw).__name__}"
            )
        if raw not in VERDICT_WHITELIST[slot]:
            raise ValueError(
                f"{slot}='{raw}' not in allowed vocabulary "
                f"{sorted(VERDICT_WHITELIST[slot])}"
            )
        return raw
    return fmt.format(raw)


def apply_substitutions(text: str, substitutions: dict[str, str]) -> tuple[str, int]:
    n = 0
    for slot, val in substitutions.items():
        token = f"[{slot}]"
        if token in text:
            n += text.count(token)
            text = text.replace(token, val)
    return text, n


def patch_figure_placeholders(
    text: str, slots: dict[str, str], experiment: str
) -> tuple[str, int]:
    """Idempotent: only replaces None RHS values; pre-filled cells stay put."""
    if experiment == "writesae_mamba2":
        row_key = "Mamba-2-370M-L24"
        # Only int-shaped values map cleanly onto the figure dict from formatted strings.
        patches = {
            "registers": slots.get("SLOT_M2_REGISTERS"),
        }
    elif experiment == "writesae_rwkv7":
        row_key = "RWKV-7-rank2-L?"
        patches = {
            "registers": slots.get("SLOT_R7_REGISTERS"),
            "cos":       slots.get("SLOT_R7_COS"),
        }
    else:
        return text, 0

    marker = f'"{row_key}":'
    idx = text.find(marker)
    if idx == -1:
        return text, 0
    end = text.find("},", idx)
    if end == -1:
        return text, 0
    block = text[idx:end]

    n = 0
    for field, formatted in patches.items():
        if formatted is None:
            continue
        needle = f'"{field}":'
        f_idx = block.find(needle)
        if f_idx == -1:
            continue
        line_end = block.find(",", f_idx)
        if line_end == -1:
            continue
        line = block[f_idx:line_end]
        if "None" not in line:
            continue
        new_line = line.replace("None", formatted, 1)
        block = block[:f_idx] + new_line + block[line_end:]
        n += 1

    if n > 0:
        text = text[:idx] + block + text[end:]
    return text, n


def build_substitutions(
    experiment: str, summary: dict
) -> tuple[dict[str, str], list[str]]:
    slots = EXPERIMENT_TO_SLOTS[experiment]
    substitutions: dict[str, str] = {}
    warnings: list[str] = []
    for slot, (path, fmt) in slots.items():
        try:
            raw = get_dotted(summary, path)
            substitutions[slot] = format_value(slot, raw, fmt)
        except (KeyError, ValueError) as e:
            warnings.append(f"[warn] {slot}: {e}; leaving placeholder unchanged")
    return substitutions, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_TO_SLOTS),
        help="HF subfolder name; also used to pick the slot mapping",
    )
    ap.add_argument(
        "--json",
        help="local path to summary JSON (overrides HF fetch)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print substitutions without writing")
    args = ap.parse_args()

    if not args.experiment and not args.json:
        ap.error("specify --experiment or --json")

    if args.json:
        summary = json.loads(Path(args.json).read_text())
        experiment = args.experiment or summary.get("experiment_name")
        if not experiment:
            ap.error(
                "--json without --experiment requires JSON to include "
                "'experiment_name' at the top level"
            )
    else:
        experiment = args.experiment
        summary = load_summary_from_hf(experiment)

    if experiment not in EXPERIMENT_TO_SLOTS:
        ap.error(
            f"unknown experiment '{experiment}'; "
            f"choices: {sorted(EXPERIMENT_TO_SLOTS)}"
        )

    substitutions, warnings = build_substitutions(experiment, summary)

    print(f"# slotfill plan ({experiment}):", file=sys.stderr)
    for k, v in substitutions.items():
        print(f"  [{k}] -> {v}", file=sys.stderr)
    for w in warnings:
        print(w, file=sys.stderr)

    if not substitutions:
        print("# no slots filled (all paths missing or invalid)", file=sys.stderr)
        return 1 if warnings else 0

    if args.dry_run:
        print("# dry run, no files modified", file=sys.stderr)
        return 0

    total_replacements = 0
    files_changed = 0
    for fpath in TARGET_FILES:
        if not fpath.exists():
            print(f"[warn] missing target file: {fpath}", file=sys.stderr)
            continue
        original = fpath.read_text()
        new_text, n = apply_substitutions(original, substitutions)
        if n > 0 and new_text != original:
            fpath.write_text(new_text)
            files_changed += 1
            total_replacements += n
            print(f"  wrote {fpath.relative_to(REPO)} ({n} replacements)",
                  file=sys.stderr)

    if experiment in {"writesae_mamba2", "writesae_rwkv7"} and FIGURE_FILE.exists():
        original = FIGURE_FILE.read_text()
        new_text, n = patch_figure_placeholders(original, substitutions, experiment)
        if n > 0 and new_text != original:
            FIGURE_FILE.write_text(new_text)
            files_changed += 1
            total_replacements += n
            print(f"  wrote {FIGURE_FILE.relative_to(REPO)} "
                  f"({n} PLACEHOLDERS cells patched)",
                  file=sys.stderr)

    print(
        f"# total: {total_replacements} placeholders filled across "
        f"{files_changed} files",
        file=sys.stderr,
    )
    if experiment in {"writesae_mamba2", "writesae_rwkv7"}:
        print(
            "# next: rebuild fig_cross_arch_writesae.pdf:\n"
            "#   python paper-9pager/src/figures/build_fig_cross_arch_writesae.py",
            file=sys.stderr,
        )
    print(
        "# rebuild paper:\n"
        "#   cd paper-9pager/source && tectonic --untrusted main.tex "
        "&& bash check_paper.sh",
        file=sys.stderr,
    )
    return 0


def _self_test() -> int:
    """Synthetic-JSON smoke test, run with --self-test. Does not touch files."""
    print("=== self-test ===")

    m2_summary = {
        "partition": {"register_count": 138, "null_count": 1910, "delta_bic": 1234.7},
        "firing_level": {
            "head_idx": 3,
            "kl_atom_over_kl_ablate": 0.41,
            "n_firings": 5234,
        },
    }
    subs, warns = build_substitutions("writesae_mamba2", m2_summary)
    assert subs == {
        "SLOT_M2_REGISTERS": "138",
        "SLOT_M2_NULL":      "1910",
        "SLOT_M2_BIC":       "1235",
        "SLOT_M2_HEAD":      "3",
        "SLOT_M2_KL_RATIO":  "0.41",
        "SLOT_M2_N":         "5,234",
    }, f"unexpected: {subs}"
    assert warns == [], warns
    print("  writesae_mamba2: ok")

    r2_summary = {
        "rank2": {
            "partition": {"register_count": 218, "median_register_cosine": 0.985},
            "firing_level": {"atom_vs_ablate_pct": 91.7},
            "factorization": {"block_rank1_fraction_pct": 84.0},
            "verdict_word": "matches",
        }
    }
    subs, warns = build_substitutions("rank2_downstream_pilot", r2_summary)
    assert subs["SLOT_R2_REGISTERS"] == "218", subs
    assert subs["SLOT_R2_COS"] == "0.985", subs
    assert subs["SLOT_R2_WIN"] == "91.7", subs
    assert subs["SLOT_R2_BLOCK"] == "84", subs
    assert subs["SLOT_R2_VERDICT"] == "matches", subs
    assert warns == [], warns
    print("  rank2_downstream_pilot (valid verdict): ok")

    r2_bad = json.loads(json.dumps(r2_summary))
    r2_bad["rank2"]["verdict_word"] = "nope"
    subs, warns = build_substitutions("rank2_downstream_pilot", r2_bad)
    assert "SLOT_R2_VERDICT" not in subs, subs
    assert any("SLOT_R2_VERDICT" in w for w in warns), warns
    print("  rank2_downstream_pilot (bad verdict caught): ok")

    r2_missing = {
        "rank2": {
            "partition": {"register_count": 218},
            "firing_level": {"atom_vs_ablate_pct": 91.7},
            "factorization": {"block_rank1_fraction_pct": 84.0},
            "verdict_word": "matches",
        }
    }
    subs, warns = build_substitutions("rank2_downstream_pilot", r2_missing)
    assert "SLOT_R2_COS" not in subs, subs
    assert "SLOT_R2_REGISTERS" in subs, subs
    assert any("SLOT_R2_COS" in w for w in warns), warns
    print("  rank2_downstream_pilot (missing key caught): ok")

    r7_summary = {
        "layer": 18,
        "partition": {"register_count": 174, "median_register_cosine": 0.412},
        "firing_level": {"atom_vs_ablate_pct": 88.2, "n_firings": 4023},
    }
    subs, warns = build_substitutions("writesae_rwkv7", r7_summary)
    assert subs == {
        "SLOT_R7_LAYER":     "18",
        "SLOT_R7_REGISTERS": "174",
        "SLOT_R7_COS":       "0.412",
        "SLOT_R7_WIN":       "88.2",
        "SLOT_R7_N":         "4,023",
    }, subs
    assert warns == [], warns
    print("  writesae_rwkv7: ok")

    l9h7_summary = {
        "firing_level": {"atom_vs_ablate_pct": 89.4, "n_firings": 3812},
        "verdict_word": "replicating",
    }
    subs, warns = build_substitutions("rank2_l9h7_pilot", l9h7_summary)
    assert subs == {
        "SLOT_L9H7_WIN":     "89.4",
        "SLOT_L9H7_N":       "3,812",
        "SLOT_L9H7_VERDICT": "replicating",
    }, subs
    print("  rank2_l9h7_pilot: ok")

    sample = "result: [SLOT_M2_REGISTERS] regs, head [SLOT_M2_HEAD], n=[SLOT_M2_N]."
    subs = {
        "SLOT_M2_REGISTERS": "138",
        "SLOT_M2_HEAD":      "3",
        "SLOT_M2_N":         "5,234",
    }
    out, n = apply_substitutions(sample, subs)
    assert out == "result: 138 regs, head 3, n=5,234.", out
    assert n == 3, n
    out2, n2 = apply_substitutions(out, subs)
    assert out2 == out and n2 == 0
    print("  text replacement + idempotency: ok")

    fig_text = (
        '    "Mamba-2-370M-L24": {\n'
        '        "short":     "Mamba-2",\n'
        '        "registers": None,\n'
        '        "cos":       None,\n'
        '    },\n'
        '    "RWKV-7-rank2-L?": {\n'
        '        "registers": None,\n'
        '        "cos":       None,\n'
        '    },\n'
    )
    new_fig, n = patch_figure_placeholders(
        fig_text,
        {"SLOT_M2_REGISTERS": "138"},
        "writesae_mamba2",
    )
    assert '"registers": 138' in new_fig, new_fig
    assert n == 1
    new_fig2, n2 = patch_figure_placeholders(
        new_fig,
        {"SLOT_M2_REGISTERS": "138"},
        "writesae_mamba2",
    )
    assert new_fig2 == new_fig and n2 == 0
    print("  figure placeholder patch + idempotency: ok")

    print("=== all self-tests passed ===")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    sys.exit(main())

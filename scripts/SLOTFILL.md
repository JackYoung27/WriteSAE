# slotfill.py — fill SLOT_* placeholders from experiment summaries

One Python script that takes an experiment-summary JSON and substitutes the 19 named placeholders across the §4 source, the two affected rebuttal blocks, and (where applicable) the cross-architecture figure builder. Run it once per experiment as Akash results land. Re-running is idempotent: only unfilled `[SLOT_X]` brackets are touched.

## Quick start

```bash
# rank-2 trained decoder pilot (Qwen3.5-0.8B L9 H4)
python scripts/slotfill.py --experiment rank2_downstream_pilot

# L9 H7 pilot (second cell at the same model)
python scripts/slotfill.py --experiment rank2_l9h7_pilot

# matched-substrate WriteSAE on Mamba-2 370M L24
python scripts/slotfill.py --experiment writesae_mamba2

# rank-2 WriteSAE on RWKV-7
python scripts/slotfill.py --experiment writesae_rwkv7

# from a local JSON instead of HF
python scripts/slotfill.py --json /path/to/summary.json --experiment writesae_mamba2

# preview substitutions without writing
python scripts/slotfill.py --experiment rank2_downstream_pilot --dry-run

# self-test (synthetic JSONs, no files touched)
python scripts/slotfill.py --self-test
```

The HF source is `<anon-handle>/matrix-sae-akash-ckpts/<experiment>/`. The script looks for `partition_summary.json`, `summary.json`, `results.json`, or `<experiment>_summary.json` in that order.

## Files written

| File | Slots used |
|---|---|
| `paper-9pager/source/sections/04_experiments.tex` (SLOT-INDEX block lines ~150-169) | all 19 |
| `paper-9pager/drafts/rebuttal_drafts/03_rank_2_attack_response.md` | 8 (R2_*, L9H7_*) |
| `paper-9pager/drafts/rebuttal_drafts/06_cross_arch_writesae_response.md` | 11 (M2_*, R7_*) |
| `paper-9pager/src/figures/build_fig_cross_arch_writesae.py` (PLACEHOLDERS dict — best-effort) | M2/R7 only |

The figure-file patch only fires for `writesae_mamba2` and `writesae_rwkv7`. It walks the row block for `Mamba-2-370M-L24` or `RWKV-7-rank2-L?` and rewrites `None` cells in place. KL triple values (`kl_atom`, `kl_ablate`, `kl_random`) and MSE cells in the figure are not auto-patched yet — fill those by hand from the same summary, since the dotted-path table here only carries scalars used in §4 prose.

## The 19 slots

### Cross-arch Mamba-2 (writesae_mamba2)

| Slot | JSON path | Format |
|---|---|---|
| `SLOT_M2_REGISTERS` | `partition.register_count` | `{:d}` |
| `SLOT_M2_NULL` | `partition.null_count` | `{:d}` |
| `SLOT_M2_BIC` | `partition.delta_bic` | `{:.0f}` |
| `SLOT_M2_HEAD` | `firing_level.head_idx` | `{:d}` |
| `SLOT_M2_KL_RATIO` | `firing_level.kl_atom_over_kl_ablate` | `{:.2f}` |
| `SLOT_M2_N` | `firing_level.n_firings` | `{:,}` |

### Cross-arch RWKV-7 (writesae_rwkv7)

| Slot | JSON path | Format |
|---|---|---|
| `SLOT_R7_LAYER` | `layer` | `{:d}` |
| `SLOT_R7_REGISTERS` | `partition.register_count` | `{:d}` |
| `SLOT_R7_COS` | `partition.median_register_cosine` | `{:.3f}` |
| `SLOT_R7_WIN` | `firing_level.atom_vs_ablate_pct` | `{:.1f}` |
| `SLOT_R7_N` | `firing_level.n_firings` | `{:,}` |

### Rank-2 L9 H4 (rank2_downstream_pilot)

| Slot | JSON path | Format |
|---|---|---|
| `SLOT_R2_REGISTERS` | `rank2.partition.register_count` | `{:d}` |
| `SLOT_R2_COS` | `rank2.partition.median_register_cosine` | `{:.3f}` |
| `SLOT_R2_WIN` | `rank2.firing_level.atom_vs_ablate_pct` | `{:.1f}` |
| `SLOT_R2_BLOCK` | `rank2.factorization.block_rank1_fraction_pct` | `{:.0f}` |
| `SLOT_R2_VERDICT` | `rank2.verdict_word` | string in `{matches, degrades, splits}` |

### L9 H7 pilot (rank2_l9h7_pilot)

| Slot | JSON path | Format |
|---|---|---|
| `SLOT_L9H7_WIN` | `firing_level.atom_vs_ablate_pct` | `{:.1f}` |
| `SLOT_L9H7_N` | `firing_level.n_firings` | `{:,}` |
| `SLOT_L9H7_VERDICT` | `verdict_word` | string in `{matching, replicating, extending}` |

### Verdict whitelist

`SLOT_R2_VERDICT` and `SLOT_L9H7_VERDICT` reject anything outside their allowed vocabulary; the slot is skipped and a `[warn]` is printed. This catches typos before they land in the rebuttal block.

## Idempotency

`slotfill.py` rewrites a target file only if it actually contains an unfilled bracket for one of that experiment's slots. Running the same experiment twice is a no-op the second time. Running a different experiment afterward fills the slots from that second experiment without disturbing anything already filled.

The figure-builder patch is also idempotent: it only replaces `None` on the right-hand side of a known field; once the cell holds a number, it is left alone.

## Schema mismatch — actual runner output vs. expected paths

The runners in `paper-9pager/src/flagship/` (`akash_rank2_downstream.py`, `akash_mamba2_pipeline.py`, `akash_rwkv7_pipeline.py`) emit JSON with a deeper, per-step shape. `akash_rank2_downstream.py` writes `rank2_downstream_summary.json` with `train_results[]` and `downstream.L{layer}` blocks; `akash_mamba2_pipeline.py` writes a `meta` + `per_feature` dict. Neither matches the flat slot paths above.

You have two equally good options:

1. **Edit the slot paths in `slotfill.py`.** The `EXPERIMENT_TO_SLOTS` dict at the top of the script is the only source of truth. Change a path from `"rank2.partition.register_count"` to whatever the runner actually emits, save, re-run.

2. **Pre-process the runner JSON into the expected shape.** Three recipes covering the current runners:

### Recipe A: rank2_downstream_pilot from `rank2_downstream_summary.json`

```python
import json
from pathlib import Path

raw = json.loads(Path("rank2_downstream_summary.json").read_text())
L9 = raw["downstream"]["L9"]              # layer 9 block
# Pull the rank-2 row from the per-rank result table; key shape depends on
# `evaluate_downstream_perhead_matched`. Inspect once and adapt.
r2_row = L9["per_rank"]["r2"]             # adapt key after first inspection

summary = {
    "experiment_name": "rank2_downstream_pilot",
    "rank2": {
        "partition":     {"register_count": r2_row["n_register"],
                          "median_register_cosine": r2_row["register_median_cos"]},
        "firing_level":  {"atom_vs_ablate_pct": r2_row["atom_vs_ablate_pct"]},
        "factorization": {"block_rank1_fraction_pct": r2_row["block_rank1_pct"]},
        "verdict_word":  "matches",   # set by hand from the headline number
    },
}
Path("rank2_pilot_slotfill.json").write_text(json.dumps(summary, indent=2))
```

Then: `python scripts/slotfill.py --json rank2_pilot_slotfill.json`.

### Recipe B: writesae_mamba2 from the runner's `result["meta"]` block

```python
import json
from pathlib import Path

raw = json.loads(Path("mamba2_writesae_result.json").read_text())
m  = raw["meta"]

# delta_bic, kl ratios, and head_idx come from a separate substitution-test
# stage (currently produced by hand or by a downstream script). Plug those in.
summary = {
    "experiment_name": "writesae_mamba2",
    "partition": {
        "register_count": m["n_register"],
        "null_count":     m["n_bundle"],
        "delta_bic":      <fill>,
    },
    "firing_level": {
        "head_idx":               m["head"],
        "kl_atom_over_kl_ablate": <fill>,
        "n_firings":              <fill>,
    },
}
Path("writesae_mamba2_slotfill.json").write_text(json.dumps(summary, indent=2))
```

### Recipe C: writesae_rwkv7

The RWKV-7 runner is least mature; once it lands, mirror Recipe B and add the top-level `"layer"` key.

The verdict words are author calls, not runner outputs. Set them by hand in the pre-processed JSON before running `slotfill.py`.

## Manual fallback

If a slot's JSON path is missing or the runner emits a wholly different shape, you have three escape hatches in order of preference:

1. Edit `EXPERIMENT_TO_SLOTS` in `slotfill.py` to match the actual shape, re-run.
2. Pre-process the runner JSON into the expected shape (recipes above).
3. Open the four target files and replace the brackets by hand. Slot names are unique strings, so `grep -nR '\[SLOT_' paper-9pager/` enumerates everything that still needs filling.

## After slot-fill

```bash
# if M2 or R7 slots changed, rebuild the cross-arch figure
python paper-9pager/src/figures/build_fig_cross_arch_writesae.py

# rebuild paper + run the line-budget / TODO checks
cd paper-9pager/source && tectonic --untrusted main.tex && bash check_paper.sh
```

`check_paper.sh` will flag any remaining `[SLOT_*]` brackets, so a clean compile after slotfill confirms every expected placeholder for that experiment landed.

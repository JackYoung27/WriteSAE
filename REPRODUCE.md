# Reproducing the paper

Each experiment runs on one GPU via argparse. Run from the repo root so `core.*` and `experiments.*` resolve.

## 1. Install

```
git clone https://github.com/JackYoung27/writesae writesae && cd writesae
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`flash-linear-attention` is required for DeltaNet, GLA, and the Qwen3.5 GDN reference. `mamba-ssm` plus `causal-conv1d` is required for the Mamba-2 spectral audit. The wheel URLs in git history document the exact versions; installing fresh on torch 2.10 triggers an ABI break in `causal-conv1d`, so pin to torch 2.8 with the corresponding wheels or rebuild from source.

```
python -m experiments.<name> --help
```

## 2. Hardware

| experiment | script | GPU | wall time |
| --- | --- | --- | --- |
| Extract 0.8B GDN states, one layer, 50K sequences | `experiments.extraction.extract_states` | 1× A10G or 3090 | 15 min |
| Train one SAE, 0.8B, one head | `core.train` | 1× A10G | 10 min |
| 720-run encoder-swap sweep | `experiments.ablations.encoder_swap_ablation` | 16× A100 parallel | 2–3 h (days sequential) |
| Per-layer five-variant sweep (L1, L9, L17) | `experiments.ablations.layer_encoder_swap_ablation` | 8× A100 parallel | 1–2 h |
| 4B cross-scale | `experiments.run_9pager_overnight` | 1× A100 80GB | ~1 h |
| DeltaNet / GLA validation | `experiments.run_{deltanet,gla}_validation` | 1× A100 | ~30 min each |
| Mamba-2 spectral audit (no training) | `experiments.mamba2.mamba2_write_geometry` | 1× A100 | ~20 min |
| Substitution test (firing-level KL) | `scripts/clean_amplified_kl.py` | 1× CPU | ~5 min/feature |
| Behavioral steering | `experiments.behavioral_steering` | 1× A100 | 4 h |

Sweeps run sequentially by default. Parallelize by launching the script multiple times with different `--seed`, `--head`, or `--layer`, or wrap with joblib or SLURM.

## 3. Data

- Corpus: OpenWebText for training, UltraChat 50K slice for cross-corpus checks. Both load via `datasets`.
- Models: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-27B`, `fla-hub/delta_net-1.3B-100B`, `fla-hub/gla-1.3B-100B`, `state-spaces/mamba-2.8b`.
- SAE checkpoints: [JackYoung27/writesae-ckpts](https://huggingface.co/JackYoung27/writesae-ckpts) — front-facing layout with `writesae/<base_model>/L#_H#/` packs (9 cells), `flat_baseline/` (3 cells), `results/<claim>/` (organized by paper claim), and `pre_registration/predictions/` (192 NPZ shards). See the repo's top-level README for navigation.

## 4. Reproduction levels

- **L1 inspect.** Open `paper/WriteSAE.pdf`, read `core/sae.py`. No GPU.
- **L2 load a published SAE.** `torch.load("best.pt", weights_only=False, map_location="cpu")`. 5 min, CPU.
- **L3 rerun one downstream eval.** Extract 500 sequences of states, load the matching SAE pack from the HuggingFace release, run `experiments.run_batchtopk_downstream`. 1 h on one A10G.
- **L4 retrain one SAE.** `python -m core.train --sae_type bilinear --data_dir states --layer 9 --head 0 --n_features 2048 --k 32 --output_dir ckpt`. 10 min on one A10G.
- **L5 paper replication.** Run every row in §2. ~400 H100-hours across the main SAE sweep (~80), the 720-run encoder-swap ablation (~300), and state extraction (~18).

## 5. Expected outputs

Each script writes a JSON under `--output-dir`. JSONs match those cited in the paper. Last-digit variation is expected from non-deterministic tensor ops. Hypothesis-test conclusions (atom beats ablation, σ₁/σ₂ predicts decoder choice) are stable across seeds.

## 6. Loading an HF checkpoint

```python
import torch
ckpt = torch.load("writesae_L9_H4_nf2048_k32_s42/best.pt", weights_only=False, map_location="cpu")
print(ckpt["config"])
print(ckpt["val_mse"])
```

The `manifest.json` in the HF repo maps tags to SHA256 and metadata.

## 7. Known pitfalls

- SAE checkpoints save with `torch.save` using a full module pickle. Load with `weights_only=False`.
- Older rank-1 checkpoints have 2D decoder atoms `(d_k, d_v)` rather than 3D `(1, d_k, d_v)`. `core.sae.load_sae_checkpoint` upgrades these on load.
- State `.npy` files are memory-mapped; put them on SSD.
- The 4B model in fp16 uses ~10 GB VRAM for the transformer alone, before SAE replacement.

## 8. Pre-registration

Experiment C was pre-registered on 2026-04-17 before any prediction or observation pass ran. The pre-registration document, the `R²`-based hypotheses, and the audit script live at:

- `paper-9pager/pre_registration.md`
- `scripts/prereg_r2_computation.py`
- 192 prediction NPZ shards on HF: `JackYoung27/writesae-ckpts/pre_registration/predictions/`

The Stage 3 observed pass populates via `scripts/slotfill.py --experiment exp_c_observe`.

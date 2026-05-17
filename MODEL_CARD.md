# WriteSAE checkpoints — model card

## Artifact

Sparse autoencoders trained on cached GatedDeltaNet write activations, accompanying the paper *WriteSAE: Sparse Atoms that Substitute for Recurrent State Writes*.

- HF repo: [JackYoung27/writesae-ckpts](https://huggingface.co/JackYoung27/writesae-ckpts)
- Code: [https://github.com/JackYoung27/writesae](https://github.com/JackYoung27/writesae)
- Paper: see `paper/` in the code repo

## Variants released

| variant | encoder | decoder | downstream PPL Δ (L9 H4, 0.8B) |
| --- | --- | --- | ---: |
| WriteSAE (primary) | bilinear $v_i^\top S w_i$ | rank-1 $v_i w_i^\top$ | +0.58% |
| FlatSAE | linear on vec($S$) | flat | +3.31% |
| MatrixSAE | linear on vec($S$) | rank-1 | +11.18% |
| BilinearSAE | bilinear | bilinear | +1.33% |

Each variant carries TopK sparsity (per-variant *k*; BatchTopK supported via `--use_batchtopk`).

## Base models

- Qwen/Qwen3.5-0.8B (primary, all main-text results)
- Qwen/Qwen3.5-4B (cross-scale replication)
- Qwen/Qwen3.5-27B (cross-scale replication)
- fla-hub/delta_net-1.3B-100B, fla-hub/gla-1.3B-100B (cross-arch)
- state-spaces/mamba-2.8b (cross-arch, no SAE training; spectral audit only)

## Layer / head coverage

- L9 H4 — primary substitution site (0.8B)
- L1 H4, L17 H4 — cross-layer firing-ordering experiments
- Full L9, all 16 heads — bestiary distribution and selectivity sweep
- 47 cells × 8 features — uniqueness controls

## Training

- Architecture: rank-1 decoder atoms $v_i w_i^\top$, bilinear encoder
- Dictionary size: 16,384 features (`--n_features`; expansion factor configurable)
- Sparsity: TopK (per-variant *k*)
- Data: OpenWebText (`Skylion007/openwebtext`, streaming) tokenized with the Qwen3.5 tokenizer; states extracted via `experiments/extraction/extract_states.py`
- Split: 80/20 train/val, seed 42 (deterministic)
- Compute: ~180 H100-hours single-GPU total across variants (paper App. B.3; full L5 replication including the 720-run encoder-swap ablation is ~400 H100-hours per `REPRODUCE.md`)

## Evaluation

- Substitution: forward KL at firing positions under three matched-Frobenius-norm conditions (atom, ablation, random rank-1). Pooled *n*=4,851 firings (L1 1,500 + L9 1,851 + L17 1,500) at Qwen3.5-0.8B L9 H4.
- Closed-form factorization: per-firing logit shift predicted at median *R²*=0.98 across 200 atom×ε cells.
- Steering: 30 prompts × 5 install positions × 8 target tokens × 3 magnitudes = 3,600 trials.

## Files on the HF repo

```
JackYoung27/writesae-ckpts/
  writesae_L{1,9,17}_H{0..15}/                # WriteSAE checkpoints
  flatsae_L{1,9,17}_H{0..15}/                 # FlatSAE controls
  matrixsae_L{1,9,17}_H{0..15}/               # MatrixSAE controls
  bilinearsae_L{1,9,17}_H{0..15}/             # BilinearSAE controls
  exp_c_full_seed2026/exp_c/predictions/      # 192 prediction NPZ shards
  memory_edit_L9_H4/                          # F412 ERASE records
  behavioral_steering_L9_H4/                  # INSTALL records
  flat_sae_svd_{gdn,mamba2,rwkv7}/            # Cross-arch SVD
  manifest.json                               # tag → SHA256 + metadata
```

## Loading

```python
import torch
ckpt = torch.load("writesae_L9_H4_nf2048_k32_s42/best.pt", weights_only=False, map_location="cpu")
print(ckpt["config"], ckpt["val_mse"])
```

## License

- Code and SAE checkpoints: MIT
- Base models retain the upstream Tongyi Qianwen license (Qwen3.5) and the licenses of the FLA models (DeltaNet, GLA) and Mamba-2.
- We do not redistribute base-model weights; all base models are loaded from their public HF release at runtime.

## Citation

```bibtex
@article{young2026writesae,
  title  = {WriteSAE: Sparse Atoms that Substitute for Recurrent State Writes},
  author = {Jack Young},
  year   = {2026},
  journal= {arXiv preprint},
  url    = {https://github.com/JackYoung27/writesae},
}
```

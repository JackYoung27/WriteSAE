# Bilinear SAE variants: Gated and JumpReLU

Two non-TopK sparsity mechanisms added to `core/sae.py`, both preserving the
existing rank-1 bilinear decoder (`V_dec` outer `W_dec`) used by
`BilinearMatrixSAE` and `BilinearEncoderFlatSAE`.

## Class summary

| `sae_type`            | Class                  | Sparsity mechanism                | Key params                                                 |
|-----------------------|------------------------|-----------------------------------|------------------------------------------------------------|
| `bilinear_gated`      | `BilinearGatedSAE`     | Dual-path gate + magnitude        | `lambda_sparsity`, `lambda_aux`                            |
| `bilinear_jumprelu`   | `BilinearJumpReLUSAE`  | JumpReLU + learnable threshold    | `lambda_sparsity`, `bandwidth`, `init_threshold`           |

Both share the rank-1 bilinear decoder (`V_dec`, `W_dec`) and expose the
standard `SAEOutput` (`reconstruction`, `coefficients`, `loss`, `mse`,
`aux_loss`, `n_dead`). Both accept `use_batchtopk` for signature parity but
ignore it (their activation replaces TopK entirely).

---

## 1. `BilinearGatedSAE` — Rajamanoharan et al. 2024, [arXiv:2404.16014](https://arxiv.org/abs/2404.16014)

### Architecture

A single bilinear encoder `W = V_enc outer W_enc` is shared by two paths:

```
shared  = einsum(V_enc, x, W_enc)                 # (batch, n_features)
gate_pre = shared + b_gate
mag_pre  = exp(r_mag) * shared + b_mag
a(x)     = 1[gate_pre > 0] * ReLU(mag_pre)        # final coefficients
```

`r_mag` is a learnable per-feature log-scaling (paper's exact parameterization);
this lets the magnitude path scale independently of the gating path without
extra matrices.

### Losses

`forward(x)` returns:

- `mse` — standard decoder reconstruction of `a(x)` vs `x`
- `aux_loss` = `lambda_sparsity * |ReLU(gate_pre)|_1 + lambda_aux * ||decoder_frozen(ReLU(gate_pre)) - x||^2`
- `loss = mse + aux_loss`

`decoder_frozen` stops gradient on `V_dec`, `W_dec`, and `bias`, so the
auxiliary reconstruction only updates the gate path's weights. This is the
"auxiliary task" construction from the paper — it prevents the L1 term on
the gate path from shrinking the decoder atoms.

You can also call `sae.aux_loss(x)` directly if you want the auxiliary term
without running the full forward.

### Hyperparameters

| name              | default | notes                                                         |
|-------------------|---------|---------------------------------------------------------------|
| `lambda_sparsity` | `1e-3`  | L1 coefficient on the gating path                             |
| `lambda_aux`      | `1.0`   | Weight on the frozen-decoder gate-path reconstruction         |
| `rank`            | `1`     | Rank of bilinear atoms (matches `BilinearMatrixSAE`)          |

### Instantiate from config

```python
from core.sae import build_sae_from_config

cfg = {
    "sae_type": "bilinear_gated",
    "d_k": 128,
    "d_v": 128,
    "n_features": 16384,
    "rank": 1,
    "lambda_sparsity": 1e-3,
    "lambda_aux": 1.0,
}
sae = build_sae_from_config(cfg)
```

---

## 2. `BilinearJumpReLUSAE` — Rajamanoharan et al. 2024, [arXiv:2407.14435](https://arxiv.org/abs/2407.14435)

### Architecture

```
z       = einsum(V_enc, x, W_enc) + b_enc
theta   = exp(log_threshold)                # per-feature, strictly positive
a_i     = z_i * 1[z_i > theta_i]            # hard jump; no ReLU smoothing
```

Threshold is parameterized in log-space so it stays positive and the
optimizer sees well-behaved gradients through `exp`.

### Gradients (straight-through estimator, rectangle kernel)

Two custom autograd functions in `core/sae.py`:

- `_JumpReLUSTE` — handles gradient of `a_i` through `z` and `theta`.
  Forward: `z * 1[z > theta]`. Backward:
  - `dL/dz = grad_out * 1[z > theta]`  (pass-through on active)
  - `dL/dtheta = -(theta / h) * rect((z - theta)/h) * grad_out`
- `_L0HeavisideSTE` — handles gradient of the L0 penalty through `theta`.
  Forward: `1[z > theta]`. Backward:
  - `dL/dz = 0`  (L0 must not push pre-activations)
  - `dL/dtheta = -(1/h) * rect((z - theta)/h) * grad_out`

`rect(u)` is 1 for `|u| <= 1/2`, else 0. The bandwidth `h` controls how
close `z` has to be to `theta` for the STE to pass gradient. Both
backward paths apply the `theta -> log_threshold` chain rule (multiply by
`theta`).

### Losses

`forward(x)` returns:

- `mse` — reconstruction MSE of `decode(a(x))` vs `x`
- `aux_loss` = `lambda_sparsity * mean_batch sum_i H_STE((z_i - theta_i)/h)` — an L0 surrogate whose gradient drives the thresholds into the tails of the `z` distribution
- `loss = mse + aux_loss`

Call `sae.aux_loss(x)` for the standalone sparsity term.

### Hyperparameters

| name              | default | notes                                                                 |
|-------------------|---------|-----------------------------------------------------------------------|
| `lambda_sparsity` | `1e-3`  | L0 coefficient                                                        |
| `bandwidth`       | `1e-3`  | Rectangle kernel half-width `h` for the STE                           |
| `init_threshold`  | `1e-3`  | Initial `theta_i` (stored as `log_threshold = log(init_threshold)`)   |
| `rank`            | `1`     | Rank of bilinear atoms                                                |

Picking `bandwidth`: needs to cover the spread of `z - theta` across
training. `1e-3` works once activations settle to small magnitudes near
the threshold. For early training you may want wider (`1e-2`), then
anneal.

### Instantiate from config

```python
from core.sae import build_sae_from_config

cfg = {
    "sae_type": "bilinear_jumprelu",
    "d_k": 128,
    "d_v": 128,
    "n_features": 16384,
    "rank": 1,
    "lambda_sparsity": 1e-3,
    "bandwidth": 1e-3,
    "init_threshold": 1e-3,
}
sae = build_sae_from_config(cfg)
```

---

## Training integration notes (for `core/train.py`)

Both variants work with the existing `SAEOutput`-consuming training loop
without changes: `output.loss` already includes `aux_loss`. If train.py
wants to log the sparsity and auxiliary terms separately it can read
`output.aux_loss` (already aggregated) or call `sae.aux_loss(x)` directly.

Dead-feature resampling is implemented for both variants via
`resample_dead_features(x_batch)`, following the same SVD-initialized
path as `BilinearMatrixSAE`. For JumpReLU, dead features also get their
`log_threshold` reset to the mean of alive features so the threshold
doesn't stay stuck above the signal after the feature direction has
moved.

## Inference / `build_sae_from_config` round-trip

`infer_sae_type(state_dict=...)` detects the variants by distinguishing
parameters:

- `b_gate` + `r_mag` + `V_dec` -> `bilinear_gated`
- `log_threshold` + `V_dec` -> `bilinear_jumprelu`

so saved checkpoints load correctly even if the config is missing.

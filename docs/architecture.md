# The architecture: encoder, flow head and latent, on one page

Read from `src/chai/models/encoder.py`, `src/chai/models/flow.py`, `src/chai/models/gaussian.py`,
`src/chai/models/latent.py` and `src/chai/train/loop.py`, with the numbers of the shipped
members (`configs/train_flow_v22_100.yaml`, `configs/train_flow_v22_250.yaml`). **The two members
differ in depth and length only**: the 100 Ma member (`assets/flow_v22_100.pt`) has **12**
couplings and a 95-dimensional latent, the 250 Ma member (`assets/flow_v22_250.pt`) has **16**
and 159, each with its own corpus and latent. The sketch is drawn for the 100 Ma member.
[family.md](family.md) has the prior, the observation model, the training and the calibration.

Encoder and head are one `nn.Module` (`AmortizedPosterior`, whose forward is
`head(encoder(batch))`), trained end to end by one AdamW on one loss. The latent travels inside
it but is fitted once from the corpus and never trained.

```
  ONE SAMPLE (its measurements, and the user's present-day temperature)
  ┌────────────────────┬───────────────────────┬──────────────────────────┬──────────────┐
  │ AFT scalars (9)    │ length histogram      │ He grains: a set of G,   │ T0, °C       │
  │ log age, log sd,   │ 200 bins of 0.1 µm,   │ each 4 numbers: log age, │ one draw per │
  │ log1p counts,      │ summed in pairs to    │ log sd, log radius,      │ path from    │
  │ mean length, Dpar, │ 100 bin probabilities │ log eU; plus a mask for  │ the user's   │
  │ 4 has_* flags      │ × 100 (flat = 1)      │ which slots are filled   │ range        │
  └─────────┬──────────┴───────────┬───────────┴────────────┬─────────────┴──────┬───────┘
            │                      │                        │                    │
 ═══════════╪══════════════════════╪════ ENCODER ═══════════╪════════════════════╪═══════
            ▼                      ▼                        ▼                    ▼
    Standardize             MLP 100→64→64          Standardize (fitted)      Standardize
    (moments fitted         (1 hidden layer)       per-grain MLP 4→128→128
     once; flags left                              ×mask ─► masked MEAN ‖ masked MAX
     untouched)                                            ‖ log1p(count) ‖ any_grain
            │                      │                        │ (Deep Sets: order-free)
            │                      │                  MLP 258→128→128
            │ 9                    │ 64                     │ 128                │ 1
            └──────────── concat (202) ──────────────────────┴────────────────────┘
                                   ▼
                  TRUNK MLP 202 → 256 → 256 → 256 → 256
                  (3 hidden layers)
                                   ▼
                        embedding x  (256-vector)
 ═══════════════════════════════════╪═══════════ FLOW HEAD ════════════════
                                    │ (x conditions every layer below)
    base  u ~ Student-t, 95 independent coordinates (159 on the 250 Ma member)
          each with its own learned df (starts at 30, floor 2.5)
                                    ▼
    12 × rational-quadratic spline coupling (Durkan et al. 2019)   [16 on the 250 Ma member]
      • half the coordinates are transformed, the other half condition them
      • the split alternates by parity and by position
      • conditioner: MLP [other half ‖ x] → 256 → 256 → spline params
      • 8 bins on [−5, 5]; identity outside (so tails = base tails)
      • end slopes pinned at 1; starts as the identity (output × 1e-3)
                                    ▼
    affine layer = the GaussianHead:  z = μ(x) + L(x)·v
      • MLP 256 → 256 → (95 means + 95·96/2 Cholesky entries)
      • diag(L) = softplus(·) + 0.001; lower triangle taken as is
                                    ▼
                    z  (95-vector, whitened coordinates)
 ═══════════════════════════════════╪═══════════ LATENT (fixed) ═══════════
                                    ▼
    un-whiten:  T_1..T_95 − T_0 = μ + W⁻¹ z   (Cholesky whitening of the path's
                                               excess over the present day; fitted
                                               to the corpus once, not trained)
      • 95 temperature nodes on the 1.05–100 Ma grid, present → past
      • T_0 is the user's present-day temperature, handed in, not drawn
                                    ▼
    remove_floor: the 1 K training noise taken back out of each batch of draws
                                    ▼
             a thermal history T(t) in °C on the member's grid
```

## The choices, component by component

**Encoder.**
- Every MLP is `Linear → LayerNorm → GELU`, repeated, then a final `Linear`.
- Widths are small on purpose, since the input is a few hundred numbers: embedding 256, trunk
  256 × 3 hidden layers, grain branch 128, histogram branch 64.
- The length histogram is read at 100 bins of 0.2 µm (`hist_pool: 2`): what laboratories
  report, and measured to cost nothing against 200.
- Missing data is zero plus a `has_*` flag, never NaN. Training drops the lengths of 30% of
  samples, all fission-track data of 20% and helium of 30%, so incomplete samples get their own,
  wider posterior.
- Grains are a set (Deep Sets, masked mean and max). The mask multiplies the embeddings rather
  than the features, so padding cannot leak into the pool.
- The present-day temperature is one more standardised scalar. It is not data but the user's
  surface temperature; `api` draws one per path from the range the user gives.

**Flow head.**
- Student-t base with learned degrees of freedom per coordinate; the Gaussian-base twin matches
  every coverage to within a point, so the shape lives in the splines.
- Initialisation is splines at the identity, the Gaussian layer at N(0, I) and a near-Gaussian
  base, so training starts exactly at "posterior = prior".
- 6.8 M parameters in the 100 Ma member (27 MB), 13.8 M in the 250 Ma member (56 MB). Densities
  are exact both ways: one pass for `log_prob`, one for `sample`.

**Latent.**
- An affine map fitted once from the corpus, with nothing learned by gradient.
- Coordinates are the nodes from the present into the past, each as its excess over `T_0`.
  With Cholesky whitening, `z_k` is the innovation at node k given everything more recent, so
  where the data say nothing, N(0, 1) already reproduces the prior.
- The pulse prior has no hyperparameters a density can hold (two of its latents are counts), so
  the latent is the path alone: 95 coordinates on the 100 Ma grid, 159 on the 250 Ma grid.

**Training.**
- Loss: the negative log-likelihood of the true `z`, end to end through head and encoder.
- AdamW, lr 1e-3, cosine to 1e-5 after 500 warmup steps; no weight decay; gradient clip 10;
  batch 4 096; 200 000 steps, about 341 passes over the 2.4 M training rows of each corpus.
- Every batch is a fresh noisy measurement of the stored truths (observation model and effort
  prior), so each epoch is a new dataset.
- The target carries 1 K of Gaussian noise per node, without which a flow fitted by maximum
  likelihood has no minimum. The API takes that floor back out of the draws (`remove_floor`,
  from the draws' own covariance), which is why a member is asked for at least 200 (100 Ma) or
  318 (250 Ma) draws at once and subsamples when fewer are wanted.
- The shipped weights are the 200 000-step snapshots.

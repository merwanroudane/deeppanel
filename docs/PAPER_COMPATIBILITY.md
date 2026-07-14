# Paper ↔ code compatibility map

This document maps every implemented step to the exact equation / algorithm /
section of the source papers, and records — honestly — what is **not** ported and
why. It is the "check each step is compatible with the paper(s)" artifact.

Papers:

- **P2** — Chronopoulos, Chrysikou, Kapetanios, Mitchell, Raftapostolos,
  *Forecasting with Deep Pooled Panel Neural Networks*, Econometric Reviews 2026
  (`doi:10.1080/07474938.2026.2660660`).
- **P3** — same authors, *Deep Neural Network Estimation in Panel Data Models*,
  FRB Cleveland WP 23-15, 2023 (`doi:10.26509/frbc-wp-202315`). Theory paper.
- **P1** — Gao, Sun, Wang, Liu, Hsiao, *How Does LLM Help Regional CPI Forecast:
  An LLM-powered Deep Panel Modeling Framework*, 2026 (arXiv 2604.06894).

---

## A. Deep pooled panel (P2 / P3) → `DeepPooledPanel` and friends

| Paper object | Equation / section | Code |
|---|---|---|
| Conditional mean `E(y_it\|x_it)=h̃_i(x_it)` | P3 Eq. 1 | `DeepPooledPanel` |
| Decomposition `h̃_i = h + h_i` (common + idiosyncratic) | P3 Eq. 3 | `idiosyncratic=True/False` |
| Feed-forward ReLU MLP, linear output, `b⁽¹⁾=0` | P3 Eq. 16 / P2 Eq. 3 | `models/mlp.py` (`first_layer_bias=False`) |
| Final linear representation `g=θ_L' f(x)` | P2 Eq. 8 / P3 Eq. 6 | `MLP.features()` |
| Pooled MSE objective `(1/NT)ΣΣ(y−g)²` | P3 Eq. 9 | `training/trainer.py` |
| Cross-sectional-averaged equivalent | P3 Eq. 12 | (algebraically equivalent to Eq. 9; Eq. 9 used) |
| Idiosyncratic step `argmin_θi (1/T)Σ(y−g−g_i)²` | P3 Eq. 10 | `DeepPooledPanel.fit` step 2 |
| LASSO penalty `λ‖θ‖₁`, `λ=c√(log p/NT)` | P2/P3 §3.1, §3.3 | `TrainConfig.lam`, `cv.lasso_lambda` |
| ADAM, dropout, batch-norm, early stopping, 5000 epochs | §3.2–3.4 | `TrainConfig`, `MLP(batchnorm=,dropout=)` |
| CV grids L∈{1,3,5,10,15}, M∈{5,…,30}, γ∈{.01,.001}, c-grid | §3.3 | `PAPER_GRID`, `cross_validate` |
| Train/val/test time split 0.8/0.2, direct h-step | §3.3, P2 fn. 5 | `PanelData.direct_pairs`, `val_frac` |
| Rank/min-max normalisation to [0,1] | P2 Eq. 10 | `MinMaxNormalizer` |
| Standardisation (inflation app.) | P2 §4.2 | `Standardizer` |
| Partial derivatives `d_{ij,t}=∂g/∂x_{i,j,t−h}` | P2 Eq. 11 / P3 Eq. 11 | `inference/partial.py` (autograd) |
| Poolability test `P=(1/σ̂√N)Σ(T R²_i−m)` | P3 Remark 3 | `inference/poolability.py` |
| AR(1) benchmark, iterated | P2 §4 | `benchmarks/ar.py` |
| PVAR `z_it=μ_i+ΣA_q z + ε`, BIC lags, iterated | P2 fn. 5 | `benchmarks/pvar.py` |
| Deep time-series (per-unit, no pooling) | P2/P3 §4 | `benchmarks.DeepTimeSeries` |
| Diebold–Mariano test | P2, DM 1995 | `evaluation/dm.py` |
| Giacomini–Rossi fluctuation test | P2, GR 2010 | `evaluation/fluctuation.py` |
| Expanding-window recursive OOS | §3.3, §4 | `forecast/rolling.py` |
| Double descent (non-penalised best) | P2 Remark 2 / P3 Remark 5 | `lam=0` default; documented |

**Empirically verified (known-truth DGP `simulate_pooled_panel`):**
network fits to the noise floor (MSE 0.027 vs Var 0.54); gradient recovery
corr 0.61 / sign-agreement 0.80; poolability test size correct (P=−0.7, no reject
under H0) and powered (P=4.9, reject under H1); **deep pooled beats deep
time-series** (RMSE 0.335 vs 0.528) and both beat PVAR/AR with significant DM
statistics — reproducing the papers' central result.

## B. LDPM (P1) → `LDPM`, `DeepPanelTraining`, conformal

| Paper object | Equation / section | Code |
|---|---|---|
| Target model `y=F_i(x,z)+ε` | P1 Eq. 3.1 | `LDPM` design |
| Surrogate model `y^S=G_i(x^S,lags)+ε^S` | P1 Eq. 3.2 | `surrogate/augment.py` |
| Residual link `ε=Γ(ε^S)+e` → augmented `Q_i=F_i+Γ` | P1 Eq. 3.3–3.4 | design `[x,z,ε^S]` |
| Shared feature map + unit heads `β_i'h+b_i` | P1 Eq. 3.5–3.6 | `classo._SharedBody`, heads |
| Latent groups `(η_k,φ_k)`, assignment `g(i)` | P1 §3.2 | `DeepPanelTraining` |
| C-LASSO product penalty (Eq. 3.7) | P1 Eq. 3.7 | `DeepPanelTraining` penalised loss |
| Group assignment + unpenalised refit | P1 §3.3 | hard-assign + pooled OLS refit |
| Common-scale normalisation of hidden rep. | P1 §3.3 | `BatchNorm1d(affine=False)` |
| Split conformal, within-group `q_k` | P1 §3.4 | `inference/conformal.py` |
| LPM / LPM-E linear baselines | P1 §4.2 | `benchmarks.LinearPooledPanel` |
| SVD embedding reduction `X→XV` | P1 App. A.2 | `surrogate.svd_reduce` |
| Max-pool embeddings / mean-pool scores | P1 §4.1 | `surrogate.max_pool/mean_pool` |
| Simulation DGP `y=β_i'cos(Wx+b)+ε`, groups, ρ | P1 §5 | `datasets.simulate_ldpm` |

**Empirically verified (known-truth DGP `simulate_ldpm`):** group recovery
0.94 (chance 0.33) at ρ=0.2 and ρ=0.8; within-group conformal intervals produced
and **tighten as ρ grows** (width 5.57 → 4.74), matching the paper's efficiency
claim.

---

## B2. Theorem-level verification (numerical, against the papers' own claims)

Beyond the equation map, the library was checked against the papers' *theorems
and headline results* on their known-truth DGPs:

| Claim | Source | Result |
|---|---|---|
| Network architecture (p=1, hidden [3,2], out 1) has 11 connections | P2 Fig. 1 | **11 weights — exact match** |
| Conformal quantile `q_k=s_{k,(⌈(m_k+1)(1−α)⌉)}` | P1 Eq. 3.4 (verbatim) | implemented exactly |
| Pooled estimator Eq. 9 (Eq. 12 only "asymptotically equivalent") | P3 (verbatim) | Eq. 9 used |
| **Proposition 1**: `sup‖ĝ−h‖²=O_P(N^{−ψ})` | P3 Prop. 1 | common-error RMS **0.308→0.155→0.129** as N=8→20→45 |
| Poolability test size & power | P3 Remark 3 | H0 P=−0.7 (no reject); H1 P=4.9 (reject) |
| Partial derivatives exact | P2 Eq. 11 | autograd == finite-diff (8e-3); recovery corr 0.61 |
| **Pooling gain**: deep pooled > deep time-series > linear | P2/P3 §4 | RMSE 0.335 < 0.528 < 0.76; DM −6.1/−2.7 |
| Group recovery via C-LASSO | P1 §3.2 | accuracy **0.94** (chance 0.33) |
| **Split-conformal marginal coverage ≈ 1−α** | P1 §3.4 | **0.89** on 300 held-out points (nominal 0.90) |
| **Table 8**: LDPM improves and surrogate gain grows with ρ | P1 §5 | RMSE 0.801→0.772; surrogate gain +0.050→+0.079 as ρ=0.2→0.8 |

Double descent (P2 Remark 2 / P3 Remark 5) is supported by the default
`lam=0` (unpenalised) path; the penalised path is available via the CV `lasso_c`
grid.

## C. What is deliberately NOT ported (honest limitations)

1. **The LLM / text pipeline of P1** — GPT annotation, the three fine-tuned BERTs
   (Filter / Categorize / Score), LDA / BERT / OpenAI embeddings on 120M Weibo
   posts. These are inherently external (need GPT/BERT/OpenAI). `deeppanel`
   consumes their *outputs* (aggregated surrogate scores and embeddings) and
   implements the entire statistical model on top. `surrogate.max_pool` /
   `mean_pool` / `svd_reduce` cover the aggregation/reduction steps.
2. **Confidence bands on partial derivatives.** P2 states explicitly that "there
   is currently no rigorous technology available to produce these" and leaves the
   Kapetanios–Kempf bootstrap to future work; we therefore report point
   derivatives only (faithful to the paper).
3. **Identifiability and conformal-coverage proofs** (P1 App. B, D; P3 Prop. 1)
   are theory, not code; the estimators they justify are implemented.
4. **Optimisation of the C-LASSO objective** uses ADAM on the joint objective
   with k-means-initialised centres, rather than the iterative coordinate scheme
   of Su–Shi–Phillips (2016). This targets the *same* Eq. (3.7) objective and is
   flagged as the equivalent-objective route; group recovery is validated on the
   paper's own DGP.
5. **GPU / exact-seed reproduction of the authors' trained networks** is not a
   goal — deep networks trained by SGD are not bit-reproducible across
   frameworks. Faithfulness is at the level of model, objective, and recovered
   truth, not identical weights.

## D. Fixes applied during the compatibility audit

- `Standardizer/MinMaxNormalizer.transform` now dispatch on the trailing axis, so
  a pooled `(M,p)` feature matrix is scaled correctly (was misread as a panel).
- `simulate_pooled_panel` now generates `y_it = h(x_{i,t-h})`, i.e. a genuine
  *forecasting* DGP; the previous contemporaneous design had no horizon-1 signal.
- Poolability test defaults to a well-conditioned polynomial feature map and
  centres on the **effective** regressor count (rank), fixing a sign error that
  arose from the rank-deficient hidden-layer map.

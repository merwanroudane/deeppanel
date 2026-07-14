# deeppanel

**Deep neural network estimation, forecasting and inference in panel data models.**

`deeppanel` is a faithful Python implementation of three recent papers on deep
learning for panel data:

| Estimator | Paper |
|---|---|
| `DeepPooledPanel` (+ poolability test, partial derivatives, benchmarks) | Chronopoulos, Chrysikou, Kapetanios, Mitchell & Raftapostolos, *Deep Neural Network Estimation in Panel Data Models*, FRB Cleveland WP 23-15 (2023); *Forecasting with Deep Pooled Panel Neural Networks*, **Econometric Reviews** (2026). |
| `LDPM` (surrogate augmentation, homogeneity pursuit, conformal intervals) | Gao, Sun, Wang, Liu & Hsiao, *How Does LLM Help Regional CPI Forecast: An LLM-powered Deep Panel Modeling Framework* (2026). |

Every estimator is checked equation-by-equation against its source and validated
on the papers' own known-truth simulation designs — see
[`docs/PAPER_COMPATIBILITY.md`](docs/PAPER_COMPATIBILITY.md).

Author: **Dr Merwan Roudane**. Licensed MIT.

---

## Installation

```bash
pip install deeppanel            # from PyPI (once published)
# or, from a local clone:
pip install -e .
```

Requires Python ≥ 3.9 and `numpy`, `pandas`, `scipy`, `torch`, `matplotlib`.
Optional extras (`statsmodels`, `scikit-learn`) install with `pip install deeppanel[extras]`.

---

## Quickstart

### 1. Deep pooled panel forecasting (Chronopoulos et al.)

```python
import numpy as np
from deeppanel import DeepPooledPanel, PanelData, TrainConfig

# y: (N, T) outcome; X: (N, T, p) regressors -- a balanced panel
panel = PanelData(y, X)                      # or PanelData.from_long(df, "id","t","y",["x1","x2"])

model = DeepPooledPanel(
    horizon=1,                 # direct h-step forecasting: y_it ~ x_{i,t-h}
    depth=3, width=20,         # L hidden ReLU layers of width M (or pass cv=PAPER_GRID)
    idiosyncratic=False,       # add per-unit component h_i (WP Eq. 10) if True
    feature_scale="standardize",
    config=TrainConfig(max_epochs=1000, lr=0.01, batch_size=14),
    seed=0,
).fit(panel)

yhat_next = model.forecast(panel)            # one direct h-step forecast per unit
grad      = model.partial_derivatives(X_rows)  # d g / d x  (Eq. 11, "opens the black box")
test      = model.poolability_test(panel)      # is an idiosyncratic component needed?
```

### 2. LLM-powered Deep Panel Modeling (Gao et al.)

```python
from deeppanel import LDPM, surrogate_residuals

# The LLM/BERT text pipeline is external; feed its aggregated outputs here.
eps_S = surrogate_residuals(y_surrogate, x_surrogate, n_lags=1)   # Eq. 3.2 residuals

ldpm = LDPM(n_groups=3, lam=0.05, alpha=0.10, seed=0)
ldpm.fit(y, X, Z=macro, surrogate_resid=eps_S, calibrate=True)

groups              = ldpm.groups                       # recovered latent groups (C-LASSO)
yhat, lower, upper  = ldpm.forecast_interval(X_last, Z_last, eps_S_last)  # split conformal
```

---

## What's implemented

**Deep pooled panel (Papers 2 & 3)**

- `DeepPooledPanel` — pooled MSE estimator of the common component `h` (Eq. 9),
  optional idiosyncratic component `h_i` (Eq. 10); ReLU MLP with linear output
  (Eq. 16); LASSO penalty; ADAM, dropout, batch-norm, early stopping.
- `cross_validate` / `PAPER_GRID` — data-driven selection of depth `L`, width `M`,
  learning rate, and LASSO `λ = c√(log p / NT)` over the paper's grids.
- `partial_derivatives` — exact `∂g/∂x` via autograd (Eq. 11).
- `poolability_test` — nonlinear test of `h̃_i = h` (Remark 3).
- Benchmarks: `ARBenchmark`, `PanelVAR` (BIC lags, iterated), `LinearPooledPanel`
  (LPM/LPM-E), `DeepTimeSeries` (pooling switched off).
- Evaluation: `diebold_mariano`, `fluctuation_test` (Giacomini–Rossi), `rmse`/`pmse`.
- `rolling_forecast` — recursive expanding-window OOS harness producing RMSE-ratio
  tables and DM comparisons.

**LDPM (Paper 1)**

- `surrogate_residuals` — the surrogate model `G_i` and its residuals (Eq. 3.2).
- `LDPM` — surrogate-augmented target model (Eqs. 3.1–3.4) fit by Deep Panel
  Training with classifier-LASSO homogeneity pursuit (Eqs. 3.5–3.7) recovering
  latent groups, plus within-group split-conformal intervals (§3.4).
- `DeepPanelTraining` — the standalone homogeneity-pursuit estimator.
- `svd_reduce`, `max_pool`, `mean_pool` — the embedding reduction/aggregation of §4.1 / App. A.2.

**Visualization** (`deeppanel.viz`) — journal-style tables (`forecast_table`,
`coef_table`, `to_latex`) and figures (`plot_forecasts`, `plot_partial_derivatives`,
`plot_conformal_intervals`, `plot_rmse_heatmap`, `plot_group_map`), plus MATLAB
colour maps (`parula_colors`, …) with Parula as the default.

---

## Syntax reference

### `DeepPooledPanel(horizon=1, depth=3, width=20, idiosyncratic=False, demean=True, feature_scale="standardize", target_scale="none", cv=None, val_frac=0.2, config=None, first_layer_bias=False, batchnorm=False, dropout=0.0, seed=None)`

| Method | Returns |
|---|---|
| `.fit(panel)` | fits the common (+ optional idiosyncratic) network(s) |
| `.predict(X, units=None)` | predictions on the original target scale |
| `.forecast(panel)` | direct `h`-step forecast per unit, `(N,)` |
| `.partial_derivatives(X, feature=None)` | `∂g/∂x` in original units |
| `.poolability_test(panel, feature_map="poly", split=False)` | `PoolabilityResult(statistic, p_value, r2, ...)` |
| `.architecture` | the selected `{depth, width, lr, lambda, dropout}` |

### `LDPM(n_groups=3, d_h=8, depth=3, width=20, lam=0.05, alpha=0.10, feature_scale="standardize", calib_frac=0.2, warmup=200, epochs=400, lr=0.01, seed=None)`

| Method | Returns |
|---|---|
| `.fit(y, X, Z=None, surrogate_resid=None, calibrate=True)` | fits the augmented DPT model + calibrates conformal radii |
| `.forecast(X_last, Z_last=None, surrogate_resid_last=None)` | one-step point forecast per unit |
| `.forecast_interval(...)` | `(yhat, lower, upper)` with within-group conformal radii |
| `.groups` | estimated latent group labels `(N,)` |

See full docstrings (`help(DeepPooledPanel)`, `help(LDPM)`) for every argument.

---

## Examples

Runnable scripts in [`examples/`](examples/) (each writes figures to `examples/_output/`):

```bash
python examples/01_deep_pooled_forecasting.py       # RMSE table + DM vs benchmarks
python examples/02_partial_derivatives_poolability.py
python examples/03_ldpm_conformal.py                # groups + conformal intervals
```

Example 1 prints a journal-style table such as:

```
Out-of-sample forecast accuracy
--------------------------------------
Model             RMSE       Ratio
DeepPooled       0.256    0.356***
DeepTS           0.413    0.574***
PVAR             0.733       1.019
AR1              0.719       1.000
--------------------------------------
Stars: Diebold-Mariano test vs base.  *** p<.01  ** p<.05  * p<.10
```

---

## Faithfulness to the papers

`deeppanel` reproduces the papers' theorems and headline results on their own
DGPs: Proposition 1 consistency, the pooling gain (deep pooled ≻ deep
time-series ≻ linear), correct poolability-test size and power, exact partial
derivatives, C-LASSO group recovery ≈ 0.94, split-conformal coverage ≈ nominal,
and the Table-8 result that LDPM's surrogate gain grows with the target/surrogate
error correlation. Details and the equation-by-equation map are in
[`docs/PAPER_COMPATIBILITY.md`](docs/PAPER_COMPATIBILITY.md).

The **LLM/BERT text pipeline** of Paper 1 (GPT annotation, fine-tuned BERTs, LDA
/ OpenAI embeddings) is intentionally out of scope — it is inherently external.
`deeppanel` implements the full statistical model on top of its outputs.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

## Citation

If you use this package, please cite the package and the underlying papers (see
[`CITATION.cff`](CITATION.cff)).

## License

MIT © Merwan Roudane.

# deeppanel — usage & syntax cookbook

Complete, copy-paste-ready code for every workflow. Each block is a full script:
import, prepare data, fit, read the outputs. Expected array shapes are shown in
comments. All arrays are NumPy; the panel is **balanced** (every unit observed in
every period).

- [1. Install & import](#1-install--import)
- [2. Preparing your data → `PanelData`](#2-preparing-your-data--paneldata)
- [3. Deep pooled panel: fit, forecast, read output](#3-deep-pooled-panel)
- [4. Every option of `DeepPooledPanel` and `TrainConfig`](#4-every-option)
- [5. Cross-validation over the paper grids](#5-cross-validation)
- [6. Benchmarks + recursive forecast comparison + tables](#6-benchmarks--rolling-forecast)
- [7. Partial derivatives (interpretability)](#7-partial-derivatives)
- [8. Poolability test](#8-poolability-test)
- [9. Diebold–Mariano & Giacomini–Rossi tests](#9-forecast-comparison-tests)
- [10. LDPM: surrogate augmentation + groups + conformal](#10-ldpm)
- [11. Journal-style tables & figures](#11-tables--figures)
- [12. What every result object contains](#12-result-objects)

---

## 1. Install & import

```bash
pip install deeppanel
```

```python
import numpy as np
import deeppanel as dp
from deeppanel import (
    PanelData, DeepPooledPanel, LDPM, TrainConfig, CVGrid, PAPER_GRID,
    rolling_forecast, diebold_mariano, fluctuation_test, surrogate_residuals, viz,
)
from deeppanel.benchmarks import ARBenchmark, PanelVAR, LinearPooledPanel, DeepTimeSeries
```

---

## 2. Preparing your data → `PanelData`

`deeppanel` works on a balanced panel held in a `PanelData` object:
`y` is `(N, T)` and `X` is `(N, T, p)` — `N` units (countries, regions, firms…),
`T` periods, `p` regressors.

### 2a. From NumPy arrays

```python
N, T, p = 7, 80, 3
y = np.random.randn(N, T)               # outcome, shape (N, T)
X = np.random.randn(N, T, p)            # regressors, shape (N, T, p)

panel = PanelData(
    y, X,
    units=["CA", "FR", "DE", "IT", "JP", "UK", "US"],  # optional labels (len N)
    times=list(range(2005, 2005 + T)),                 # optional labels (len T)
    feature_names=["unemp", "core_cpi", "energy"],     # optional (len p)
)
print(panel)            # PanelData(N=7, T=80, p=3)
print(panel.N, panel.T, panel.p)
```

### 2b. From a long / tidy `pandas` DataFrame

Your data is usually one row per (unit, period). Convert it in one call:

```python
import pandas as pd
# df columns: country, year, inflation, unemployment, core, energy
panel = PanelData.from_long(
    df,
    unit="country",           # column identifying the unit
    time="year",              # column identifying the period
    y="inflation",            # outcome column
    x=["unemployment", "core", "energy"],   # regressor columns
)
```

The panel must be balanced (same periods for every unit). If it is not, filter or
fill it first, e.g. `df = df.dropna(subset=[...])` and keep the common years.

### 2c. From a CSV

```python
df = pd.read_csv("mydata.csv")
panel = PanelData.from_long(df, "country", "year", "inflation",
                            ["unemployment", "core", "energy"])
```

---

## 3. Deep pooled panel

Fit the deep pooled estimator and produce a **direct** `h`-step-ahead forecast
(one value per unit). "Direct" means `y_it` is related to `x_{i,t-h}` and the
fitted map is applied to the latest regressors to forecast `y_{i,T+h}`.

```python
model = DeepPooledPanel(
    horizon=4,                     # forecast 4 periods ahead
    depth=3, width=20,             # 3 hidden ReLU layers of 20 neurons
    feature_scale="standardize",   # z-score the regressors on the training window
    config=TrainConfig(max_epochs=1000, lr=0.01, batch_size=14, seed=0),
    seed=0,
)
model.fit(panel)                   # returns self, so you can chain .fit(panel)

yhat = model.forecast(panel)       # shape (N,) — y_{i,T+4} for each unit
print(dict(zip(panel.units, np.round(yhat, 3))))

# Predict for arbitrary regressor rows (original units in, original units out):
X_rows = panel.X[:, -1, :]         # (N, p) latest regressors
preds  = model.predict(X_rows, units=np.arange(panel.N))   # (N,)

# What architecture was used?
print(model.architecture)          # {'depth':3,'width':20,'lr':0.01,'lambda':0.0,'dropout':0.0}
```

Add a **unit-specific (idiosyncratic) component** on top of the common one:

```python
model = DeepPooledPanel(horizon=1, depth=3, width=20, idiosyncratic=True, seed=0)
model.fit(panel)
# predict now REQUIRES units so the right h_i is added:
preds = model.predict(panel.X[:, -1, :], units=np.arange(panel.N))
```

---

## 4. Every option

### `DeepPooledPanel(...)`

| Argument | Default | Meaning |
|---|---|---|
| `horizon` | `1` | direct forecast horizon `h`; pairs `y_it` with `x_{i,t-h}` |
| `depth` | `3` | number of hidden ReLU layers `L` |
| `width` | `20` | neurons per hidden layer `M` |
| `idiosyncratic` | `False` | also fit per-unit component `h_i` (WP Eq. 10) |
| `demean` | `True` | unit-by-unit demean `y` first (`E(y_it)=0`) |
| `feature_scale` | `"standardize"` | `"standardize"`, `"minmax"` (→[0,1]) or `"none"` for X |
| `target_scale` | `"none"` | same options, applied to `y` |
| `cv` | `None` | a `CVGrid` to choose depth/width/lr/λ by validation |
| `val_frac` | `0.2` | fraction of newest periods held out for early stopping / CV |
| `config` | `TrainConfig()` | training hyper-parameters (below) |
| `first_layer_bias` | `False` | keep `b⁽¹⁾=0` as in the paper |
| `batchnorm` | `False` | batch normalisation in hidden layers |
| `dropout` | `0.0` | dropout probability |
| `seed` | `None` | reproducibility |

### `TrainConfig(...)`

| Argument | Default | Meaning |
|---|---|---|
| `lr` | `0.001` | learning rate γ (paper grid {.01,.001}) |
| `lam` | `0.0` | LASSO penalty on `‖θ‖₁` (0 = none; paper's "double descent") |
| `batch_size` | `14` | mini-batch size (paper fixes 14) |
| `max_epochs` | `5000` | epoch budget |
| `patience` | `50` | early-stopping patience |
| `optimizer` | `"adam"` | `"adam"` or `"sgd"` |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `seed` | `None` | reproducibility |

```python
cfg = TrainConfig(lr=0.01, lam=0.0, batch_size=14, max_epochs=2000, patience=60, seed=0)
model = DeepPooledPanel(horizon=1, depth=5, width=30, config=cfg, seed=0).fit(panel)
```

---

## 5. Cross-validation

Let the data choose depth `L`, width `M`, learning rate, LASSO `c`, dropout —
exactly the paper's grids — instead of fixing them:

```python
from deeppanel import PAPER_GRID          # full 350-combination paper grid
# or a custom, smaller grid:
grid = CVGrid(depth=(1, 3, 5), width=(10, 20, 30), lr=(0.01, 0.001),
              lasso_c=(0.0, 0.1, 1.0), dropout=(0.0,))

model = DeepPooledPanel(horizon=1, cv=grid,
                        config=TrainConfig(max_epochs=800, seed=0), seed=0)
model.fit(panel)
print("selected:", model.architecture)   # winning depth/width/lr/lambda/dropout
```

---

## 6. Benchmarks & rolling forecast

Compare the deep pooled model against the paper's benchmarks over an expanding
(recursive) out-of-sample window, then read RMSE ratios and DM tests.

```python
cfg = TrainConfig(max_epochs=500, lr=0.01, batch_size=14, seed=0)

# Each entry is a *factory*: a zero-arg function returning a fresh estimator,
# re-created at every origin so no information leaks across windows.
models = {
    "DeepPooled": lambda: DeepPooledPanel(horizon=1, depth=3, width=20, config=cfg, seed=0),
    "DeepTS":     lambda: DeepTimeSeries(horizon=1, depth=3, width=20, config=cfg, seed=0),
    "PVAR":       lambda: PanelVAR(max_lags=4),
    "AR1":        lambda: ARBenchmark(1),
}

res = rolling_forecast(panel, models, horizon=1, start_frac=0.7, step=1)

# Pooled RMSE per model:
for name in models:
    print(f"{name:12s} RMSE = {res.rmse(name):.4f}")

# Per-unit RMSE ratios relative to a benchmark (DataFrame, units × models):
print(res.rmse_table(relative_to="AR1").round(3))

# Diebold–Mariano test of each model vs AR1 (H1: model more accurate):
print(res.dm_table(base="AR1").round(3))
```

Run a single benchmark on its own:

```python
ar = ARBenchmark(order=1).fit(panel)
print(ar.forecast(steps=1))               # (N,) iterated 1-step forecast

pvar = PanelVAR(max_lags=4).fit(panel)    # BIC picks the lag order
print(pvar.forecast(steps=4))             # (N,) iterated 4-step forecast

lpm = LinearPooledPanel(horizon=1).fit(panel)
print(lpm.forecast(panel))                # (N,) direct linear forecast
```

---

## 7. Partial derivatives

"Open the black box": the marginal effect `∂g/∂x` of each regressor, evaluated at
any set of points, in the original units of your data.

```python
model = DeepPooledPanel(horizon=1, depth=3, width=24,
                        config=TrainConfig(max_epochs=800, seed=0), seed=0).fit(panel)

X_eval = panel.X[:, -1, :]                 # (N, p) evaluate at the latest period
d = model.partial_derivatives(X_eval)      # (N, p): d yhat / d x_j for every unit
print(d)

# Just one regressor (e.g. feature 0 = unemployment):
d_unemp = model.partial_derivatives(X_eval, feature=0)   # (N,)

# Track the effect over time: derivative w.r.t. feature 0 at each period, per unit
D = np.stack([model.partial_derivatives(panel.X[:, t, :], feature=0)
              for t in range(panel.T)], axis=1)          # (N, T)
```

---

## 8. Poolability test

Is a common function enough, or do you need unit-specific components? (Nonlinear
poolability test, WP Remark 3.)

```python
model = DeepPooledPanel(horizon=1, depth=3, width=24,
                        config=TrainConfig(max_epochs=800, seed=0), seed=0).fit(panel)

pt = model.poolability_test(panel)         # feature_map="poly" by default
print(f"P = {pt.statistic:.2f},  p-value = {pt.p_value:.3f}")
if pt.p_value < 0.05:
    print("Reject pooling → an idiosyncratic component is warranted.")
else:
    print("Do not reject → the common (pooled) model suffices.")
print("per-unit R^2:", np.round(pt.r2, 3))
```

---

## 9. Forecast-comparison tests

Use these directly on any two forecast-error series (`error = actual − forecast`).

```python
e_model = res.errors("DeepPooled").ravel()   # from a RollingResult, or your own
e_base  = res.errors("AR1").ravel()

# Diebold–Mariano (H1: model 1 more accurate):
dm = diebold_mariano(e_model, e_base, horizon=1, alternative="less")
print(f"DM = {dm.statistic:.3f},  p = {dm.p_value:.3f}")

# Giacomini–Rossi fluctuation test (WHEN does one model dominate?):
fl = fluctuation_test(e_model, e_base, window=0.3, horizon=1, alpha=0.05)
print(f"max|F| = {fl.statistic:.2f}, crit = {fl.crit_value:.2f}, reject = {fl.reject}")
# fl.path / fl.centers give the time-varying statistic for plotting
```

---

## 10. LDPM

The LLM/BERT text pipeline is external; you supply the aggregated surrogate
outcome and embeddings. Then:

```python
from deeppanel import LDPM, surrogate_residuals

# y : (N, T) target (e.g. regional CPI)
# X : (N, T, dx) aggregated text embeddings
# Z : (N, T, dz) macro covariates (optional)
# y_surr : (N, T) aggregated surrogate outcome (e.g. sentiment score)
# x_surr : (N, T, ds) surrogate predictors (optional)

# Step 1 — surrogate residuals (Eq. 3.2):
eps_S = surrogate_residuals(y_surr, x_surr, n_lags=1)     # (N, T)

# Step 2 — surrogate-augmented Deep Panel Training + conformal calibration:
ldpm = LDPM(
    n_groups=3,        # number of latent groups K0
    d_h=8,             # hidden representation dimension
    depth=3, width=20, # shared feature map size
    lam=0.05,          # homogeneity-pursuit (classifier-LASSO) penalty
    alpha=0.10,        # 1-alpha = 90% conformal coverage
    calib_frac=0.2,    # last 20% of periods reserved for calibration
    seed=0,
)
ldpm.fit(y[:, :-1], X[:, :-1], Z=Z[:, :-1], surrogate_resid=eps_S[:, :-1], calibrate=True)

print("estimated groups:", ldpm.groups)                  # (N,) latent group labels

# One-step-ahead point forecast + conformal interval, per unit:
yhat, lower, upper = ldpm.forecast_interval(
    X[:, -1, :], Z_last=Z[:, -1, :], surrogate_resid_last=eps_S[:, -1]
)
for i in range(len(yhat)):
    print(f"unit {i}: {yhat[i]:.2f}  [{lower[i]:.2f}, {upper[i]:.2f}]")
```

Without a surrogate (`surrogate_resid=None`) LDPM reduces to Deep Panel Training
with homogeneity pursuit on `[x, z]` alone.

Embedding helpers for the linear baselines:

```python
from deeppanel.surrogate import svd_reduce, max_pool, mean_pool
XV, V = svd_reduce(embeddings_2d, r0=10)   # rank-10 SVD reduction (App. A.2)
day_vec  = max_pool([post_vec1, post_vec2, ...])   # embeddings → region-day vector
day_score = mean_pool([0.9, 0.2, 0.5])             # surrogate scores → region-day
```

---

## 11. Tables & figures

Journal-style outputs in `deeppanel.viz`.

```python
from deeppanel import viz

# --- console/Markdown table with significance stars ---
rmse   = {n: res.rmse(n) for n in models}
ratios = {n: rmse[n] / rmse["AR1"] for n in models}
pvals  = res.dm_table(base="AR1").set_index("model")["p_value"].to_dict()
print(viz.forecast_table(rmse, ratios=ratios, pvalues=pvals, base="AR1"))

# --- LaTeX (booktabs) for a paper ---
tex = viz.to_latex(res.rmse_table(relative_to="AR1"),
                   caption="RMSE ratios to AR(1)", label="tab:rmse")

# --- figures (matplotlib) ---
import matplotlib.pyplot as plt
viz.set_style()                                   # clean top-journal style
viz.plot_forecasts(res.actuals[0], res.forecasts["DeepPooled"][0], times=res.origins)
viz.plot_rmse_heatmap(res.rmse_table(relative_to="AR1"))     # Parula colormap
viz.plot_conformal_intervals(yhat, lower, upper, actual=y[:, -1])
viz.plot_group_map(ldpm.groups)
plt.show()   # or plt.savefig("figure.png", dpi=300)

# MATLAB Parula palette for your own plots:
colors = viz.parula_colors(5)     # 5 hex colours
```

---

## 12. Result objects

| Object | Key attributes / methods |
|---|---|
| `PanelData` | `.N .T .p .y .X .units .times`, `.demean_y()`, `.direct_pairs(h)`, `.last_features()`, `.slice_time(a,b)` |
| `DeepPooledPanel` (fitted) | `.forecast(panel)`, `.predict(X, units)`, `.partial_derivatives(X, feature)`, `.poolability_test(panel)`, `.architecture` |
| `RollingResult` | `.forecasts` (dict), `.actuals`, `.origins`, `.rmse(name)`, `.rmse_by_unit(name)`, `.rmse_table(relative_to)`, `.dm_table(base)`, `.errors(name)` |
| `PoolabilityResult` | `.statistic`, `.p_value`, `.r2`, `.m`, `.n_units` |
| `DMResult` | `.statistic`, `.p_value`, `.mean_loss_diff`, `.horizon`, `.n` |
| `FluctuationResult` | `.statistic`, `.crit_value`, `.reject`, `.path`, `.centers`, `.mu`, `.window` |
| `LDPM` (fitted) | `.groups`, `.forecast(...)`, `.forecast_interval(...)`, `.conformal` |
| `ConformalCalibration` | `.radius(group)`, `.interval(yhat, groups)`, `.radii`, `.counts` |

For any object, `help(obj)` prints the full docstring with every argument.

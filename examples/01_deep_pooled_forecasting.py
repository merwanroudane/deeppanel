"""Example 1 -- Deep pooled panel forecasting vs benchmarks.

Reproduces the core exercise of Chronopoulos et al. (2023, 2026): forecast a
nonlinear panel with the deep pooled estimator and compare it, by recursive
out-of-sample RMSE and Diebold-Mariano tests, against a deep time-series model
(pooling switched off), a panel VAR, and an AR(1).

Run::

    python examples/01_deep_pooled_forecasting.py

Outputs a journal-style RMSE-ratio table to the console and saves two figures to
``examples/_output/``.
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from deeppanel import DeepPooledPanel, TrainConfig, rolling_forecast, viz
from deeppanel.benchmarks import DeepTimeSeries, PanelVAR, ARBenchmark
from deeppanel.datasets import simulate_pooled_panel

OUT = os.path.join(os.path.dirname(__file__), "_output")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    # A nonlinear forecasting panel: y_it = h(x_{i,t-1}) + h_i(x_{i,t-1}) + eps
    sim = simulate_pooled_panel(N=14, T=60, p=3, idio_scale=0.05,
                                noise_sd=0.2, horizon=1, seed=42)
    panel = sim.panel

    cfg = TrainConfig(max_epochs=300, patience=40, lr=0.01, batch_size=64, seed=0)
    models = {
        "DeepPooled": lambda: DeepPooledPanel(horizon=1, depth=3, width=24, config=cfg, seed=0),
        "DeepTS":     lambda: DeepTimeSeries(horizon=1, depth=3, width=24, config=cfg, seed=0),
        "PVAR":       lambda: PanelVAR(max_lags=2),
        "AR1":        lambda: ARBenchmark(1),
    }

    print("Running recursive out-of-sample forecasts ...")
    res = rolling_forecast(panel, models, horizon=1, start_frac=0.75, step=2)

    # ---- journal-style RMSE table (ratios to AR1) + DM stars -----------------
    rmse = {n: res.rmse(n) for n in models}
    ratios = {n: rmse[n] / rmse["AR1"] for n in models}
    dm = res.dm_table(base="AR1").set_index("model")["p_value"].to_dict()
    print()
    print(viz.forecast_table(rmse, ratios=ratios, pvalues=dm, base="AR1"))

    # ---- figure 1: forecasts vs outturns for one unit ------------------------
    unit = 0
    fig, ax = plt.subplots(figsize=(8, 4.2))
    viz.plot_forecasts(res.actuals[unit], res.forecasts["DeepPooled"][unit],
                       times=res.origins, label="Deep pooled", ax=ax,
                       title=f"Deep pooled forecasts vs outturns (unit {unit})")
    ax.plot(res.origins, res.forecasts["AR1"][unit], color="#999999", ls=":",
            label="AR(1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_forecasts.png"), dpi=150)

    # ---- figure 2: RMSE-ratio heatmap (units x models) -----------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    viz.plot_rmse_heatmap(res.rmse_table(relative_to="AR1"),
                          title="RMSE ratios to AR(1)", ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_rmse_heatmap.png"), dpi=150)

    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    main()

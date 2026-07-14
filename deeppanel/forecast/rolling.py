"""Recursive (expanding-window) out-of-sample forecasting harness.

Reproduces the forecasting design of Chronopoulos et al. (2023, 2026): at each
origin the models are re-estimated on an expanding window and used to produce a
direct ``h``-step-ahead forecast; accuracy is then summarised by RMSE ratios and
Diebold-Mariano tests.

Any estimator with ``fit(panel)`` plus a forecast method is accepted:

* direct models -- :class:`~deeppanel.DeepPooledPanel`,
  :class:`~deeppanel.benchmarks.LinearPooledPanel`,
  :class:`~deeppanel.benchmarks.DeepTimeSeries` -- expose ``forecast(panel)`` and
  carry the horizon internally;
* iterated models -- :class:`~deeppanel.benchmarks.ARBenchmark`,
  :class:`~deeppanel.benchmarks.PanelVAR` -- expose ``forecast(steps)``.

The harness dispatches to whichever signature the estimator provides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..evaluation.dm import diebold_mariano
from ..evaluation.metrics import rmse
from ..utils.panel import PanelData

__all__ = ["RollingResult", "rolling_forecast"]


def _forecast(est, window: PanelData, horizon: int) -> np.ndarray:
    """Call the estimator's forecast method, whichever signature it uses."""
    try:
        return np.asarray(est.forecast(window), dtype=float)     # direct models
    except TypeError:
        return np.asarray(est.forecast(horizon), dtype=float)    # iterated models


@dataclass
class RollingResult:
    forecasts: Dict[str, np.ndarray]           # name -> (N, n_origins)
    actuals: np.ndarray                        # (N, n_origins)
    origins: List[int]                         # target period index of each column
    horizon: int
    units: Optional[List] = None
    _errors: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def errors(self, name: str) -> np.ndarray:
        return self.actuals - self.forecasts[name]

    def rmse(self, name: str) -> float:
        """Pooled RMSE over all units and origins."""
        return rmse(self.actuals, self.forecasts[name])

    def rmse_by_unit(self, name: str) -> np.ndarray:
        e = self.errors(name)
        return np.sqrt(np.nanmean(e ** 2, axis=1))

    def rmse_table(self, relative_to: Optional[str] = None):
        """Per-unit RMSE (or RMSE ratio to a base model) as a DataFrame."""
        import pandas as pd
        data = {}
        base = self.rmse_by_unit(relative_to) if relative_to else None
        for name in self.forecasts:
            col = self.rmse_by_unit(name)
            data[name] = col / base if relative_to else col
        idx = self.units if self.units is not None else list(range(self.actuals.shape[0]))
        return pd.DataFrame(data, index=idx)

    def dm_table(self, base: str, alternative: str = "less"):
        """Pooled DM test of each model vs ``base`` (H1: model more accurate)."""
        import pandas as pd
        rows = []
        eb = self.errors(base).ravel()
        for name in self.forecasts:
            if name == base:
                continue
            em = self.errors(name).ravel()
            try:
                dm = diebold_mariano(em, eb, horizon=self.horizon, alternative=alternative)
                rows.append({"model": name, "vs": base, "DM": dm.statistic,
                             "p_value": dm.p_value, "mean_dloss": dm.mean_loss_diff})
            except ValueError:
                rows.append({"model": name, "vs": base, "DM": np.nan, "p_value": np.nan,
                             "mean_dloss": np.nan})
        return pd.DataFrame(rows)


def rolling_forecast(
    panel: PanelData,
    models: Dict[str, Callable[[], object]],
    horizon: int = 1,
    start_frac: float = 0.7,
    step: int = 1,
    min_train: Optional[int] = None,
    verbose: bool = False,
) -> RollingResult:
    """Run the recursive expanding-window forecast comparison.

    Parameters
    ----------
    panel : PanelData
    models : dict[str, callable]
        Maps a model name to a zero-argument factory returning a *fresh* estimator
        (re-created each origin so no information leaks across windows).
    horizon : int, default 1
        Forecast horizon ``h``.
    start_frac : float, default 0.7
        First origin uses ``round(start_frac * T)`` periods of training data.
    step : int, default 1
        Advance the origin by this many periods each iteration (use >1 to speed
        up expensive deep models, as the papers do for COVID-19).
    min_train : int, optional
        Override the initial training length.

    Returns
    -------
    RollingResult
    """
    T = panel.T
    L0 = min_train if min_train is not None else max(3, int(round(start_frac * T)))
    origins_t0 = list(range(L0, T - horizon + 1, step))
    if not origins_t0:
        raise ValueError("No valid forecast origins; reduce start_frac/horizon or add periods.")

    forecasts: Dict[str, List[np.ndarray]] = {name: [] for name in models}
    actual_cols: List[np.ndarray] = []
    target_periods: List[int] = []

    for t0 in origins_t0:
        window = panel.slice_time(0, t0)
        target_t = t0 - 1 + horizon
        actual_cols.append(panel.y[:, target_t].copy())
        target_periods.append(target_t)
        for name, factory in models.items():
            est = factory()
            est.fit(window)
            forecasts[name].append(_forecast(est, window, horizon))
        if verbose:
            print(f"origin t0={t0} -> target period {target_t}")

    fc = {name: np.column_stack(cols) for name, cols in forecasts.items()}
    return RollingResult(
        forecasts=fc, actuals=np.column_stack(actual_cols), origins=target_periods,
        horizon=horizon, units=panel.units,
    )

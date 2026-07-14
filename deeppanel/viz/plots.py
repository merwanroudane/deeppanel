"""Journal-style figures for the deep-panel models."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .style import parula_colors, set_style

__all__ = [
    "plot_forecasts", "plot_partial_derivatives", "plot_conformal_intervals",
    "plot_rmse_heatmap", "plot_group_map",
]


def _ax(ax):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    return ax


def plot_forecasts(actual, forecast, times=None, label="Deep pooled",
                   title="Forecasts vs outturns", ax=None):
    """Overlay a forecast path on the realised outturns for one unit."""
    set_style()
    ax = _ax(ax)
    actual = np.asarray(actual, float)
    forecast = np.asarray(forecast, float)
    x = np.arange(actual.size) if times is None else times
    cols = parula_colors(3)
    ax.plot(x, actual, color="#111111", label="Actual", lw=1.8)
    ax.plot(x, forecast, color=cols[0], ls="--", marker="o", ms=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("y")
    ax.legend()
    return ax


def plot_partial_derivatives(dates, deriv_by_unit, unit_labels=None,
                             title="Partial derivatives over time",
                             ylabel=r"$\partial g / \partial x$", ax=None):
    """Plot time-varying partial derivatives, one line per unit (WP Eq. 11)."""
    set_style()
    ax = _ax(ax)
    D = np.asarray(deriv_by_unit, float)     # (n_units, n_time)
    n_units = D.shape[0]
    cols = parula_colors(max(n_units, 2))
    x = np.arange(D.shape[1]) if dates is None else dates
    for i in range(n_units):
        lab = None if unit_labels is None else unit_labels[i]
        ax.plot(x, D[i], color=cols[i], label=lab, lw=1.4)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    if unit_labels is not None and n_units <= 12:
        ax.legend(ncol=2, fontsize=8)
    return ax


def plot_conformal_intervals(yhat, lower, upper, actual=None, labels=None,
                             title="Conformal prediction intervals", ax=None):
    """Point forecasts with conformal bands across units (Gao et al. Fig. 4)."""
    set_style()
    ax = _ax(ax)
    yhat = np.asarray(yhat, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    x = np.arange(yhat.size)
    cols = parula_colors(3)
    ax.errorbar(x, yhat, yerr=[yhat - lower, upper - yhat], fmt="o", ms=4,
                color=cols[0], ecolor=cols[1], elinewidth=1.4, capsize=3,
                label="Forecast +/- conformal")
    if actual is not None:
        ax.scatter(x, np.asarray(actual, float), color="#b2182b", marker="x",
                   s=30, zorder=5, label="Actual")
    ax.set_title(title)
    ax.set_ylabel("y")
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend()
    return ax


def plot_rmse_heatmap(table, title="RMSE ratios", cmap="parula", ax=None):
    """Heatmap of an RMSE(-ratio) table (rows=units, cols=models)."""
    import matplotlib.pyplot as plt
    from .style import get_cmap
    set_style()
    ax = _ax(ax)
    import pandas as pd
    if isinstance(table, pd.DataFrame):
        data = table.values
        rows = list(table.index)
        cols = list(table.columns)
    else:
        data = np.asarray(table, float)
        rows = list(range(data.shape[0]))
        cols = list(range(data.shape[1]))
    im = ax.imshow(data, aspect="auto", cmap=get_cmap(cmap))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_group_map(groups, unit_labels=None, title="Estimated latent groups", ax=None):
    """Bar chart colouring each unit by its estimated group (LDPM/DPT)."""
    set_style()
    ax = _ax(ax)
    groups = np.asarray(groups, int)
    K = int(groups.max()) + 1 if groups.size else 1
    cols = parula_colors(max(K, 2))
    x = np.arange(groups.size)
    ax.bar(x, np.ones_like(groups), color=[cols[g] for g in groups])
    ax.set_yticks([])
    ax.set_title(title)
    if unit_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(unit_labels, rotation=90, fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[k], label=f"Group {k}") for k in range(K)],
              ncol=min(K, 4), fontsize=8)
    return ax

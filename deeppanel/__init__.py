"""deeppanel -- deep neural network estimation, forecasting and inference in
panel data models.

Faithful Python implementations of

* **Deep pooled panel networks** -- Chronopoulos, Chrysikou, Kapetanios, Mitchell
  and Raftapostolos, *Deep Neural Network Estimation in Panel Data Models*
  (FRB Cleveland WP 23-15, 2023) and *Forecasting with Deep Pooled Panel Neural
  Networks* (Econometric Reviews, 2026): :class:`DeepPooledPanel`, the poolability
  test, partial derivatives, and the PVAR/AR/deep-time-series benchmarks.

* **LLM-powered Deep Panel Modeling (LDPM)** -- Gao, Sun, Wang, Liu and Hsiao,
  *How Does LLM Help Regional CPI Forecast* (2026): :class:`LDPM`, Deep Panel
  Training with classifier-LASSO homogeneity pursuit, and within-group split
  conformal prediction intervals.

Author: Dr Merwan Roudane.
"""
from __future__ import annotations

from .utils.panel import PanelData
from .models.deep_pooled import DeepPooledPanel
from .models.ldpm import LDPM
from .models.classo import DeepPanelTraining
from .models.mlp import MLP
from .training.trainer import TrainConfig
from .training.cv import CVGrid, PAPER_GRID, DEFAULT_GRID, cross_validate
from .forecast.rolling import rolling_forecast, RollingResult
from .inference.partial import partial_derivatives
from .inference.poolability import poolability_test
from .inference.conformal import conformal_radii, coverage
from .evaluation.metrics import rmse, mse, pmse, mae
from .evaluation.dm import diebold_mariano
from .evaluation.fluctuation import fluctuation_test
from .surrogate.augment import surrogate_residuals

from . import benchmarks
from . import datasets
from . import viz

__version__ = "0.1.1"
__author__ = "Merwan Roudane"

__all__ = [
    "PanelData",
    "DeepPooledPanel",
    "LDPM",
    "DeepPanelTraining",
    "MLP",
    "TrainConfig",
    "CVGrid",
    "PAPER_GRID",
    "DEFAULT_GRID",
    "cross_validate",
    "rolling_forecast",
    "RollingResult",
    "partial_derivatives",
    "poolability_test",
    "conformal_radii",
    "coverage",
    "rmse",
    "mse",
    "pmse",
    "mae",
    "diebold_mariano",
    "fluctuation_test",
    "surrogate_residuals",
    "benchmarks",
    "datasets",
    "viz",
    "__version__",
]

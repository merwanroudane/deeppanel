"""Linear pooled panel (LPM / LPM-E) and deep *time-series* benchmarks.

* :class:`LinearPooledPanel` -- the linear pooled panel model (LPM) of Hsiao
  (2022) used as a baseline in Gao et al. (2026) and, with embeddings appended
  to the regressors, the LPM-E variant.  Direct ``h``-step pooled OLS of ``y_it``
  on ``x_{i,t-h}`` with optional unit fixed effects.
* :class:`DeepTimeSeries` -- the deep *time-series* benchmark of Chronopoulos et
  al. (2023, 2026): the same feed-forward network estimated *separately for each
  unit*, i.e. pooling switched off.  Comparing it with :class:`DeepPooledPanel`
  isolates the value of pooling cross-sectional information.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..models.mlp import MLP
from ..training.normalize import identity, MinMaxNormalizer, Standardizer
from ..training.trainer import TrainConfig, predict as _predict, train_mlp
from ..utils.panel import PanelData

__all__ = ["LinearPooledPanel", "DeepTimeSeries"]


def _scaler(kind):
    if kind in (None, "none"):
        return identity()
    if kind in ("standardize", "standard"):
        return Standardizer()
    if kind in ("minmax", "rank"):
        return MinMaxNormalizer()
    raise ValueError(kind)


class LinearPooledPanel:
    """Direct ``h``-step linear pooled panel (LPM / LPM-E)."""

    def __init__(self, horizon: int = 1, unit_effects: bool = True, demean: bool = True) -> None:
        self.horizon = int(horizon)
        self.unit_effects = bool(unit_effects)
        self.demean = bool(demean)
        self._beta = None
        self._mu = None
        self._N = None

    def fit(self, panel: PanelData) -> "LinearPooledPanel":
        work = panel.demean_y() if self.demean else panel
        self._means = work.unit_means
        feats, targ, uidx, _ = work.direct_pairs(self.horizon)
        N = panel.N
        if self.unit_effects:
            D = np.zeros((len(uidx), N))
            D[np.arange(len(uidx)), uidx] = 1.0
            design = np.column_stack([D, feats])
            beta, *_ = np.linalg.lstsq(design, targ, rcond=None)
            self._mu = beta[:N]
            self._beta = beta[N:]
        else:
            design = np.column_stack([np.ones(len(uidx)), feats])
            beta, *_ = np.linalg.lstsq(design, targ, rcond=None)
            self._mu = np.full(N, beta[0])
            self._beta = beta[1:]
        self._N = N
        return self

    def forecast(self, panel: PanelData) -> np.ndarray:
        if self._beta is None:
            raise RuntimeError("Call fit() first.")
        XT = panel.last_features()
        yhat = self._mu + XT @ self._beta
        if self.demean and self._means is not None:
            yhat = yhat + self._means
        return yhat


class DeepTimeSeries:
    """Deep time-series benchmark: one network per unit (no pooling)."""

    def __init__(self, horizon: int = 1, depth: int = 3, width: int = 20,
                 demean: bool = True, feature_scale: str = "standardize",
                 config: Optional[TrainConfig] = None, seed: Optional[int] = None) -> None:
        self.horizon = int(horizon)
        self.depth = int(depth)
        self.width = int(width)
        self.demean = bool(demean)
        self.feature_scale = feature_scale
        self.config = config or TrainConfig(seed=seed)
        self.seed = seed
        self._models: List[Optional[MLP]] = []
        self._scalers: List[object] = []
        self._means = None

    def fit(self, panel: PanelData) -> "DeepTimeSeries":
        work = panel.demean_y() if self.demean else panel
        self._means = work.unit_means
        self._models, self._scalers = [], []
        for i in range(panel.N):
            sub = PanelData(work.y[i:i + 1], work.X[i:i + 1])
            feats, targ, _, _ = sub.direct_pairs(self.horizon)
            sc = _scaler(self.feature_scale).fit(sub.X)
            Xs = sc.transform(feats)
            if len(targ) < 3:
                self._models.append(None)
                self._scalers.append(sc)
                continue
            model = MLP(p=panel.p, depth=self.depth, width=self.width)
            train_mlp(model, Xs, targ, None, None, self.config)
            self._models.append(model)
            self._scalers.append(sc)
        return self

    def forecast(self, panel: PanelData) -> np.ndarray:
        XT = panel.last_features()
        out = np.empty(panel.N)
        for i in range(panel.N):
            model, sc = self._models[i], self._scalers[i]
            if model is None:
                out[i] = 0.0
            else:
                out[i] = float(_predict(model, sc.transform(XT[i:i + 1]))[0])
            if self.demean and self._means is not None:
                out[i] += self._means[i]
        return out

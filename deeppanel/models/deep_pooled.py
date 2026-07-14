"""Deep pooled panel estimator (Chronopoulos, Chrysikou, Kapetanios, Mitchell &
Raftapostolos, 2023 FRB Cleveland WP 23-15 and 2026 Econometric Reviews).

The model is the nonlinear panel conditional mean

    E(y_it | x_it) = h~_i(x_it) = h(x_it) + h_i(x_it),                    (WP Eq. 3)

a *common* nonlinear component ``h`` shared by all units plus an optional
*idiosyncratic* component ``h_i``.  Both are feed-forward ReLU networks.

Estimation (two steps, WP Eqs. 9-10)
------------------------------------
1. **Common / pooled step.**  Fit one network ``g(x; theta_hat)`` by pooled MSE
   over *all* ``(i, t)`` pairs (Eq. 9).  This is the deep pooled estimator of
   ``h``; because the objective pools cross-sectional information it is the
   estimator whose forecasting gains the papers document.
2. **Idiosyncratic step (optional).**  For each unit ``i`` fit a second network
   ``g(x; theta_hat_i)`` on that unit's residuals ``y_it - g(x_it; theta_hat)``
   (Eq. 10).  Switched off (``idiosyncratic=False``) this reduces to the "deep
   pooled" model of the 2026 paper; switched on it is the common+idiosyncratic
   model of WP 23-15.

Forecasts are **direct** (2026, footnote 5): the ``h``-step target ``y_it`` is
regressed on ``x_{i,t-h}`` and the fitted map is applied to the known time-``T``
features to obtain ``y_{i,T+h}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..training.cv import CVGrid, cross_validate
from ..training.normalize import MinMaxNormalizer, Standardizer, identity
from ..training.trainer import TrainConfig, predict as _predict, train_mlp
from ..utils.panel import PanelData
from .mlp import MLP

__all__ = ["DeepPooledPanel"]


def _make_scaler(kind: Optional[str]):
    if kind in (None, "none", "identity"):
        return identity()
    if kind in ("standardize", "standard", "zscore"):
        return Standardizer()
    if kind in ("minmax", "rank", "01"):
        return MinMaxNormalizer()
    raise ValueError(f"Unknown scaler {kind!r}.")


@dataclass
class _Fitted:
    common: MLP
    idio: Optional[List[Optional[MLP]]]
    feat_scaler: object
    targ_scaler: object
    unit_means: Optional[np.ndarray]
    arch: Dict
    N: int
    T: int
    p: int
    horizon: int


class DeepPooledPanel:
    """Deep pooled panel estimator with optional idiosyncratic component.

    Parameters
    ----------
    horizon : int, default 1
        Direct forecast horizon ``h``.  ``fit`` pairs ``y_it`` with ``x_{i,t-h}``.
    depth, width : int
        Network depth ``L`` and width ``M``.  Ignored when ``cv`` is given.
    idiosyncratic : bool, default False
        Whether to add the per-unit component ``h_i`` (WP Eq. 10).
    demean : bool, default True
        Unit-by-unit demean the target first (the ``E(y_it)=0`` normalisation).
    feature_scale : {'standardize','minmax','none'}, default 'standardize'
        Transform applied to the regressors (fit on the training window).
    target_scale : {'none','standardize','minmax'}, default 'none'
        Optional transform applied to the (demeaned) target.
    cv : CVGrid, optional
        If given, the architecture/regularisation is chosen by cross-validation
        instead of the fixed ``depth``/``width``.
    val_frac : float, default 0.2
        Fraction of the (time-ordered) training pairs held out for early
        stopping / CV, matching the paper's 0.8/0.2 train/validation split.
    config : TrainConfig, optional
    first_layer_bias, batchnorm, dropout : network options.
    seed : int, optional
    """

    def __init__(
        self,
        horizon: int = 1,
        depth: int = 3,
        width: int = 20,
        idiosyncratic: bool = False,
        demean: bool = True,
        feature_scale: str = "standardize",
        target_scale: str = "none",
        cv: Optional[CVGrid] = None,
        val_frac: float = 0.2,
        config: Optional[TrainConfig] = None,
        first_layer_bias: bool = False,
        batchnorm: bool = False,
        dropout: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.horizon = int(horizon)
        self.depth = int(depth)
        self.width = int(width)
        self.idiosyncratic = bool(idiosyncratic)
        self.demean = bool(demean)
        self.feature_scale = feature_scale
        self.target_scale = target_scale
        self.cv = cv
        self.val_frac = float(val_frac)
        self.config = config or TrainConfig(seed=seed)
        self.first_layer_bias = bool(first_layer_bias)
        self.batchnorm = bool(batchnorm)
        self.dropout = float(dropout)
        self.seed = seed
        self._fit: Optional[_Fitted] = None

    # ------------------------------------------------------------------ fit
    def _time_split(self, tidx: np.ndarray):
        """Split pooled rows into train/validation by *time* (no look-ahead)."""
        order = np.unique(tidx)
        n_val = max(1, int(round(self.val_frac * len(order))))
        val_times = set(order[-n_val:])
        is_val = np.array([t in val_times for t in tidx])
        return ~is_val, is_val

    def fit(self, panel: PanelData) -> "DeepPooledPanel":
        """Estimate the common (and optional idiosyncratic) network(s)."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        work = panel.demean_y() if self.demean else panel
        unit_means = work.unit_means

        # direct h-step pairs on the (possibly demeaned) panel
        feats, targ, uidx, tidx = work.direct_pairs(self.horizon)

        # scalers fit on training features/targets only
        feat_scaler = _make_scaler(self.feature_scale).fit(work.X)
        targ_scaler = _make_scaler(self.target_scale).fit(work.y[:, :, None])
        Xs = feat_scaler.transform(feats)
        ys = targ_scaler.transform(targ[:, None]).ravel()

        tr_mask, va_mask = self._time_split(tidx)
        p = panel.p
        nt = panel.N * panel.T

        # -- architecture: CV or fixed --------------------------------------
        if self.cv is not None:
            cvres = cross_validate(
                Xs[tr_mask], ys[tr_mask], Xs[va_mask], ys[va_mask],
                p=p, nt=nt, grid=self.cv, base_config=self.config,
                first_layer_bias=self.first_layer_bias, batchnorm=self.batchnorm,
                seed=self.seed,
            )
            arch = cvres.best
            depth, width, drop = arch["depth"], arch["width"], arch["dropout"]
            cfg = TrainConfig(
                lr=arch["lr"], lam=arch["lambda"], batch_size=self.config.batch_size,
                max_epochs=self.config.max_epochs, patience=self.config.patience,
                optimizer=self.config.optimizer, device=self.config.device, seed=self.seed,
            )
        else:
            depth, width, drop = self.depth, self.width, self.dropout
            cfg = self.config
            arch = {"depth": depth, "width": width, "lr": cfg.lr, "lambda": cfg.lam,
                    "dropout": drop}

        # -- common step (Eq. 9) --------------------------------------------
        common = MLP(p=p, depth=depth, width=width, dropout=drop,
                     batchnorm=self.batchnorm, first_layer_bias=self.first_layer_bias)
        train_mlp(common, Xs[tr_mask], ys[tr_mask], Xs[va_mask], ys[va_mask], cfg)
        # refit on the full window at the chosen architecture (no held-out block)
        train_mlp(common, Xs, ys, None, None,
                  TrainConfig(lr=cfg.lr, lam=cfg.lam, batch_size=cfg.batch_size,
                              max_epochs=min(cfg.max_epochs, 1500), optimizer=cfg.optimizer,
                              device=cfg.device, seed=self.seed))

        # -- idiosyncratic step (Eq. 10) ------------------------------------
        idio: Optional[List[Optional[MLP]]] = None
        if self.idiosyncratic:
            resid = ys - _predict(common, Xs, cfg.device)
            idio = []
            for i in range(panel.N):
                sel = uidx == i
                if sel.sum() < 3:
                    idio.append(None)
                    continue
                gi = MLP(p=p, depth=max(1, depth // 2), width=max(5, width // 2),
                         first_layer_bias=self.first_layer_bias)
                train_mlp(gi, Xs[sel], resid[sel], None, None,
                          TrainConfig(lr=cfg.lr, lam=cfg.lam, batch_size=cfg.batch_size,
                                      max_epochs=min(cfg.max_epochs, 800),
                                      optimizer=cfg.optimizer, device=cfg.device, seed=self.seed))
                idio.append(gi)

        self._fit = _Fitted(
            common=common, idio=idio, feat_scaler=feat_scaler, targ_scaler=targ_scaler,
            unit_means=unit_means, arch=arch, N=panel.N, T=panel.T, p=panel.p,
            horizon=self.horizon,
        )
        return self

    # -------------------------------------------------------------- predict
    def _check_fitted(self) -> _Fitted:
        if self._fit is None:
            raise RuntimeError("Estimator is not fitted; call fit(panel) first.")
        return self._fit

    def predict(self, X, units: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict the (transformed-back) target for feature rows ``X``.

        Parameters
        ----------
        X : array_like, shape (M, p)
            Regressor rows *in original units*.
        units : array_like of int, optional
            Unit index for each row; required to add the idiosyncratic component
            when the model was fit with ``idiosyncratic=True``.

        Returns
        -------
        yhat : ndarray, shape (M,)
            Predictions on the original target scale (unit means added back).
        """
        f = self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = f.feat_scaler.transform(X)
        yhat_s = _predict(f.common, Xs, self.config.device)
        if f.idio is not None:
            if units is None:
                raise ValueError("units is required for idiosyncratic predictions.")
            units = np.asarray(units, dtype=int)
            add = np.zeros_like(yhat_s)
            for i, gi in enumerate(f.idio):
                if gi is None:
                    continue
                sel = units == i
                if sel.any():
                    add[sel] = _predict(gi, Xs[sel], self.config.device)
            yhat_s = yhat_s + add
        yhat = f.targ_scaler.inverse_target(yhat_s, var=0)
        if f.unit_means is not None and units is not None:
            yhat = yhat + f.unit_means[np.asarray(units, dtype=int)]
        return yhat

    def forecast(self, panel: PanelData) -> np.ndarray:
        """Direct ``h``-step-ahead forecast ``y_{i,T+h}`` for every unit.

        Uses the time-``T`` regressors ``X_T`` (Chronopoulos et al. 2026,
        footnote 5) and returns one forecast per unit on the original scale.
        """
        f = self._check_fitted()
        XT = panel.last_features()  # (N, p)
        units = np.arange(f.N)
        return self.predict(XT, units=units)

    # ----------------------------------------------------------- diagnostics
    @property
    def architecture(self) -> Dict:
        return dict(self._check_fitted().arch)

    def torch_model(self) -> MLP:
        """The fitted common network (for partial derivatives etc.)."""
        return self._check_fitted().common

    def feature_scaler(self):
        return self._check_fitted().feat_scaler

    def partial_derivatives(self, X, feature: Optional[int] = None):
        """Partial derivatives ``d g / d x`` of the common map (WP Eq. 11).

        Thin wrapper around :func:`deeppanel.inference.partial.partial_derivatives`.
        """
        from ..inference.partial import partial_derivatives as _pd
        f = self._check_fitted()
        Xs = f.feat_scaler.transform(np.atleast_2d(np.asarray(X, dtype=float)))
        d = _pd(f.common, Xs, feature=feature)
        # chain rule: bring the derivative back to original input/target units
        in_scale = np.array([f.feat_scaler.scale_target(j) for j in range(f.p)])
        out_scale = f.targ_scaler.scale_target(0)
        if feature is None:
            return d * (out_scale / in_scale)[None, :]
        return d * (out_scale / in_scale[feature])

    def poolability_test(self, panel: PanelData, **kwargs):
        """Nonlinear poolability test of ``h~_i = h`` (WP Remark 3).

        Thin wrapper around
        :func:`deeppanel.inference.poolability.poolability_test`.
        """
        from ..inference.poolability import poolability_test as _pt
        return _pt(self, panel, **kwargs)

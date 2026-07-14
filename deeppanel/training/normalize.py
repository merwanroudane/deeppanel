"""Data transforms used before deep-panel estimation.

Two transforms appear in the papers, both fit on the *training* information only
so that the recursive out-of-sample exercise never looks ahead:

* **Rank / min-max normalisation to [0, 1]** -- Chronopoulos et al. (2026),
  Eq. (10), used for the COVID-19 application because case counts have heavy
  tails: ``z_tilde = (z - min z) / (max z - min z)``.
* **Standardisation (z-score)** -- used for the inflation application.

Each transform is fit per variable by *stacking a variable over time within each
unit* (the paper stacks ``z_i`` "over time", Eq. (10)).  ``fit`` therefore takes
an array shaped ``(N, T)`` (one variable) or ``(N, T, p)`` (several) and stores
one statistic per variable.  Both are invertible on the target so forecasts can
be reported in original units.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MinMaxNormalizer", "Standardizer", "identity"]


def _as_ntp(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        return a[:, :, None]
    if a.ndim == 3:
        return a
    raise ValueError("Expected an array shaped (N, T) or (N, T, p).")


class _BaseTransform:
    """Common fit/transform plumbing; subclasses set ``_a`` and ``_b``.

    Transform is affine: ``z_tilde = (z - a) / b`` with per-variable ``a, b``.
    """

    def __init__(self) -> None:
        self._a = None  # per-variable offset, shape (p,)
        self._b = None  # per-variable scale,  shape (p,)

    @property
    def fitted(self) -> bool:
        return self._a is not None

    def transform(self, a: np.ndarray) -> np.ndarray:
        """Apply the affine transform.

        Accepts any layout whose trailing axis is the ``p`` variables -- a pooled
        feature matrix ``(M, p)``, a panel tensor ``(N, T, p)``, or a single
        feature vector ``(p,)`` -- as well as a single-variable panel ``(N, T)``
        when ``p == 1``.
        """
        if not self.fitted:
            raise RuntimeError("Transform must be fit before transform.")
        arr = np.asarray(a, dtype=float)
        p = self._a.shape[0]
        if arr.ndim >= 1 and arr.shape[-1] == p:
            return (arr - self._a) / self._b            # broadcasts over leading dims
        if p == 1:                                      # single-variable panel (N, T)
            return (arr - self._a[0]) / self._b[0]
        raise ValueError(
            f"Cannot align array with trailing shape {arr.shape[-1]} to p={p} variables."
        )

    def fit_transform(self, a: np.ndarray) -> np.ndarray:
        return self.fit(a).transform(a)

    # -- target (single-variable) inverse ----------------------------------
    def inverse_target(self, y: np.ndarray, var: int = 0) -> np.ndarray:
        """Invert the transform for the target variable (column ``var``)."""
        if not self.fitted:
            raise RuntimeError("Transform must be fit before inverse_target.")
        return np.asarray(y, dtype=float) * self._b[var] + self._a[var]

    def scale_target(self, var: int = 0) -> float:
        """The multiplicative factor ``b`` for target column ``var``.

        Handy for rescaling derivatives/marginal effects back to original units.
        """
        return float(self._b[var])


class MinMaxNormalizer(_BaseTransform):
    """Min-max normalisation to ``[0, 1]`` (Chronopoulos et al. 2026, Eq. 10)."""

    def fit(self, a: np.ndarray) -> "MinMaxNormalizer":
        arr = _as_ntp(a)
        lo = arr.min(axis=(0, 1))
        hi = arr.max(axis=(0, 1))
        rng = hi - lo
        rng[rng == 0] = 1.0  # guard constant columns
        self._a = lo
        self._b = rng
        return self


class Standardizer(_BaseTransform):
    """Per-variable standardisation to zero mean and unit variance."""

    def fit(self, a: np.ndarray) -> "Standardizer":
        arr = _as_ntp(a)
        mu = arr.mean(axis=(0, 1))
        sd = arr.std(axis=(0, 1))
        sd[sd == 0] = 1.0
        self._a = mu
        self._b = sd
        return self


def identity() -> _BaseTransform:
    """A no-op transform (offset 0, scale 1) inferred from data at fit time."""

    class _Identity(_BaseTransform):
        def fit(self, a: np.ndarray) -> "_Identity":
            p = _as_ntp(a).shape[2]
            self._a = np.zeros(p)
            self._b = np.ones(p)
            return self

    return _Identity()

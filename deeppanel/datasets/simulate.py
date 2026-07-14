"""Known-truth simulation designs taken from the papers.

Two generators are provided so every code path can be validated against a
data-generating process whose truth is known:

* :func:`simulate_pooled_panel` -- the common + idiosyncratic nonlinear panel of
  Chronopoulos et al. (2023): ``y_it = h(x_it) + h_i(x_it) + eps``.  The common
  ``h`` has a closed-form gradient, so partial-derivative recovery can be checked;
  ``h_i`` is small, so the deep *pooled* model should beat the per-unit model and
  the poolability test should (weakly) detect the idiosyncratic part.

* :func:`simulate_ldpm` -- the design of Gao et al. (2026, Section 5):
  ``y_it = beta_i' Z(x_it) + eps_it`` with ``Z(x) = cos(Wx + b)`` and a latent
  group structure in ``beta_i`` / ``theta_i^S``; the target and surrogate errors
  are correlated with parameter ``rho``.  This lets us verify group recovery and
  that the surrogate signal helps more as ``rho`` grows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..utils.panel import PanelData

__all__ = ["simulate_pooled_panel", "simulate_ldpm", "PooledSim", "LDPMSim"]


@dataclass
class PooledSim:
    panel: PanelData
    common_weight: np.ndarray          # a in h(x)=sin(a.x)
    idio_weight: np.ndarray            # (N, p) unit-specific linear slopes
    noise_sd: float
    horizon: int                       # lag at which x drives y

    def true_partial(self, X: np.ndarray) -> np.ndarray:
        """Closed-form common gradient d h / d x = a * cos(a . x)."""
        X = np.atleast_2d(X)
        z = X @ self.common_weight
        return np.cos(z)[:, None] * self.common_weight[None, :]


def simulate_pooled_panel(
    N: int = 20,
    T: int = 60,
    p: int = 3,
    idio_scale: float = 0.15,
    noise_sd: float = 0.3,
    horizon: int = 1,
    seed: Optional[int] = None,
) -> PooledSim:
    """Common + idiosyncratic nonlinear panel with a known common gradient.

    The target is driven by the regressors *at lag ``horizon``*, matching the
    direct multi-step forecasting design of the papers (``y_it`` is predicted
    from ``x_{i,t-h}``):

        y_it = h(x_{i,t-h}) + h_i(x_{i,t-h}) + eps_it,   h(x) = sin(a . x).

    so that ``DeepPooledPanel(horizon=h)`` can recover ``h`` and its gradient.
    """
    rng = np.random.default_rng(seed)
    h = int(horizon)
    X = rng.normal(size=(N, T, p))
    a = rng.normal(size=p)
    a = a / np.linalg.norm(a) * 1.5              # moderate curvature
    idio = idio_scale * rng.normal(size=(N, p))
    y = np.zeros((N, T))
    # y_t built from x_{t-h}; the first h periods have no valid driver (unused as
    # targets by direct_pairs(h)) so they are filled with noise only.
    driver = X[:, : T - h, :]                    # x_{t-h} aligned to targets t=h..T-1
    common = np.sin(np.einsum("ntp,p->nt", driver, a))
    idio_part = np.einsum("ntp,np->nt", driver, idio)
    y[:, h:] = common + idio_part + noise_sd * rng.normal(size=(N, T - h))
    y[:, :h] = noise_sd * rng.normal(size=(N, h))
    panel = PanelData(y, X)
    return PooledSim(panel=panel, common_weight=a, idio_weight=idio,
                     noise_sd=noise_sd, horizon=h)


@dataclass
class LDPMSim:
    y: np.ndarray                      # (N, T) target
    X: np.ndarray                      # (N, T, p) embeddings (predictors)
    y_surrogate: np.ndarray            # (N, T) surrogate outcome
    groups: np.ndarray                 # (N,) true group labels
    rho: float
    W: np.ndarray
    b: np.ndarray

    @property
    def N(self) -> int:
        return self.y.shape[0]

    @property
    def T(self) -> int:
        return self.y.shape[1]


def simulate_ldpm(
    N: int = 24,
    T: int = 60,
    p: int = 6,
    p_prime: int = 4,
    n_groups: int = 3,
    rho: float = 0.5,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> LDPMSim:
    """Gao et al. (2026, Section 5) DGP with correlated target/surrogate errors.

    ``y_it = beta_i' cos(W x_it + b) + eps_it`` and
    ``y^S_it = theta_i^S' cos(W x_it + b) + eps^S_it`` with
    ``corr(eps, eps^S) = rho`` and a latent group structure in the coefficients.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(p_prime, p))
    b = rng.uniform(0, 2 * np.pi, size=p_prime)
    # group-level coefficient vectors (distinct per group)
    group_beta = rng.normal(size=(n_groups, p_prime)) * 1.5
    group_theta = rng.normal(size=(n_groups, p_prime)) * 1.5
    groups = rng.integers(0, n_groups, size=N)
    beta = group_beta[groups]                    # (N, p')
    theta = group_theta[groups]                  # (N, p')

    X = rng.normal(size=(N, T, p))
    Z = np.cos(np.einsum("ntp,qp->ntq", X, W) + b)   # (N, T, p') basis cos(Wx+b)

    # correlated errors: Sigma = (1-rho) I + rho J  on the (eps, eps^S) pair
    cov = np.array([[1.0, rho], [rho, 1.0]]) * (noise_sd ** 2)
    L = np.linalg.cholesky(cov)
    E = rng.normal(size=(N, T, 2)) @ L.T
    y = np.einsum("ntq,nq->nt", Z, beta) + E[:, :, 0]
    y_surr = np.einsum("ntq,nq->nt", Z, theta) + E[:, :, 1]
    return LDPMSim(y=y, X=X, y_surrogate=y_surr, groups=groups, rho=rho, W=W, b=b)

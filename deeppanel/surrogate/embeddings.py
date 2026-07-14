"""Embedding utilities for LDPM (Gao et al. 2026, Appendix A.2).

High-dimensional text embeddings (``p = 3072`` for OpenAI, ``768`` for BERT) are
impractical inside the *linear* baselines, so the paper reduces them by a
rank-``r0`` singular value decomposition: with ``X = U diag(sigma) V'`` the
compact representation is ``X V`` (an ``NT x r0`` matrix).  The nonlinear LDPM
does not need this step -- it handles the raw embeddings through representation
learning -- so this helper is meant for the LPM/LPM-E comparators.

Two aggregation helpers are also provided because the paper aggregates post-level
signals to the region-day level in two different ways (Section 4.1): **max
pooling** for embeddings (keep the strongest semantic activation per dimension)
and **averaging** for the surrogate score.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["svd_reduce", "max_pool", "mean_pool"]


def svd_reduce(X: np.ndarray, r0: int):
    """Rank-``r0`` SVD reduction ``X -> X V`` (Appendix A.2).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Stacked embedding matrix (rows = observations).
    r0 : int
        Target dimension ``1 <= r0 < min(n, p)``.

    Returns
    -------
    XV : ndarray, shape (n, r0)
        The reduced features ``X V``.
    V : ndarray, shape (p, r0)
        The right singular vectors (apply to new data as ``X_new @ V``).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    r0 = int(min(r0, min(n, p) - 1)) if min(n, p) > 1 else 1
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:r0].T                       # (p, r0)
    return X @ V, V


def max_pool(posts: Sequence[np.ndarray]) -> np.ndarray:
    """Max-pool a list of post-level embedding vectors to one region-day vector.

    Retains the strongest activation along each embedding dimension (Section 4.1).
    """
    if len(posts) == 0:
        raise ValueError("max_pool received no posts.")
    M = np.asarray(posts, dtype=float)
    return M.max(axis=0)


def mean_pool(scores: Sequence[float]) -> float:
    """Average post-level surrogate scores to a region-day surrogate outcome."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        raise ValueError("mean_pool received no scores.")
    return float(s.mean())

"""Partial derivatives of a fitted network -- the interpretability device of
Chronopoulos et al. (2023, 2026).

The papers "open the black box" by reporting

    d_{ij,t} = d g(X_t; theta_hat) / d x_{i,j,t-h}                        (Eq. 11)

the partial derivative of the fitted conditional mean with respect to the
``j``-th characteristic.  Because ``g`` is (almost everywhere) differentiable
these are obtained exactly by automatic differentiation -- no finite differences.
ReLU is non-differentiable only at 0; as the papers note, autograd returns a
valid sub-gradient there, so this is faithful to their construction.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..models.mlp import MLP

__all__ = ["partial_derivatives"]


def partial_derivatives(
    model: MLP,
    X,
    feature: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Return ``d g / d x`` evaluated at every row of ``X``.

    Parameters
    ----------
    model : MLP
        A fitted scalar-output network.
    X : array_like, shape (M, p)
        Points (already on the network's input scale) at which to differentiate.
    feature : int, optional
        If given, return only the derivative w.r.t. that regressor (shape
        ``(M,)``).  Otherwise return the full Jacobian ``(M, p)``.

    Returns
    -------
    ndarray
        ``(M, p)`` full gradient, or ``(M,)`` if ``feature`` is specified.
    """
    model = model.to(device).eval()
    Xt = torch.as_tensor(np.atleast_2d(np.asarray(X, dtype=np.float32)), device=device)
    Xt.requires_grad_(True)
    out = model(Xt)                      # (M,)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=Xt,
        grad_outputs=torch.ones_like(out),
        retain_graph=False,
        create_graph=False,
    )[0]                                  # (M, p)
    g = grad.detach().cpu().numpy()
    return g if feature is None else g[:, int(feature)]

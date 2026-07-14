"""Feed-forward ReLU multilayer perceptron ``g(x; theta)``.

This is the functional parameterisation used to approximate the (common)
nonlinear conditional mean in all three papers:

    g(X_t; theta) = sigma_L( ... sigma_2( sigma_1(X_t W1' + b1') W2' + b2' ) ... ) WL' + bL'

with ReLU activations on the ``L-1`` hidden layers and a *linear* output layer
(Chronopoulos et al. 2023, Eq. (16); 2026, Eq. (3)).  Depth ``L`` (number of
hidden layers) and width ``M`` (neurons per hidden layer) are constant positive
integers selected by cross-validation from the paper's grids.

Faithfulness notes
------------------
* ``b^{(1)} = 0``.  The papers fix the first hidden-layer bias to zero; we honour
  this with ``first_layer_bias=False`` by default.  With standardised/normalised
  inputs this is immaterial, but the default matches the text.
* The output layer is linear (no activation), so -- exactly as the papers stress
  (Eq. (8) 2026 / Eq. (6) 2023) -- the network has a final linear representation
  ``g(x;theta) = theta_L' f(x)`` in the last-hidden-layer features ``f(x)``.
  :meth:`MLP.features` returns that ``f(x)`` for the poolability test.
* Optional dropout and batch normalisation are provided because the papers use
  them as regularisers (Section 3); both default to off so the plain estimator is
  reproducible.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn

__all__ = ["MLP"]

ActLike = Union[str, nn.Module]


def _resolve_activation(act: ActLike) -> nn.Module:
    if isinstance(act, nn.Module):
        return act
    key = str(act).lower()
    table = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    if key not in table:
        raise ValueError(f"Unknown activation {act!r}; choose from {sorted(table)}.")
    return table[key]()


class MLP(nn.Module):
    """A depth-``L`` width-``M`` feed-forward network with a linear output head.

    Parameters
    ----------
    p : int
        Number of input regressors.
    depth : int
        Number of hidden layers ``L`` (paper grid ``[1, 3, 5, 10, 15]``).
    width : int or sequence of int
        Neurons per hidden layer ``M`` (paper grid ``[5, 10, 15, 20, 30]``).  A
        scalar uses the same width for every hidden layer.
    activation : str or nn.Module, default ``"relu"``
        Hidden-layer activation.  The output layer is always linear.
    dropout : float, default 0.0
        Dropout probability applied after each hidden activation.
    batchnorm : bool, default False
        Whether to insert 1-D batch normalisation after each hidden linear map
        (Ioffe and Szegedy 2015; Section 3 of the papers).
    first_layer_bias : bool, default False
        If ``False`` the first hidden layer has no bias, matching ``b^{(1)} = 0``.
    out_dim : int, default 1
        Output dimension (1 for a scalar conditional mean).
    """

    def __init__(
        self,
        p: int,
        depth: int = 3,
        width: Union[int, Sequence[int]] = 20,
        activation: ActLike = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        first_layer_bias: bool = False,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth (number of hidden layers L) must be >= 1.")
        if isinstance(width, int):
            widths: List[int] = [width] * depth
        else:
            widths = list(width)
            if len(widths) != depth:
                raise ValueError("len(width) must equal depth when width is a sequence.")
        self.p = int(p)
        self.depth = int(depth)
        self.widths = widths
        self.out_dim = int(out_dim)

        body: List[nn.Module] = []
        in_dim = p
        for l, m in enumerate(widths):
            use_bias = first_layer_bias if l == 0 else True
            body.append(nn.Linear(in_dim, m, bias=use_bias))
            if batchnorm:
                body.append(nn.BatchNorm1d(m))
            body.append(_resolve_activation(activation))
            if dropout > 0:
                body.append(nn.Dropout(dropout))
            in_dim = m
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(in_dim, out_dim, bias=True)  # linear output layer

    # -- forward ------------------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Last-hidden-layer features ``f(x)`` (the linear representation)."""
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.body(x))
        return out.squeeze(-1) if self.out_dim == 1 else out

    # -- introspection ------------------------------------------------------
    def n_parameters(self) -> int:
        """Total number of trainable parameters ``d = |theta|``."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                module.reset_parameters()

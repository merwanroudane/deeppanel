"""Cross-validation for the deep-panel networks.

Selects the architecture and regularisation entirely from the data, from the
grids used in Chronopoulos et al. (2023, Section 3.3; 2026, Section 3.2):

    depth   L        in {1, 3, 5, 10, 15}
    width   M        in {5, 10, 15, 20, 30}
    lr      gamma     in {0.01, 0.001}
    lasso   lambda    = c * sqrt(log p / (N*T)),  c in {.001,.01,.1,.5,1,5,10}
    dropout          up to 0.10

The sample is split into three disjoint, time-ordered blocks (train / validation
/ test).  Hyper-parameters are tuned on the validation block given parameters
estimated on the training block; the winning configuration minimises validation
MSE (Section 3.3).  The full grid has 350 combinations, so a lighter default is
provided; pass :data:`PAPER_GRID` to reproduce the paper exactly.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..models.mlp import MLP
from .trainer import TrainConfig, train_mlp

__all__ = ["CVGrid", "PAPER_GRID", "DEFAULT_GRID", "cross_validate", "CVResult"]


@dataclass
class CVGrid:
    """A grid of candidate hyper-parameters for :func:`cross_validate`."""

    depth: Sequence[int] = (1, 3, 5)
    width: Sequence[int] = (5, 10, 20)
    lr: Sequence[float] = (0.01, 0.001)
    lasso_c: Sequence[float] = (0.0,)     # multiplier c; 0 => no penalty
    dropout: Sequence[float] = (0.0,)

    def combinations(self):
        return itertools.product(self.depth, self.width, self.lr, self.lasso_c, self.dropout)

    def size(self) -> int:
        return (len(self.depth) * len(self.width) * len(self.lr)
                * len(self.lasso_c) * len(self.dropout))


#: The exact grids from the papers (350 combinations).
PAPER_GRID = CVGrid(
    depth=(1, 3, 5, 10, 15),
    width=(5, 10, 15, 20, 30),
    lr=(0.01, 0.001),
    lasso_c=(0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0),
    dropout=(0.0, 0.1),
)

#: A small, fast default suitable for examples and tests.
DEFAULT_GRID = CVGrid(depth=(1, 3), width=(5, 10), lr=(0.01, 0.001), lasso_c=(0.0,), dropout=(0.0,))


@dataclass
class CVResult:
    best: Dict = field(default_factory=dict)
    best_val: float = float("inf")
    table: List[Dict] = field(default_factory=list)  # every config + its val loss

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.table).sort_values("val_loss").reset_index(drop=True)


def lasso_lambda(c: float, p: int, nt: int) -> float:
    """LASSO ``lambda = c * sqrt(log p / (N*T))`` (paper scaling)."""
    if c <= 0:
        return 0.0
    return float(c) * math.sqrt(math.log(max(p, 2)) / max(nt, 1))


def cross_validate(
    X_tr,
    y_tr,
    X_val,
    y_val,
    p: int,
    nt: int,
    grid: Optional[CVGrid] = None,
    base_config: Optional[TrainConfig] = None,
    first_layer_bias: bool = False,
    batchnorm: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> CVResult:
    """Grid-search the architecture/regularisation, minimising validation MSE.

    Parameters
    ----------
    X_tr, y_tr : training pooled features/targets.
    X_val, y_val : validation pooled features/targets.
    p : int
        Number of input regressors (for the ``lambda`` scaling).
    nt : int
        ``N * T`` of the estimation window (for the ``lambda`` scaling).
    grid : CVGrid, optional
        Candidate grid (default :data:`DEFAULT_GRID`).
    base_config : TrainConfig, optional
        Base training config; ``lr`` and ``lam`` are overridden per candidate.

    Returns
    -------
    CVResult
        The winning configuration, its validation loss, and the full table.
    """
    grid = grid or DEFAULT_GRID
    base = base_config or TrainConfig()
    result = CVResult()

    for depth, width, lr, c, drop in grid.combinations():
        lam = lasso_lambda(c, p, nt)
        cfg = TrainConfig(
            lr=lr, lam=lam, batch_size=base.batch_size, max_epochs=base.max_epochs,
            patience=base.patience, min_delta=base.min_delta,
            weight_decay=base.weight_decay, optimizer=base.optimizer,
            device=base.device, seed=seed, verbose=False,
        )
        model = MLP(p=p, depth=depth, width=width, dropout=drop,
                    batchnorm=batchnorm, first_layer_bias=first_layer_bias)
        tr = train_mlp(model, X_tr, y_tr, X_val, y_val, cfg)
        row = {
            "depth": depth, "width": width, "lr": lr, "lasso_c": c,
            "lambda": lam, "dropout": drop, "val_loss": tr.best_val,
            "best_epoch": tr.best_epoch, "n_params": model.n_parameters(),
        }
        result.table.append(row)
        if verbose:
            print(f"L={depth:2d} M={width:2d} lr={lr:<6} c={c:<5} drop={drop:<4} "
                  f"-> val {tr.best_val:.5f}")
        if tr.best_val < result.best_val:
            result.best_val = tr.best_val
            result.best = row.copy()
    return result

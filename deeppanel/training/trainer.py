"""Training loop for the deep-panel networks.

Minimises the pooled mean-squared-error objective

    (1 / NT) * sum_i sum_t ( y_it - g(x_it; theta) )^2                    (Eq. 9)

optionally augmented with the LASSO penalty ``lambda * ||theta||_1`` on all
trainable parameters (Section 3.1 of the papers).  Optimisation uses ADAM
(Kingma and Ba 2014) -- "a more efficient version of SGD" -- with early stopping
on a validation split, up to a large epoch budget (the papers use 5,000 epochs
with early stopping).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.mlp import MLP

__all__ = ["TrainConfig", "train_mlp"]


@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`train_mlp`.

    Defaults follow the papers: batch size 14, ADAM, up to 5,000 epochs with
    early stopping, LASSO off (``lam=0``) since the papers report that the
    non-penalised network usually forecasts best (the "double descent" remark).
    """

    lr: float = 0.001            # learning rate gamma, paper grid {0.01, 0.001}
    lam: float = 0.0             # LASSO penalty lambda on ||theta||_1
    batch_size: int = 14         # paper fixes the batch size to 14
    max_epochs: int = 5000       # paper epoch budget
    patience: int = 50           # early-stopping patience (epochs w/o val gain)
    min_delta: float = 1e-6      # minimum val-loss improvement to reset patience
    weight_decay: float = 0.0    # optional L2 (kept 0 to match the paper's L1-only)
    optimizer: str = "adam"      # "adam" or "sgd"
    device: str = "cpu"
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class TrainResult:
    model: MLP
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val: float = float("inf")


def _to_tensor(a, device: str) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def _l1_penalty(model: nn.Module) -> torch.Tensor:
    total = None
    for param in model.parameters():
        if param.requires_grad:
            term = param.abs().sum()
            total = term if total is None else total + term
    return total if total is not None else torch.zeros(1)


def train_mlp(
    model: MLP,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    config: Optional[TrainConfig] = None,
) -> TrainResult:
    """Fit ``model`` by pooled MSE (+ optional LASSO) with ADAM + early stopping.

    Parameters
    ----------
    model : MLP
        The network to train (modified in place; best weights restored on exit).
    X_train, y_train : array_like
        Pooled training features ``(M, p)`` and targets ``(M,)``.
    X_val, y_val : array_like, optional
        Validation split for early stopping.  If omitted, training runs for
        ``max_epochs`` and the final weights are kept (no early stopping).
    config : TrainConfig, optional

    Returns
    -------
    TrainResult
        Fitted model plus per-epoch train/validation loss history.
    """
    cfg = config or TrainConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    device = cfg.device
    model = model.to(device)

    Xtr = _to_tensor(X_train, device)
    ytr = _to_tensor(y_train, device)
    has_val = X_val is not None and y_val is not None
    if has_val:
        Xva = _to_tensor(X_val, device)
        yva = _to_tensor(y_val, device)

    ds = TensorDataset(Xtr, ytr)
    batch = min(cfg.batch_size, len(ds)) or 1
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    if cfg.optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'.")
    mse = nn.MSELoss()

    res = TrainResult(model=model)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wait = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            if cfg.lam > 0:
                loss = loss + cfg.lam * _l1_penalty(model)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(xb)
        res.train_loss.append(running / len(ds))

        if has_val:
            model.eval()
            with torch.no_grad():
                vloss = float(mse(model(Xva), yva))
            res.val_loss.append(vloss)
            if vloss < res.best_val - cfg.min_delta:
                res.best_val = vloss
                res.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    if cfg.verbose:
                        print(f"[train_mlp] early stop at epoch {epoch} (best {res.best_epoch}).")
                    break
        if cfg.verbose and epoch % 100 == 0:
            msg = f"epoch {epoch:5d}  train {res.train_loss[-1]:.5f}"
            if has_val:
                msg += f"  val {res.val_loss[-1]:.5f}"
            print(msg)

    if has_val:
        model.load_state_dict(best_state)  # restore best-validation weights
    model.eval()
    return res


def predict(model: MLP, X, device: str = "cpu") -> np.ndarray:
    """Convenience: evaluate ``model`` on ``X`` and return a NumPy array."""
    model.eval()
    with torch.no_grad():
        out = model(_to_tensor(X, device))
    return out.detach().cpu().numpy()

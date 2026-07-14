"""Deep Panel Training with classifier-LASSO homogeneity pursuit
(Gao, Sun, Wang, Liu & Hsiao 2026, Section 3.2-3.3).

A shared feed-forward feature map ``h(.; W, gamma)`` is combined with *unit-
specific linear heads* ``(beta_i, b_i)`` (Eq. 3.5):

    Q_i(v) = beta_i' h(v; W, gamma) + b_i.

To borrow strength across units, the heads are pushed toward ``K0`` latent group
centres ``{(eta_k, phi_k)}`` by the classifier-LASSO *product* penalty of Su, Shi
and Phillips (2016), extended to the network output layer (Eq. 3.7):

    (1/NT) sum_i sum_t ( y_it - beta_i' h_it - b_i )^2
        + (lambda/N) sum_i prod_{k=1}^{K0} || (eta_k, phi_k) - (beta_i, b_i) ||.

Because the penalty is a *product* over centres, it vanishes as soon as a head
coincides with *any* one centre -- this is what drives each unit into exactly one
group.  After the penalised fit each unit is hard-assigned to its nearest centre
(``g_hat``) and the group heads are re-estimated **without** penalty (the final
"debiasing" refit of Section 3.3).

Optimisation note (faithful-but-practical).  The paper's objective is optimised
here by ADAM on all parameters jointly, with the centres warm-started by k-means
on the unpenalised heads; the hidden representation is standardised on a common
scale before the heads (as the paper requires, Section 3.3) so the grouped
penalty is well defined.  This is the equivalent-objective route flagged in the
build brief, not a different estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .mlp import MLP

__all__ = ["DeepPanelTraining", "DPTResult"]


class _SharedBody(nn.Module):
    """Shared feature map ``h(.; W, gamma)`` returning ``d_h`` standardised features."""

    def __init__(self, q: int, depth: int, width: int, d_h: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(p=q, depth=depth, width=width, dropout=dropout, out_dim=d_h)
        # replace scalar head by a d_h-dim linear so `forward` yields the feature vector
        self.norm = nn.BatchNorm1d(d_h, affine=False)  # common-scale normalisation

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        feat = self.mlp(v)                 # (B, d_h)
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)
        return self.norm(feat)


@dataclass
class DPTResult:
    groups: np.ndarray                 # g_hat(i), shape (N,)
    centers: np.ndarray                # (K0, d_h + 1) stacked (eta_k, phi_k)
    loss_path: List[float] = field(default_factory=list)
    n_groups_used: int = 0


class DeepPanelTraining:
    """Deep Panel Training estimator with hidden homogeneity pursuit.

    Parameters
    ----------
    n_groups : int, default 3
        Number of latent groups ``K0``.
    d_h : int, default 8
        Dimension of the shared hidden representation feeding the linear heads.
    depth, width : int
        Depth/width of the shared feature map.
    lam : float, default 0.05
        Homogeneity-pursuit penalty ``lambda``.
    warmup : int, default 200
        Unpenalised epochs before k-means initialisation of the centres.
    epochs : int, default 400
        Penalised epochs.
    lr : float, default 0.01
    dropout : float, default 0.0
    seed : int, optional
    """

    def __init__(self, n_groups: int = 3, d_h: int = 8, depth: int = 3, width: int = 20,
                 lam: float = 0.05, warmup: int = 200, epochs: int = 400, lr: float = 0.01,
                 dropout: float = 0.0, seed: Optional[int] = None) -> None:
        self.K0 = int(n_groups)
        self.d_h = int(d_h)
        self.depth = int(depth)
        self.width = int(width)
        self.lam = float(lam)
        self.warmup = int(warmup)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.seed = seed
        self._body = None
        self._heads = None            # (N, d_h+1) tensor: [beta_i, b_i]
        self._result: Optional[DPTResult] = None

    # ------------------------------------------------------------------ fit
    def fit(self, y: np.ndarray, design: np.ndarray) -> "DeepPanelTraining":
        """Fit the shared map, unit heads, and latent groups.

        Parameters
        ----------
        y : ndarray, shape (N, T)
            Target.
        design : ndarray, shape (N, T, q)
            Per-observation design entering the shared feature map (e.g.
            ``[x_it, z_it, eps_hat^S_it]`` in LDPM).
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        y = np.asarray(y, dtype=float)
        design = np.asarray(design, dtype=float)
        N, T, q = design.shape

        Vt = torch.as_tensor(design.reshape(N * T, q), dtype=torch.float32)
        yt = torch.as_tensor(y.reshape(N * T), dtype=torch.float32)
        unit_idx = torch.repeat_interleave(torch.arange(N), T)

        body = _SharedBody(q, self.depth, self.width, self.d_h, self.dropout)
        heads = nn.Parameter(0.1 * torch.randn(N, self.d_h + 1))
        centers = nn.Parameter(0.1 * torch.randn(self.K0, self.d_h + 1))

        def head_pred():
            feat = body(Vt)                              # (NT, d_h)
            feat1 = torch.cat([feat, torch.ones(feat.size(0), 1)], dim=1)  # add const
            h_i = heads[unit_idx]                        # (NT, d_h+1)
            return (feat1 * h_i).sum(dim=1)

        mse = nn.MSELoss()
        result = DPTResult(groups=np.zeros(N, dtype=int), centers=np.zeros((self.K0, self.d_h + 1)))

        # -- warm-up: unpenalised (body + heads) ---------------------------
        opt = torch.optim.Adam(list(body.parameters()) + [heads], lr=self.lr)
        for _ in range(self.warmup):
            opt.zero_grad()
            loss = mse(head_pred(), yt)
            loss.backward()
            opt.step()
            result.loss_path.append(float(loss.detach()))

        # -- k-means init of centres on the (warm) heads -------------------
        with torch.no_grad():
            H = heads.detach().numpy()
            centers.data = torch.as_tensor(_kmeans(H, self.K0, seed=self.seed), dtype=torch.float32)

        # -- penalised training: body + heads + centres --------------------
        opt = torch.optim.Adam(list(body.parameters()) + [heads, centers], lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            fit_loss = mse(head_pred(), yt)
            # product penalty (1/N) sum_i prod_k || c_k - head_i ||
            diff = centers.unsqueeze(0) - heads.unsqueeze(1)          # (N, K0, d_h+1)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)          # (N, K0)
            pen = dist.prod(dim=1).mean()
            loss = fit_loss + self.lam * pen
            loss.backward()
            opt.step()
            result.loss_path.append(float(fit_loss.detach()))

        # -- hard assignment + unpenalised group refit ---------------------
        with torch.no_grad():
            feat = body(Vt).numpy()
            feat1 = np.column_stack([feat, np.ones(feat.shape[0])])
            H = heads.detach().numpy()
            C = centers.detach().numpy()
            g = np.argmin(((H[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)

        # refit each group's centre by pooled OLS of y on [h, 1] (Eq. after 3.7)
        yv = y.reshape(N * T)
        uidx = unit_idx.numpy()
        refit_centers = C.copy()
        used = 0
        for k in range(self.K0):
            members = np.where(g == k)[0]
            if members.size == 0:
                continue
            used += 1
            rows = np.isin(uidx, members)
            beta, *_ = np.linalg.lstsq(feat1[rows], yv[rows], rcond=None)
            refit_centers[k] = beta

        # assign heads to refit centres
        final_heads = refit_centers[g]
        self._body = body
        self._heads = torch.as_tensor(final_heads, dtype=torch.float32)
        result.groups = g
        result.centers = refit_centers
        result.n_groups_used = used
        self._result = result
        return self

    # -------------------------------------------------------------- predict
    def _feat1(self, design_rows: np.ndarray) -> np.ndarray:
        self._body.eval()
        with torch.no_grad():
            feat = self._body(torch.as_tensor(design_rows, dtype=torch.float32)).numpy()
        return np.column_stack([feat, np.ones(feat.shape[0])])

    def predict(self, unit: int, design_rows: np.ndarray) -> np.ndarray:
        """Predict ``y`` for a single unit at the given design rows."""
        if self._heads is None:
            raise RuntimeError("Call fit() first.")
        f1 = self._feat1(np.atleast_2d(design_rows))
        head = self._heads[int(unit)].numpy()
        return f1 @ head

    def forecast_last(self, design_last: np.ndarray) -> np.ndarray:
        """One prediction per unit from a design matrix of shape (N, q)."""
        design_last = np.atleast_2d(design_last)
        f1 = self._feat1(design_last)
        heads = self._heads.numpy()
        return np.einsum("nq,nq->n", f1, heads)

    @property
    def result(self) -> DPTResult:
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result

    @property
    def groups(self) -> np.ndarray:
        return self.result.groups


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: Optional[int] = None) -> np.ndarray:
    """Tiny Lloyd's k-means used to initialise the group centres."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        newC = C.copy()
        for j in range(k):
            m = X[lab == j]
            if len(m):
                newC[j] = m.mean(axis=0)
        if np.allclose(newC, C):
            break
        C = newC
    # pad if fewer clusters than requested
    if C.shape[0] < k:
        C = np.vstack([C, C[-1:].repeat(k - C.shape[0], axis=0)])
    return C

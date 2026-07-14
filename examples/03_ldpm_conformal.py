"""Example 3 -- LLM-powered Deep Panel Modeling (LDPM).

Reproduces the LDPM pipeline of Gao et al. (2026) on the paper's own simulation
design (Section 5): build surrogate residuals, fit the surrogate-augmented Deep
Panel Training model with classifier-LASSO homogeneity pursuit, recover the
latent groups, and produce within-group split-conformal prediction intervals.

The LLM/BERT text pipeline is external; here the surrogate outcome and embeddings
come from the simulator, standing in for the aggregated Weibo signals.

Run::

    python examples/03_ldpm_conformal.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deeppanel import LDPM, TrainConfig, surrogate_residuals, viz
from deeppanel.datasets import simulate_ldpm

OUT = os.path.join(os.path.dirname(__file__), "_output")
os.makedirs(OUT, exist_ok=True)


def group_accuracy(true, est):
    from itertools import permutations
    K = int(max(true.max(), est.max())) + 1
    return max(np.mean([p[e] for e in est] == true) for p in permutations(range(K)))


def main() -> None:
    s = simulate_ldpm(N=20, T=70, p=6, p_prime=4, n_groups=3, rho=0.6, seed=2)

    # Stage 1: surrogate residuals (Eq. 3.2)
    eps = surrogate_residuals(s.y_surrogate, s.X, n_lags=1, depth=2, width=12,
                              config=TrainConfig(max_epochs=200, seed=0))

    # Stage 2: surrogate-augmented Deep Panel Training + conformal calibration
    ld = LDPM(n_groups=3, d_h=6, depth=2, width=16, lam=0.05, alpha=0.10,
              calib_frac=0.2, warmup=150, epochs=250, lr=0.01, seed=0)
    ld.fit(s.y[:, :-1], s.X[:, :-1], surrogate_resid=eps[:, :-1], calibrate=True)

    acc = group_accuracy(s.groups, ld.groups)
    print(f"Latent group recovery accuracy: {acc:.2f}  (chance = {1/3:.2f})")

    # one-step-ahead forecast with conformal intervals
    yhat, lo, hi = ld.forecast_interval(s.X[:, -1, :], surrogate_resid_last=eps[:, -1])
    y_true = s.y[:, -1]
    cover = np.mean((y_true >= lo) & (y_true <= hi))
    print(f"One-step forecast: mean interval width = {np.mean(hi - lo):.2f}, "
          f"coverage = {cover:.2f}")

    # ---- figure: conformal intervals across units ---------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = np.argsort(yhat)
    viz.plot_conformal_intervals(yhat[order], lo[order], hi[order], actual=y_true[order],
                                 labels=[f"u{u}" for u in order], ax=ax,
                                 title="LDPM one-step forecasts with conformal intervals")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_ldpm_conformal.png"), dpi=150)

    # ---- figure: estimated latent groups ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 2.6))
    viz.plot_group_map(ld.groups, unit_labels=[f"u{i}" for i in range(s.N)], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_ldpm_groups.png"), dpi=150)

    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    main()

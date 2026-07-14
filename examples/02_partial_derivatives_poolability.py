"""Example 2 -- Interpretability: partial derivatives and the poolability test.

Two devices from Chronopoulos et al. (2023, 2026):

* **Partial derivatives** ``d g / d x`` (Eq. 11) "open the black box" of the
  network -- here we recover the known common gradient of the DGP.
* **Poolability test** (Remark 3) checks whether an idiosyncratic component is
  needed on top of the common one.

Run::

    python examples/02_partial_derivatives_poolability.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deeppanel import DeepPooledPanel, TrainConfig, viz
from deeppanel.datasets import simulate_pooled_panel

OUT = os.path.join(os.path.dirname(__file__), "_output")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    cfg = TrainConfig(max_epochs=400, patience=50, lr=0.01, batch_size=64, seed=0)

    # ---- partial derivatives recover the true common gradient ----------------
    sim = simulate_pooled_panel(N=25, T=70, p=3, idio_scale=0.05,
                                noise_sd=0.15, horizon=1, seed=1)
    model = DeepPooledPanel(horizon=1, depth=3, width=32, config=cfg, seed=0).fit(sim.panel)

    feats, _, _, _ = sim.panel.direct_pairs(1)
    d_hat = model.partial_derivatives(feats)
    d_true = sim.true_partial(feats)
    corr = np.corrcoef(d_hat.ravel(), d_true.ravel())[0, 1]
    print(f"Partial-derivative recovery: corr(estimated, true common gradient) = {corr:.3f}")

    fig, axes = plt.subplots(1, sim.panel.p, figsize=(4 * sim.panel.p, 3.6))
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.scatter(d_true[:, j], d_hat[:, j], s=8, alpha=0.4,
                   color=viz.parula_colors(sim.panel.p)[j])
        lim = [min(d_true[:, j].min(), d_hat[:, j].min()),
               max(d_true[:, j].max(), d_hat[:, j].max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(f"x{j}")
        ax.set_xlabel("true dh/dx")
        ax.set_ylabel("estimated dg/dx")
    fig.suptitle("Partial-derivative recovery (Eq. 11)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_partial_derivatives.png"), dpi=150)

    # ---- poolability test: size (no idiosyncratic) vs power (idiosyncratic) ---
    print("\nPoolability test (H0: pooling is enough):")
    for label, idio in [("no idiosyncratic (H0 true)", 0.0),
                        ("strong idiosyncratic (H0 false)", 0.6)]:
        s = simulate_pooled_panel(N=25, T=70, p=3, idio_scale=idio,
                                  noise_sd=0.2, horizon=1, seed=3)
        m = DeepPooledPanel(horizon=1, depth=3, width=32, config=cfg, seed=0).fit(s.panel)
        pt = m.poolability_test(s.panel)
        verdict = "REJECT pooling" if pt.p_value < 0.05 else "do not reject"
        print(f"  {label:32s}: P={pt.statistic:6.2f}  p={pt.p_value:.3f}  -> {verdict}")

    print(f"\nFigure written to {OUT}")


if __name__ == "__main__":
    main()

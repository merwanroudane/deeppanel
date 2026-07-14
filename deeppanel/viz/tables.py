"""Journal-style result tables (console, Markdown and LaTeX).

Formats estimation/forecast results the way top econometrics journals do:
aligned columns, a horizontal rule, significance stars from p-values, and a
footer with the significance legend.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["stars", "forecast_table", "coef_table", "to_latex"]


def stars(p: float, thresholds=(0.01, 0.05, 0.10), symbols=("***", "**", "*")) -> str:
    """Significance stars for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    for thr, sym in zip(thresholds, symbols):
        if p <= thr:
            return sym
    return ""


def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def forecast_table(
    rmse: Dict[str, float],
    ratios: Optional[Dict[str, float]] = None,
    pvalues: Optional[Dict[str, float]] = None,
    base: Optional[str] = None,
    nd: int = 3,
    title: str = "Out-of-sample forecast accuracy",
) -> str:
    """Render an RMSE / RMSE-ratio comparison table as aligned text.

    Parameters
    ----------
    rmse : dict[str, float]
        Model -> RMSE (absolute).
    ratios : dict[str, float], optional
        Model -> RMSE ratio relative to ``base``.
    pvalues : dict[str, float], optional
        Model -> DM p-value vs ``base`` (adds stars).
    base : str, optional
        Name of the benchmark model whose absolute RMSE anchors the ratios.
    """
    names = list(rmse.keys())
    width = max(len(n) for n in names) + 2
    lines: List[str] = [title, "=" * (width + 26)]
    header = f"{'Model':<{width}}{'RMSE':>10}"
    if ratios is not None:
        header += f"{'Ratio':>12}"
    lines.append(header)
    lines.append("-" * (width + 26))
    for n in names:
        row = f"{n:<{width}}{_fmt(rmse[n], nd):>10}"
        if ratios is not None:
            r = ratios.get(n, np.nan)
            star = stars(pvalues.get(n, np.nan)) if pvalues else ""
            row += f"{_fmt(r, nd) + star:>12}"
        lines.append(row)
    lines.append("-" * (width + 26))
    if base is not None:
        lines.append(f"Ratios relative to: {base}  (ratio < 1 => more accurate).")
    if pvalues is not None:
        lines.append("Stars: Diebold-Mariano test vs base.  *** p<.01  ** p<.05  * p<.10")
    return "\n".join(lines)


def coef_table(
    names: Sequence[str],
    coefs: Sequence[float],
    ses: Optional[Sequence[float]] = None,
    pvalues: Optional[Sequence[float]] = None,
    nd: int = 3,
    title: str = "Estimates",
    footer: Optional[str] = None,
) -> str:
    """Render a coefficient table with SEs (in parentheses) and stars."""
    width = max(len(n) for n in names) + 2
    lines = [title, "=" * (width + 24)]
    lines.append(f"{'Term':<{width}}{'Estimate':>12}")
    lines.append("-" * (width + 24))
    for k, n in enumerate(names):
        star = stars(pvalues[k]) if pvalues is not None else ""
        lines.append(f"{n:<{width}}{_fmt(coefs[k], nd) + star:>12}")
        if ses is not None:
            lines.append(f"{'':<{width}}{'(' + _fmt(ses[k], nd) + ')':>12}")
    lines.append("-" * (width + 24))
    if footer:
        lines.append(footer)
    lines.append("*** p<.01  ** p<.05  * p<.10")
    return "\n".join(lines)


def to_latex(df, caption: str = "", label: str = "", nd: int = 3) -> str:
    """Booktabs LaTeX for a pandas DataFrame (journal-ready)."""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError("to_latex expects a pandas DataFrame.")
    body = df.to_latex(float_format=lambda x: f"{x:.{nd}f}", escape=False)
    if caption or label:
        head = "\\begin{table}[htbp]\n\\centering\n"
        if caption:
            head += f"\\caption{{{caption}}}\n"
        if label:
            head += f"\\label{{{label}}}\n"
        return head + body + "\\end{table}\n"
    return body

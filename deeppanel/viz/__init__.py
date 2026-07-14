"""Journal-style tables and figures."""
from .style import (
    set_style, parula_colors, matlab_jet_colors, turbo_colors,
    bluered_colors, get_cmap, PARULA,
)
from .tables import stars, forecast_table, coef_table, to_latex
from .plots import (
    plot_forecasts, plot_partial_derivatives, plot_conformal_intervals,
    plot_rmse_heatmap, plot_group_map,
)

__all__ = [
    "set_style", "parula_colors", "matlab_jet_colors", "turbo_colors",
    "bluered_colors", "get_cmap", "PARULA",
    "stars", "forecast_table", "coef_table", "to_latex",
    "plot_forecasts", "plot_partial_derivatives", "plot_conformal_intervals",
    "plot_rmse_heatmap", "plot_group_map",
]

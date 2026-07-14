# Changelog

All notable changes to `deeppanel` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-07-14

### Fixed
- README links (cookbook, compatibility doc, license, examples, citation) are now
  absolute GitHub URLs so they resolve on the PyPI project page (PyPI does not
  follow relative Markdown links). No code changes.

## [0.1.1] - 2026-07-14

### Added
- **`docs/USAGE.md`** — a full syntax cookbook: copy-paste-ready scripts for every
  workflow (data preparation from arrays / long DataFrame / CSV, every option of
  every estimator, cross-validation, benchmarks, recursive forecasting, partial
  derivatives, poolability, LDPM, journal tables and figures, and a reference of
  what every result object contains).
- Expanded README quickstart with a data-preparation step and a benchmark
  comparison, plus prominent links to the cookbook.

### Changed
- No API changes; documentation only. Code is identical in behaviour to 0.1.0.

## [0.1.0] - 2026-07-14

### Added
- Initial release.
- `DeepPooledPanel`: deep pooled panel estimator with optional idiosyncratic
  component, direct multi-step forecasting, LASSO penalty, cross-validation over
  the paper grids.
- `partial_derivatives`, `poolability_test`.
- Benchmarks: `ARBenchmark`, `PanelVAR`, `LinearPooledPanel`, `DeepTimeSeries`.
- Evaluation: `diebold_mariano`, `fluctuation_test`, metrics; `rolling_forecast`.
- `LDPM`: surrogate augmentation, Deep Panel Training with classifier-LASSO
  homogeneity pursuit, within-group split conformal intervals; `surrogate_residuals`,
  `svd_reduce`, pooling helpers.
- `deeppanel.viz`: journal-style tables and figures with MATLAB colour maps.
- Simulation designs, pytest suite, and an equation-by-equation compatibility map.

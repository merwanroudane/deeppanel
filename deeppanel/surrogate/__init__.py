"""Surrogate model / residual construction and embedding utilities for LDPM."""
from .augment import surrogate_residuals
from .embeddings import svd_reduce, max_pool, mean_pool

__all__ = ["surrogate_residuals", "svd_reduce", "max_pool", "mean_pool"]

"""Cosine similarity helpers for TF-CV-DS vectors."""

from __future__ import annotations

import numpy as np

from .utils import cosine_similarity_dense


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0.0 when either is zero)."""
    return cosine_similarity_dense(a, b)


def pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """Full pairwise cosine matrix for row-vector matrix ``X``."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1)
    safe = np.maximum(norms, 1e-12)
    Xn = X / safe[:, None]
    S = Xn @ Xn.T
    zero = norms == 0.0
    S[zero, :] = 0.0
    S[:, zero] = 0.0
    return S

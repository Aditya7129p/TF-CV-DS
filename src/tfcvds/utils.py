"""Shared utilities for TF-CV-DS."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def chunk_sequence(tokens: Sequence[str], n_chunks: int) -> List[List[str]]:
    """Split a token sequence into ``n_chunks`` contiguous chunks.

    Chunks are as equal as possible (first chunks get one extra token
    when the length does not divide evenly). If there are fewer tokens
    than requested chunks, the number of chunks is reduced so no empty
    chunk is produced. An empty input returns an empty list.
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    n = len(tokens)
    if n == 0:
        return []
    c = min(n_chunks, n)
    base, rem = divmod(n, c)
    chunks: List[List[str]] = []
    start = 0
    for i in range(c):
        size = base + (1 if i < rem else 0)
        chunks.append(list(tokens[start : start + size]))
        start += size
    return chunks


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of ``X``; zero rows stay zero."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    out = X / norms
    out[np.linalg.norm(X, axis=1) == 0.0] = 0.0
    return out


def l1_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L1-normalize each row of ``X``; zero rows stay zero."""
    norms = np.abs(X).sum(axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    out = X / norms
    out[np.abs(X).sum(axis=1) == 0.0] = 0.0
    return out


def cosine_similarity_dense(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1-D vectors (0.0 if either is zero)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_finite_vector(v: np.ndarray) -> bool:
    """Return True when every entry of ``v`` is finite."""
    return bool(np.all(np.isfinite(v)))


def summarize_array(v: np.ndarray) -> dict:
    """Small helper for experiments: sparsity / shape statistics."""
    v = np.asarray(v, dtype=float)
    return {
        "nnz": int(np.count_nonzero(v)),
        "dim": int(v.size),
        "sparsity": float(1.0 - np.count_nonzero(v) / max(v.size, 1)),
        "min": float(np.min(v)) if v.size else 0.0,
        "max": float(np.max(v)) if v.size else 0.0,
    }

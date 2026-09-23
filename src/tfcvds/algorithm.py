"""Core TF-CV-DS mathematics.

Final production formula (per spec)::

    p_{t,c}   = f(t,c) / N_c              (relative frequency in chunk c)
    bar_p_t   = mean_c p_{t,c}
    sigma2_t  = sample variance (C - 1 denominator)
    sigma_t   = sqrt(sigma2_t)
    CV_t      = sigma_t / (bar_p_t + delta)
    W_final   = log(1 + TF(t,d)) / (CV_t + eps_cv)

Also provided for analysis: raw inverse-variance weight
``W_DS = 1 / (sigma2 + epsilon)`` (scale-biased) and the
scale-invariant ``W_CV_DS = 1 / (CV + eps_cv)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .utils import chunk_sequence


@dataclass
class TFCVDSConfig:
    """Hyperparameters for TF-CV-DS."""

    n_chunks: int = 8
    epsilon: float = 1e-8      # guards raw inverse-variance denominator
    delta: float = 1e-8        # guards CV denominator (mean -> 0)
    eps_cv: float = 1e-6       # guards stability inversion
    min_freq: int = 1          # per-document TF floor for ranking
    use_log_tf: bool = True    # False -> raw TF multiplier


@dataclass
class TermStats:
    """Per-term dispersion statistics for one document."""

    term: str
    tf: float
    mean: float
    var: float
    std: float
    cv: float
    w_ds_raw: float
    w_cv_ds: float
    w_final: float


def compute_term_stats(
    candidate_tokens: Sequence[str],
    config: TFCVDSConfig | None = None,
    weights: Sequence[float] | None = None,
) -> Dict[str, TermStats]:
    """Compute TF-CV-DS statistics for every distinct candidate term.

    Args:
        candidate_tokens: post-filter token stream (order preserved).
        config: hyperparameters (defaults to :class:`TFCVDSConfig`).
        weights: optional per-token count weights (e.g. soft-match
            confidence from :func:`tokenizer.map_oov_token`, so shaky
            matches contribute fractionally). Defaults to 1.0 each.

    Chunking uses fixed-token windows over the candidate stream
    (mathematically uniform). ``N_c`` is the candidate-token count of
    chunk ``c``; empty chunks contribute ``p = 0``.
    """
    cfg = config or TFCVDSConfig()
    tokens = list(candidate_tokens)
    if len(tokens) == 0:
        return {}
    if weights is None:
        wts = [1.0] * len(tokens)
    else:
        wts = list(weights)
        if len(wts) != len(tokens):
            raise ValueError("weights must parallel candidate_tokens")
    if cfg.n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")

    chunks = chunk_sequence(tokens, cfg.n_chunks)
    C = len(chunks)
    vocab = sorted(set(tokens))
    index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    counts = np.zeros((C, V), dtype=float)
    chunk_lens = np.zeros(C, dtype=float)
    offset = 0
    for ci, ch in enumerate(chunks):
        chunk_lens[ci] = len(ch)
        for w, wt in zip(ch, wts[offset : offset + len(ch)]):
            counts[ci, index[w]] += wt
        offset += len(ch)

    # Relative frequency per chunk; guard N_c == 0 -> p = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(chunk_lens[:, None] > 0, counts / np.maximum(chunk_lens[:, None], 1.0), 0.0)

    tf = counts.sum(axis=0)
    mean = P.mean(axis=0)
    if C >= 2:
        var = P.var(axis=0, ddof=1)  # unbiased sample variance
    else:
        # Single chunk: no dispersion information; stability is neutral.
        var = np.zeros(V)
    var = np.maximum(var, 0.0)  # numerical guard
    std = np.sqrt(var)
    cv = std / (mean + cfg.delta)

    w_ds_raw = 1.0 / (var + cfg.epsilon)
    if C < 2:
        # Avoid 1/epsilon blowup carrying meaning; neutral stability.
        w_ds_raw = np.ones(V)
        w_cv_ds = np.ones(V)
        cv = np.zeros(V)
    else:
        w_cv_ds = 1.0 / (cv + cfg.eps_cv)

    if cfg.use_log_tf:
        tf_mult = np.log1p(tf)
    else:
        tf_mult = tf
    # Terms absent from this doc cannot occur here (tf>0 always since
    # vocab is doc-derived), but keep the guard for vectorizer reuse.
    w_final = np.where(tf > 0, tf_mult * w_cv_ds, 0.0)

    stats: Dict[str, TermStats] = {}
    for t, i in index.items():
        stats[t] = TermStats(
            term=t,
            tf=float(tf[i]),
            mean=float(mean[i]),
            var=float(var[i]),
            std=float(std[i]),
            cv=float(cv[i]),
            w_ds_raw=float(w_ds_raw[i]),
            w_cv_ds=float(w_cv_ds[i]),
            w_final=float(w_final[i]),
        )
    return stats


def rank_terms(
    stats: Dict[str, TermStats],
    top_k: int | None = None,
    min_freq: int = 1,
) -> List[TermStats]:
    """Sort terms by ``w_final`` descending, applying a TF floor."""
    items = [s for s in stats.values() if s.tf >= min_freq]
    items.sort(key=lambda s: s.w_final, reverse=True)
    if top_k is not None:
        items = items[: max(top_k, 0)]
    return items


def score_document(
    candidate_tokens: Sequence[str],
    config: TFCVDSConfig | None = None,
    top_k: int | None = None,
) -> List[TermStats]:
    """One-call helper: stats + ranking for a single document."""
    cfg = config or TFCVDSConfig()
    stats = compute_term_stats(candidate_tokens, cfg)
    return rank_terms(stats, top_k=top_k, min_freq=cfg.min_freq)

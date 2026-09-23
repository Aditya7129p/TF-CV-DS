"""Edge-case tests: stability, degenerate inputs, determinism."""

import numpy as np

from tfcvds import TFCVDSConfig, TFCVDSVectorizer, compute_term_stats


def test_empty_tokens_no_crash():
    assert compute_term_stats([]) == {}


def test_single_chunk_neutral_stability():
    stats = compute_term_stats(["alpha", "beta"], TFCVDSConfig(n_chunks=1))
    for s in stats.values():
        assert s.w_final == float(np.log1p(s.tf))


def test_more_chunks_than_tokens():
    stats = compute_term_stats(["a", "b", "c"], TFCVDSConfig(n_chunks=10))
    assert len(stats) == 3
    assert all(np.isfinite(s.w_final) for s in stats.values())


def test_singleton_term_does_not_explode():
    toks = ["theme"] * 12 + ["once"] + ["pad"] * 12
    stats = compute_term_stats(toks, TFCVDSConfig(n_chunks=6))
    assert np.isfinite(stats["once"].w_final)
    assert stats["theme"].w_final > stats["once"].w_final


def test_determinism():
    toks = ["learning", "data", "models"] * 10
    s1 = compute_term_stats(toks, TFCVDSConfig(n_chunks=5))
    s2 = compute_term_stats(toks, TFCVDSConfig(n_chunks=5))
    assert {k: v.w_final for k, v in s1.items()} == {k: v.w_final for k, v in s2.items()}


def test_numerical_stability_long_repetition():
    vec = TFCVDSVectorizer(n_chunks=8)
    X = vec.fit_transform(["word " * 5000])
    assert np.isfinite(X).all()

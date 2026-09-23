"""Tests for core TF-CV-DS mathematics."""

import math

import numpy as np

from tfcvds import TFCVDSConfig, compute_term_stats, score_document
from tfcvds.tokenizer import candidate_tokens


def test_chunk_relative_frequency_math():
    # 8 candidates -> 4 chunks of 2: ["a b","a b","a b","a b"] fully stable
    toks = ["a", "b"] * 4
    stats = compute_term_stats(toks, TFCVDSConfig(n_chunks=4))
    assert stats["a"].tf == 4
    assert stats["a"].var == 0.0
    assert stats["a"].cv == 0.0
    assert math.isclose(stats["a"].w_final, math.log1p(4) / 1e-6, rel_tol=1e-6)


def test_stuffed_term_penalized():
    # Build the stream chunk-by-chunk with equal chunk sizes so the
    # fixed-window chunking aligns with logical chunks: 6 chunks of
    # 6 tokens; 'stable' appears 2x in every chunk, 'stuffed' 4x only
    # in chunk 0, 'pad' fills the remaining slots elsewhere.
    toks = []
    for c in range(6):
        toks += ["stable"] * 2
        if c == 0:
            toks += ["stuffed"] * 4
        else:
            toks += ["pad"] * 4
    stats = compute_term_stats(toks, TFCVDSConfig(n_chunks=6))
    assert stats["stable"].cv < stats["stuffed"].cv
    assert stats["stable"].w_final > stats["stuffed"].w_final


def test_log_tf_dampening():
    toks = ["w"] * 20 + ["v"] * 5 + ["pad"] * 30
    stats = compute_term_stats(toks, TFCVDSConfig(n_chunks=5))
    ratio_raw = stats["w"].tf / stats["v"].tf
    ratio_w = stats["w"].w_final / max(stats["v"].w_final, 1e-12)
    assert ratio_w < ratio_raw  # log dampening compresses the gap


def test_scores_finite_and_nonneg():
    toks = candidate_tokens("Machine learning enables learning from data every day")
    stats = compute_term_stats(toks, TFCVDSConfig(n_chunks=3))
    for s in stats.values():
        for v in (s.mean, s.var, s.std, s.cv, s.w_ds_raw, s.w_cv_ds, s.w_final):
            assert math.isfinite(v) and v >= 0.0


def test_fit_transform_shapes():
    from tfcvds import TFCVDSVectorizer

    docs = ["cat sat mat cat dog", "dog barked loudly at night"]
    vec = TFCVDSVectorizer(n_chunks=3)
    X = vec.fit_transform(docs)
    assert X.shape[0] == 2 and X.shape[1] == len(vec.get_feature_names_out())


def test_vocabulary_and_repeated_words():
    from tfcvds import TFCVDSVectorizer

    vec = TFCVDSVectorizer(n_chunks=4)
    X = vec.fit_transform(["apple apple apple banana"])
    names = vec.get_feature_names_out()
    assert "apple" in names and X[0, names.index("apple")] > 0


def test_similarity_identical_and_orthogonal():
    from tfcvds import TFCVDSVectorizer

    vec = TFCVDSVectorizer(n_chunks=4)
    vec.fit(["cats dogs birds fish", "quantum chromodynamics gauge theory"])
    assert vec.similarity("cats dogs birds", "cats dogs birds") > 0.99
    assert vec.similarity("cats dogs birds", "quantum chromodynamics gauge") < 0.5

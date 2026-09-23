"""Tests for the vectorizer API."""

import numpy as np

from tfcvds import TFCVDSVectorizer


def test_basic_fit_transform():
    vec = TFCVDSVectorizer(n_chunks=4)
    docs = ["machine learning data", "football sport ball"]
    X = vec.fit_transform(docs)
    assert X.shape == (2, len(vec.get_feature_names_out()))
    assert (X >= 0).all()


def test_transform_unknown_words_zero():
    vec = TFCVDSVectorizer(n_chunks=4)
    vec.fit(["machine learning data"])
    X = vec.transform(["zyxqwv unknownword"])
    assert (X == 0).all()


def test_duplicate_documents_equal_vectors():
    vec = TFCVDSVectorizer(n_chunks=4)
    X = vec.fit_transform(["dogs cats birds fish pets"] * 2)
    np.testing.assert_allclose(X[0], X[1])


def test_single_document_corpus():
    vec = TFCVDSVectorizer(n_chunks=5)
    X = vec.fit_transform(["only one document here with content words"])
    assert X.shape[0] == 1 and np.isfinite(X).all()


def test_empty_document_row_is_zero():
    vec = TFCVDSVectorizer(n_chunks=4)
    vec.fit(["some content words here"])
    X = vec.transform([""])
    assert (X == 0).all()
    X2 = vec.transform(["the and of"])
    assert (X2 == 0).all()


def test_very_long_document():
    vec = TFCVDSVectorizer(n_chunks=8)
    long_doc = " ".join(["learning data models"] * 2000)
    X = vec.fit_transform([long_doc, "football sport"])
    assert np.isfinite(X).all() and X.shape[0] == 2

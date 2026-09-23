"""Tests for stemming and OOV soft-matching."""

from tfcvds import TFCVDSVectorizer
from tfcvds.tokenizer import (
    char_trigrams,
    map_oov_token,
    stem_token,
    trigram_jaccard,
)


def test_stem_inflections():
    assert stem_token("reactions") == "reaction"
    assert stem_token("compounds") == "compound"
    assert stem_token("players") == "player"
    assert stem_token("studied") == "study"
    assert stem_token("stories") == "story"
    # Derivations intentionally NOT merged by the stemmer.
    assert stem_token("chemical") != stem_token("chemistry")


def test_stem_short_and_nondestructive():
    assert stem_token("bat") == "bat"
    assert stem_token("data") == "data"
    assert stem_token("bus") == "bus"


def test_trigram_jaccard_known_values():
    assert trigram_jaccard("chemical", "chemistry") > 0.25
    assert trigram_jaccard("play", "player") >= 0.5
    assert trigram_jaccard("space", "sauce") == 0.0


def test_map_oov_prefix_compound():
    term, conf = map_oov_token("batsman", {"bat", "ball", "game"}, threshold=0.5)
    assert term == "bat" and conf == 0.8


def test_map_oov_derivation():
    vocab = {"chemistry", "microscope", "cells"}
    term, conf = map_oov_token("chemical", vocab, threshold=0.25)
    assert term == "chemistry"
    term, _ = map_oov_token("chemical", vocab, threshold=0.5)
    assert term is None  # 0.30 < 0.5: abstains at the strict threshold


def test_map_oov_abstains_on_unrelated():
    term, conf = map_oov_token("zyxqwv", {"bat", "ball"}, threshold=0.5)
    assert term is None and conf == 0.0


def test_map_oov_deterministic_tiebreak():
    # Both candidates score identically; alphabetical order decides.
    term, _ = map_oov_token("abcx", {"abcq", "abcw"}, threshold=0.0)
    assert term == "abcq"


def test_soft_match_fixes_morphological_unknown():
    vec = TFCVDSVectorizer(n_chunks=4, soft_match=True, trigram_threshold=0.5)
    vec.fit(["cricket bat ball game play match"])
    X = vec.transform(["the batsman hit a six"])
    assert X.sum() > 0  # baseline (exact match only) gives all zeros


def test_soft_match_off_abstains():
    vec = TFCVDSVectorizer(n_chunks=4)
    vec.fit(["cricket bat ball game play match"])
    X = vec.transform(["the batsman hit a six"])
    assert X.sum() == 0


def test_fractional_weight_below_tf_floor_gate():
    # A single low-confidence soft match must still pass the occurrence
    # gate (min_freq counts occurrences, not confidence mass).
    vec = TFCVDSVectorizer(
        n_chunks=2, soft_match=True, trigram_threshold=0.5, min_freq=2
    )
    vec.fit(["cricket bat ball game play match"])
    X = vec.transform(["batsman"])  # one occurrence -> below floor of 2
    assert X.sum() == 0
    X2 = vec.transform(["batsman batsman"])
    assert X2.sum() > 0

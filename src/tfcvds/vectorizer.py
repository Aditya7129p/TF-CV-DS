"""Scikit-learn-style vectorizer wrapping TF-CV-DS.

The score itself is purely intra-document (no IDF / no corpus needed).
``fit`` only builds a shared vocabulary so documents can be compared as
aligned vectors; all weights are computed per document independently.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .algorithm import TFCVDSConfig, compute_term_stats
from .similarity import cosine_sim
from .tokenizer import candidate_tokens, map_oov_token
from .utils import l1_normalize_rows, l2_normalize_rows


class TFCVDSVectorizer:
    """Transform documents into TF-CV-DS vectors.

    Args:
        n_chunks: number of sub-document chunks (recommended 5..10).
        epsilon: guards raw inverse-variance denominator.
        delta: guards CV denominator.
        eps_cv: guards stability inversion.
        min_freq: per-document TF floor (terms below it get weight 0).
        min_df: global document-frequency floor for vocabulary pruning.
        use_log_tf: use ``log(1+TF)`` multiplier (recommended).
        norm: ``'l2'``, ``'l1'`` or ``None`` row normalization.
        lowercase, min_len, drop_numeric, drop_adverbs, use_nltk_pos:
            forwarded to :func:`tokenizer.candidate_tokens`.
        stemming: stem tokens before filtering (fixes inflectional OOV:
            ``reactions`` -> ``reaction``).
        soft_match: map remaining OOV tokens onto vocabulary via prefix
            containment + char-trigram Jaccard (fixes compounds like
            ``batsman`` -> ``bat`` and derivations like
            ``chemical`` -> ``chemistry``); unmapped tokens are dropped.
        trigram_threshold: minimum Jaccard for a soft match.
    """

    def __init__(
        self,
        n_chunks: int = 8,
        epsilon: float = 1e-8,
        delta: float = 1e-8,
        eps_cv: float = 1e-6,
        min_freq: int = 1,
        min_df: int = 1,
        use_log_tf: bool = True,
        norm: Optional[str] = "l2",
        lowercase: bool = True,
        min_len: int = 2,
        drop_numeric: bool = True,
        drop_adverbs: bool = True,
        use_nltk_pos: bool = False,
        stemming: bool = False,
        soft_match: bool = False,
        trigram_threshold: float = 0.5,
    ) -> None:
        if n_chunks < 1:
            raise ValueError("n_chunks must be >= 1")
        if norm not in (None, "l2", "l1"):
            raise ValueError("norm must be 'l2', 'l1' or None")
        self.config = TFCVDSConfig(
            n_chunks=n_chunks,
            epsilon=epsilon,
            delta=delta,
            eps_cv=eps_cv,
            min_freq=min_freq,
            use_log_tf=use_log_tf,
        )
        self.min_df = min_df
        self.norm = norm
        self.lowercase = lowercase
        self.min_len = min_len
        self.drop_numeric = drop_numeric
        self.drop_adverbs = drop_adverbs
        self.use_nltk_pos = use_nltk_pos
        self.stemming = stemming
        self.soft_match = soft_match
        self.trigram_threshold = trigram_threshold
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self._fitted = False

    # -- internal helpers -------------------------------------------------
    def _candidates(self, doc: str) -> List[str]:
        return candidate_tokens(
            doc,
            lowercase=self.lowercase,
            min_len=self.min_len,
            drop_numeric=self.drop_numeric,
            drop_adverbs=self.drop_adverbs,
            use_nltk_pos=self.use_nltk_pos,
            stemming=self.stemming,
        )

    def _map_oov(self, toks: List[str]) -> tuple[List[str], List[float]]:
        """Map OOV tokens onto vocabulary with confidence weights.

        Returns ``(mapped_tokens, weights)``: exact hits weigh 1.0,
        prefix/trigram matches weigh their confidence (see
        :func:`tokenizer.map_oov_token`), unmapped tokens are dropped
        (abstention). No-op returning all-1.0 weights when
        ``soft_match`` is off or the vectorizer is unfitted.
        """
        if not self.soft_match or not self._fitted:
            return toks, [1.0] * len(toks)
        vocab = set(self.vocabulary_)
        out: List[str] = []
        wts: List[float] = []
        for t in toks:
            if t in vocab:
                out.append(t)
                wts.append(1.0)
                continue
            m, conf = map_oov_token(t, vocab, threshold=self.trigram_threshold)
            if m is not None:
                out.append(m)
                wts.append(conf)
        return out, wts

    def _vectorize_tokens(self, toks: List[str]) -> np.ndarray:
        dim = len(self.feature_names_)
        vec = np.zeros(dim, dtype=float)
        if not toks or dim == 0:
            return vec
        toks, wts = self._map_oov(toks)
        if not toks:
            return vec
        stats = compute_term_stats(toks, self.config, weights=wts)
        occ = Counter(toks)
        for term, st in stats.items():
            j = self.vocabulary_.get(term)
            if j is None:
                continue
            if occ[term] < self.config.min_freq:
                continue
            vec[j] = st.w_final
        return vec

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.norm == "l2":
            return l2_normalize_rows(X)
        if self.norm == "l1":
            return l1_normalize_rows(X)
        return X

    # -- public API --------------------------------------------------------
    def fit(self, documents: Sequence[str]) -> "TFCVDSVectorizer":
        """Build vocabulary from a corpus (weights stay per-document)."""
        df: Dict[str, int] = {}
        for doc in documents:
            for t in set(self._candidates(doc)):
                df[t] = df.get(t, 0) + 1
        names = sorted(t for t, c in df.items() if c >= self.min_df)
        self.feature_names_ = names
        self.vocabulary_ = {t: i for i, t in enumerate(names)}
        self._fitted = True
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transform documents to TF-CV-DS vectors."""
        if not self._fitted:
            raise RuntimeError("Vectorizer is not fitted; call fit() first.")
        X = np.zeros((len(documents), len(self.feature_names_)), dtype=float)
        for i, doc in enumerate(documents):
            X[i] = self._vectorize_tokens(self._candidates(doc))
        return self._normalize(X)

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit vocabulary and return document vectors."""
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self) -> List[str]:
        """Return vocabulary in column order."""
        if not self._fitted:
            raise RuntimeError("Vectorizer is not fitted.")
        return list(self.feature_names_)

    def extract_keywords(self, doc: str, top_k: int = 10) -> List[tuple]:
        """Rank keywords of a single document as ``(term, score)``."""
        toks = self._candidates(doc)
        if not toks:
            return []
        toks, wts = self._map_oov(toks)
        if not toks:
            return []
        stats = compute_term_stats(toks, self.config, weights=wts)
        occ = Counter(toks)
        ranked = sorted(stats.values(), key=lambda s: s.w_final, reverse=True)
        ranked = [s for s in ranked if occ[s.term] >= self.config.min_freq]
        return [(s.term, s.w_final) for s in ranked[: max(top_k, 0)]]

    def similarity(self, doc_a: str, doc_b: str) -> float:
        """Cosine similarity between two raw documents."""
        X = self.transform([doc_a, doc_b])
        return cosine_sim(X[0], X[1])

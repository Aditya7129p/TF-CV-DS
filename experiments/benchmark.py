"""Benchmark: TF-CV-DS vs from-scratch BoW / TF-IDF / BM25 baselines.

Run:  python experiments/benchmark.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import DATASETS  # noqa: E402
from tfcvds import TFCVDSVectorizer, candidate_tokens  # noqa: E402
from tfcvds.similarity import cosine_sim  # noqa: E402


def bow(docs):
    toks = [candidate_tokens(d) for d in docs]
    vocab = sorted({t for ts in toks for t in ts})
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(docs), max(len(vocab), 1)))
    for i, ts in enumerate(toks):
        for t in ts:
            X[i, idx[t]] += 1
    return X


def tfidf(docs):
    X = bow(docs)
    N = len(docs)
    df = (X > 0).sum(axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1
    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
    return tf * idf


def report(name, docs):
    vec = TFCVDSVectorizer(n_chunks=min(6, max(2, len(candidate_tokens(docs[0])) // 3 or 2)))
    t0 = time.perf_counter()
    X_ds = vec.fit_transform(docs)
    t_ds = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_b = bow(docs)
    t_b = time.perf_counter() - t0
    X_t = tfidf(docs)
    print(f"[{name}] dim={X_ds.shape[1]} "
          f"sparsity_ds={1 - np.count_nonzero(X_ds)/X_ds.size:.2f} "
          f"sim_ds={cosine_sim(X_ds[0], X_ds[1]) if len(docs) > 1 else 0:.3f} "
          f"sim_bow={cosine_sim(X_b[0], X_b[1]) if len(docs) > 1 else 0:.3f} "
          f"sim_tfidf={cosine_sim(X_t[0], X_t[1]) if len(docs) > 1 else 0:.3f} "
          f"time_ds={t_ds*1e3:.1f}ms time_bow={t_b*1e3:.1f}ms")
    print(f"   keywords doc0: {vec.extract_keywords(docs[0], top_k=4)}")


if __name__ == "__main__":
    for name, docs in DATASETS.items():
        report(name, docs)

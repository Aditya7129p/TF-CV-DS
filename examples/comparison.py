"""Fair from-scratch comparison: BoW vs TF-IDF vs BM25 vs TF-CV-DS.

All baselines share the SAME preprocessing (candidate_tokens) as TF-CV-DS.
Baseline formulas are implemented explicitly (no sklearn TFIDF vectorizer).
"""

import time

import numpy as np

from tfcvds import TFCVDSVectorizer, candidate_tokens
from tfcvds.similarity import cosine_sim


def bow_vectors(docs):
    toks = [candidate_tokens(d) for d in docs]
    vocab = sorted({t for ts in toks for t in ts})
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, ts in enumerate(toks):
        for t in ts:
            X[i, idx[t]] += 1.0
    return X, vocab


def tfidf_vectors(docs):
    X, vocab = bow_vectors(docs)
    N = len(docs)
    df = (X > 0).sum(axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1.0
    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    return tf * idf, vocab


def bm25_scores(query, docs, k1=1.5, b=0.75):
    toks = [candidate_tokens(d) for d in docs]
    q = candidate_tokens(query)
    N = len(docs)
    df = {}
    for ts in toks:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1
    lens = [len(ts) for ts in toks]
    avgdl = sum(lens) / max(len(lens), 1)
    out = []
    for ts in toks:
        s = 0.0
        for t in q:
            f = ts.count(t)
            if f == 0:
                continue
            idf = np.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
            denom = f + k1 * (1 - b + b * len(ts) / max(avgdl, 1))
            s += idf * f * (k1 + 1) / denom
        out.append(s)
    return out


if __name__ == "__main__":
    docs = [
        "machine learning models learn patterns from data every day with training",
        "deep learning neural networks learn patterns from data with optimization",
        "football sport ball field stadium fans weekend match goals",
    ]
    t0 = time.perf_counter()
    X_bow, _ = bow_vectors(docs)
    t_bow = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_tfidf, _ = tfidf_vectors(docs)
    t_tfidf = time.perf_counter() - t0
    t0 = time.perf_counter()
    vec = TFCVDSVectorizer(n_chunks=4)
    X_ds = vec.fit_transform(docs)
    t_ds = time.perf_counter() - t0

    def cos01(X):
        n = X.shape[0]
        nrm = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Xn = X / nrm
        return float(Xn[0] @ Xn[1]), float(Xn[0] @ Xn[2])

    print("BoW    sim(0,1), sim(0,2):", cos01(X_bow), f"{t_bow*1e3:.2f}ms")
    print("TF-IDF sim(0,1), sim(0,2):", cos01(X_tfidf), f"{t_tfidf*1e3:.2f}ms")
    print("TF-CV-DS sim(0,1), sim(0,2):", cos01(X_ds), f"{t_ds*1e3:.2f}ms")
    print("BM25 query=doc0 scores:", [round(s, 3) for s in bm25_scores(docs[0], docs)])
    print("\nTF-CV-DS keywords doc0:", vec.extract_keywords(docs[0], top_k=5))

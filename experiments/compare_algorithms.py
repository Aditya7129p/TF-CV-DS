"""Head-to-head: default NLP representations vs TF-CV-DS.

Same preprocessing (candidate_tokens) and same classifier (nearest
centroid, cosine; zero vector -> 'unknown') for every model — only the
vectors differ. All baselines implemented from scratch in this file.

Models: BoW, Binary BoW, TF, TF-IDF, Sublinear TF-IDF, BM25-weighted
vectors, TF-CV-DS (plain defaults: no stemming, no soft-match, no
topic-IDF — the extensions are ablated elsewhere).

Test sets (imported, frozen, never added to training):
  IN_VOCAB (n=18): familiar vocabulary.
  HARD/OOV  (n=27): unseen-but-related wording.

Run:  python experiments/compare_algorithms.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classification_eval import HARD, IN_VOCAB, PROTOTYPES  # noqa: E402
from tfcvds import TFCVDSVectorizer, candidate_tokens  # noqa: E402
from tfcvds.similarity import cosine_sim  # noqa: E402


# --------------------------------------------------------------------------
# Shared preprocessing + vocabulary
# --------------------------------------------------------------------------
def prepare(topics):
    """Return (flat_docs, labels, token_lists) for prototype topics."""
    docs, labels = [], []
    for topic, ds in topics.items():
        docs.extend(ds)
        labels.extend([topic] * len(ds))
    return docs, labels, [candidate_tokens(d) for d in docs]


def build_vocab(train_toks, min_df=1):
    df = {}
    for ts in train_toks:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1
    names = sorted(t for t, c in df.items() if c >= min_df)
    return names, {t: i for i, t in enumerate(names)}, df


def count_matrix(toks_list, idx):
    X = np.zeros((len(toks_list), len(idx)))
    for i, ts in enumerate(toks_list):
        for t in ts:
            j = idx.get(t)
            if j is not None:
                X[i, j] += 1.0
    return X


# --------------------------------------------------------------------------
# Model vectorizers: each exposes fit(train_toks) -> state and
# transform(toks_list) -> matrix. State carries train-only statistics.
# --------------------------------------------------------------------------
class BoW:
    name = "BoW"

    def fit(self, train_toks, vocab, df):
        self.vocab = vocab
        return self

    def transform(self, toks_list):
        return count_matrix(toks_list, self.vocab)


class BinaryBoW:
    name = "BinaryBoW"

    def fit(self, train_toks, vocab, df):
        self.vocab = vocab
        return self

    def transform(self, toks_list):
        return (count_matrix(toks_list, self.vocab) > 0).astype(float)


class TF:
    name = "TF"

    def fit(self, train_toks, vocab, df):
        self.vocab = vocab
        return self

    def transform(self, toks_list):
        X = count_matrix(toks_list, self.vocab)
        return X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)


class TFIDF:
    name = "TF-IDF"

    def fit(self, train_toks, vocab, df):
        self.vocab = vocab
        N = len(train_toks)
        self.idf = np.array(
            [np.log((1 + N) / (1 + df.get(t, 0))) + 1.0 for t in sorted(vocab)]
        )
        return self

    def transform(self, toks_list):
        X = count_matrix(toks_list, self.vocab)
        tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        return tf * self.idf


class SublinearTFIDF:
    name = "SublinearTF-IDF"

    def fit(self, train_toks, vocab, df):
        self.vocab = vocab
        N = len(train_toks)
        self.idf = np.array(
            [np.log((1 + N) / (1 + df.get(t, 0))) + 1.0 for t in sorted(vocab)]
        )
        return self

    def transform(self, toks_list):
        X = count_matrix(toks_list, self.vocab)
        sub = np.zeros_like(X)
        nz = X > 0
        sub[nz] = 1.0 + np.log(X[nz])
        return sub * self.idf


class BM25Vec:
    name = "BM25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, train_toks, vocab, df):
        self.vocab = vocab
        N = len(train_toks)
        self.idf = np.array(
            [
                np.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
                for t in sorted(vocab)
            ]
        )
        lens = np.array([len(ts) for ts in train_toks], dtype=float)
        self.avgdl = float(lens.mean()) if len(lens) else 1.0
        return self

    def _weigh(self, X, lens):
        denom = X + self.k1 * (
            1 - self.b + self.b * lens[:, None] / max(self.avgdl, 1e-9)
        )
        return self.idf * X * (self.k1 + 1) / np.maximum(denom, 1e-12)

    def transform(self, toks_list):
        X = count_matrix(toks_list, self.vocab)
        lens = np.array([len(ts) for ts in toks_list], dtype=float)
        return self._weigh(X, lens)


class TFCVDS:
    name = "TF-CV-DS"

    def fit(self, train_docs, vocab=None, df=None):
        # train_docs: raw strings (vectorizer tokenizes internally).
        self.model = TFCVDSVectorizer(n_chunks=4)
        self.X_train = self.model.fit_transform(train_docs)
        return self

    def transform(self, raw_docs):
        return self.model.transform(raw_docs)


# --------------------------------------------------------------------------
# Nearest-centroid evaluation (identical for every model)
# --------------------------------------------------------------------------
def evaluate(model, train_toks, train_labels, topics, dataset, raw_train=None):
    if isinstance(model, TFCVDS):
        model.fit(raw_train)
        X_train = model.X_train
    else:
        names, idx, df = build_vocab(train_toks)
        model.fit(train_toks, idx, df)
        X_train = model.transform(train_toks)
        names = sorted(idx)

    centroids = {}
    for topic in topics:
        rows = X_train[[i for i, lab in enumerate(train_labels) if lab == topic]]
        centroids[topic] = rows.mean(axis=0)

    correct = wrong = unknown = 0
    for text, expected in dataset:
        if isinstance(model, TFCVDS):
            v = model.transform([text])[0]
        else:
            v = model.transform([candidate_tokens(text)])[0]
        if v.sum() == 0:
            unknown += 1
            continue
        scores = {t: float(cosine_sim(v, c)) for t, c in centroids.items()}
        pred = max(scores, key=scores.get)
        if pred == expected:
            correct += 1
        else:
            wrong += 1
    total = len(dataset)
    covered = total - unknown
    return {
        "acc": correct / total,
        "cov": covered / total,
        "unk": unknown / total,
        "conf": wrong / total,
        "acc_cov": correct / max(covered, 1),
        "dim": X_train.shape[1],
        "sparsity": 1.0 - np.count_nonzero(X_train) / X_train.size,
    }


def main():
    docs, labels, train_toks = prepare(PROTOTYPES)
    models = [BoW(), BinaryBoW(), TF(), TFIDF(), SublinearTFIDF(), BM25Vec(), TFCVDS()]
    rows = []
    for model in models:
        t0 = time.perf_counter()
        iv = evaluate(model, train_toks, labels, PROTOTYPES, IN_VOCAB, raw_train=docs)
        h = evaluate(model, train_toks, labels, PROTOTYPES, HARD, raw_train=docs)
        dt = (time.perf_counter() - t0) * 1e3
        rows.append((model.name, iv, h, dt))

    header = (
        f"{'model':16s} {'IN_acc':>7s} {'HARD_acc':>8s} {'HARD_cov':>8s} "
        f"{'HARD_unk':>8s} {'HARD_conf':>9s} {'dim':>4s} {'spars':>6s} {'ms':>7s}"
    )
    print(header)
    print("-" * len(header))
    out = ["# Default algorithms vs TF-CV-DS (shared preprocessing + centroid classifier)\n", "", header, "-" * len(header)]
    for name, iv, h, dt in rows:
        line = (
            f"{name:16s} {iv['acc']:7.3f} {h['acc']:8.3f} {h['cov']:8.3f} "
            f"{h['unk']:8.3f} {h['conf']:9.3f} {h['dim']:4d} {h['sparsity']:6.2f} {dt:7.1f}"
        )
        print(line)
        out.append(line)
    out += [
        "",
        "IN_VOCAB n=18, HARD n=27. 'unknown' = zero vector (no vocab overlap).",
        "TF-CV-DS here uses plain defaults (no stemming/soft-match/topic-IDF).",
    ]
    Path(__file__).resolve().parent.joinpath("results", "algorithm_comparison.md").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("\nSaved experiments/results/algorithm_comparison.md")


if __name__ == "__main__":
    main()

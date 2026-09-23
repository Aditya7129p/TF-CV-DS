# TF-CV-DS — Distributional Stability Keyword Extraction

> **Research prototype — potentially novel, not established.** A corpus-free
> term representation that rewards terms spread *evenly* across a document.
> Package: `tfcvds` · Python ≥ 3.9 · NumPy/SciPy/pandas/matplotlib only.

TF-IDF asks *"how rare is this word in the collection?"* TF-CV-DS asks a
different question: *"how steadily does this word appear throughout this one
document?"* Terms that are both **prominent** and **stably distributed** form
the thematic spine of long documents — and terms stuffed into a single
paragraph are explicitly penalised.

## Contents

- [Problem & Motivation](#problem--motivation)
- [Intuition](#intuition)
- [Mathematical Formulation](#mathematical-formulation)
- [Algorithm & Pseudocode](#algorithm--pseudocode)
- [Installation](#installation)
- [Quick Start & Code Example](#quick-start--code-example)
- [API Reference](#api-reference)
- [Comparison with BoW / TF-IDF / BM25](#comparison-with-bow--tf-idf--bm25)
- [Experiments & Results](#experiments--results)
- [Complexity](#complexity)
- [Limitations & Failure Cases](#limitations--failure-cases)
- [Project Structure](#project-structure)
- [Research Status & Future Work](#research-status--future-work)
- [License](#license)

## Problem & Motivation

Classical sparse representations derive global term importance from
*collection* statistics (document frequency). That leaves three gaps for
single-document analysis:

1. **No corpus, no weights** — one paper/report/book alone cannot be scored.
2. **Repetition is rewarded without bound** — TF-IDF has no saturation story;
   keyword stuffing scores highly.
3. **Distribution shape is ignored** — burstiness models reward spikes, but
   nothing rewards the opposite: persistent, even coverage (the theme).

## Intuition

Split the document into chunks. A theme word (*learning* in an ML paper)
shows similar density in every chunk (low variance). A stuffed or incidental
word (*keyword* dumped in one paragraph) is dense in one chunk and absent
elsewhere (high variance). Score = prominence × steadiness. Stop-word
filtering is mandatory beforehand — otherwise function words, the most
uniform tokens in any language, win by definition.

## Mathematical Formulation

$$p_{t,c} = \frac{f(t,c)}{N_c} \qquad
\bar p_t = \frac{1}{C}\sum_c p_{t,c} \qquad
\sigma^2_t = \frac{1}{C-1}\sum_c (p_{t,c}-\bar p_t)^2$$

$$\mathrm{CV}_t = \frac{\sigma_t}{\bar p_t+\delta} \qquad
\boxed{W(t,d)=\log\big(1+\mathrm{TF}(t,d)\big)\cdot\frac{1}{\mathrm{CV}_t+\epsilon_{CV}}}$$

The coefficient of variation (not raw inverse variance) makes steadiness
comparable across high- and low-frequency terms. Document vector:
$R(d)=[W(t_1,d),\dots,W(t_M,d)]$, optionally L2/L1-normalised. Full
derivation, guards ($\epsilon,\delta,\epsilon_{CV}$), and edge-case table:
`docs/mathematical_formulation.md`.

## Algorithm & Pseudocode

```text
INPUT:  document string d, config (C, eps, delta, eps_cv, min_freq)
OUTPUT: ranked keywords / vector R(d)

1. toks = candidate_tokens(d)        # stop/pronoun/adverb removal (+POS approx)
2. if toks empty: return empty
3. chunks = split toks into C contiguous fixed-token windows
4. p[t,c] = count(t in c) / max(len(c), 1)      # per-chunk density
5. bar = mean_c p; var = sample variance (0 if C == 1)
   CV = sqrt(var) / (bar + delta); stab = 1 / (CV + eps_cv)
6. W[t] = log(1 + TF[t]) * stab      # 0 if TF < min_freq
7. sort by W; align to global vocab for vectors
```

Classifier extensions (see `examples/text_to_text.py`): OOV soft-matching
(prefix + char-trigram Jaccard ≥ 0.5, confidence-weighted counts),
topic-IDF reweighting, near-tie reject option, opt-in stemmer.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start & Code Example

```python
from tfcvds import TFCVDSVectorizer

documents = [
    "machine learning models learn patterns from data every day",
    "deep learning neural networks learn patterns from data",
    "football sport ball field stadium fans weekend match",
]

model = TFCVDSVectorizer(n_chunks=5)
X = model.fit_transform(documents)          # (3, vocab) L2-normalised array

print(model.get_feature_names_out())
print(model.extract_keywords(documents[0], top_k=5))
# [('learning', ...), ('machine', ...), ('data', ...), ...]
print(model.similarity(documents[0], documents[1]))  # 0.77 (related)
print(model.similarity(documents[0], documents[2]))  # 0.08 (unrelated)
```

More: `examples/basic_example.py`, `examples/document_similarity.py`,
`examples/comparison.py`, `examples/text_to_text.py` (text-in → topic label).

## API Reference

```python
TFCVDSVectorizer(n_chunks=8, epsilon=1e-8, delta=1e-8, eps_cv=1e-6,
                 min_freq=1, min_df=1, use_log_tf=True, norm='l2',
                 lowercase=True, min_len=2, drop_numeric=True,
                 drop_adverbs=True, use_nltk_pos=False,
                 stemming=False, soft_match=False, trigram_threshold=0.5)
    .fit(docs)                  # build shared vocab (weights stay per-doc)
    .transform(docs)            # -> np.ndarray, rows normalised per `norm`
    .fit_transform(docs)
    .get_feature_names_out()    # vocab in column order
    .extract_keywords(doc, top_k=10)  # [(term, score), ...]
    .similarity(doc_a, doc_b)   # cosine float
```

Lower level: `candidate_tokens`, `stem_token`, `map_oov_token`,
`compute_term_stats` (+`weights=` for confidence-weighted counts),
`score_document`, `cosine_sim`, `pairwise_cosine`. See docstrings.

## Comparison with BoW / TF-IDF / BM25

`experiments/compare_algorithms.py` pits 7 representations against each
other with **identical preprocessing and an identical centroid classifier**
(all baselines re-implemented from scratch):

| model | IN_VOCAB acc (n=18) | HARD acc (n=27) | HARD unknown |
|---|---|---|---|
| BoW / Binary / TF / TF-IDF / Sublinear / BM25 / TF-CV-DS (plain) | 1.000 | 0.519 | 0.444 |

The tie *is* the finding: on short single-evidence texts, weighting schemes
can't separate — coverage decides, and all lexical models go blind together.
TF-CV-DS pulls ahead via its extensions (OOV soft-match: HARD 0.519 →
0.630) and in long-document behaviour (below), not in plain short-text
weighting. Details: `experiments/results/algorithm_comparison.md`.

- **vs BoW:** adds a global importance signal; repetition dampened via
  log-TF and stability instead of raw counts.
- **vs TF-IDF:** no corpus needed; rare-typo overweighting fixed by
  TF-floor + confidence shrinkage; repetition penalised when localised.
- **vs BM25:** BM25 remains the stronger tuned retrieval scorer; TF-CV-DS
  is a document *representation* (symmetric vectors, no query side, no
  $k_1/b$ tuning) with an explicit anti-stuffing property BM25 lacks.

## Experiments & Results

| Harness | Result |
|---|---|
| `pytest` (29 tests: math, API, edge cases, fuzz) | **29/29 pass** |
| Synthetic benchmark (7 probes: repetition, rare/noise, stuffing, length, subtopic…) | sane; length-invariance exact (1.000); stuffing *dampened* (30× repeat still leads 4.43 vs 1.42 — partial, reported) |
| 20-topic × 3 demo classifier | **59/60** (baseline 58/60, 2 abstentions) |
| Frozen IN_VOCAB / HARD eval | 1.000 / **0.630** (from 0.519); unknown 0.444 → 0.296 |
| Ablation | soft-match@0.5 + topic-IDF help; **stemming rejected** for crowded classifiers (59/60 → 56/60); threshold 0.5 halves confusion vs 0.25 |

Run everything: `python -m pytest tests -v`,
`python examples/text_to_text.py` (prints DEMO ACCURACY),
`python experiments/benchmark.py`,
`python experiments/classification_eval.py`.
Full logs: `experiments/results/REPORT.md`. Full paper: `docs/research_paper.md`.

## Complexity

Fit $O(\sum L)$; transform $O(L+V_d)$ per doc (+$O(U{\cdot}M)$ trigram scan
only with `soft_match` on); similarity $O(M)$; memory $O(M)$ per vector.
Same order as TF-IDF; no co-occurrence matrix. Measured: ~0.5 ms/transform
vs 0.1–0.2 ms for BoW/TF-IDF at demo scale.

## Limitations & Failure Cases

- **Lexical ceiling:** pure synonyms with no surface overlap (`six`,
  `overs`) are invisible; perfect evidence ties (`bat`→sports vs
  `player`→gaming, both 0.8/​df=1) are unsolvable without world knowledge.
- **Short texts:** ~8 tokens over 4 chunks makes CV estimates noisy.
- **Subtopic penalty:** section-localised keywords score ~⅓ of spine terms
  — the burstiness view is right there; our polarity is wrong by design.
- **Stuffing suppression is partial** (see numbers above).
- No synonymy/polysemy handling; $C$, TF floor, threshold need choosing;
  trigram scan needs indexing at large vocabularies.

## Project Structure

```text
tf-cv-ds/
├── README.md  pyproject.toml  requirements.txt
├── src/tfcvds/          # tokenizer, algorithm, vectorizer, similarity, utils
├── examples/            # basic, similarity, baseline comparison, text-to-text
├── tests/               # 29 tests (algorithm, vectorizer, edge cases, tokenizer)
├── experiments/         # benchmark, datasets, compare_algorithms,
│   └── results/         # classification eval, REPORT, comparison logs
└── docs/                # formulation, algorithm, research notes, paper
```

## Research Status & Future Work

Proposed method; closest prior art to check: Juilland's $D$ / Gries DP
dispersion, LSA log-entropy weighting, Katz burstiness (opposite polarity).
No priority claimed — see `docs/research_notes.md` for the cautious novelty
statement. Next: held-out threshold validation, $C$-scaling law, CV-noise
vs vocabulary-gap separation, section-aware chunking, IDF-hybrid retrieval,
co-occurrence extension.

## License

MIT.

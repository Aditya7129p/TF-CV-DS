# TF-CV-DS — Full Test & Experiment Report

**Project:** Distributional Stability Keyword Extraction (TF-CV-DS), a proposed
potentially-novel intra-document term representation.
**Date:** 2026-09-23. **Status:** research prototype — no novelty claimed.
**Location:** `tf-cv-ds/` (48 files incl. caches; git-untracked).

## 1. Environment

| Item | Value |
|---|---|
| Platform | win32, Python 3.14.7 |
| numpy / scipy / pandas / matplotlib / pytest | 2.5.2 / 1.18.1 / 3.0.5 / 3.11.1 / 9.1.1 |
| External NLP deps | none (NLTK/WordNet optional, not installed, not required) |
| Model defaults (shipped) | `n_chunks=8` (lib) / 4 (classifier demo), `stemming=False`, `soft_match=False`, `trigram_threshold=0.5`, `norm='l2'`, `min_freq=1`; demo enables `soft_match=True` + topic-IDF |

## 2. Unit tests — 29/29 PASS (0.25 s)

| File | Tests | Result |
|---|---|---|
| `tests/test_algorithm.py` | chunk math exactness, anti-stuffing (`stable` > `stuffed`), log-TF dampening, finiteness, fit/transform shapes, vocab/repetition, similarity identical-vs-orthogonal | 7/7 pass |
| `tests/test_edge_cases.py` | empty input, single-chunk neutral stability, C > n tokens, singleton no-explosion, determinism, 5000× repetition stability | 6/6 pass |
| `tests/test_vectorizer.py` | fit/transform API, unknown→zero, duplicates equal, single-doc corpus, empty/stopword-only docs, very long doc | 6/6 pass |
| `tests/test_tokenizer.py` | stemmer inflections + non-merging derivations, trigram values, OOV prefix/derivation/abstain/tiebreak mapping, soft-match on/off, occurrence-based `min_freq` gate | 10/10 pass |

Command: `python -m pytest tests -v` (config in `pyproject.toml`).

## 3. Examples (all execute cleanly)

### 3.1 `basic_example.py`
Related ML docs similarity **0.7744**, unrelated ML-vs-football **0.0757**.
Top keywords correct (`learning`, `machine` lead both ML docs).

### 3.2 `document_similarity.py`
Near-duplicate thematic docs **0.9908**; unrelated pairs **0.0000**.

### 3.3 `comparison.py` (shared preprocessing, from-scratch baselines)
| Pair | BoW | TF-IDF | TF-CV-DS | BM25 (query=doc0) |
|---|---|---|---|---|
| sim(doc0, doc1) related | 0.471 | 0.340 | 0.471 | 1.947 |
| sim(doc0, doc2) unrelated | 0.0 | 0.0 | 0.0 | 0.0 |
Runtimes per transform: BoW 0.08 ms, TF-IDF 0.19 ms, TF-CV-DS 0.52–0.66 ms
(same complexity class; no co-occurrence matrix anywhere).

### 3.4 `text_to_text.py` (20 topics × 3 inputs, self-scoring)
**59/60 = 0.983** (baseline before improvements: 58/60 with 2 `unknown`).
Previously-failing cases now:
- `the chemical reaction produced a new compound` → **science 0.331** (was `unknown`)
- `the model learns patterns…` → **technology 0.229 vs health 0.091** (was an exact 0.218 tie)
- `the video became viral…` → social_media, correct but thin (0.150 vs gaming 0.142)
- Sole error: `the batsman hit a six…` → gaming. Evidence is a perfect tie
  (`bat`→sports vs `player`→gaming, both confidence 0.8, both df=1) —
  unsolvable lexically; kept as the documented lexical-ceiling exhibit.

## 4. Synthetic benchmark (`experiments/benchmark.py`, 7 probes)

| Dataset | sim_ds | sim_bow | sim_tfidf | Note |
|---|---|---|---|---|
| repeated | 0.603 | 0.632 | 0.549 | sane |
| common_rare | 0.617 | 0.617 | 0.450 | sane |
| cooccurrence | 0.571 | 0.571 | 0.403 | sane |
| stuffing | 0.291 | 0.250 | 0.150 | penalty works but partial: 30× stuffed `keyword` still leads (4.43 vs 1.42) — log-TF outweighs CV on short docs |
| lengths | 1.000 | 1.000 | 1.000 | length-invariant ✓ |
| noisy | 0.668 | 0.668 | 0.506 | typo tokens treated honestly as-is |
| subtopic | 0.929 | 0.806 | 0.678 | confirms known weakness: localized `genome` 1.26 vs spine 3.87 |

## 5. Classification eval (`experiments/classification_eval.py`, frozen sets)

Final shipped flags (`soft_match=True, thr=0.5, topic-IDF on, stemming off`):

| Set | Coverage | Acc (total) | Acc (covered) | Unknown | Confusion |
|---|---|---|---|---|---|
| IN_VOCAB (n=18) | 1.000 | **1.000** | 1.000 | 0.000 | 0.000 |
| HARD/OOV (n=27) | 0.704 | **0.630** | 0.895 | 0.296 | 0.074 |
| Baseline HARD | 0.556 | 0.519 | 0.933 | 0.444 | 0.037 |

Improvement: +0.111 accuracy, unknowns nearly halved; cost: +0.037 confusion
(the price of guessing instead of abstaining).

## 6. Ablation matrix (HARD eval, n=27; IN_VOCAB = 1.000 in all rows)

| stem | soft | thr | idf | acc | cov | unk | conf |
|---|---|---|---|---|---|---|---|
| 0 | 0 | – | 0 | 0.519 | 0.556 | 0.444 | 0.037 |
| 1 | 0 | – | 0 | 0.519 | 0.556 | 0.444 | 0.037 |
| 1 | 1 | 0.25 | 0 | 0.704 | 0.889 | 0.111 | 0.185 |
| 1 | 1 | 0.50 | 0 | 0.704 | 0.778 | 0.222 | 0.074 |
| 1 | 1 | 0.25 | 1 | 0.704 | 0.889 | 0.111 | 0.185 |
| 1 | 1 | 0.50 | 1 | 0.704 | 0.778 | 0.222 | 0.074 |
| 1 | 0 | – | 1 | 0.519 | 0.556 | 0.444 | 0.037 |

Demo-scale ablation (n=60): baseline 58/60 → no-stem+soft@0.5+idf **59/60**;
adding stemming drops it to 56/60 (cross-topic inflection merges:
`dish`→food, `players`→gaming, `studied`→`study`→education). Stemming effect
is small and harness-dependent (±2–3 items); default OFF chosen on the
larger sample plus the understood mechanism. Threshold 0.5 over 0.25 keeps
equal accuracy with confusion 0.074 instead of ~0.19 (avoids
`microscope`→`telescope` mistargets, Jaccard 0.25).

## 7. Failure analysis (not hidden)

1. **OOV semantic gap (partially fixed):** pure synonyms with no surface
   similarity (`six`, `overs`) remain invisible — lexical ceiling.
2. **Perfect-tie confusion:** `batsman` (bat vs player) — abstention would be
   more honest than the current gaming guess; margin reject option exists.
3. **Stuffing dampened, not eliminated** (§4 stuffing row).
4. **Subtopic penalty:** localized keywords score ~⅓ of spine terms — genuine
   trade-off of the evenness criterion, disclosed by design.
5. **Short texts:** 4 chunks over ~8 tokens makes CV estimates noisy; part of
   the unknown rate may be instability noise, not vocabulary gaps.
6. **No semantics:** synonyms, polysemy, paraphrase unhandled (classical scope).

## 8. Mathematical sanity (covered by tests + runs)

Finite/non-negative scores everywhere; empty docs → zero vectors, no crash;
single-chunk docs → neutral stability (weight = log(1+TF)); duplicate docs →
identical vectors; unknown words → 0; 5000× repetition finite; all
deterministic across reruns.

## 9. Complexity (as implemented)

Fit (vocab) O(ΣL); transform O(L + V_d) per doc (+O(U·M) trigram scan only
when `soft_match` is on: U OOV tokens × vocab M — negligible at demo scale,
needs a cap/index at large M); cosine O(M); memory O(M) per vector. Same
order as TF-IDF; no co-occurrence matrix.

## 10. Research status

Proposed method, potentially novel as a packaged scheme; components (CV,
log-TF, chunk dispersion, centroid + topic-IDF classification, fuzzy OOV
backoff) are standard. Must-check prior art before any stronger claim:
Juilland's D / Gries DP dispersion, LSA log-entropy weighting, Katz
burstiness (opposite polarity), centroid classifiers, SymSpell-style
symmetric-delete matching. No fabricated results, citations, or novelty
claims in this report — every number above came from the runs on 2026-09-23.

## 11. Reproduction

```bash
cd tf-cv-ds
pip install -r requirements.txt
python -m pytest tests -v            # §2
python examples/basic_example.py     # §3.1 (needs PYTHONPATH=src on some shells)
python examples/comparison.py        # §3.3
python examples/text_to_text.py      # §3.4, prints DEMO ACCURACY
python experiments/benchmark.py      # §4
python experiments/classification_eval.py  # §5, writes classification_eval.md
```

Ablation env overrides for `text_to_text.py`: `TF_STEM` / `TF_SOFT` /
`TF_IDF` ∈ {0,1}, `TF_TH`, `TF_MARGIN`.

# Algorithm — TF-CV-DS

## Pseudocode

```text
INPUT: document string d, config (C, eps, delta, eps_cv, min_freq)
OUTPUT: ranked keyword list / vector R(d)

1. toks = candidate_tokens(d)          # stop/pronoun/adverb removal, POS approx
2. if toks empty: return empty
3. chunks = split toks into C contiguous fixed-token windows
4. for each term t, each chunk c:
       p[t,c] = count(t in c) / max(len(c), 1)
5. bar  = mean_c p[t,c]
   var  = sample variance over c (0 if C == 1)
   std  = sqrt(max(var, 0))
   CV   = std / (bar + delta)
   stab = 1 / (CV + eps_cv)            # 1.0 if C == 1
6. W[t] = log(1 + TF[t]) * stab  (0 if TF < min_freq)
7. sort terms by W descending; align to global vocab for vectors
```

## Implementation map

- `tokenizer.py` — steps 1 (regex tokenize, stop/pronoun/adverb lists,
  optional NLTK `NN*`/`JJ*` filter, heuristic fallback).
- `algorithm.py` — steps 2–7 (`compute_term_stats`, `rank_terms`,
  `score_document`, `TFCVDSConfig`).
- `vectorizer.py` — corpus vocabulary alignment + `fit` / `transform` /
  `fit_transform` / `extract_keywords` / `similarity`.
- `similarity.py`, `utils.py` — cosine, chunking, normalization.

## Relation to existing methods

- TF-IDF: shares the TF factor; replaces global IDF rarity with
  intra-document stability. No corpus needed.
- TextRank: both corpus-free; TextRank uses graph centrality, this uses
  spatial dispersion.
- Katz K-mixture / burstiness models: inverted logic — they reward spikes,
  this rewards evenness.
- LSA entropy weighting: shares dispersion intuition across a corpus;
  this operates within one document over chunks.

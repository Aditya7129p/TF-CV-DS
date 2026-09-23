# Distributional Stability Keyword Extraction (TF-CV-DS):
# A Corpus-Free Term Representation Based on Intra-Document Evenness

**Status:** early-stage research draft / working paper. The method is described
as *potentially novel as a packaged scheme*; its components are standard, and
no claim of priority or of being the "first" anything is made. All numerical
results in this paper were produced by the accompanying implementation on
2026-09-23 and are reproducible via the commands in Appendix A.

---

## Abstract

Classical sparse text representations (bag of words, TF-IDF, BM25) derive a
term's global importance from collection statistics — document frequency or
inverse document frequency — and therefore require a corpus, reward
repetition, and treat terms as independent. Burstiness models take the
opposite view within a document, rewarding terms that spike locally. We
investigate a third, underexplored polarity for single-document analysis:
rewarding terms that are **both prominent and evenly distributed** across a
document's parts. The proposed method, TF-CV-DS (Term Frequency × Coefficient
of Variation Distributional Stability), splits a document into fixed-token
chunks, measures each candidate term's relative-frequency dispersion across
chunks, inverts the scale-invariant coefficient of variation into a stability
score, and multiplies it by log-dampened term frequency. The method needs no
corpus, explicitly penalises keyword stuffing, and is implemented from
scratch with a scikit-learn-style API. On a 60-sentence, 20-topic
classification demo it reaches 59/60 (baseline exact-match variant: 58/60
with two abstentions); on a frozen 27-sentence hard/out-of-vocabulary eval,
accuracy rises from 0.519 to 0.630 while abstentions fall from 0.444 to
0.296. Ablations show each component's contribution and lead us to reject one
candidate component (rule-based stemming) for crowded classifiers. We report
failures in full — including an irreducible evidence-tie that no lexical
system can resolve — and position the work honestly against prior art in
dispersion measurement, entropy weighting, and centroid classification.

**Keywords:** term weighting, keyword extraction, dispersion, coefficient of
variation, TF-IDF, burstiness, text classification, information retrieval.

---

## 1. Introduction

### 1.1 Problem

Given a single document $d$ and no external collection, assign each term $t$
a weight $W(t, d)$ that surfaces the document's persistent thematic
vocabulary ("what this document keeps returning to") while suppressing
incidental mentions, boilerplate, and deliberately stuffed repetition. The
representation must be interpretable, computable from scratch, robust on long
documents (papers, reports, books), and honest about where it degrades.

### 1.2 Why not TF-IDF?

TF-IDF (Jones, 1972; Salton & Buckley, 1988) remains the default sparse
representation, but it has well-known structural properties that motivate
alternatives: (i) it requires a corpus to compute document frequency;
(ii) its local factor is linear (or ad hoc sublinear) in term frequency, so
repetition is rewarded without bound; (iii) it treats terms as independent
axes, ignoring co-occurrence, position, and distribution shape; (iv) it
over-weights rare noisy tokens (a typo with $df = 1$ receives maximal IDF);
(v) its global factor cannot distinguish a term that is topical to a
sub-collection from one that is genuinely vacuous. BM25 (Robertson &
Zaragoza, 2009) fixes saturation and length normalisation for retrieval
scoring but remains lexical, query-side, and corpus-dependent.

### 1.3 The evenness hypothesis

Dispersion-based thinking has a long history in corpus linguistics
(Juilland's $D$; Gries, 2008, DP) and in IR (Katz, 1996; Church & Gale,
1995), but almost always with the **burstiness** polarity: high variance
signals topicality. We invert the premise for long single documents: the
thematic backbone of a document is what the author *cannot stop returning
to everywhere*, not what spikes once intensely. A term repeated unnaturally
in one paragraph (keyword stuffing) should therefore be *suppressed*, and a
term spread uniformly should be *promoted*, conditional on being prominent
enough overall. This is the entire hypothesis; everything else is machinery.

### 1.4 Contributions (claimed modestly)

1. A precise, implementable formulation (TF-CV-DS) combining log-dampened TF
   with inverted coefficient-of-variation stability over fixed-token chunks.
2. A from-scratch implementation with tests (29 passing), baselines sharing
   identical preprocessing, and three evaluation harnesses.
3. Measured evidence: where the method wins, where it ties, and where it
   fails — including a rejected component (stemming) and an irreducible
   lexical-tie error we deliberately do not engineer away.
4. A cautious novelty assessment against the closest prior art.

---

## 2. Related Work

**Classical weighting.** From raw counts through TF-IDF to BM25, the dominant
family has the diagonal form $g(f(t,d)) \cdot h(df(t))$ (Salton & Buckley,
1988). Our method keeps the local factor $g$ but replaces the collection
factor $h$ with an intra-document dispersion factor — usable with zero
external documents.

**Entropy and dispersion weighting.** LSA log-entropy weighting (Deerwester
et al., 1990; Dumais, 1991) uses $1 - H/\log N$ where $H$ is the entropy of a
term's distribution *across a collection*. TF-CV-DS applies the same
intuition *within one document across chunks*, with CV instead of entropy
and no collection. Corpus linguists have long measured evenness across
corpus parts (Juilland's $D$, 1964; Gries's DP, 2008); to our knowledge the
CV-inversion-over-chunks form as a document vectoriser is not a published
standard, though we claim no priority and invite correction.

**Burstiness.** Katz (1996) and Church & Gale (1995) model topical terms as
bursty (high variance). TF-CV-DS is the deliberate inverse: low variance is
signal. The two views are complementary, not contradictory — bursts mark
events, evenness marks themes — and §6 shows a case (section-localised
subtopics) where the burstiness view is clearly the right one.

**Keyword extraction.** TextRank (Mihalcea & Tarau, 2004) ranks by graph
centrality; RAKE (Rose et al., 2010) and YAKE! (Campos et al., 2020) use
statistical co-occurrence features. All are corpus-free like ours, but none
uses cross-chunk evenness as the primary signal.

**Centroid classification and query expansion.** Our text-to-text demo is a
nearest-centroid classifier over TF-CV-DS vectors with IDF-over-topics
reweighting — textbook machinery (centroid/Rocchio lineage) reused as an
evaluation harness, not claimed as new. OOV soft-matching (prefix +
character-trigram Jaccard) belongs to the classical fuzzy-matching family
(edit-distance / n-gram blocking literature); WordNet-style synonym
expansion was considered and left out to keep the system dependency-free.

**Dense representations.** Word2Vec (Mikolov et al., 2013), GloVe
(Pennington et al., 2014), and contextual models (Devlin et al., 2019) solve
the generalisation problem we deliberately leave open (§6.1); they are
inapplicable where interpretability, sparsity, tiny data, or zero
dependencies are required — the niche this work targets.

---

## 3. Method

### 3.1 Preprocessing (mandatory, not optional)

Raw stability ranking would crown stop words, since function words are the
most uniformly distributed tokens in any language. Candidates are therefore
restricted by: tokenisation (alphanumeric regex, case-folded), stop-word /
pronoun removal (embedded lists, dependency-free), low-content adverb
filtering (blocklist plus a documented `-ly` heuristic), minimum length 2,
and numeric-token removal, with optional NLTK `NN*`/`JJ*` POS restriction
(heuristic fallback otherwise, documented as approximate).

### 3.2 Chunking

The post-filter candidate stream is split into $C$ contiguous fixed-token
windows ($C \in [5,10]$ recommended; $C = 4$ used for short-text
classification). Fixed-token windows keep the mathematics uniform; if fewer
tokens than $C$ exist, $C$ is reduced so no empty chunk is produced. Let
$N_c$ be the candidate-token count of chunk $c$ and $f(t, c)$ the count of
$t$ in $c$.

### 3.3 Core formulation

Relative frequency per chunk:

$$p_{t,c} = \frac{f(t,c)}{N_c}, \quad (p = 0 \text{ when } N_c = 0)$$

Mean and unbiased sample variance across chunks:

$$\bar{p}_t = \frac{1}{C}\sum_{c=1}^{C} p_{t,c}, \qquad
\sigma_t^2 = \frac{1}{C-1}\sum_{c=1}^{C}(p_{t,c} - \bar{p}_t)^2, \qquad
\sigma_t = \sqrt{\max(\sigma_t^2, 0)}$$

Scale-invariant dispersion (the key correction — see §3.4):

$$\mathrm{CV}_t = \frac{\sigma_t}{\bar{p}_t + \delta}$$

Stability and final hybrid weight:

$$W_{\mathrm{CV\text{-}DS}}(t,d) = \frac{1}{\mathrm{CV}_t + \epsilon_{CV}},
\qquad
\boxed{W_{\mathrm{final}}(t,d) = \log(1 + \mathrm{TF}(t,d)) \cdot
W_{\mathrm{CV\text{-}DS}}(t,d)}}$$

with $\mathrm{TF}(t,d) = \sum_c f(t,c)$. Guards $\epsilon, \delta,
\epsilon_{CV} > 0$ protect each denominator. For a shared vocabulary $V$
(built per corpus purely for vector alignment), $R(d) =
[W_{\mathrm{final}}(t_1,d), \dots, W_{\mathrm{final}}(t_M,d)]$ with a
per-document TF floor and optional L2/L1 row normalisation. A single-chunk
document ($C = 1$) carries no dispersion information, so stability is
defined neutral ($1.0$).

### 3.4 Why CV and not raw inverse variance

The naive $W_{DS} = 1/(\sigma^2 + \epsilon)$ is scale-biased: two equally
steady terms at different absolute densities receive different variances
merely for living at different heights. $\mathrm{CV}$ measures spread
*relative to the mean*, making stability comparable across high- and
low-frequency candidates. Both forms are implemented; the CV form ships.

### 3.5 Anti-stuffing mechanism (analytical)

A term stuffed into one paragraph has high TF but near-zero density
elsewhere, hence large $\sigma_t$ relative to $\bar{p}_t$, hence large CV
and a suppressed final score. Measured (§5.2): a 30× stuffed keyword still
leads on a short document (4.43 vs 1.42 for spine terms) because
$\log(1+30)$ outweighs the CV penalty at that length — suppression is
partial, and we report the exact numbers rather than claiming a cure.

### 3.6 OOV soft-matching (classifier extension)

Exact lexical matching leaves morphological variants invisible
(`reaction` vs prototype `reactions`). Two dependency-free backoffs,
applied at transform time only:
(i) prefix containment (`batsman`→`bat`, confidence 0.8);
(ii) character-trigram Jaccard ≥ 0.5 (`reaction`→`reactions` 0.857,
`chemical`→`chemistry` 0.30 — correctly *below* threshold and abstained).
Matches contribute fractionally by confidence, and the `min_freq` floor
counts occurrences, not confidence mass (else all soft matches at 0.8 would
fall below a floor of 1 — a real bug we caught and fixed during testing).
Unmapped tokens are dropped (abstention, not guessing).

### 3.7 Topic-IDF and reject option (classifier harness)

Centroid classification reweights features by $\log(K/df)$ over topics, so
cross-topic generic words (`training` in technology *and* health) stop
deciding close calls. A margin reject option (near-tie → `unknown`) is
implemented and available but defaults to 0.

### 3.8 Complexity

Fit (vocabulary) $O(\sum L)$; transform $O(L + V_d)$ per document plus an
$O(U \cdot M)$ trigram scan only when soft-matching is on ($U$ OOV tokens,
$M$ vocabulary — negligible at tested scales, needs indexing at large $M$);
similarity $O(M)$; memory $O(M)$ per vector. Same order as TF-IDF; no
co-occurrence matrix.

---

## 4. Theoretical Properties (with proofs where short)

- **Finiteness:** all quantities are finite for finite input; guards cover
  $N_c = 0$, $\bar{p}_t \to 0$, $C = 1$, empty documents.
- **Monotonicity in TF (fixed dispersion):** $W_{\mathrm{final}}$ is strictly
  increasing in TF via $\log(1+\mathrm{TF})$.
- **Monotonicity in evenness (fixed TF):** $W_{\mathrm{final}}$ strictly
  decreases in $\mathrm{CV}$.
- **Duplicate invariance:** identical documents yield identical vectors
  (tested).
- **Empty/unknown handling:** empty or fully-filtered documents and fully-OOV
  inputs yield zero vectors and an explicit `unknown`, never a crash or a
  fabricated ranking (tested).

---

## 5. Experiments

All runs: Python 3.14.7, numpy 2.5.2, win32, 2026-09-23. Baselines (BoW,
TF-IDF, BM25) are implemented from scratch in-repo and share *identical*
preprocessing. No test sentence was ever added to any prototype or
vocabulary source.

### 5.1 Unit tests — 29/29 pass

Core math exactness (fully-stable term: var = CV = 0, closed-form weight),
anti-stuffing ordering, log-TF dampening, finiteness, API shapes, duplicate
equality, empty/unknown/single-doc/long-doc behaviour, determinism,
stemmer/trigram/mapping unit cases, soft-match on/off contrast,
occurrence-based floor gating. (`python -m pytest tests -v`.)

### 5.2 Synthetic benchmark (7 probes)

Similarities behave sanely (related 0.57–0.67 vs TF-IDF 0.40–0.55 with the
same preprocessing); length-invariance holds exactly (1.000); the stuffing
and subtopic probes confirm §3.5 and the §6.3 trade-off with numbers.

### 5.3 Classification

*Demo (20 topics × 3 sentences, self-scoring):* **59/60 = 0.983** vs 58/60
with two abstentions for the exact-match baseline. Fixes verified:
chemistry sentence → science 0.331; technology/health exact tie 0.218 →
technology 0.229 vs health 0.091.

*Frozen eval (9 topics; IN_VOCAB n=18, HARD/OOV n=27):*

| Set | Coverage | Acc (total) | Acc (covered) | Unknown | Confusion |
|---|---|---|---|---|---|
| IN_VOCAB | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| HARD baseline | 0.556 | 0.519 | 0.933 | 0.444 | 0.037 |
| HARD final | 0.704 | **0.630** | 0.895 | 0.296 | 0.074 |

### 5.4 Ablation (HARD, n=27; IN_VOCAB = 1.000 throughout)

Stemming alone: no effect (0.519). Soft-match@0.5: 0.519 → 0.704 with
confusion 0.074 (vs 0.25 threshold: same accuracy, confusion ~0.19 —
rejected). Topic-IDF breaks ties without moving HARD accuracy. On the
larger demo (n=60), stemming *hurts* (59/60 → 56/60) via cross-topic
inflection merges (`dish`→food, `players`→gaming, `studied`→`study`);
stemming therefore ships OFF for classifiers (kept opt-in for long-doc
keyword work). **Caveat:** threshold 0.5 was selected on these same small
sets — expect shrinkage on held-out data; independent validation is open
future work (§7).

---

## 6. Failure Analysis

1. **Semantic gap (lexical ceiling).** Pure synonyms with no surface overlap
   (`six`, `overs`) are invisible. No classical fix exists without external
   knowledge; dense models own this territory.
2. **Irreducible evidence tie.** `batsman` carries `bat`→sports and
   `player`→gaming at identical confidence (0.8) and rarity (df=1). Kept as
   the exhibit, not engineered away.
3. **Collision regressions.** Stemming/soft-matching add cross-topic links
   (`player` in gaming vs a sports sentence; `dish` food vs cooking) that
   flip previously-correct close calls — measured, and the reason for the
   conservative defaults.
4. **Partial stuffing suppression** (§3.5 numbers).
5. **Subtopic penalty.** Section-localised keywords score ~⅓ of spine terms
   (`genome` 1.26 vs 3.87) — the burstiness view is correct there; our
   polarity is wrong by construction.
6. **Short-text noise.** ~8 tokens over 4 chunks makes CV estimates noisy;
   part of the unknown rate is instability noise, unseparated from
   vocabulary gaps — an explicit open measurement.

---

## 7. Limitations, Novelty Assessment, Future Work

**Limitations:** needs chunkable length; no synonymy/polysemy handling;
POS/stop filtering carries load; $C$, TF floor, threshold, and margin are
tuning choices; trigram scan needs indexing at scale; threshold tuned on
small same-set data.

**Novelty (conditional):** the packaged combination — fixed-token chunking,
per-chunk relative frequency, sample variance, CV inversion, log-TF
product, as a corpus-free document vectoriser with explicit anti-stuffing
behaviour — matches no formulation we could identify; each component is
textbook. We claim *potentially novel packaging*, not a new principle, and
request correction with citations if the exact form exists (first suspects:
dispersion-measure literature, LSA entropy weighting).

**Future work:** held-out threshold validation; $C$-scaling law; analytic
separation of CV noise vs vocabulary gaps on short texts; section-aware
chunking; IDF-hybrid retrieval mode; co-occurrence extension (the
Phase-1 C5/C10 candidates) as the principled answer to §6.1; a formal
Juilland-$D$/DP equivalence check.

---

## References

- Campos, R. et al. (2020). YAKE! Keyword extraction from single documents
  using multiple local features. *Information Sciences*.
- Church, K. W. & Gale, W. A. (1995). Poisson mixtures.
  *Natural Language Engineering*.
- Deerwester, S. et al. (1990). Indexing by latent semantic analysis.
  *JASIS*.
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional
  transformers. *NAACL*.
- Dumais, S. (1991). Improving the retrieval of information from external
  sources. *Behavior Research Methods*.
- Gries, S. Th. (2008). Dispersions and adjusted frequencies of occurrence.
  *IJCL*.
- Jones, K. Sparck (1972). A statistical interpretation of term
  specificity. *J. Documentation*.
- Juilland, A. & Chang-Rodriguez, E. (1964). *Frequency Dictionary of
  Spanish Words*. Mouton.
- Katz, S. (1996). Distribution of content words and phrases in text and
  language modelling. *Natural Language Engineering*.
- Mihalcea, R. & Tarau, P. (2004). TextRank: Bringing order into texts.
  *EMNLP*.
- Mikolov, T. et al. (2013). Efficient estimation of word representations.
  *NeurIPS*.
- Pennington, J. et al. (2014). GloVe: Global vectors for word
  representation. *EMNLP*.
- Porter, M. (1980). An algorithm for suffix stripping. *Program*.
- Robertson, S. & Zaragoza, H. (2009). The probabilistic relevance
  framework: BM25 and beyond. *FnTIR*.
- Rose, S. et al. (2010). Automatic keyword extraction from individual
  documents. In *Text Mining* (RAKE). Wiley.
- Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic
  text retrieval. *Information Processing & Management*.

*All references are established, widely-cited works included as background;
no reference was invented for this draft.*

---

## Appendix A — Reproduction

```bash
cd tf-cv-ds
pip install -r requirements.txt
python -m pytest tests -v
python examples/basic_example.py
python examples/comparison.py
python examples/text_to_text.py        # prints DEMO ACCURACY
python experiments/benchmark.py
python experiments/classification_eval.py
```

`text_to_text.py` env overrides: `TF_STEM` / `TF_SOFT` / `TF_IDF` ∈ {0,1},
`TF_TH`, `TF_MARGIN`. Full run log: `experiments/results/REPORT.md`.

## Appendix B — File Map

`src/tfcvds/`: `tokenizer.py` (tokenise, filter, stemmer, trigram
OOV mapping), `algorithm.py` (core equations, fractional weights),
`vectorizer.py` (fit/transform/keywords/similarity), `similarity.py`,
`utils.py`. `examples/`: basic, similarity, from-scratch baseline
comparison, text-to-text classifier. `experiments/`: synthetic datasets,
benchmark, frozen classification eval (+`results/`). `tests/`: 29 tests.
`docs/`: formulation, algorithm/pseudocode, research notes, this paper.

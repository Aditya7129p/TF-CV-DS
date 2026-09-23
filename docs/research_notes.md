# Research Notes — TF-CV-DS

## Proposed contribution

A corpus-free term weight combining log-dampened TF with an inverted,
scale-invariant dispersion score (1/CV over sub-document chunks) to surface
the persistent thematic "spine" of a long single document and explicitly
suppress keyword stuffing.

## Closest existing methods

1. TF-IDF / TF-ICF family — same TF factor, different specificity signal.
2. TextRank — corpus-free graph alternative.
3. Katz K-mixture / burstiness / Church–Gale models — opposite polarity
   (reward bursts; we reward evenness).
4. LSA log-entropy weighting (Dumais) — entropy/CV-style global weights;
   ours is intra-document over chunks.
5. Dispersion / Juilland's D / DP in corpus linguistics — evenness
   measures over corpus parts; closest conceptual cousin.

## Differences

The exact combination — fixed-token chunking + per-chunk relative
frequency + sample variance + CV inversion + log-TF product as a document
vectorizer with stuffing penalty — was not identified as an identical
published formulation at prototype time. No priority claimed.

## Potential novelty

Cautiously: *potentially novel as a packaged weighting scheme*, built from
well-known components (CV, log-TF, chunk dispersion). Must be checked
against Juilland's D / Gries DP literature before any stronger claim.

## Known limitations

- Needs long-enough documents (5–10 meaningful chunks); weak on tweets.
- Penalizes legitimate section-localized subtopic keywords.
- Stop-word / POS filtering does heavy lifting; heuristic fallback is lax.
- Chunk count $C$ and TF floor $\tau$ are corpus-dependent choices.

## Open questions

- Optimal $C$ vs document length scaling law?
- Section-aware vs fixed-window chunking trade-off?
- Hybrid with IDF for cross-document retrieval?
- Formal equivalence/normalization vs Juilland's D?

## Update: generalization improvements (measured, not claimed)

Failure mode found via `examples/text_to_text.py`: OOV sentences
(`batsman/six/overs`, `chemical/reaction/compound` vs prototypes holding
`bat`, `reactions/compounds/chemistry`) scored zero vectors, and generic
words (`training` in technology AND health) produced exact ties.

Fixes, all ablated on frozen sets (20-topic demo n=60; IN_VOCAB n=18 /
HARD n=27):

1. OOV soft-matching (`tokenizer.map_oov_token`): prefix containment
   (`batsman`->`bat`) + char-trigram Jaccard >= 0.5, with
   confidence-weighted fractional counts and occurrence-based
   `min_freq` gating. Unmapped tokens still abstain.
2. Topic-IDF (`log(K/df)` over topics) in the classifier: breaks
   generic-word ties (technology/health 0.218=0.218 ->
   technology 0.229 vs health 0.091).
3. Reject option (near-tie margin -> `unknown`): kept, defaults off.
4. Rule-based stemming: REJECTED for crowded classifiers — ablation
   showed it merges cross-topic inflections (`dish`->food,
   `players`->gaming, `studied`->`study`) and drops the demo 59/60 ->
   56/60. Kept in the library (opt-in) for long-doc keyword work.

Results: demo 58/60 (2 unknown) -> **59/60**; HARD acc **0.519 ->
0.630**, unknown 0.444 -> 0.296 (confusion 0.037 -> 0.074 — the price
of guessing); IN_VOCAB stays 1.000. Sole demo error is a perfect
evidence tie (`bat`->sports vs `player`->gaming, both confidence 0.8,
both df=1): correctly unsolvable lexically, kept as the exhibit of the
lexical ceiling rather than engineered away.

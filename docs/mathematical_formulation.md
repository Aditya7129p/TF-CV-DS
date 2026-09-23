# Mathematical Formulation — TF-CV-DS

Candidate research name: **Distributional Stability Keyword Extraction (TF-CV-DS)**.
Status: *proposed method, potentially novel — not claimed as established.*

## 1. Problem definition

Given a **single** document $d$ (no corpus required), assign each candidate
term $t$ a weight that is large when $t$ is both **prominent** (high global
frequency in $d$) and **stably distributed** (similar density in every part
of $d$).

## 2. Preprocessing (mandatory)

Stop words, pronouns and low-content adverbs are removed and candidates are
restricted (ideally by POS tag) to nouns / proper nouns / adjectives.
Without this, the stability criterion would rank function words highest,
since they are the most uniformly distributed tokens in any language.

## 3. Chunking

Partition the post-filter candidate stream into $C$ contiguous fixed-token
windows $\{c_1,\dots,c_C\}$, recommended $C\in[5,10]$. Fixed-token windows
keep the mathematics uniform; section-based chunks need the relative
frequency normalization below to remain comparable.

## 4. Core equations

$$p_{t,c} = \frac{f(t,c)}{N_c}$$

relative frequency of $t$ in chunk $c$ ($N_c$ = candidate-token count of $c$;
$p=0$ when $N_c=0$).

$$\bar p_t = \frac{1}{C}\sum_c p_{t,c}$$

$$\sigma^2_t = \frac{1}{C-1}\sum_c (p_{t,c}-\bar p_t)^2,\quad
\sigma_t=\sqrt{\sigma^2_t}$$

unbiased sample variance / std. Single chunk ($C=1$): dispersion is
undefined and stability is defined neutral ($=1$).

$$\mathrm{CV}_t = \frac{\sigma_t}{\bar p_t+\delta}$$

coefficient of variation — scale-invariant dispersion. $\delta>0$ guards
$\bar p_t\to 0$. The naive $W_{DS}=1/(\sigma^2+\epsilon)$ is scale-biased:
two equally steady terms at different absolute densities get different
variances; CV fixes this.

$$W_{\mathrm{CV}\text{-}\mathrm{DS}}(t,d)=\frac{1}{\mathrm{CV}_t+\epsilon_{CV}}$$

$$\mathrm{TF}(t,d)=\sum_c f(t,c)$$

$$\boxed{W_{\mathrm{final}}(t,d)=\log(1+\mathrm{TF}(t,d))\cdot
W_{\mathrm{CV}\text{-}\mathrm{DS}}(t,d)}$$

Log dampening stops hyper-frequent stable terms from dominating on volume
alone.

## 5. Vector form

For a shared vocabulary $V$ (built per corpus only for alignment), document
$d$ is represented as

$$R(d)=[W_{\mathrm{final}}(t_1,d),\dots,W_{\mathrm{final}}(t_M,d)]$$

with per-document TF floor ($t$ with $\mathrm{TF}<\tau$ set to 0) and
optional L2/L1 row normalization. Unknown terms map to 0.

## 6. Edge cases

| Case | Behaviour |
|---|---|
| Empty / fully-filtered doc | all-zero vector, no keywords |
| $C=1$ (very short doc) | stability neutral $=1$, weight $=\log(1+TF)$ |
| Unknown word | weight 0 |
| One-chunk-per-token ($C=n$) | valid, high variance expected |
| Common words | removed by stop filter, else high mean + low CV |
| Rare words (TF=1–2) | TF floor + $\delta$ guard prevent spurious extremes |
| Very long docs | fixed $C$ keeps chunks large; statistics stable |
| Very short docs | method degrades (needs length to chunk); documented weakness |
| Division by zero | $\epsilon,\delta,\epsilon_{CV}$ guards + $N_c=0\to p=0$ |
| Numerical stability | all quantities finite for finite input; variance clipped at 0 |

## 7. Complexity

$L$ = candidate tokens in $d$, $V_d$ distinct terms, $C$ chunks.
Fit (vocab build): $O(\sum L)$. Transform per doc: $O(L+V_d)$.
Similarity: $O(M)$ dense / $O(\mathrm{nnz})$ sparse. Memory: $O(M+C\cdot V_d)$
transient, $O(M)$ per vector. No co-occurrence matrix needed.

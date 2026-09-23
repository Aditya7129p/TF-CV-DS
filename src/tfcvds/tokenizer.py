"""Tokenization and candidate filtering for TF-CV-DS.

The model is intra-document: it rewards terms spread evenly across a
document's chunks. Raw (unfiltered) text would rank stop words highest
because function words are the most uniformly distributed tokens in any
language. Filtering is therefore mandatory, not optional.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")

# A compact embedded English stop-word list (incl. pronouns, auxiliaries,
# determiners, prepositions, conjunctions). Kept dependency-free on purpose.
DEFAULT_STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "also", "may", "one", "two", "first", "new",
    "used", "using", "often", "however", "within", "without",
}

# Low-content adverbs frequently kept by naive filters; blocked explicitly.
LOW_CONTENT_ADVERBS: Set[str] = {
    "very", "really", "highly", "extremely", "quite", "rather",
    "simply", "merely", "barely", "hardly", "mostly", "largely",
    "generally", "usually", "often", "sometimes", "always", "never",
    "well", "much", "many", "even", "still", "yet", "already",
}

PRONOUNS: Set[str] = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs", "myself", "yourself",
    "himself", "herself", "itself", "ourselves", "yourselves",
    "themselves",
}


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Split text into alphanumeric tokens.

    Args:
        text: raw document string.
        lowercase: fold case (recommended; proper-noun distinction is
            then approximated, see :func:`candidate_tokens`).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


_VOWELS = set("aeiou")


def stem_token(word: str) -> str:
    """Conservative rule-based stemmer (dependency-free).

    Handles inflections only (plurals, -ed, -ing, -ies/-ied -> -y):
    ``reactions`` -> ``reaction``, ``studied`` -> ``study``.
    Derivational forms (``chemical``/``chemistry``) are intentionally NOT
    merged here — that is the job of :func:`map_oov_token`.
    """
    w = word
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5 and any(c in _VOWELS for c in w[:-3]):
        return w[:-3]
    if w.endswith("ed") and len(w) > 4 and any(c in _VOWELS for c in w[:-2]):
        return w[:-2]
    if (
        w.endswith("es")
        and len(w) > 4
        and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh"))
    ):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("us", "ss")) and len(w) > 3:
        return w[:-1]
    return w


def char_trigrams(word: str) -> Set[str]:
    """Character trigram set of a word (for fuzzy OOV matching)."""
    if len(word) < 3:
        return {word} if word else set()
    return {word[i : i + 3] for i in range(len(word) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of character-trigram sets (0.0..1.0)."""
    sa, sb = char_trigrams(a), char_trigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def map_oov_token(
    token: str, vocab: Set[str], threshold: float = 0.5
) -> tuple[Optional[str], float]:
    """Map an out-of-vocabulary token to the closest vocab term.

    Strategy (deterministic; best score wins, ties broken alphabetically):

    1. Prefix containment (morphological compounds): ``batsman`` starts
       with vocab term ``bat`` (confidence 0.8).
    2. Character-trigram Jaccard >= ``threshold`` (derivational
       similarity): ``chemical`` -> ``chemistry`` (~0.30).

    Returns ``(term, confidence)`` with confidence 1.0 for exact hits,
    0.8 for prefix matches, and the Jaccard value for trigram matches.
    Returns ``(None, 0.0)`` when nothing passes the threshold — the
    token is then dropped (honest abstention, not a guess). Confidence
    is meant to scale the token's count weight downstream, so shaky
    matches whisper instead of shouting.
    """
    if token in vocab:
        return (token, 1.0)
    scored: List[tuple] = []
    for v in vocab:
        if len(v) < 3 or len(token) < 3:
            continue
        if (token.startswith(v) and len(v) >= 3) or (
            v.startswith(token) and len(token) >= 4
        ):
            score = 0.8
        else:
            score = trigram_jaccard(token, v)
        if score >= threshold:
            scored.append((score, v))
    if not scored:
        return (None, 0.0)
    # Best score wins; alphabetical tie-break keeps it deterministic.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return (scored[0][1], scored[0][0])


def _try_nltk_pos_filter(tokens: List[str]) -> Optional[List[str]]:
    """Keep nouns / proper nouns / adjectives using NLTK, if installed.

    Returns None when NLTK (or its tagger data) is unavailable so the
    caller can fall back to the heuristic filter.
    """
    try:
        import nltk  # type: ignore
    except ImportError:
        return None
    try:
        tagged = nltk.pos_tag(tokens)
    except Exception:
        return None
    keep_prefixes = ("NN", "JJ")  # NN*, JJ* (includes NNP, JJR, JJS)
    return [w for w, tag in tagged if tag.startswith(keep_prefixes)]


def candidate_tokens(
    text: str,
    *,
    lowercase: bool = True,
    stopwords: Optional[Set[str]] = None,
    min_len: int = 2,
    drop_numeric: bool = True,
    drop_adverbs: bool = True,
    use_pos_filter: bool = True,
    use_nltk_pos: bool = False,
    stemming: bool = False,
) -> List[str]:
    """Tokenize text and return content-candidate tokens.

    Pipeline: tokenize -> optional stem -> stop-word / pronoun / adverb
    removal -> length + numeric filter -> optional POS restriction.

    Because this package is dependency-free by default, POS restriction
    works in two modes:

    * ``use_nltk_pos=True`` and NLTK + tagger data installed: keep
      tokens tagged ``NN*`` / ``JJ*`` (true nouns/adjectives).
    * otherwise (default): heuristic approximation — stop-word,
      pronoun, length and adverb-blocklist filtering. This keeps mostly
      nouns/adjectives in practice but is documented as approximate.

    Args:
        text: raw document string.
        lowercase: fold case before filtering.
        stopwords: custom stop set (defaults to :data:`DEFAULT_STOPWORDS`).
        min_len: drop tokens shorter than this.
        drop_numeric: drop purely numeric tokens.
        drop_adverbs: drop tokens in :data:`LOW_CONTENT_ADVERBS` and,
            heuristically, common ``-ly`` adverbs longer than 5 chars.
        use_pos_filter: apply filtering at all (must stay True for the
            model to behave; disabling is only for ablation).
        use_nltk_pos: opt into NLTK POS tagging when available.
        stemming: apply :func:`stem_token` before filtering (stem-first
            ordering, so e.g. ``overs`` -> ``over`` is still caught by
            the stop-word list; raw and stemmed forms are both checked
            so ``having`` -> ``hav`` cannot leak back in).
    """
    """Tokenize text and return content-candidate tokens.

    Pipeline: tokenize -> stop-word / pronoun / adverb removal ->
    length + numeric filter -> optional POS restriction.

    Because this package is dependency-free by default, POS restriction
    works in two modes:

    * ``use_nltk_pos=True`` and NLTK + tagger data installed: keep
      tokens tagged ``NN*`` / ``JJ*`` (true nouns/adjectives).
    * otherwise (default): heuristic approximation — stop-word,
      pronoun, length and adverb-blocklist filtering. This keeps mostly
      nouns/adjectives in practice but is documented as approximate.

    Args:
        text: raw document string.
        lowercase: fold case before filtering.
        stopwords: custom stop set (defaults to :data:`DEFAULT_STOPWORDS`).
        min_len: drop tokens shorter than this.
        drop_numeric: drop purely numeric tokens.
        drop_adverbs: drop tokens in :data:`LOW_CONTENT_ADVERBS` and,
            heuristically, common ``-ly`` adverbs longer than 5 chars.
        use_pos_filter: apply filtering at all (must stay True for the
            model to behave; disabling is only for ablation).
        use_nltk_pos: opt into NLTK POS tagging when available.
    """
    sw = DEFAULT_STOPWORDS if stopwords is None else stopwords
    tokens = tokenize(text, lowercase=lowercase)
    if not use_pos_filter:
        return tokens
    if use_nltk_pos:
        kept = _try_nltk_pos_filter(tokens)
        if kept is not None:
            out = []
            for w in kept:
                wl = w.lower()
                if wl in sw or wl in PRONOUNS:
                    continue
                if len(wl) < min_len:
                    continue
                if drop_numeric and wl.isdigit():
                    continue
                out.append(wl if lowercase else w)
            if stemming:
                out = [
                    t
                    for t in (stem_token(x) for x in out)
                    if t not in sw and len(t) >= min_len
                ]
            return out
        # fall through to heuristic when NLTK is unavailable
    out: List[str] = []
    for w in tokens:
        raw = w
        if stemming:
            w = stem_token(w)
        if w in sw or raw in sw or w in PRONOUNS:
            continue
        if len(w) < min_len:
            continue
        if drop_numeric and w.isdigit():
            continue
        if drop_adverbs:
            if w in LOW_CONTENT_ADVERBS:
                continue
            # Heuristic: most longer "-ly" tokens are manner adverbs
            # ("quickly", "clearly"); a few adjectives ("friendly")
            # are collateral — documented limitation.
            if len(w) > 5 and w.endswith("ly"):
                continue
        out.append(w)
    return out

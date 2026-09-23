"""Synthetic datasets for TF-CV-DS experiments."""

REPEATED = [
    " ".join(["data"] * 20 + ["model"] * 2),
    " ".join(["data"] * 2 + ["model"] * 2 + ["extra"] * 2),
]

COMMON_RARE = [
    "learning data model systems analysis methods research",
    "learning zyxqwv data model qwerty systems",
]

COOCCURRENCE = [
    "neural networks deep learning gradient descent optimization",
    "neural networks deep learning football stadium goals",
]

STUFFING = [
    " ".join(["machine", "learning"] * 8 + ["keyword"] * 30),  # stuffed in tail
    " ".join(["machine", "learning", "data", "models"] * 8),
]

LENGTHS = [
    "learning data models",
    " ".join(["learning data models"] * 50),
]

NOISY = [
    "learning data models syst3m learn1ng modle research analysis",
    "learning data models systems research analysis methods",
]

SUBTOPIC = [
    # 'genome' is section-localized inside a broad ML-flavoured doc (known weakness probe)
    " ".join(["learning data models"] * 10 + ["genome sequencing"] * 6),
    " ".join(["learning data models"] * 10 + ["cooking recipes"] * 6),
]

DATASETS = {
    "repeated": REPEATED,
    "common_rare": COMMON_RARE,
    "cooccurrence": COOCCURRENCE,
    "stuffing": STUFFING,
    "lengths": LENGTHS,
    "noisy": NOISY,
    "subtopic": SUBTOPIC,
}

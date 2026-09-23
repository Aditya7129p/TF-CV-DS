"""Document similarity with TF-CV-DS on a longer thematic document."""

from tfcvds import TFCVDSVectorizer

corpus = [
    " ".join(["neural networks learn patterns from data"] * 6)
    + " optimization gradient descent training epochs",
    " ".join(["neural networks learn patterns from data"] * 6)
    + " football stadium goals fans weekend match",
    "cooking recipes need salt pepper herbs spices oven baking flavor kitchen",
]

model = TFCVDSVectorizer(n_chunks=6)
model.fit(corpus)
print("Pairwise similarities:")
docs = corpus
for i in range(len(docs)):
    for j in range(i + 1, len(docs)):
        print(f"sim(d{i+1}, d{j+1}) = {model.similarity(docs[i], docs[j]):.4f}")
print("\nKeywords doc1:", model.extract_keywords(docs[0], top_k=6))

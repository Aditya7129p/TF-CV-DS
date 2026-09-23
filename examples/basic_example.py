"""Basic TF-CV-DS example: fit, transform, inspect, compare."""

import numpy as np

from tfcvds import TFCVDSVectorizer

documents = [
    "machine learning is powerful and machine learning models learn from data every day",
    "machine learning is useful and deep learning models improve machine learning systems",
    "football is a popular sport played with a ball on a large field every weekend",
]

model = TFCVDSVectorizer(n_chunks=5)
vectors = model.fit_transform(documents)

print("Vocabulary:")
print(model.get_feature_names_out())
print("\nVectors (rounded):")
print(np.round(vectors, 4))
print("\nTop keywords per document:")
for i, doc in enumerate(documents):
    print(f"doc{i+1}:", model.extract_keywords(doc, top_k=5))
print("\nSimilarity(doc1, doc2) =", round(model.similarity(documents[0], documents[1]), 4))
print("Similarity(doc1, doc3) =", round(model.similarity(documents[0], documents[2]), 4))

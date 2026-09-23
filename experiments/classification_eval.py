"""Text-classification evaluation: in-vocabulary vs. OOV generalization.

Two strictly separated test sets share one frozen prototype representation:

  IN_VOCAB: sentences reusing prototype vocabulary (tests familiar-word
      classification).
  HARD (OOV): sentences deliberately avoiding obvious class words (tests
      whether the representation generalizes to unseen but related words).

Metrics (tracked separately per set, never blended):
  coverage  = predicted != 'unknown' / total
  accuracy  = correct / total, plus correct / covered
  unknown   = 'unknown' / total
  confusion = wrong (non-unknown) / total

Run:  python experiments/classification_eval.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfcvds import TFCVDSVectorizer  # noqa: E402
from tfcvds.similarity import cosine_sim  # noqa: E402

# Ablation switches for the improvement study. Baseline (all off) reproduces
# the pre-fix model; toggle to measure each component's contribution.
# Current winner: soft-match@0.5 + topic-IDF, stemming OFF (stemming merges
# cross-topic inflections in crowded topic sets; neutral on these sets).
USE_STEMMING = False
USE_SOFT_MATCH = True
TRIGRAM_THRESHOLD = 0.5
USE_TOPIC_IDF = True

# --------------------------------------------------------------------------
# Frozen topic prototypes (the representation). Test sentences below are
# NEVER added here.
# --------------------------------------------------------------------------
PROTOTYPES = {
    "sports": [
        "football match stadium fans goals weekend league victory",
        "cricket bat ball innings runs wickets stadium crowd",
        "tennis court racket serve volley match point tournament",
    ],
    "technology": [
        "machine learning neural networks data training optimization",
        "software programming code algorithms database server",
        "hardware processor memory chip circuit performance",
    ],
    "cooking": [
        "recipe salt pepper herbs spices oven baking flavor",
        "pasta sauce garlic olive oil cheese cooking pan",
        "cake flour sugar butter eggs bake dessert sweet",
    ],
    "science": [
        "chemical reaction compound laboratory experiment solution",
        "biology cells microscope organism tissue growth",
        "physics energy particles motion force gravity",
    ],
    "travel": [
        "airport flight departure arrival delay gate",
        "hotel beach coast resort holiday booking",
        "train trip voyage itinerary passport luggage",
    ],
    "education": [
        "teacher classroom lesson students homework school",
        "exam test study grades curriculum semester",
        "library books reading essay knowledge degree",
    ],
    "finance": [
        "bank account money savings deposit customer",
        "investor assets returns stocks profit loss",
        "budget spending expenses debt loan interest",
    ],
    "environment": [
        "waste pollution recycling garbage landfill plastic",
        "species habitat wildlife forest conservation nature",
        "clean energy solar wind emissions climate carbon",
    ],
    "space": [
        "spacecraft astronaut orbit mission launch rocket",
        "planet telescope galaxy stars universe distant",
        "crew station satellite signal surface probe",
    ],
}

# --------------------------------------------------------------------------
# Set A: in-vocabulary (overlap with prototypes by design).
# --------------------------------------------------------------------------
IN_VOCAB = [
    ("the football team scored a goal", "sports"),
    ("the tennis match went to a final point", "sports"),
    ("the processor improves computer performance", "technology"),
    ("neural networks need training data", "technology"),
    ("the chef prepared pasta with garlic", "cooking"),
    ("bake the cake with flour and sugar", "cooking"),
    ("the chemical reaction bubbled in the laboratory", "science"),
    ("cells were observed under the microscope", "science"),
    ("the flight departure was delayed at the gate", "travel"),
    ("they booked a hotel near the beach", "travel"),
    ("the teacher gave a lesson in class", "education"),
    ("she studied hard for the exam", "education"),
    ("the customer deposited money into a savings account", "finance"),
    ("the investor expects future returns on assets", "finance"),
    ("recycling plastic reduces landfill waste", "environment"),
    ("solar panels provide clean energy", "environment"),
    ("the astronaut launched aboard the rocket", "space"),
    ("the telescope revealed a distant galaxy", "space"),
]

# --------------------------------------------------------------------------
# Set B: hard / OOV (no obvious class words; conceptually related only).
# --------------------------------------------------------------------------
HARD = [
    ("the batsman hit a six after several overs of play", "sports"),
    ("the goalkeeper stopped the shot in the final minute", "sports"),
    ("the athlete returned to competition after training", "sports"),
    ("the system adjusted its parameters using repeated feedback", "technology"),
    ("the application stored records and retrieved them efficiently", "technology"),
    ("the computer completed the calculation much faster", "technology"),
    ("the ingredients were mixed before being placed inside the oven", "cooking"),
    ("the dish was simmered until the sauce became thick", "cooking"),
    ("the mixture was allowed to cool before serving", "cooking"),
    ("the substance changed after being exposed to heat", "science"),
    ("the researchers observed microscopic structures", "science"),
    ("the experiment produced a measurable change", "science"),
    ("they arrived several hours before departure", "travel"),
    ("the visitors stayed near the coast for a week", "travel"),
    ("their journey continued through the mountain region", "travel"),
    ("she spent the evening preparing for tomorrow's test", "education"),
    ("the instructor explained the concept using an example", "education"),
    ("he submitted his work before the deadline", "education"),
    ("the investor purchased assets expecting future returns", "finance"),
    ("she tracked everything she spent during the month", "finance"),
    ("the money remained in the account for several years", "finance"),
    ("the amount of waste has increased significantly", "environment"),
    ("several species are losing their natural habitat", "environment"),
    ("cleaner energy could reduce harmful emissions", "environment"),
    ("the object continued moving around the distant world", "space"),
    ("the crew performed research far above Earth's surface", "space"),
    ("astronomers observed light from extremely distant objects", "space"),
]


def build_centroids(model, prototypes):
    docs, labels = [], []
    for topic, ds in prototypes.items():
        docs.extend(ds)
        labels.extend([topic] * len(ds))
    X = model.fit_transform(docs)
    if USE_TOPIC_IDF:
        K = len(prototypes)
        df = np.zeros(X.shape[1])
        for ds in prototypes.values():
            present = set()
            for d in ds:
                present.update(model._candidates(d))
            for t in present:
                j = model.vocabulary_.get(t)
                if j is not None:
                    df[j] += 1
        idf = np.log(K / np.maximum(df, 1.0))
    else:
        idf = np.ones(X.shape[1])
    centroids = {}
    for topic in prototypes:
        rows = X[[i for i, lab in enumerate(labels) if lab == topic]]
        centroids[topic] = rows.mean(axis=0) * idf
    return centroids, idf


def classify(text, model, centroids, idf):
    vec = model.transform([text])[0] * idf
    if vec.sum() == 0:
        return "unknown", {}
    scores = {t: float(cosine_sim(vec, c)) for t, c in centroids.items()}
    return max(scores, key=scores.get), scores


def evaluate(model, centroids, idf, dataset):
    rows = []
    for text, expected in dataset:
        pred, scores = classify(text, model, centroids, idf)
        rows.append((text, expected, pred, scores))
    total = len(rows)
    correct = sum(1 for _, e, p, _ in rows if p == e)
    unknown = sum(1 for _, _, p, _ in rows if p == "unknown")
    wrong = total - correct - unknown
    return {
        "rows": rows,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "coverage": (total - unknown) / total,
        "accuracy_total": correct / total,
        "accuracy_covered": correct / max(total - unknown, 1),
        "unknown_rate": unknown / total,
        "confusion_rate": wrong / total,
    }


def print_report(name, res):
    print(f"--- {name} (n={res['total']}) ---")
    for text, expected, pred, scores in res["rows"]:
        top = max(scores.values()) if scores else 0.0
        mark = "OK " if pred == expected else ("?? " if pred == "unknown" else "ERR")
        print(f"[{mark}] exp={expected:11s} pred={pred:11s} top={top:.3f} :: {text}")
    print(
        f"=> coverage={res['coverage']:.3f} acc_total={res['accuracy_total']:.3f} "
        f"acc_covered={res['accuracy_covered']:.3f} unknown={res['unknown_rate']:.3f} "
        f"confusion={res['confusion_rate']:.3f}\n"
    )


if __name__ == "__main__":
    model = TFCVDSVectorizer(
        n_chunks=4, stemming=USE_STEMMING, soft_match=USE_SOFT_MATCH,
        trigram_threshold=TRIGRAM_THRESHOLD,
    )
    centroids, idf = build_centroids(model, PROTOTYPES)
    in_vocab = evaluate(model, centroids, idf, IN_VOCAB)
    hard = evaluate(model, centroids, idf, HARD)
    print_report("IN_VOCAB", in_vocab)
    print_report("HARD/OOV", hard)

    out = ["# Classification eval\n"]
    for name, res in (("IN_VOCAB", in_vocab), ("HARD", hard)):
        out.append(f"## {name} (n={res['total']})")
        out.append(
            f"coverage={res['coverage']:.3f} acc_total={res['accuracy_total']:.3f} "
            f"acc_covered={res['accuracy_covered']:.3f} "
            f"unknown={res['unknown_rate']:.3f} confusion={res['confusion_rate']:.3f}"
        )
        for text, exp, pred, scores in res["rows"]:
            top = max(scores.values()) if scores else 0.0
            out.append(f"- [{exp} -> {pred} {top:.3f}] {text}")
        out.append("")
    Path(__file__).resolve().parent.joinpath("results", "classification_eval.md").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("Saved experiments/results/classification_eval.md")

"""
Stage 1-2 baseline: character n-gram TF-IDF + logistic regression.

What it does
1. Loads the Naver Shopping reviews (rating <TAB> text).
2. Labels sentiment from the rating: 1-2 stars = negative (0), 4-5 stars = positive (1).
   The dataset has no 3-star reviews.
3. Trains TF-IDF (character 2-5-grams) + logistic regression and reports
   accuracy and macro F1 on a held-out test set.
4. Finds rating-text mismatch candidates with out-of-fold predictions:
   every review is scored by a model that never saw it, and reviews whose
   text strongly disagrees with their rating are saved for human labeling.

Usage
    python baseline.py --data data/naver_shopping.txt --out results
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from common import SEED, load_reviews





def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",   # character n-grams inside word boundaries
            ngram_range=(2, 5),
            min_df=3,
            max_features=300_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")),
    ])


def evaluate(df, out):
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    model = make_model()
    t0 = time.time()
    model.fit(train["text"], train["label"])
    train_sec = time.time() - t0
    pred = model.predict(test["text"])

    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": round(accuracy_score(test["label"], pred), 4),
        "macro_f1": round(f1_score(test["label"], pred, average="macro"), 4),
        "train_seconds": round(train_sec, 1),
    }
    report = classification_report(test["label"], pred, target_names=["negative", "positive"], digits=4)
    (out / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "baseline_report.txt").write_text(report)
    print(json.dumps(metrics, indent=2))
    print(report)

    # The character chunks that push hardest towards each label (useful for the paper).
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    top = pd.DataFrame({
        "most_negative": vocab[order[:30]],
        "most_positive": vocab[order[::-1][:30]],
    })
    top.to_csv(out / "top_features.csv", index=False, encoding="utf-8-sig")
    return model


def find_mismatches(df, out, folds=5, threshold=0.9):
    """Score every review with a model that did not train on it."""
    p_pos = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for i, (tr, te) in enumerate(skf.split(df["text"], df["label"]), 1):
        model = make_model()
        model.fit(df["text"].iloc[tr], df["label"].iloc[tr])
        p_pos[te] = model.predict_proba(df["text"].iloc[te])[:, 1]
        print(f"fold {i}/{folds} done")

    df = df.copy()
    df["p_text_positive"] = p_pos.round(4)
    # How strongly the text disagrees with the rating (0 = agrees, 1 = fully disagrees).
    df["disagreement"] = np.where(df["label"] == 1, 1 - p_pos, p_pos).round(4)
    df["mismatch_candidate"] = df["disagreement"] >= threshold

    cand = df[df["mismatch_candidate"]].sort_values("disagreement", ascending=False)
    cols = ["review_id", "rating", "p_text_positive", "disagreement", "text"]
    cand[cols].to_csv(out / "mismatch_candidates.csv", index=False, encoding="utf-8-sig")
    df[cols].to_csv(out / "all_scores.csv", index=False, encoding="utf-8-sig")

    summary = {
        "threshold": threshold,
        "n_reviews": len(df),
        "n_candidates": int(len(cand)),
        "share_candidates": round(len(cand) / len(df), 4),
        "candidates_by_rating": {int(k): int(v) for k, v in cand["rating"].value_counts().sort_index().items()},
    }
    (out / "mismatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/naver_shopping.txt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--skip-mismatch", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_reviews(args.data)
    print(f"{len(df)} reviews after removing duplicates")
    evaluate(df, out)
    if not args.skip_mismatch:
        find_mismatches(df, out, threshold=args.threshold)


if __name__ == "__main__":
    main()

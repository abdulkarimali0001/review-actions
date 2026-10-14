"""
Ablation: does the baseline work better with character pieces or with Korean morphemes?

Same data cleaning, same 80/20 split and same classifier (logistic regression) as baseline.py.
Only the way text is cut into pieces changes:

  char_ngrams      character 2-5-grams inside words (our baseline)
  words            whitespace words + word pairs (naive; Korean endings stay attached)
  morphemes        Kiwi morpheme analyzer: 배송은 -> 배송/NNG + 은/JX, word parts + pairs
  morphemes+char   both feature sets together

Usage:  pip install kiwipiepy
        python ablation_tokenizer.py
Output: results/ablation_tokenizer.json
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from common import SEED, load_reviews





def morpheme_strings(texts):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    out = []
    for toks in kiwi.tokenize(texts):
        out.append(" ".join(f"{t.form}/{t.tag}" for t in toks))
    return out


def vectorizers():
    return {
        "char_ngrams": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                               max_features=300_000, sublinear_tf=True),
        "words": lambda: TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                                         max_features=300_000, sublinear_tf=True),
        "morphemes": lambda: TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                                             max_features=300_000, sublinear_tf=True),
    }


def fit_eval(Xtr, Xte, ytr, yte):
    t0 = time.time()
    clf = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {"accuracy": round(accuracy_score(yte, pred), 4),
            "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
            "train_seconds": round(time.time() - t0, 1), "features": int(Xtr.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/naver_shopping.txt")
    ap.add_argument("--out", default="results/ablation_tokenizer.json")
    args = ap.parse_args()

    df = load_reviews(args.data)
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    ytr, yte = train["label"].values, test["label"].values
    print(f"train {len(train)} / test {len(test)}")

    t0 = time.time()
    morph_tr = morpheme_strings(train["text"].tolist())
    morph_te = morpheme_strings(test["text"].tolist())
    morph_secs = round(time.time() - t0, 1)
    print(f"Kiwi morpheme analysis: {morph_secs}s")

    V = vectorizers()
    inputs = {"char_ngrams": (train["text"], test["text"]), "words": (train["text"], test["text"]),
              "morphemes": (morph_tr, morph_te)}
    mats, results = {}, {}
    for name, (a, b) in inputs.items():
        vec = V[name]()
        Xtr, Xte = vec.fit_transform(a), vec.transform(b)
        mats[name] = (Xtr, Xte)
        results[name] = fit_eval(Xtr, Xte, ytr, yte)
        print(name, results[name])
    Xtr = hstack([mats["morphemes"][0], mats["char_ngrams"][0]]).tocsr()
    Xte = hstack([mats["morphemes"][1], mats["char_ngrams"][1]]).tocsr()
    results["morphemes+char"] = fit_eval(Xtr, Xte, ytr, yte)
    print("morphemes+char", results["morphemes+char"])
    results["morpheme_analysis_seconds_all_reviews"] = morph_secs

    results["note"] = ("Same split as baseline.py. Differences under ~0.003 accuracy are within "
                       "run-to-run noise for a 40k test set (95% CI half-width ≈ ±0.0026).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

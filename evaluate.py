"""
Evaluate mismatch detection against a set of labels (RQ1 + RQ2).

Labels can be the human gold labels or the LLM labels; the script reports
which one it used, so results are never mixed up in the paper.

What it reports
- RQ1: estimated mismatch rate across ALL reviews (sampling weights undo the
  oversampling of likely mismatches), overall and per star rating.
- RQ2: how well each detector finds mismatches: precision, recall, F1,
  both on the sample and weighted to the full dataset.
  Detectors: the TF-IDF baseline (always), plus any prediction files you
  pass with --pred (e.g. KcELECTRA or Gemini output).
- Which aspects carry the opposite sentiment in mismatched reviews (preview of RQ4).
- If human labels exist for some items, agreement between the LLM labels and humans.

Definitions
- mismatch: 4-5 stars with negative text, or 1-2 stars with positive text.
- mixed text and neutral text are NOT counted as mismatches (reported separately).

Usage
    python evaluate.py --labels annotation/claude_labels.csv --name claude
    python evaluate.py --labels annotation/gold_labels.csv --name human
    python evaluate.py --labels annotation/claude_labels.csv --pred results/kcelectra_scores.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ASPECTS = ["배송_delivery", "가격_price", "품질_quality", "포장_packaging", "서비스_service", "기타_other"]


def load_labels(path):
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "gold_sentiment" in lab.columns:  # output of agreement.py
        lab = lab.rename(columns={"gold_sentiment": "text_sentiment"})
    lab["item"] = lab["item"].astype(int)
    lab["text_sentiment"] = lab["text_sentiment"].str.strip().str.lower()
    return lab


def mismatch_from(rating, sentiment):
    high = rating >= 4
    return (high & (sentiment == "neg")) | (~high & (sentiment == "pos"))


def scores(y_true, y_pred, w):
    """Precision/recall/F1 on the sample and weighted to the full dataset."""
    out = {}
    for name, ww in [("sample", np.ones(len(w))), ("weighted", w)]:
        tp = (ww * (y_true & y_pred)).sum()
        fp = (ww * (~y_true & y_pred)).sum()
        fn = (ww * (y_true & ~y_pred)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    return out


def weighted_rate(flag, w):
    return round(float((flag * w).sum() / w.sum()), 4)


def evaluate_detector(df, p_text_positive, name):
    """Turn a text-positivity score into mismatch flags and score them."""
    disagreement = np.where(df["rating"] >= 4, 1 - p_text_positive, p_text_positive)
    res = {}
    for t in [0.5, 0.7, 0.9]:
        pred = disagreement >= t
        res[f"threshold_{t}"] = scores(df["mismatch"].values, pred, df["weight"].values)
    # Best threshold on the sample (report it as tuned, not as a fair test number).
    best = max(np.arange(0.3, 0.99, 0.01),
               key=lambda t: scores(df["mismatch"].values, disagreement >= t, df["weight"].values)["sample"]["f1"])
    res["best_threshold_on_sample"] = round(float(best), 2)
    res["best"] = scores(df["mismatch"].values, disagreement >= best, df["weight"].values)
    print(f"\n[{name}] F1 at 0.9: {res['threshold_0.9']['sample']['f1']}  "
          f"(weighted {res['threshold_0.9']['weighted']['f1']}); best threshold {res['best_threshold_on_sample']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="annotation/sample_key_DO_NOT_SHARE.csv")
    ap.add_argument("--labels", default="annotation/claude_labels.csv")
    ap.add_argument("--name", default=None, help="label source name used in output files, e.g. claude or human")
    ap.add_argument("--pred", nargs="*", default=[],
                    help="CSV files with columns item,p_text_positive from other models")
    ap.add_argument("--consistency", nargs="*", default=[],
                    help="CSV files with columns item,consistent (LLM shown stars + text)")
    ap.add_argument("--human", default="annotation/gold_labels.csv",
                    help="human gold labels, compared with --labels when present")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    name = args.name or Path(args.labels).stem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(args.key)
    lab = load_labels(args.labels)
    df = key.merge(lab.drop(columns=[c for c in ["text", "pilot"] if c in lab.columns]), on="item")
    df = df[df["text_sentiment"].isin(["pos", "neg", "mixed", "neutral"])].copy()
    df["mismatch"] = mismatch_from(df["rating"], df["text_sentiment"])
    df["mixed"] = df["text_sentiment"] == "mixed"
    w = df["weight"]

    report = {"label_source": name, "labels_file": str(args.labels), "items_used": int(len(df))}

    # RQ1: prevalence across the whole dataset
    report["rq1_prevalence"] = {
        "mismatch_rate_all_reviews": weighted_rate(df["mismatch"], w),
        "mixed_rate_all_reviews": weighted_rate(df["mixed"], w),
        "mismatch_rate_by_rating": {int(r): weighted_rate(g["mismatch"], g["weight"]) for r, g in df.groupby("rating")},
        "mismatches_in_sample": int(df["mismatch"].sum()),
        "estimated_mismatched_reviews_in_dataset": int(round((df["mismatch"] * w).sum())),
    }

    # RQ2: detectors
    detectors = {"tfidf_baseline": evaluate_detector(df, df["p_text_positive"].values, "tfidf_baseline")}
    for p in args.pred:
        pr = pd.read_csv(p)[["item", "p_text_positive"]].rename(columns={"p_text_positive": "p_other"})
        d2 = df.merge(pr, on="item")
        detectors[Path(p).stem] = evaluate_detector(d2, d2["p_other"].values, Path(p).stem)
    for p in args.consistency:
        c = pd.read_csv(p)[["item", "consistent"]]
        d2 = df.merge(c, on="item")
        pred = ~d2["consistent"].astype(bool).values
        detectors[Path(p).stem] = {"direct_yes_no": scores(d2["mismatch"].values, pred, d2["weight"].values)}
        print(f"\n[{Path(p).stem}] F1: {detectors[Path(p).stem]['direct_yes_no']['sample']['f1']}")
    report["rq2_detection"] = detectors

    # Baseline sentiment accuracy on clearly positive / negative texts
    clear = df[df["text_sentiment"].isin(["pos", "neg"])]
    correct = (clear["p_text_positive"] >= 0.5) == (clear["text_sentiment"] == "pos")
    report["baseline_text_sentiment_accuracy_on_clear_texts"] = {
        "sample (enriched with hard cases)": round(float(correct.mean()), 3),
        "weighted to all reviews": weighted_rate(correct, clear["weight"]),
    }

    # RQ4 preview: aspects with sentiment opposite to the stars, inside mismatches
    mm = df[df["mismatch"]]
    asp = {}
    for a in ASPECTS:
        opposite = np.where(mm["rating"] >= 4, mm[a] == "neg", mm[a] == "pos")
        asp[a] = round(float(opposite.mean()), 3) if len(mm) else None
    report["aspects_opposite_to_rating_in_mismatches"] = asp
    hi = df[(df["rating"] >= 4) & (df["text_sentiment"].isin(["neg", "mixed"]))]
    report["complaints_hidden_in_4_5_star_reviews"] = {
        a: round(float((hi[a] == "neg").mean()), 3) for a in ASPECTS} if len(hi) else {}

    # Agreement with human labels, when they exist
    hp = Path(args.human)
    if hp.exists() and hp.resolve() != Path(args.labels).resolve():
        hum = load_labels(hp)[["item", "text_sentiment"]].rename(columns={"text_sentiment": "human"})
        both = df.merge(hum, on="item")
        both = both[both["human"].isin(["pos", "neg", "mixed", "neutral"])]
        if len(both):
            report["agreement_with_humans"] = {
                "items": int(len(both)),
                "kappa_sentiment": round(cohen_kappa_score(both["text_sentiment"], both["human"]), 3),
                "raw_agreement": round(float((both["text_sentiment"] == both["human"]).mean()), 3),
                "kappa_mismatch": round(cohen_kappa_score(
                    both["mismatch"], mismatch_from(both["rating"], both["human"])), 3),
            }

    path = out / f"evaluation_{name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "rq2_detection"}, indent=2, ensure_ascii=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

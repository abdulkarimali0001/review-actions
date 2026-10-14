"""
Build the human-annotation sample (about 3,000 reviews).

Why stratify
Random sampling would give very few mismatches (they are ~2% of reviews),
so we oversample reviews the baseline flags as likely mismatches. Each
stratum (rating x disagreement band) gets a sampling weight, so prevalence
(RQ1) can still be estimated correctly for the whole dataset later.

Blind labeling
Annotators do NOT see the star rating or any model score. They label the
text only; mismatch is computed afterwards by comparing their label with
the rating. This avoids anchoring on the stars or on our model.

Usage
    python sample_for_annotation.py --scores results/all_scores.csv --out annotation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# Reviews per rating in each disagreement band (4 ratings -> x4 in total).
PLAN = {"high": 250, "mid": 200, "low": 300}  # 1,000 + 800 + 1,200 = 3,000
PILOT_SIZE = 200
ASPECTS = ["배송_delivery", "가격_price", "품질_quality", "포장_packaging", "서비스_service", "기타_other"]


def band(d):
    if d >= 0.9:
        return "high"
    if d >= 0.5:
        return "mid"
    return "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/all_scores.csv")
    ap.add_argument("--out", default="annotation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    df["band"] = df["disagreement"].apply(band)

    parts = []
    for (rating, b), grp in df.groupby(["rating", "band"]):
        n = min(PLAN[b], len(grp))
        s = grp.sample(n=n, random_state=SEED).copy()
        s["stratum"] = f"{rating}star_{b}"
        s["stratum_population"] = len(grp)
        s["stratum_sample"] = n
        s["weight"] = len(grp) / n  # how many real reviews each sampled one stands for
        parts.append(s)
    sample = pd.concat(parts)

    # Shuffle so annotators never see reviews grouped by stratum.
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["item"] = np.arange(1, len(sample) + 1)
    sample["pilot"] = sample["item"] <= PILOT_SIZE

    # Hidden key: everything needed later, never given to annotators.
    key_cols = ["item", "review_id", "rating", "p_text_positive", "disagreement",
                "band", "stratum", "stratum_population", "stratum_sample", "weight", "pilot", "text"]
    sample[key_cols].to_csv(out / "sample_key_DO_NOT_SHARE.csv", index=False, encoding="utf-8-sig")

    # Annotator sheet: text only, empty label columns.
    sheet = sample[["item", "pilot", "text"]].copy()
    sheet["text_sentiment"] = ""   # pos / neg / mixed / neutral
    for a in ASPECTS:
        sheet[a] = ""              # pos / neg / blank (not mentioned)
    sheet["unclear"] = ""          # 1 if you cannot judge
    sheet["note"] = ""
    for name in ["annotator_A", "annotator_B"]:
        sheet.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary = sample.groupby("stratum").agg(sampled=("item", "size"), population=("stratum_population", "first"))
    summary.to_csv(out / "sample_summary.csv", encoding="utf-8-sig")
    print(summary.to_string())
    print(f"\nTotal sampled: {len(sample)} (pilot: {int(sample['pilot'].sum())})")


if __name__ == "__main__":
    main()

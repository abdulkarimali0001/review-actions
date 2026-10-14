"""
Check annotator agreement, list disagreements, and compute mismatch labels.

Steps
1. Reads pilot_annotator_A.xlsx / pilot_annotator_B.xlsx (Excel sheets with dropdowns) if they
   contain labels, otherwise annotator_A.csv / annotator_B.csv (same items, labeled independently).
2. Reports Cohen's kappa for text sentiment and for each aspect.
3. Writes disagreements.csv for a third person to resolve (fill `final_sentiment`).
4. If adjudicated.csv exists, merges everything with the hidden key and computes:
   - mismatch: rating 4-5 with negative text, or rating 1-2 with positive text
   - partial:  text is "mixed" (praise and complaint together)
   - weighted prevalence estimates for the whole dataset (RQ1)

Usage
    python agreement.py --dir annotation            # all items labeled so far
    python agreement.py --dir annotation --pilot    # pilot items only
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

SENTIMENTS = {"pos", "neg", "mixed", "neutral"}
ASPECTS = ["배송_delivery", "가격_price", "품질_quality", "포장_packaging", "서비스_service", "기타_other"]
ASPECT_VALUES = {"pos", "neg", ""}


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        df[c] = df[c].str.strip().str.lower() if c not in ("text", "note") else df[c]
    return df


def load_annotator(d, who):
    """Pilot labels, in order of preference: the export of label_tool.html (pilot_annotator_X.csv),
    the Excel sheet (pilot_annotator_X.xlsx), then the plain blind sheet (annotator_X.csv)."""
    for x in [d / f"pilot_annotator_{who}.csv", d / f"pilot_annotator_{who}.xlsx"]:
        if not x.exists():
            continue
        if x.suffix == ".csv":
            df = pd.read_csv(x, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        else:
            df = pd.read_excel(x, sheet_name="Label here", dtype=str).fillna("")
        if (df["text_sentiment"].str.strip() != "").any():
            df["pilot"] = "true"
            for c in df.columns:
                df[c] = df[c].astype(str).str.strip()
                if c not in ("text", "note"):
                    df[c] = df[c].str.lower()
            df["item"] = df["item"].str.replace(r"\.0$", "", regex=True)
            print(f"Annotator {who}: reading {x.name}")
            return df
    return load(d / f"annotator_{who}.csv")


def check_values(df, name, aspects):
    bad = df[(df["text_sentiment"] != "") & (~df["text_sentiment"].isin(SENTIMENTS))]
    for a in aspects:
        bad = pd.concat([bad, df[~df[a].isin(ASPECT_VALUES)]])
    if len(bad):
        print(f"WARNING {name}: {len(bad)} rows have values outside the allowed set "
              f"(sentiment: pos/neg/mixed/neutral; aspects: pos/neg/blank). Items: "
              f"{sorted(bad['item'].astype(int).unique())[:20]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="annotation")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)

    A, B = load_annotator(d, "A"), load_annotator(d, "B")
    aspects = [c for c in A.columns if c in ASPECTS]
    check_values(A, "A", aspects)
    check_values(B, "B", aspects)

    m = A.merge(B, on="item", suffixes=("_a", "_b"))
    if args.pilot:
        m = m[m["pilot_a"] == "true"]
    done = m[(m["text_sentiment_a"] != "") & (m["text_sentiment_b"] != "")]
    if done.empty:
        print("No items labeled by both annotators yet.")
        return

    report = {"items_labeled_by_both": len(done)}
    report["kappa_text_sentiment"] = round(cohen_kappa_score(done["text_sentiment_a"], done["text_sentiment_b"]), 3)
    report["raw_agreement_text_sentiment"] = round((done["text_sentiment_a"] == done["text_sentiment_b"]).mean(), 3)
    for a in aspects:
        x, y = done[f"{a}_a"].replace("", "none"), done[f"{a}_b"].replace("", "none")
        report[f"kappa_{a}"] = round(cohen_kappa_score(x, y), 3) if len(set(x) | set(y)) > 1 else None
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nRule of thumb: kappa >= 0.6 is substantial agreement. Below that, revise the guide before labeling more.")

    dis = done[done["text_sentiment_a"] != done["text_sentiment_b"]]
    out = dis[["item", "text_a", "text_sentiment_a", "text_sentiment_b", "note_a", "note_b"]].rename(columns={"text_a": "text"})
    out["final_sentiment"] = ""
    out.to_csv(d / "disagreements.csv", index=False, encoding="utf-8-sig")
    print(f"{len(dis)} sentiment disagreements written to {d / 'disagreements.csv'}")
    (d / ("agreement_pilot.json" if args.pilot else "agreement.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    adj_path = d / "adjudicated.csv"
    if not adj_path.exists():
        print("When disagreements are resolved, save the file as adjudicated.csv and run again.")
        return

    adj = pd.read_csv(adj_path, dtype=str, keep_default_na=False)[["item", "final_sentiment"]]
    done = done.merge(adj, on="item", how="left")
    agree = done["text_sentiment_a"] == done["text_sentiment_b"]
    done["gold_sentiment"] = done["text_sentiment_a"].where(agree, done["final_sentiment"].str.strip().str.lower())
    done = done[done["gold_sentiment"].isin(SENTIMENTS)]

    key = pd.read_csv(d / "sample_key_DO_NOT_SHARE.csv")
    done["item"] = done["item"].astype(int)
    gold = key.merge(done[["item", "gold_sentiment"]], on="item")
    high = gold["rating"] >= 4
    gold["mismatch"] = (high & (gold["gold_sentiment"] == "neg")) | (~high & (gold["gold_sentiment"] == "pos"))
    gold["partial"] = gold["gold_sentiment"] == "mixed"
    gold.to_csv(d / "gold_labels.csv", index=False, encoding="utf-8-sig")

    w = gold["weight"]
    prev = {
        "gold_items": len(gold),
        "mismatch_in_sample": round(gold["mismatch"].mean(), 4),
        "estimated_mismatch_rate_all_reviews": round((gold["mismatch"] * w).sum() / w.sum(), 4),
        "estimated_mixed_rate_all_reviews": round((gold["partial"] * w).sum() / w.sum(), 4),
        "by_rating": {
            int(r): round((g["mismatch"] * g["weight"]).sum() / g["weight"].sum(), 4)
            for r, g in gold.groupby("rating")
        },
    }
    (d / "prevalence.json").write_text(json.dumps(prev, indent=2))
    print(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()

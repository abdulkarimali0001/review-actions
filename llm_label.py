"""
Zero-shot LLM experiments on the 3,000-review sample (RQ2 + reliability).

Two tasks:
  label        Blind labeling, same as the human annotators: text only, no stars.
               Gives sentiment + aspects. Mismatch is computed afterwards from the rating.
  consistency  The LLM sees stars AND text and answers "do they match?"
               (the setup used in the Brazilian Portuguese study).

Reliability: --runs 3 repeats the task with a "run" tag. With temperature 0
runs are often identical; a second prompt wording (--variant B) tests whether
the answer depends on how the question is asked. Fleiss' kappa is reported.

Outputs
  annotation/gemini_labels.csv       same columns as the annotator sheets (task=label)
  results/gemini_label_scores.csv    item, p_text_positive -> use with evaluate.py --pred
  results/gemini_consistency.csv     item, consistent (task=consistency)
  results/llm_reliability.json       Fleiss' kappa across runs / variants

Usage
  export GEMINI_API_KEY=...            (Windows: set GEMINI_API_KEY=...)
  python llm_label.py --task label --runs 3
  python llm_label.py --task label --variant B
  python llm_label.py --task consistency
  python llm_label.py --task label --mock --limit 100     # test without a key
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_client import DEFAULT_LABEL_MODEL, LLM

ASPECT_COLS = {"delivery": "배송_delivery", "price": "가격_price", "quality": "품질_quality",
               "packaging": "포장_packaging", "service": "서비스_service"}
SENT_TO_P = {"pos": 1.0, "neg": 0.0, "mixed": 0.5, "neutral": 0.5}

PROMPT_LABEL = {
    "A": """TASK: LABEL_REVIEWS
You label Korean e-commerce product reviews. For each review give:
- "sentiment": overall feeling of the TEXT: "pos", "neg", "mixed" (clear praise AND clear complaint, no winner),
  "neutral" (no opinion), or "unclear" (meaningless text).
- "aspects": for delivery, price, quality, packaging, service: "pos", "neg" or "none" (not mentioned).
  quality = the product itself (taste, size, material, function, design). service = seller response, exchange, refund.
Judge meaning, not keywords ("나쁘지 않아요" is positive). Sarcasm counts as what the writer means.
Return JSON: {{"labels": [{{"id": <id>, "sentiment": "...", "aspects": {{"delivery": "...", "price": "...", "quality": "...", "packaging": "...", "service": "..."}}}}]}}
One entry per review, same ids.

REVIEWS:
{reviews}""",
    "B": """TASK: LABEL_REVIEWS
아래는 한국 온라인 쇼핑몰 상품 리뷰입니다. 각 리뷰에 대해 판단하세요.
- "sentiment": 리뷰 글 전체의 감정. "pos"(만족), "neg"(불만), "mixed"(칭찬과 불만이 비슷하게 섞임), "neutral"(의견 없음), "unclear"(의미 없는 글)
- "aspects": delivery(배송), price(가격), quality(상품 자체: 맛, 사이즈, 소재, 기능, 디자인), packaging(포장), service(판매자 응대, 교환, 환불)
  각각 "pos", "neg", "none"(언급 없음)
JSON으로만 답하세요: {{"labels": [{{"id": <id>, "sentiment": "...", "aspects": {{"delivery": "...", "price": "...", "quality": "...", "packaging": "...", "service": "..."}}}}]}}

REVIEWS:
{reviews}""",
}

PROMPT_CONSISTENCY = """TASK: RATING_TEXT_CONSISTENCY
Each Korean product review below has a star rating (1-5) and a text.
Decide if the text is consistent with the star rating.
Inconsistent means: high stars (4-5) but the text is clearly negative, or low stars (1-2) but the text is clearly positive.
Mixed or neutral texts are consistent.
Return JSON: {{"labels": [{{"id": <id>, "consistent": true or false}}]}}

REVIEWS:
{reviews}"""


def fleiss_kappa(table):
    """table: items x categories counts."""
    table = np.asarray(table, dtype=float)
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def run_label(llm, sheet, variant, run, batch):
    rows = []
    for i in range(0, len(sheet), batch):
        chunk = sheet.iloc[i:i + batch]
        reviews = "\n".join(f"[{r.item}] {str(r.text).replace(chr(10), ' ')}" for r in chunk.itertuples())
        prompt = PROMPT_LABEL[variant].format(reviews=reviews)
        try:
            ans = llm.json(prompt, tag=f"label-{variant}-run{run}")
        except Exception as e:
            print(f"  batch {i // batch} failed: {e}")
            continue
        for lab in ans.get("labels", []):
            asp = lab.get("aspects", {}) or {}
            rows.append({"item": int(lab["id"]), "sentiment": str(lab.get("sentiment", "")).lower(),
                         **{ASPECT_COLS[a]: str(asp.get(a, "none")).lower() for a in ASPECT_COLS}})
        done = min(i + batch, len(sheet))
        if (i // batch) % 20 == 0 or done == len(sheet):
            print(f"  prompt {variant}, run {run}: {done}/{len(sheet)} reviews")
    return pd.DataFrame(rows).drop_duplicates("item")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["label", "consistency"], default="label")
    ap.add_argument("--variant", choices=["A", "B"], default="A")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--sheet", default="annotation/annotator_A.csv", help="blind sheet (text only)")
    ap.add_argument("--key", default="annotation/sample_key_DO_NOT_SHARE.csv", help="needed for the consistency task")
    ap.add_argument("--model", default=DEFAULT_LABEL_MODEL)
    ap.add_argument("--batch", type=int, default=25, help="reviews per request (saves free-tier quota)")
    ap.add_argument("--rpm", type=int, default=10, help="requests per minute allowed by your tier")
    ap.add_argument("--limit", type=int, default=None, help="only the first N reviews (for a cheap test)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    tag = "mock_" if args.mock else ""
    llm = LLM(args.model, f"results/llm_cache/{tag}{args.model}.jsonl", rpm=args.rpm, mock=args.mock)
    sheet = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    if args.limit:
        sheet = sheet.head(args.limit)
    Path("results").mkdir(exist_ok=True)
    name = f"{tag}gemini"

    if args.task == "consistency":
        key = pd.read_csv(args.key)[["item", "rating"]]
        df = sheet[["item", "text"]].assign(item=lambda d: d["item"].astype(int)).merge(key, on="item")
        rows = []
        for i in range(0, len(df), args.batch):
            chunk = df.iloc[i:i + args.batch]
            reviews = "\n".join(f"[{r.item}] stars={r.rating} | {str(r.text).replace(chr(10), ' ')}"
                                for r in chunk.itertuples())
            ans = llm.json(PROMPT_CONSISTENCY.format(reviews=reviews), tag="consistency")
            rows += [{"item": int(x["id"]), "consistent": bool(x["consistent"])} for x in ans.get("labels", [])]
        out = pd.DataFrame(rows).drop_duplicates("item")
        out["p_text_positive"] = np.nan  # not a sentiment score; evaluate with --consistency instead
        out.to_csv(f"results/{name}_consistency.csv", index=False)
        print(f"Saved results/{name}_consistency.csv ({len(out)} reviews, {int((~out['consistent']).sum())} inconsistent)")
        return

    runs = [run_label(llm, sheet, args.variant, r, args.batch) for r in range(1, args.runs + 1)]
    first = runs[0]

    # Same format as the human sheets, so evaluate.py and agreement.py can read it.
    lab = sheet[["item", "pilot", "text"]].copy()
    lab["item"] = lab["item"].astype(int)
    lab = lab.merge(first.rename(columns={"sentiment": "text_sentiment"}), on="item", how="left")
    lab["unclear"] = np.where(lab["text_sentiment"] == "unclear", "1", "")
    lab["text_sentiment"] = lab["text_sentiment"].where(lab["text_sentiment"].isin(SENT_TO_P), "")
    for c in ASPECT_COLS.values():
        lab[c] = lab[c].replace("none", "").fillna("")
    lab["기타_other"] = ""
    lab["note"] = f"LLM label ({args.model}, prompt {args.variant}), not human"
    suffix = "" if args.variant == "A" else f"_{args.variant}"
    lab_path = f"annotation/{name}_labels{suffix}.csv"
    lab.fillna("").to_csv(lab_path, index=False, encoding="utf-8-sig")

    scores = lab[lab["text_sentiment"] != ""][["item"]].copy()
    scores["p_text_positive"] = lab.loc[lab["text_sentiment"] != "", "text_sentiment"].map(SENT_TO_P).values
    scores.to_csv(f"results/{name}_label_scores{suffix}.csv", index=False)
    print(f"Saved {lab_path} and results/{name}_label_scores{suffix}.csv "
          f"({int((lab['text_sentiment'] != '').sum())} of {len(lab)} labeled)")

    # Reliability across runs
    rel_path = Path("results/llm_reliability.json")
    rel = json.loads(rel_path.read_text()) if rel_path.exists() else {}
    if len(runs) > 1:
        m = runs[0][["item", "sentiment"]]
        for j, r in enumerate(runs[1:], 2):
            m = m.merge(r[["item", "sentiment"]].rename(columns={"sentiment": f"s{j}"}), on="item")
        cols = [c for c in m.columns if c != "item"]
        cats = sorted(set(m[cols].values.ravel()))
        table = [[list(row).count(c) for c in cats] for row in m[cols].values]
        rel[f"{name}_{args.variant}_runs"] = {
            "runs": len(runs), "items": len(m), "fleiss_kappa_sentiment": round(fleiss_kappa(table), 3),
            "identical_across_runs": round(float((m[cols].nunique(axis=1) == 1).mean()), 3)}
    other = Path(f"annotation/{name}_labels{'_B' if args.variant == 'A' else ''}.csv")
    if other.exists():
        o = pd.read_csv(other, dtype=str, keep_default_na=False)[["item", "text_sentiment"]]
        o["item"] = o["item"].astype(int)
        both = lab[["item", "text_sentiment"]].merge(o, on="item", suffixes=("_x", "_y"))
        both = both[(both["text_sentiment_x"] != "") & (both["text_sentiment_y"] != "")]
        from sklearn.metrics import cohen_kappa_score
        rel[f"{name}_prompt_A_vs_B"] = {
            "items": len(both),
            "cohen_kappa_sentiment": round(cohen_kappa_score(both["text_sentiment_x"], both["text_sentiment_y"]), 3),
            "agreement": round(float((both["text_sentiment_x"] == both["text_sentiment_y"]).mean()), 3)}
    if rel:
        rel_path.write_text(json.dumps(rel, indent=2))
        print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()

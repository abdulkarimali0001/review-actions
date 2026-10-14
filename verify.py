"""
Stage 6: check recommendations against the reviews, then write the company report (RQ5).

For every claim:
  grounded    the judge sees the claim and ONLY the reviews it cites
              -> supported / partially_supported / not_supported
              + which cited reviews actually mention the problem (citation precision)
  ungrounded  no citations, so the judge sees the claim and ALL evidence reviews of the issue
              (the most generous fair test for the baseline)

Metrics (results/verification_summary.json)
  faithfulness        share of claims judged supported (partially = 0.5)
  citation_precision  share of cited reviews that the judge confirms (grounded only)
  abstention_rate     share of issues where the grounded system declined
  unsupported_recommendations  recommendations with no supported claim behind them

Also writes
  results/company_report.md             verified recommendations per category, with evidence
  results/human_rating_sheet.csv        blind sheet: raters score each recommendation 1-5
  results/human_rating_key_DO_NOT_SHARE.csv  which row is grounded / ungrounded

Usage
  python verify.py
  python verify.py --mock
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from llm_client import DEFAULT_REASON_MODEL, LLM

PROMPT_VERIFY = """TASK: VERIFY_CLAIM
You check whether customer reviews support a claim. Be strict: a claim is supported only if the reviews
clearly state it. Do not use outside knowledge.

CLAIM: {claim}

REVIEWS:
{reviews}

Return JSON: {{"verdict": "supported" | "partially_supported" | "not_supported",
  "relevant_review_ids": [ids of the reviews above that clearly mention the claimed problem],
  "reason": "one short sentence"}}"""

SCORE = {"supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0}


def load(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def fmt(reviews):
    return "\n".join(f"[R{r['review_id']}] ({r['rating']}★) {r['text']}" for r in reviews)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="results/issues/issue_evidence.jsonl")
    ap.add_argument("--model", default=DEFAULT_REASON_MODEL)
    ap.add_argument("--rpm", type=int, default=5)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    tag = "mock_" if args.mock else ""
    llm = LLM(args.model, f"results/llm_cache/{tag}{args.model}.jsonl", rpm=args.rpm, mock=args.mock)
    evidence = {e["issue_id"]: e for e in load(args.evidence)}
    recs = load(f"results/{tag}recommendations_grounded.jsonl") + load(f"results/{tag}recommendations_ungrounded.jsonl")
    if not recs:
        raise SystemExit("Run recommend.py first.")

    claim_rows, rec_rows = [], []
    for rec in recs:
        ev = evidence[rec["issue_id"]]["reviews"]
        by_id = {r["review_id"]: r for r in ev}
        abstained = bool(rec.get("abstain"))
        verdicts = []
        for c in rec.get("claims", []) if not abstained else []:
            if rec["mode"] == "grounded":
                cited = [by_id[i] for i in c.get("review_ids", []) if i in by_id]
                invented = [i for i in c.get("review_ids", []) if i not in by_id]
                shown = cited
            else:
                cited, invented, shown = [], [], ev
            if not shown:
                v = {"verdict": "not_supported", "relevant_review_ids": [], "reason": "no valid citations"}
            else:
                v = llm.json(PROMPT_VERIFY.format(claim=c["text"], reviews=fmt(shown)), tag="verify")
            relevant = set(v.get("relevant_review_ids", [])) & {r["review_id"] for r in shown}
            claim_rows.append({
                "issue_id": rec["issue_id"], "mode": rec["mode"], "claim": c["text"],
                "verdict": v.get("verdict"), "score": SCORE.get(v.get("verdict"), 0.0),
                "cited": len(cited), "cited_confirmed": len(relevant & {r["review_id"] for r in cited}),
                "invented_ids": len(invented), "reason": v.get("reason", ""),
            })
            verdicts.append(SCORE.get(v.get("verdict"), 0.0))
        rec_rows.append({
            "issue_id": rec["issue_id"], "category": rec["category"], "mode": rec["mode"], "abstained": abstained,
            "claims": len(verdicts), "faithfulness": sum(verdicts) / len(verdicts) if verdicts else None,
            "has_supported_claim": any(s >= 1.0 for s in verdicts),
        })
        print(f"  {rec['mode']:10s} {rec['issue_id']}: " + ("abstained" if abstained else f"{len(verdicts)} claims checked"))

    claims, recs_df = pd.DataFrame(claim_rows), pd.DataFrame(rec_rows)
    claims.to_csv(f"results/{tag}verification_claims.csv", index=False, encoding="utf-8-sig")
    summary = {}
    for mode in ["grounded", "ungrounded"]:
        c, r = claims[claims["mode"] == mode], recs_df[recs_df["mode"] == mode]
        answered = r[~r["abstained"]]
        summary[mode] = {
            "issues": int(len(r)),
            "abstention_rate": round(float(r["abstained"].mean()), 3) if len(r) else None,
            "claims_checked": int(len(c)),
            "faithfulness": round(float(c["score"].mean()), 3) if len(c) else None,
            "share_claims_fully_supported": round(float((c["verdict"] == "supported").mean()), 3) if len(c) else None,
            "unsupported_recommendations": int((~answered["has_supported_claim"]).sum()),
        }
        if mode == "grounded" and len(c):
            summary[mode]["citation_precision"] = round(float(c["cited_confirmed"].sum() / max(c["cited"].sum(), 1)), 3)
            summary[mode]["invented_review_ids"] = int(c["invented_ids"].sum())
    Path(f"results/{tag}verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Blind human rating sheet: same issue, both modes, shuffled, mode hidden.
    rng = random.Random(42)
    sheet = []
    for rec in recs:
        if rec.get("abstain"):
            continue
        sheet.append({"issue": rec["issue_name"], "category": rec["category"],
                      "problem_summary": rec.get("problem_summary", ""), "recommendation": rec.get("recommendation", ""),
                      "_mode": rec["mode"], "_issue_id": rec["issue_id"]})
    rng.shuffle(sheet)
    sh = pd.DataFrame(sheet)
    sh.insert(0, "row", range(1, len(sh) + 1))
    for col in ["specific_1to5", "actionable_1to5", "matches_reviews_1to5", "comment"]:
        sh[col] = ""
    sh[["row", "_mode", "_issue_id"]].to_csv(f"results/{tag}human_rating_key_DO_NOT_SHARE.csv", index=False)
    sh.drop(columns=["_mode", "_issue_id"]).to_csv(f"results/{tag}human_rating_sheet.csv", index=False, encoding="utf-8-sig")

    # Company report: only grounded recommendations, with their verification result.
    claims_g = claims[claims["mode"] == "grounded"]
    lines = ["# What to fix: evidence-based recommendations from customer reviews", "",
             "Each recommendation cites real reviews and was checked claim by claim. "
             "Issues where the evidence was too weak are listed as *not enough evidence* instead of guessed.", ""]
    for cat in sorted({r["category"] for r in recs}):
        lines += [f"## {cat}", ""]
        for rec in [r for r in recs if r["mode"] == "grounded" and r["category"] == cat]:
            ev = evidence[rec["issue_id"]]
            lines.append(f"### {rec['issue_name']}  ({rec['complaints']} complaints, "
                         f"{ev.get('hidden_in_4_5_star', '?')} of them in 4-5★ reviews)")
            if rec.get("abstain"):
                lines += [f"*Not enough evidence:* {rec.get('abstain_reason', '')}", ""]
                continue
            cc = claims_g[claims_g["issue_id"] == rec["issue_id"]]
            ok = cc["verdict"].eq("supported").sum()
            by_id = {r["review_id"]: r for r in ev["reviews"]}
            lines.append(f"**Problem:** {rec.get('problem_summary', '')}")
            lines.append(f"**Action ({rec.get('owner', '')}, confidence {rec.get('confidence', '')}):** "
                         f"{rec.get('recommendation', '')}")
            lines.append(f"**Verification:** {ok} of {len(cc)} claims fully supported by the cited reviews.")
            for c in rec.get("claims", []):
                quotes = [f"\"{by_id[i]['text'][:60]}\" ({by_id[i]['rating']}★)" for i in c.get("review_ids", [])[:2] if i in by_id]
                lines.append(f"- {c['text']} " + ("— " + "; ".join(quotes) if quotes else ""))
            lines.append("")
    Path(f"results/{tag}company_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved results/{tag}company_report.md, results/{tag}human_rating_sheet.csv")


if __name__ == "__main__":
    main()

"""
Stage 5: turn issues into recommendations a company can act on (RQ5).

Two modes, same LLM, same issues:
  grounded    The LLM gets the issue's evidence reviews with IDs. It must base every
              claim on those reviews and cite their IDs, and it must ABSTAIN when the
              reviews do not show a concrete, fixable problem.
  ungrounded  Baseline: the LLM only gets the issue name and counts (what a dashboard
              shows) and is asked for advice. No reviews, no citations, no abstention.

Input : results/issues/issue_evidence.jsonl  (from issues.py)
Output: results/recommendations_grounded.jsonl, results/recommendations_ungrounded.jsonl

Usage
  python recommend.py --top 5                  # top 5 specific issues per category
  python recommend.py --top 5 --mock           # test without a key
"""

import argparse
import json
from pathlib import Path

from llm_client import DEFAULT_REASON_MODEL, LLM

ASPECT_EN = {"배송_delivery": "delivery", "가격_price": "price", "품질_quality": "product quality",
             "포장_packaging": "packaging", "서비스_service": "customer service"}

PROMPT_GROUNDED = """TASK: GROUNDED_RECOMMENDATION
You help an online shop decide what to fix. Below are customer reviews (Korean) that our system grouped
into one issue. Each review has an ID like [R123] and its star rating.

Product category: {category}
Aspect: {aspect}
Issue key words: {name}
Size: {complaints} complaints ({low} in 1-2 star reviews, {hidden} hidden in 4-5 star reviews)

Rules:
1. Use ONLY the reviews below. Do not add facts, numbers, causes or comparisons that the reviews do not state.
2. Every claim must cite the review IDs that support it, e.g. [123, 456]. Cite at least 2 reviews per claim.
3. If the reviews do not describe one concrete, fixable problem (for example they are vague, contradictory,
   or mostly positive), set "abstain": true and explain why. Do not guess.
4. The recommendation must be one specific action the shop can take, and must follow from the claims.
5. Write problem_summary, claims and recommendation in English (you may quote short Korean phrases).

Return JSON:
{{"abstain": true/false, "abstain_reason": "...", "problem_summary": "...",
  "claims": [{{"text": "...", "review_ids": [..]}}],
  "recommendation": "...", "owner": "logistics | product | packaging | customer service | pricing | listing/marketing",
  "confidence": "high | medium | low"}}

REVIEWS:
{reviews}"""

PROMPT_UNGROUNDED = """TASK: UNGROUNDED_RECOMMENDATION
You help an online shop decide what to fix. Our review dashboard shows this issue:

Product category: {category}
Aspect: {aspect}
Issue key words (Korean): {name}
Size: {complaints} complaints ({low} in 1-2 star reviews, {hidden} in 4-5 star reviews)

Describe the problem customers have and give one specific action the shop should take.
Write in English.
Return JSON:
{{"problem_summary": "...", "claims": [{{"text": "..."}}],
  "recommendation": "...", "owner": "logistics | product | packaging | customer service | pricing | listing/marketing",
  "confidence": "high | medium | low"}}"""


def load_issues(path, top, include_broad):
    issues = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not include_broad:
        issues = [i for i in issues if i.get("specific", True)]
    picked, seen = [], {}
    for i in sorted(issues, key=lambda x: (x["category"], x["rank_in_category"])):
        seen[i["category"]] = seen.get(i["category"], 0) + 1
        if seen[i["category"]] <= top:
            picked.append(i)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="results/issues/issue_evidence.jsonl")
    ap.add_argument("--top", type=int, default=5, help="issues per category")
    ap.add_argument("--include-broad", action="store_true", help="also send broad (mixed-topic) issues")
    ap.add_argument("--model", default=DEFAULT_REASON_MODEL)
    ap.add_argument("--rpm", type=int, default=5)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    tag = "mock_" if args.mock else ""
    llm = LLM(args.model, f"results/llm_cache/{tag}{args.model}.jsonl", rpm=args.rpm, mock=args.mock)
    issues = load_issues(args.evidence, args.top, args.include_broad)
    print(f"{len(issues)} issues -> 2 requests each")

    out_g = open(f"results/{tag}recommendations_grounded.jsonl", "w", encoding="utf-8")
    out_u = open(f"results/{tag}recommendations_ungrounded.jsonl", "w", encoding="utf-8")
    for n, iss in enumerate(issues, 1):
        info = dict(category=iss["category"], aspect=ASPECT_EN.get(iss["aspect"], iss["aspect"]),
                    name=iss["name"], complaints=iss["complaints"],
                    low=iss.get("low_star_reviews", "?"), hidden=iss.get("hidden_in_4_5_star", "?"))
        reviews = "\n".join(f"[R{r['review_id']}] ({r['rating']}★) {r['text']}" for r in iss["reviews"])
        base = {"issue_id": iss["issue_id"], "category": iss["category"], "aspect": iss["aspect"],
                "issue_name": iss["name"], "complaints": iss["complaints"], "model": args.model}
        try:
            g = llm.json(PROMPT_GROUNDED.format(reviews=reviews, **info), tag="grounded")
            u = llm.json(PROMPT_UNGROUNDED.format(**info), tag="ungrounded")
        except Exception as e:
            print(f"  {iss['issue_id']} failed: {e}")
            continue
        out_g.write(json.dumps({**base, "mode": "grounded", **g}, ensure_ascii=False) + "\n")
        out_u.write(json.dumps({**base, "mode": "ungrounded", **u}, ensure_ascii=False) + "\n")
        print(f"  {n}/{len(issues)} {iss['issue_id']}: {'ABSTAIN' if g.get('abstain') else g.get('confidence')}")
    out_g.close()
    out_u.close()
    print(f"Saved results/{tag}recommendations_grounded.jsonl and results/{tag}recommendations_ungrounded.jsonl")


if __name__ == "__main__":
    main()

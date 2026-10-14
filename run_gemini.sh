#!/bin/bash
# Runs all Gemini steps. Usage (in Terminal):  cd ~/Downloads/review-actions && bash run_gemini.sh
# Add --test to only check that the key works (50 reviews).
set -e
cd "$(dirname "$0")"
set -a; source .env; set +a
python3 -m pip install -q --user google-genai pandas numpy scikit-learn scipy openpyxl matplotlib

echo "== 1. Testing the key on 50 reviews =="
python3 llm_label.py --task label --limit 50 --rpm 10
if [ "$1" == "--test" ]; then echo "Key works. Run again without --test for everything."; exit 0; fi

echo "== 2. Gemini labels for all 3,000 reviews (3 runs) =="
python3 llm_label.py --task label --runs 3 --rpm 10
echo "== 3. Korean prompt (prompt sensitivity) =="
python3 llm_label.py --task label --variant B --rpm 10
echo "== 4. Stars + text consistency =="
python3 llm_label.py --task consistency --rpm 10 || echo ">> Consistency step stopped (probably the daily free limit). Continuing; run this file again tomorrow to finish it (answers so far are cached)."
echo "== 5. Recommendations (grounded vs ungrounded) =="
python3 recommend.py --top 5 --rpm 5
echo "== 6. Verification + company report =="
python3 verify.py --rpm 5
echo "== 7. Compare all models =="
CONS=""; [ -f results/gemini_consistency.csv ] && CONS="--consistency results/gemini_consistency.csv"
python3 evaluate.py --labels annotation/claude_labels.csv --name claude \
  --pred results/gemini_label_scores.csv results/kcelectra_scores.csv $CONS
echo "== 8. Confidence intervals, errors, cost, cascade =="
python3 analysis.py
python3 figures.py
echo "All done. See results/company_report.md, results/verification_summary.json and results/analysis_claude.md"

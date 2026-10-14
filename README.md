# From Reviews to Reliable Actions

Turning Korean e-commerce reviews into **trustworthy, evidence-backed recommendations** for companies:
filter unreliable reviews (rating–text mismatch), find what customers complain about, rank the problems,
and generate recommendations that cite real reviews and are checked claim by claim.

Data: Naver Shopping reviews, 200,000 reviews with 1, 2, 4 or 5 stars.

## 한국어 요약

네이버 쇼핑 리뷰 20만 건을 이용해 (1) 별점과 리뷰 내용이 맞지 않는 리뷰를 찾고, (2) 신뢰할 수 없는 리뷰를 걸러낸 뒤
제품 카테고리별 불만 사항을 찾아 우선순위를 매기며, (3) 실제 리뷰를 근거로 인용하는 LLM 개선 제안을 만들고
주장 하나하나를 리뷰와 대조해 검증합니다. 기준 모델은 문자 n-gram TF-IDF + 로지스틱 회귀이며,
KcELECTRA 파인튜닝과 Gemini 제로샷 분류와 비교합니다.

## Status at a glance

| Stage | Script | Status |
| --- | --- | --- |
| 1. Sentiment baseline + mismatch candidates | `baseline.py` | Done |
| Annotation sample (3,000 reviews) | `sample_for_annotation.py` | Done |
| LLM labels for the sample | `annotation/claude_labels.csv` | Done (Claude, **not human**) |
| Human check of the 200 pilot rows | `annotation/label_tool.html` (or the Excel sheets) | **To do: two group members, ~45 min each** |
| Agreement + gold labels | `agreement.py` | Ready, waits for human labels |
| RQ1 prevalence + RQ2 detection | `evaluate.py` | Done on LLM labels |
| Tokenizer ablation (characters vs morphemes) | `ablation_tokenizer.py` | Done |
| Fine-tuned KcELECTRA | `kcelectra_finetune.ipynb` (run on Colab T4: `kcelectra_colab_script.py`) | Done: `results/kcelectra_scores.csv`, `kcelectra_metrics.json` |
| Gemini zero-shot labels + reliability | `llm_label.py` | Labels done; consistency task 93/120 batches (free daily limit; finishes on the next run) |
| 2–4. Trust filter, aspects, issues, ranking, RQ3 | `issues.py` | Done |
| Confidence intervals, error analysis, cost, cascade | `analysis.py` | Done on LLM labels |
| Paper figures | `figures.py` → `results/figures/` | Done |
| 5. Grounded vs ungrounded recommendations | `recommend.py` | **To do: double-click `run_gemini.command` (resumes from cache)** |
| 6. Verification + company report + rating sheet | `verify.py` | **To do: runs after `recommend.py`** |

All LLM scripts were tested end to end with `--mock` (fake answers, no key).

## Setup

```bash
pip install -r requirements.txt
mkdir -p data
curl -L -o data/naver_shopping.txt https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt
```

For the Gemini steps, get a free key at https://aistudio.google.com/apikey and set it:

```bash
export GEMINI_API_KEY=your_key        # Windows: set GEMINI_API_KEY=your_key
```

Model IDs change; check https://ai.google.dev/gemini-api/docs/models. Defaults: `gemini-3.5-flash-lite` for labeling,
`gemini-3.8-flash` for recommendations and verification (change with `--model`). If you hit the free-tier limit,
lower `--rpm`; answers are cached in `results/llm_cache/`, so rerunning continues where it stopped.

## Run order

```bash
# Stage 1 and the annotation sample (already done; rerun only if needed)
python baseline.py
python sample_for_annotation.py

# RQ1 + RQ2 on the LLM labels
python evaluate.py --labels annotation/claude_labels.csv --name claude

# Stages 2-4: trust filter, aspects, issues, ranking, RQ3  (~2 min)
python issues.py

# Gemini zero-shot experiments (3,000 reviews, 25 per request = 120 requests per run)
python llm_label.py --task label --runs 3       # blind labels + run-to-run reliability
python llm_label.py --task label --variant B    # Korean prompt, tests prompt sensitivity
python llm_label.py --task consistency          # stars + text, "do they match?"
python evaluate.py --labels annotation/claude_labels.csv --name claude \
    --pred results/gemini_label_scores.csv results/kcelectra_scores.csv \
    --consistency results/gemini_consistency.csv

# Confidence intervals, error analysis, cost/speed, cascade, then the paper figures
python analysis.py
python figures.py

# Stages 5-6: recommendations and verification  (top 5 issues x 7 categories = 70 requests + checks)
python recommend.py --top 5
python verify.py

# Human pilot labels: open annotation/label_tool.html, export, put the CSV in annotation/
python agreement.py --dir annotation --pilot
python evaluate.py --labels annotation/claude_labels.csv --name claude   # adds LLM-vs-human agreement
python analysis.py --labels annotation/gold_labels.csv --name human      # all analyses on human labels
```

## About the labels (important for the paper)

`annotation/claude_labels.csv` was produced by an LLM (Claude), reading the same blind sheet as the annotators
(text only, no stars). Every row says so in the `note` column. **These are not human labels.**
To make the paper sound:

1. Two group members label the 200 pilot rows independently in `annotation/pilot_annotator_A.xlsx` and `pilot_annotator_B.xlsx`
   (yellow cells, dropdown menus, progress counter on the "How to label" tab; full guide in `annotation/LABELING_GUIDE.md`).
   `agreement.py` reads the Excel files directly; save them under the same names.
2. `agreement.py` gives human–human κ and the human gold labels for those 200.
3. `evaluate.py` then reports LLM–human κ. If it is high, the LLM labels can be used for the full 3,000 **with that limitation stated**.

## Results so far (LLM labels; recompute once human labels exist)

**RQ1 – how common are mismatches** (weighted to all 200k reviews)

| | Value |
| --- | --- |
| Reviews whose text contradicts the stars | 3.3% (~6,400 reviews) |
| By rating: 1★ / 2★ / 4★ / 5★ | 4.3% / 5.2% / 4.7% / 0.9% |
| Additional "mixed" reviews (praise + complaint) | 5.1% |

**RQ2 – TF-IDF baseline as a mismatch detector** (flag when disagreement ≥ 0.9)

| | Precision | Recall | F1 |
| --- | --- | --- | --- |
| On the 3,000 sample | 0.70 | 0.80 | 0.75 |
| Weighted to all reviews | 0.74 | 0.41 | 0.53 |

The baseline is precise but misses most mismatches across the full dataset. This is the gap KcELECTRA and the LLMs should close.

**KcELECTRA vs baseline** (KcELECTRA fine-tuned 1 epoch on 100k reviews, Colab T4, 6 min; annotation sample excluded from training)

| | TF-IDF baseline | KcELECTRA |
| --- | --- | --- |
| Sentiment accuracy (40k test reviews) | 0.922 | **0.944** |
| Mismatch F1, sample (threshold 0.9) | 0.745 | **0.780** |
| Mismatch F1, weighted to all reviews | 0.531 | **0.608** |
| Mismatch recall, weighted | 0.41 | **0.51** |
| Mismatch precision, weighted | 0.74 | 0.76 |

**Extra analyses** (`analysis.py` → `results/analysis_claude.md`; LLM labels, threshold 0.9, F1 weighted to all reviews)

| Detector | F1 | 95% CI |
| --- | --- | --- |
| TF-IDF baseline | 0.53 | 0.47 – 0.59 |
| KcELECTRA | 0.61 | 0.54 – 0.69 |
| Gemini zero-shot | 0.84 | 0.78 – 0.89 |

- All three differences are clear of zero (stratified bootstrap, 1,000 draws). Gemini's number is likely inflated
  because the labels also come from an LLM.
- Errors: the baseline's false alarms are mostly *mixed* reviews (40% vs 11% overall) and reviews with negation or
  contrast words; KcELECTRA mostly misses **very short** reviews (51% of its misses vs 20% overall); Gemini's misses
  often contain contrast words (지만, 는데).
- **Cascade:** KcELECTRA handles the clear cases and sends only reviews with disagreement between 0.1 and 0.99 to
  Gemini. That is 13% of reviews (about 1,000 requests instead of 8,000 for 200k reviews) for the same F1 (0.84) as
  Gemini on everything. With the TF-IDF baseline as the first step, 33% must go to Gemini.
- **Held-out check of the cascade:** thresholds chosen on a random half of the items and scored on the other half
  (200 repeats, at most 15% of reviews sent to the LLM): KcELECTRA → Gemini reaches F1 0.82 (0.76–0.88) with 13% of
  reviews sent to Gemini, 0.01 below Gemini on everything (gap interval −0.02 to +0.07, so not clearly worse).
  TF-IDF → Gemini reaches 0.75, clearly below (gap +0.09).
- Figures for the paper: `results/figures/fig1`–`fig4` (PNG and PDF).

**Stages 2–4 + RQ3** (`results/issues/issue_report.md`)

- Trust filter removed 680 gibberish reviews and 7,411 low-star reviews whose text is actually positive.
- **4–10% of 1–2★ reviews are not complaints**, and **15–30% of real complaints appear only in 4–5★ reviews**,
  depending on category (highest for baby/pet at 30% and beauty at 27%). A company reading only low-star reviews misses them.
- 146 issues found across 7 categories, e.g. *size runs small* (fashion), *ice packs melted* (food),
  *adhesive does not hold* (home), *expiry date too close* (food).

Aspect model quality (5-fold CV on the labeled sample; weak where few examples exist):

| Aspect | Negative examples | Negative F1 |
| --- | --- | --- |
| Quality | 1,262 | 0.79 |
| Delivery | 146 | 0.74 |
| Packaging | 71 | 0.55 |
| Price | 64 | 0.46 |
| Service | 53 | 0.39 |

**Tokenizer ablation** (`results/ablation_tokenizer.json`, same 80/20 split, logistic regression)

| Text pieces | Accuracy | Train time |
| --- | --- | --- |
| Character 2–5-grams (baseline) | 0.9223 | 13 s |
| Kiwi morphemes + pairs | 0.9216 | 7 s (+69 s morpheme analysis) |
| Whitespace words + pairs | 0.8725 | 1 s |
| Morphemes + characters | 0.9228 | 21 s |

Character pieces match a proper Korean morpheme analyzer without needing one; naive word splitting loses 5 points
because Korean endings stay attached to words. Differences under ~0.003 are within noise for a 40k test set.

## Known limitations

- No product IDs in the dataset, so issues are grouped by **product category inferred from keywords**; about 70% of reviews
  are too generic to name a category and are grouped as "other".
- No review dates, so "recent growth" cannot be used for prioritisation.
- Issue clusters use word stems + k-means; some clusters are broad (marked "(broad)") and ranked after specific ones.
  Korean sentence embeddings would likely give cleaner clusters (Hugging Face was not reachable from our environment).
- Aspect models for price, packaging and service are weak because those aspects are rare in the sample.

## Running tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests      # or: make test
```

The tests use tiny in-memory examples (no dataset, no API key) and run on every push via GitHub Actions
(`.github/workflows/tests.yml`). `make help` lists one command per pipeline stage.

## Project structure

```
common.py                 shared data loading + random seed (used by baseline and ablation)
baseline.py               stage 1: TF-IDF + logistic regression, mismatch candidates
ablation_tokenizer.py     characters vs words vs Kiwi morphemes
sample_for_annotation.py  3,000-review stratified annotation sample
make_pilot_xlsx.py        Excel sheets for the 200 pilot rows
agreement.py              human agreement (Cohen's kappa) and gold labels
evaluate.py               RQ1 prevalence, RQ2 detection, LLM-vs-human agreement
analysis.py               confidence intervals, error analysis, cost/speed, cascade
figures.py                paper figures (results/figures/)
make_label_tool.py        builds annotation/label_tool.html (offline labeling page)
run_gemini.command        double-click to run all Gemini steps (macOS)
issues.py                 stages 2-4: trust filter, aspects, issue discovery, ranking, RQ3
llm_client.py             Gemini wrapper with caching, retries and a --mock mode
llm_label.py              Gemini zero-shot labels, reliability, consistency task
recommend.py              stage 5: grounded vs ungrounded recommendations
verify.py                 stage 6: claim verification + company report
kcelectra_finetune.ipynb  KcELECTRA fine-tuning (Colab)
run_gemini.sh             runs all Gemini steps in order
tests/                    unit tests
annotation/  results/     labels and outputs (data/ is downloaded, not committed)
```

## Files

| Path | Contents |
| --- | --- |
| `annotation/LABELING_GUIDE.md` | Rules and examples for annotators |
| `annotation/annotator_A.csv`, `annotator_B.csv` | Blind sheets for humans |
| `annotation/claude_labels.csv` | LLM labels for all 3,000 |
| `annotation/sample_key_DO_NOT_SHARE.csv` | Ratings, model scores, sampling weights |
| `results/evaluation_<name>.json` | RQ1 + RQ2 + aspects in mismatches |
| `results/issues/issue_report.md` | Readable issue ranking + RQ3 |
| `results/issues/issues.csv` | All issues with metrics |
| `results/issues/issue_evidence.jsonl` | 25 evidence reviews per issue (input to `recommend.py`) |
| `results/issues/keyword_baseline.csv` | Keyword-count baseline (what a simple dashboard shows) |
| `results/company_report.md` | Verified recommendations (after `verify.py`) |
| `results/human_rating_sheet.csv` | Blind sheet for rating recommendation usefulness |

# One target per pipeline stage. Run `make help` to list them.
# Gemini targets need GEMINI_API_KEY (see README); everything else runs offline.

PY ?= python3

.PHONY: help data baseline ablation sample evaluate issues analysis figures label-tool gemini agreement test

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /\t/'

data:          ## Download the Naver Shopping reviews into data/
	mkdir -p data
	curl -L -o data/naver_shopping.txt https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt

baseline:      ## Stage 1: TF-IDF + logistic regression baseline and mismatch candidates
	$(PY) baseline.py

ablation:      ## Tokenizer ablation: characters vs words vs Kiwi morphemes
	$(PY) ablation_tokenizer.py

sample:        ## Build the 3,000-review annotation sample
	$(PY) sample_for_annotation.py

evaluate:      ## RQ1 prevalence + RQ2 detection on the LLM labels
	$(PY) evaluate.py --labels annotation/claude_labels.csv --name claude

issues:        ## Stages 2-4: trust filter, aspects, issues, ranking, RQ3
	$(PY) issues.py

analysis:      ## Confidence intervals, error analysis, cost/speed, cascade simulation
	$(PY) analysis.py

figures:       ## Paper figures into results/figures/
	$(PY) figures.py

label-tool:    ## Build the offline labeling page annotation/label_tool.html
	$(PY) make_label_tool.py

gemini:        ## All Gemini steps (labels, consistency, recommendations, verification)
	bash run_gemini.sh

agreement:     ## Human agreement + gold labels (after the pilot sheets are filled in)
	$(PY) agreement.py --dir annotation --pilot

test:          ## Unit tests
	$(PY) -m pytest -q tests

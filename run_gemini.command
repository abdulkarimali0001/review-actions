#!/bin/bash
# Double-click in Finder to run all Gemini steps. Output is also saved to results/gemini_run.log
cd "$(dirname "$0")"
bash run_gemini.sh 2>&1 | tee results/gemini_run.log
echo
read -p "Finished. Press Enter to close this window."

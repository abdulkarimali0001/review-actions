# Extra analyses (labels: claude)

Labels are LLM labels unless the name says human. Gemini scores against LLM labels are likely inflated.

## 1. Confidence intervals (95%, stratified bootstrap)

| Quantity | Estimate | 95% CI |
| --- | --- | --- |
| mismatch_rate | 0.033 | 0.028 – 0.039 |
| tfidf_precision | 0.739 | 0.714 – 0.763 |
| tfidf_recall | 0.415 | 0.349 – 0.493 |
| tfidf_f1 | 0.532 | 0.474 – 0.592 |
| kcelectra_precision | 0.756 | 0.675 – 0.825 |
| kcelectra_recall | 0.510 | 0.431 – 0.609 |
| kcelectra_f1 | 0.610 | 0.541 – 0.689 |
| gemini_precision | 0.893 | 0.817 – 0.952 |
| gemini_recall | 0.794 | 0.702 – 0.871 |
| gemini_f1 | 0.840 | 0.776 – 0.893 |
| f1_kcelectra_minus_tfidf | 0.078 | 0.022 – 0.142 (above 0 in 100% of draws) |
| f1_gemini_minus_kcelectra | 0.231 | 0.125 – 0.326 (above 0 in 100% of draws) |
| f1_gemini_minus_tfidf | 0.308 | 0.228 – 0.386 (above 0 in 100% of draws) |

F1 values are weighted to all 200k reviews, threshold 0.9.

## 2. Where the detectors fail

Share of each error group that has the trait (compare with the 'all items' row).

| Group | n | negation | contrast | short (<15 chars) | mixed text |
| --- | --- | --- | --- | --- | --- |
| all items | 2935 | 0.218 | 0.275 | 0.195 | 0.112 |
| tfidf: missed_mismatches (FN) | 170 | 0.112 | 0.182 | 0.347 | 0.0 |
| tfidf: false_alarms (FP) | 300 | 0.323 | 0.38 | 0.14 | 0.397 |
| kcelectra: missed_mismatches (FN) | 225 | 0.062 | 0.124 | 0.511 | 0.0 |
| kcelectra: false_alarms (FP) | 130 | 0.2 | 0.308 | 0.123 | 0.454 |
| gemini: missed_mismatches (FN) | 99 | 0.212 | 0.394 | 0.121 | 0.0 |
| gemini: false_alarms (FP) | 41 | 0.244 | 0.146 | 0.146 | 0.317 |

KcELECTRA fixes 275 of the baseline's errors, adds 160 new ones, and 195 errors are shared.

## 3. Cost and speed

```
{
  "tfidf_baseline": {
    "hardware": "laptop CPU",
    "train_seconds_incl_vectorizing": 13.0,
    "reviews_per_second": 24119,
    "minutes_for_200k_reviews": 0.14,
    "llm_requests_for_200k": 0
  },
  "kcelectra": {
    "hardware": "Colab T4",
    "train_seconds": 343,
    "train_examples": 100000,
    "llm_requests_for_200k": 0,
    "reviews_per_second": "not measured (add timing to the Colab inference cell)"
  },
  "gemini_zero_shot": {
    "requests_per_3000_reviews": 120,
    "llm_requests_for_200k": 8000,
    "reviews_per_request": 25,
    "latency_and_tokens": "not logged for the runs so far; llm_client.py now records them, so the next Gemini run fills this in"
  }
}
```

## 4. Cascade: cheap model first, LLM only for uncertain reviews

Settings where no cheaper setting is better (Pareto front). F1 weighted to all reviews.

| Cheap model | send to LLM if disagreement in | Share sent to LLM | LLM requests (200k) | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf | (0.5, 0.7) | 3.2% | 260 | 0.489 | 0.691 | 0.573 |
| tfidf | (0.5, 0.8) | 4.7% | 374 | 0.635 | 0.674 | 0.654 |
| tfidf | (0.5, 0.9) | 5.8% | 464 | 0.784 | 0.662 | 0.718 |
| tfidf | (0.5, 0.95) | 6.5% | 519 | 0.87 | 0.65 | 0.744 |
| tfidf | (0.5, 0.99) | 7.3% | 584 | 0.922 | 0.634 | 0.751 |
| tfidf | (0.5, 1.01) | 7.6% | 610 | 0.93 | 0.632 | 0.752 |
| tfidf | (0.3, 0.95) | 12.8% | 1024 | 0.842 | 0.68 | 0.753 |
| tfidf | (0.3, 0.99) | 13.6% | 1089 | 0.888 | 0.664 | 0.76 |
| tfidf | (0.3, 1.01) | 13.9% | 1115 | 0.895 | 0.662 | 0.761 |
| tfidf | (0.2, 0.95) | 17.8% | 1427 | 0.833 | 0.724 | 0.775 |
| tfidf | (0.2, 0.99) | 18.6% | 1492 | 0.874 | 0.708 | 0.783 |
| tfidf | (0.2, 1.01) | 19.0% | 1519 | 0.881 | 0.706 | 0.784 |
| tfidf | (0.1, 0.9) | 30.9% | 2469 | 0.781 | 0.824 | 0.802 |
| tfidf | (0.1, 0.95) | 31.6% | 2524 | 0.848 | 0.812 | 0.83 |
| tfidf | (0.1, 0.99) | 32.4% | 2589 | 0.887 | 0.796 | 0.839 |
| tfidf | (0.1, 1.01) | 32.7% | 2616 | 0.893 | 0.794 | 0.84 |
| kcelectra | (0.5, 0.7) | 1.3% | 102 | 0.57 | 0.71 | 0.632 |
| kcelectra | (0.5, 0.8) | 2.0% | 163 | 0.662 | 0.701 | 0.681 |
| kcelectra | (0.5, 0.9) | 2.9% | 232 | 0.763 | 0.686 | 0.722 |
| kcelectra | (0.5, 0.95) | 3.5% | 282 | 0.829 | 0.651 | 0.729 |
| kcelectra | (0.5, 0.99) | 4.6% | 368 | 0.89 | 0.624 | 0.733 |
| kcelectra | (0.3, 0.9) | 5.3% | 425 | 0.773 | 0.751 | 0.762 |
| kcelectra | (0.3, 0.95) | 5.9% | 475 | 0.835 | 0.717 | 0.771 |
| kcelectra | (0.3, 0.99) | 7.0% | 561 | 0.891 | 0.689 | 0.777 |
| kcelectra | (0.2, 0.9) | 7.3% | 583 | 0.787 | 0.815 | 0.801 |
| kcelectra | (0.2, 0.95) | 7.9% | 634 | 0.847 | 0.78 | 0.812 |
| kcelectra | (0.2, 0.99) | 9.0% | 720 | 0.899 | 0.753 | 0.82 |
| kcelectra | (0.1, 0.95) | 12.0% | 958 | 0.852 | 0.813 | 0.832 |
| kcelectra | (0.1, 0.99) | 13.0% | 1044 | 0.903 | 0.785 | 0.84 |
| (Gemini on everything) | - | 100% | 8000 | 0.893 | 0.794 | 0.840 |

**Held-out check.** Thresholds chosen on a random half of the items (best F1 while sending at most 15% of reviews to the LLM), then scored on the other half; repeated with different splits.

| Cheap model | Held-out F1 | Share sent to LLM | Gemini on everything (same halves) | Gap |
| --- | --- | --- | --- | --- |
| tfidf | 0.753 (0.691–0.821) | 12.0% | 0.842 | +0.089 (+0.008 to +0.169) |
| kcelectra | 0.822 (0.764–0.883) | 12.9% | 0.836 | +0.014 (-0.018 to +0.068) |

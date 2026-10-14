"""
Extra analyses for the paper, using only files that already exist (no API calls, no GPU).

1. Bootstrap confidence intervals
   95% intervals for the RQ1 mismatch rate and the RQ2 F1 scores, and for the
   differences between detectors (is KcELECTRA really better than TF-IDF?).
   Resampling is done inside each sampling stratum, so the sampling weights stay valid.

2. Error analysis
   Which kinds of reviews each detector gets wrong: negation (안/않/못/없), contrast
   words (지만/는데/근데/그래도 ...), very short texts, and "mixed" reviews (praise + complaint).

3. Cost and speed
   Training time, inference speed on a laptop CPU, and how many LLM requests each
   approach needs for the full 200k reviews.

4. Cascade simulation
   A cheap model handles the clear cases and only the uncertain ones go to the LLM.
   For each pair of thresholds: F1 (weighted to all reviews) and the share of reviews
   sent to the LLM. Shows how much of the LLM's accuracy can be kept at a fraction of the cost.

All detectors flag a mismatch when disagreement >= 0.9 (the threshold fixed before
looking at results), so no threshold is tuned on the test labels here.

Caveat: the labels are LLM labels (Claude). Gemini is also an LLM, so its scores against
these labels are likely inflated. Rerun with --labels annotation/gold_labels.csv once the
human labels exist.

Usage
    python analysis.py
    python analysis.py --labels annotation/gold_labels.csv --name human --boot 2000
Output
    results/analysis_<name>.json, results/analysis_<name>.md, results/error_examples_<name>.csv
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import SEED, load_reviews
from evaluate import load_labels, mismatch_from

THRESHOLD = 0.9
NEGATION = re.compile(r"안 |안됨|안돼|안되|않|못|없")
CONTRAST = re.compile(r"지만|는데|근데|그런데|그래도|다만|그러나|하지만|반면")
LLM_REVIEWS_PER_REQUEST = 25   # batch size used by llm_label.py
DATASET_SIZE = 200_000


# ----------------------------------------------------------------------------- data
def load(args):
    key = pd.read_csv(args.key)
    lab = load_labels(args.labels)
    df = key.merge(lab.drop(columns=[c for c in ["text", "pilot"] if c in lab.columns]), on="item")
    df = df[df["text_sentiment"].isin(["pos", "neg", "mixed", "neutral"])].copy()
    df["mismatch"] = mismatch_from(df["rating"], df["text_sentiment"]).values
    df = df.rename(columns={"p_text_positive": "p_tfidf"})
    for name, path in [("kcelectra", args.kcelectra), ("gemini", args.gemini)]:
        if Path(path).exists():
            pr = pd.read_csv(path)[["item", "p_text_positive"]].rename(columns={"p_text_positive": f"p_{name}"})
            df = df.merge(pr, on="item", how="left")
    detectors = [c[2:] for c in df.columns if c.startswith("p_")]
    before = len(df)
    df = df.dropna(subset=[f"p_{d}" for d in detectors])  # compare all detectors on the same items
    if len(df) < before:
        print(f"{before - len(df)} items dropped because a detector has no score for them")
    for d in detectors:
        df[f"dis_{d}"] = np.where(df["rating"] >= 4, 1 - df[f"p_{d}"], df[f"p_{d}"])
        df[f"flag_{d}"] = df[f"dis_{d}"] >= THRESHOLD
    return df.reset_index(drop=True), detectors


def prf(y, pred, w):
    tp = (w * (y & pred)).sum()
    fp = (w * (~y & pred)).sum()
    fn = (w * (y & ~pred)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ----------------------------------------------------------------------------- 1. bootstrap
def bootstrap(df, detectors, B, rng):
    y, w = df["mismatch"].values, df["weight"].values
    flags = {d: df[f"flag_{d}"].values for d in detectors}
    groups = [g.index.values for _, g in df.groupby("stratum")]

    def stats(idx):
        yy, ww = y[idx], w[idx]
        out = {"mismatch_rate": (yy * ww).sum() / ww.sum()}
        for d in detectors:
            p, r, f = prf(yy, flags[d][idx], ww)
            out[f"{d}_precision"], out[f"{d}_recall"], out[f"{d}_f1"] = p, r, f
        for a, b in [("kcelectra", "tfidf"), ("gemini", "kcelectra"), ("gemini", "tfidf")]:
            if a in detectors and b in detectors:
                out[f"f1_{a}_minus_{b}"] = out[f"{a}_f1"] - out[f"{b}_f1"]
        return out

    point = stats(np.arange(len(df)))
    draws = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        draws.append(stats(idx))
    draws = pd.DataFrame(draws)
    res = {}
    for k, v in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        res[k] = {"estimate": round(float(v), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
        if k.startswith("f1_") and "_minus_" in k:
            res[k]["share_of_draws_above_0"] = round(float((draws[k] > 0).mean()), 3)
    return res


# ----------------------------------------------------------------------------- 2. errors
def error_analysis(df, detectors):
    df = df.copy()
    df["negation"] = df["text"].str.contains(NEGATION)
    df["contrast"] = df["text"].str.contains(CONTRAST)
    df["short"] = df["text"].str.len() < 15
    df["mixed_text"] = df["text_sentiment"] == "mixed"
    traits = ["negation", "contrast", "short", "mixed_text"]

    def share(sub):
        return {t: round(float(sub[t].mean()), 3) if len(sub) else None for t in traits}

    res = {"all_items": {"n": int(len(df)), **share(df)}}
    examples = []
    for d in detectors:
        fn = df[df["mismatch"] & ~df[f"flag_{d}"]]
        fp = df[~df["mismatch"] & df[f"flag_{d}"]]
        res[d] = {"missed_mismatches (FN)": {"n": int(len(fn)), **share(fn)},
                  "false_alarms (FP)": {"n": int(len(fp)), **share(fp)},
                  "false_alarms_by_true_label": fp["text_sentiment"].value_counts().to_dict()}
        for kind, sub in [("FN", fn), ("FP", fp)]:
            for _, r in sub.sort_values("weight", ascending=False).head(15).iterrows():
                examples.append({"detector": d, "error": kind, "item": r["item"], "rating": r["rating"],
                                 "label": r["text_sentiment"], "disagreement": round(r[f"dis_{d}"], 3),
                                 **{t: bool(r[t]) for t in traits}, "text": r["text"]})
    if "kcelectra" in detectors:
        base_err = df["mismatch"] != df["flag_tfidf"]
        kc_err = df["mismatch"] != df["flag_kcelectra"]
        res["tfidf_to_kcelectra"] = {"errors_fixed": int((base_err & ~kc_err).sum()),
                                     "new_errors": int((~base_err & kc_err).sum()),
                                     "errors_both_make": int((base_err & kc_err).sum())}
    return res, pd.DataFrame(examples)


# ----------------------------------------------------------------------------- 3. cost/speed
def cost_speed(args, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    res = {}
    if Path(args.data).exists():
        df = load_reviews(args.data)
        train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
        t0 = time.time()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)
        clf = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear").fit(vec.fit_transform(train["text"]), train["label"])
        train_s = time.time() - t0
        t0 = time.time()
        clf.predict_proba(vec.transform(test["text"]))
        inf_s = time.time() - t0
        res["tfidf_baseline"] = {"hardware": "laptop CPU", "train_seconds_incl_vectorizing": round(train_s, 1),
                                 "reviews_per_second": int(len(test) / inf_s),
                                 "minutes_for_200k_reviews": round(DATASET_SIZE / (len(test) / inf_s) / 60, 2),
                                 "llm_requests_for_200k": 0}
    km = Path(args.kcelectra_metrics)
    if km.exists():
        m = json.loads(km.read_text())
        res["kcelectra"] = {"hardware": m.get("gpu", "GPU"), "train_seconds": m.get("train_seconds"),
                            "train_examples": m.get("train_size"), "llm_requests_for_200k": 0,
                            "reviews_per_second": "not measured (add timing to the Colab inference cell)"}
    usage = []
    cache = Path(args.cache)
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("usage") and str(rec.get("tag", "")).startswith("label"):
                usage.append(rec["usage"])
    g = {"requests_per_3000_reviews": 3000 // LLM_REVIEWS_PER_REQUEST,
         "llm_requests_for_200k": DATASET_SIZE // LLM_REVIEWS_PER_REQUEST,
         "reviews_per_request": LLM_REVIEWS_PER_REQUEST}
    if usage:
        u = pd.DataFrame(usage)
        g.update({"logged_requests": int(len(u)), "median_seconds_per_request": float(u["seconds"].median()),
                  "mean_prompt_tokens": float(u["prompt_tokens"].mean()),
                  "mean_output_tokens": float(u["output_tokens"].mean())})
    else:
        g["latency_and_tokens"] = ("not logged for the runs so far; llm_client.py now records them, "
                                   "so the next Gemini run fills this in")
    res["gemini_zero_shot"] = g
    return res


# ----------------------------------------------------------------------------- 4. cascade
def cascade(df, cheap, llm="gemini"):
    y, w = df["mismatch"].values, df["weight"].values
    d = df[f"dis_{cheap}"].values
    llm_flag = df[f"flag_{llm}"].values
    rows = []
    for lo, hi in GRID:
        ask = (d > lo) & (d < hi)
        pred = np.where(ask, llm_flag, d >= hi)
        p, r, f = prf(y, pred, w)
        share = (ask * w).sum() / w.sum()
        rows.append({"cheap_model": cheap, "low": lo, "high": hi, "share_sent_to_llm": round(float(share), 4),
                     "llm_requests_for_200k": int(np.ceil(share * DATASET_SIZE / LLM_REVIEWS_PER_REQUEST)),
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
    t = pd.DataFrame(rows).sort_values("share_sent_to_llm")
    # Pareto front: no other setting is both cheaper and better.
    best, front = -1, []
    for _, r in t.iterrows():
        if r["f1"] > best:
            front.append(r.to_dict())
            best = r["f1"]
    return t, front


GRID = [(lo, hi) for lo in [0.05, 0.1, 0.2, 0.3, 0.5] for hi in [0.7, 0.8, 0.9, 0.95, 0.99, 1.01] if hi > lo]


def cascade_heldout(df, cheap, rng, reps=200, budget=0.15, llm="gemini"):
    """Fairer test of the cascade: choose thresholds on one half of the items, score them on the other half.

    Thresholds are the best F1 among settings that send at most `budget` of reviews to the LLM.
    Halves are split inside each sampling stratum. Repeated `reps` times with different splits.
    """
    y, w = df["mismatch"].values, df["weight"].values
    d, llm_flag = df[f"dis_{cheap}"].values, df[f"flag_{llm}"].values
    groups = [g.index.values for _, g in df.groupby("stratum")]

    def run(idx, lo, hi):
        ask = (d[idx] > lo) & (d[idx] < hi)
        pred = np.where(ask, llm_flag[idx], d[idx] >= hi)
        f = prf(y[idx], pred, w[idx])[2]
        return f, (ask * w[idx]).sum() / w[idx].sum()

    rows = []
    for _ in range(reps):
        tune, test = [], []
        for g in groups:
            g = rng.permutation(g)
            tune.append(g[: len(g) // 2])
            test.append(g[len(g) // 2:])
        tune, test = np.concatenate(tune), np.concatenate(test)
        ok = [(run(tune, lo, hi), lo, hi) for lo, hi in GRID]
        ok = [x for x in ok if x[0][1] <= budget] or ok
        (_, _), lo, hi = max(ok, key=lambda x: x[0][0])
        f, share = run(test, lo, hi)
        f_llm = prf(y[test], llm_flag[test], w[test])[2]
        rows.append({"f1": f, "share": share, "f1_llm_all": f_llm, "gap": f_llm - f, "low": lo, "high": hi})
    r = pd.DataFrame(rows)

    def ci(col):
        return {"mean": round(float(r[col].mean()), 3),
                "ci95": [round(float(x), 3) for x in np.percentile(r[col], [2.5, 97.5])]}

    picked = r.groupby(["low", "high"]).size().sort_values(ascending=False)
    return {"cheap_model": cheap, "budget_share_to_llm": budget, "repeats": reps,
            "heldout_f1": ci("f1"), "heldout_share_sent_to_llm": ci("share"),
            "llm_on_everything_f1_same_halves": ci("f1_llm_all"), "f1_gap_to_llm_on_everything": ci("gap"),
            "most_picked_thresholds": [{"low": a, "high": b, "times": int(n)} for (a, b), n in picked.head(3).items()]}


# ----------------------------------------------------------------------------- report
def markdown(name, boot, err, cost, fronts, df, detectors, heldout=()):
    L = [f"# Extra analyses (labels: {name})", "",
         "Labels are LLM labels unless the name says human. Gemini scores against LLM labels are likely inflated.", "",
         "## 1. Confidence intervals (95%, stratified bootstrap)", "", "| Quantity | Estimate | 95% CI |", "| --- | --- | --- |"]
    for k, v in boot.items():
        extra = f" (above 0 in {v['share_of_draws_above_0']:.0%} of draws)" if "share_of_draws_above_0" in v else ""
        L.append(f"| {k} | {v['estimate']:.3f} | {v['ci95'][0]:.3f} – {v['ci95'][1]:.3f}{extra} |")
    L += ["", "F1 values are weighted to all 200k reviews, threshold 0.9.", "",
          "## 2. Where the detectors fail", "",
          "Share of each error group that has the trait (compare with the 'all items' row).", "",
          "| Group | n | negation | contrast | short (<15 chars) | mixed text |", "| --- | --- | --- | --- | --- | --- |"]
    a = err["all_items"]
    L.append(f"| all items | {a['n']} | {a['negation']} | {a['contrast']} | {a['short']} | {a['mixed_text']} |")
    for d in detectors:
        for g in ["missed_mismatches (FN)", "false_alarms (FP)"]:
            s = err[d][g]
            L.append(f"| {d}: {g} | {s['n']} | {s['negation']} | {s['contrast']} | {s['short']} | {s['mixed_text']} |")
    if "tfidf_to_kcelectra" in err:
        t = err["tfidf_to_kcelectra"]
        L += ["", f"KcELECTRA fixes {t['errors_fixed']} of the baseline's errors, adds {t['new_errors']} new ones, "
              f"and {t['errors_both_make']} errors are shared."]
    L += ["", "## 3. Cost and speed", "", "```", json.dumps(cost, indent=2, ensure_ascii=False), "```", "",
          "## 4. Cascade: cheap model first, LLM only for uncertain reviews", "",
          "Settings where no cheaper setting is better (Pareto front). F1 weighted to all reviews.", "",
          "| Cheap model | send to LLM if disagreement in | Share sent to LLM | LLM requests (200k) | P | R | F1 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for front in fronts:
        for r in front:
            L.append(f"| {r['cheap_model']} | ({r['low']}, {r['high']}) | {r['share_sent_to_llm']:.1%} | "
                     f"{r['llm_requests_for_200k']} | {r['precision']} | {r['recall']} | {r['f1']} |")
    if "gemini" in detectors:
        p, r, f = prf(df["mismatch"].values, df["flag_gemini"].values, df["weight"].values)
        L.append(f"| (Gemini on everything) | - | 100% | {DATASET_SIZE // LLM_REVIEWS_PER_REQUEST} | "
                 f"{p:.3f} | {r:.3f} | {f:.3f} |")
    if heldout:
        L += ["", "**Held-out check.** Thresholds chosen on a random half of the items (best F1 while sending at most "
              "15% of reviews to the LLM), then scored on the other half; repeated with different splits.", "",
              "| Cheap model | Held-out F1 | Share sent to LLM | Gemini on everything (same halves) | Gap |",
              "| --- | --- | --- | --- | --- |"]
        for h in heldout:
            f, sh, g, gap = (h["heldout_f1"], h["heldout_share_sent_to_llm"],
                             h["llm_on_everything_f1_same_halves"], h["f1_gap_to_llm_on_everything"])
            L.append(f"| {h['cheap_model']} | {f['mean']:.3f} ({f['ci95'][0]:.3f}–{f['ci95'][1]:.3f}) | "
                     f"{sh['mean']:.1%} | {g['mean']:.3f} | {gap['mean']:+.3f} ({gap['ci95'][0]:+.3f} to {gap['ci95'][1]:+.3f}) |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="annotation/sample_key_DO_NOT_SHARE.csv")
    ap.add_argument("--labels", default="annotation/claude_labels.csv")
    ap.add_argument("--name", default="claude")
    ap.add_argument("--kcelectra", default="results/kcelectra_scores.csv")
    ap.add_argument("--kcelectra-metrics", default="results/kcelectra_metrics.json")
    ap.add_argument("--gemini", default="results/gemini_label_scores.csv")
    ap.add_argument("--cache", default="results/llm_cache/gemini-3.5-flash-lite.jsonl")
    ap.add_argument("--data", default="data/naver_shopping.txt")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=200, help="repeated half splits for the held-out cascade test")
    ap.add_argument("--skip-speed", action="store_true", help="skip retraining the baseline for timing")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df, detectors = load(args)
    print(f"{len(df)} labeled items, detectors: {detectors}")

    boot = bootstrap(df, detectors, args.boot, rng)
    err, examples = error_analysis(df, detectors)
    cost = {} if args.skip_speed else cost_speed(args, rng)
    fronts, tables, heldout = [], [], []
    if "gemini" in detectors:
        for cheap in [d for d in ["tfidf", "kcelectra"] if d in detectors]:
            t, front = cascade(df, cheap)
            tables.append(t)
            fronts.append(front)
            heldout.append(cascade_heldout(df, cheap, rng, reps=args.reps))

    out = Path("results")
    report = {"label_source": args.name, "threshold": THRESHOLD, "bootstrap_draws": args.boot,
              "confidence_intervals": boot, "error_analysis": err, "cost_speed": cost,
              "cascade_pareto": [r for f in fronts for r in f], "cascade_heldout": heldout}
    (out / f"analysis_{args.name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if tables:
        pd.concat(tables).to_csv(out / f"cascade_{args.name}.csv", index=False)
    examples.to_csv(out / f"error_examples_{args.name}.csv", index=False, encoding="utf-8-sig")
    md = markdown(args.name, boot, err, cost, fronts, df, detectors, heldout)
    (out / f"analysis_{args.name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

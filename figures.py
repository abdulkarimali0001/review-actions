"""
Paper figures, made from the result files (run analysis.py and issues.py first).

  fig1_detector_f1.png        F1 per detector with 95% bootstrap intervals (RQ2)
  fig2_cascade.png            share of reviews sent to the LLM vs F1 (cost/accuracy trade-off)
  fig3_error_traits.png       which kinds of reviews each detector gets wrong
  fig4_hidden_complaints.png  complaints that only appear in 4-5 star reviews, by category (RQ3)

Usage:  python figures.py [--name claude]
Output: results/figures/  (PNG at 300 dpi, plus PDF for LaTeX)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette, fixed order (blue, orange, aqua); text stays in ink colours.
COLORS = {"tfidf": "#2a78d6", "kcelectra": "#eb6834", "gemini": "#1baf7a"}
NAMES = {"tfidf": "TF-IDF baseline", "kcelectra": "KcELECTRA", "gemini": "Gemini zero-shot"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, out, name):
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def fig_f1(a, out):
    ci = a["confidence_intervals"]
    dets = [d for d in ["tfidf", "kcelectra", "gemini"] if f"{d}_f1" in ci]
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    for i, d in enumerate(dets):
        e, (lo, hi) = ci[f"{d}_f1"]["estimate"], ci[f"{d}_f1"]["ci95"]
        ax.plot([lo, hi], [i, i], color=COLORS[d], lw=2, solid_capstyle="round")
        ax.plot(e, i, "o", ms=8, color=COLORS[d], mec="white", mew=1.5)
        ax.text(hi + 0.012, i, f"{e:.2f}  [{lo:.2f}–{hi:.2f}]", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(dets)), [NAMES[d] for d in dets])
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.6, len(dets) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mismatch detection F1 (weighted to all reviews), 95% CI")
    save(fig, out, "fig1_detector_f1")


def fig_cascade(name, a, out):
    path = Path(f"results/cascade_{name}.csv")
    if not path.exists():
        return
    t = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for cheap in ["tfidf", "kcelectra"]:
        s = t[t["cheap_model"] == cheap].sort_values("share_sent_to_llm")
        front = s[s["f1"] >= s["f1"].cummax()]
        ax.scatter(s["share_sent_to_llm"] * 100, s["f1"], s=10, color=COLORS[cheap], alpha=0.25, lw=0)
        ax.plot(front["share_sent_to_llm"] * 100, front["f1"], "-o", ms=4, lw=2, color=COLORS[cheap],
                label=f"{NAMES[cheap]} → Gemini")
    g = a["confidence_intervals"].get("gemini_f1")
    if g:
        ax.scatter([100], [g["estimate"]], s=60, color=COLORS["gemini"], zorder=5, edgecolor="white", lw=1.5,
                   label="Gemini on every review")
        ax.axhline(g["estimate"], color=COLORS["gemini"], lw=1, ls="--", alpha=0.6)
    for cheap in ["tfidf", "kcelectra"]:
        f1 = a["confidence_intervals"].get(f"{cheap}_f1")
        if f1:
            ax.scatter([0.3], [f1["estimate"]], s=40, marker="s", color=COLORS[cheap], edgecolor="white", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30, 100], ["0 (alone)", "1%", "3%", "10%", "30%", "100%"])
    ax.set_xlabel("Share of reviews sent to the LLM (log scale)")
    ax.set_ylabel("F1 (weighted to all reviews)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "fig2_cascade")


def fig_errors(a, out):
    err = a["error_analysis"]
    # Kept in English so the figure renders without a Korean font; the word lists are in analysis.py.
    traits = [("short", "Very short\n(<15 characters)"), ("mixed_text", "Mixed praise + complaint\n(never a mismatch)"),
              ("contrast", "Contrast connective\n(e.g. -jiman, -neunde)"), ("negation", "Negation\n(an, anh, mot, eops)")]
    dets = [d for d in ["tfidf", "kcelectra", "gemini"] if d in err]
    groups = [("missed_mismatches (FN)", "missed"), ("false_alarms (FP)", "false alarm")]
    fig, axes = plt.subplots(1, len(traits), figsize=(7.6, 2.6), sharey=True)
    for ax, (key, title) in zip(axes, traits):
        x = 0
        ticks, labels = [], []
        for gkey, glab in groups:
            for d in dets:
                ax.bar(x, err[d][gkey][key] * 100, width=0.8, color=COLORS[d], edgecolor="white", lw=1)
                x += 1
            ticks.append(x - len(dets) / 2 - 0.5)
            labels.append(glab)
            x += 0.8
        ax.axhline(err["all_items"][key] * 100, color=INK, lw=1, ls="--")
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of errors with the trait")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[d]) for d in dets]
    handles.append(plt.Line2D([0], [0], color=INK, lw=1, ls="--"))
    fig.legend(handles, [NAMES[d] for d in dets] + ["share among all reviews"], loc="upper center",
               ncol=len(dets) + 1, fontsize=8, bbox_to_anchor=(0.5, 1.16))
    save(fig, out, "fig3_error_traits")


def fig_hidden(out):
    path = Path("results/issues/rq3_filter_effect.json")
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(((k.split("_", 1)[-1].replace("_", "/"), v["share_of_complaints_hidden_in_4_5_star"] * 100,
                    v["share_of_low_star_reviews_that_are_not_complaints"] * 100) for k, v in d.items()),
                  key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = range(len(rows))
    ax.barh(y, [r[1] for r in rows], color=COLORS["tfidf"], height=0.6, edgecolor="white")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.6, i, f"{r[1]:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Complaints found only in 4–5★ reviews (%)")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    save(fig, out, "fig4_hidden_complaints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="claude", help="label source used in analysis.py")
    args = ap.parse_args()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    a = json.loads(Path(f"results/analysis_{args.name}.json").read_text(encoding="utf-8"))
    print("Saving figures to results/figures/")
    fig_f1(a, out)
    fig_cascade(args.name, a, out)
    fig_errors(a, out)
    fig_hidden(out)


if __name__ == "__main__":
    main()

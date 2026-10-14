"""
Stages 1, 3 and 4: trust filter, issue discovery and prioritisation.

Pipeline
1. Trust filter: flags reviews we should not build conclusions on
   - gibberish (mostly loose jamo / keyboard mashing, e.g. "ㅁㄴㅇㄹ")
   - rating-text mismatch where the TEXT is positive but the stars are 1-2
     (likely a misclick; counting it as a complaint would invent a problem)
2. Product category from keyword rules (the dataset has no product IDs).
3. Aspect sentiment for every review: one TF-IDF + logistic regression model
   per aspect, trained on the labeled sample (LLM labels until human labels exist).
4. Issue discovery: negative mentions of an aspect inside a category are
   clustered; each cluster is an "issue" named by its most distinctive words,
   with the closest reviews kept as evidence for the recommendation stage.
5. Prioritisation: issues ranked by how many trusted complaints support them
   and how severe they are.
6. RQ3: compares the ranking a company gets from star ratings alone
   (reading only 1-2 star reviews) with the trust-filtered ranking.
Also writes a keyword-count baseline (what a simple dashboard would show).

Usage
    python issues.py --scores results/all_scores.csv --labels annotation/claude_labels.csv
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

SEED = 42
ASPECTS = ["배송_delivery", "가격_price", "품질_quality", "포장_packaging", "서비스_service"]

# Keyword rules for product category. A review goes to the category with the most hits.
CATEGORIES = {
    "식품_food": ["맛", "먹", "신선", "고기", "김치", "과일", "싱싱", "양념", "국물", "유통기한", "간식",
                 "밥", "라면", "커피", "요리", "냉동", "아이스팩", "쌀", "새우", "귤", "사과", "포도", "두유", "우유"],
    "의류잡화_fashion": ["사이즈", "입", "옷", "바지", "핏", "원단", "신발", "팬티", "양말", "티셔츠", "니트", "모자",
                      "가방", "기장", "허리", "소매", "브라", "레깅스", "착용", "신고", "발볼", "지갑", "귀걸이"],
    "뷰티_beauty": ["피부", "발림", "트러블", "화장", "크림", "로션", "샴푸", "촉촉", "보습", "향이", "향은", "향도",
                  "쿠션", "립", "마스크팩", "각질", "염색", "두피"],
    "디지털_digital": ["충전", "케이스", "배터리", "액정", "필름", "폰", "이어폰", "케이블", "블루투스", "마우스",
                     "키보드", "화면", "전원", "음질", "유심", "데이터", "usb", "젠더", "노트북", "모니터"],
    "생활주방_home": ["조립", "설치", "수납", "냄비", "청소", "세제", "이불", "베개", "매트", "커튼", "선반", "옷걸이",
                   "휴지", "수건", "그릇", "컵", "욕실", "주방", "테이프", "접착", "방향제", "모기", "벌레", "바퀴"],
    "유아반려_baby_pet": ["아기", "아가", "애기", "강아지", "고양이", "냥이", "사료", "기저귀", "젖병", "유모차",
                       "장난감", "개월", "이유식", "치발기"],
}
# Keyword-count baseline: aspect keywords a simple dashboard might search for.
ASPECT_KEYWORDS = {
    "배송_delivery": ["배송", "택배", "도착"],
    "가격_price": ["가격", "가성비", "비싸", "저렴"],
    "품질_quality": ["품질", "불량", "마감", "사이즈", "맛"],
    "포장_packaging": ["포장", "박스", "파손"],
    "서비스_service": ["응대", "문의", "고객센터", "답변", "교환", "환불"],
}
HANGUL_SYLLABLE = re.compile(r"[가-힣]")
JAMO = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
STOP_STEMS = {"너무", "정말", "진짜", "그냥", "생각", "구매", "제품", "상품", "좋아", "좋은", "좋고", "있어", "같아",
              "했는", "했어", "해서", "하는", "하고", "재구", "재구매", "주문", "사용", "아직", "조금", "약간", "많이", "완전",
              "그래", "그런", "근데", "다른", "이번", "처음", "그리", "하나", "별로", "아니", "없어", "없네", "있네", "합니",
              "습니", "입니", "됩니", "니다", "네요", "아요", "어요", "해요", "에요", "예요", "같습", "같네", "거같", "것같",
              "괜찮", "만족", "실망", "아쉽", "아쉬", "그닥", "좀더", "보다", "때문", "받았", "받아", "왔어", "왔네", "였어",
              "이에", "있는", "없는", "되네", "되어", "되서", "되요", "돼요", "하네", "하게", "하지", "해야", "해도", "했다",
              "이거", "이건", "그거", "저는", "제가", "우리", "다시", "계속", "정도", "부분", "느낌", "엄청", "제일", "역시"}
STOP_TERMS = {"아요", "어요", "네요", "해요", "니다", "습니", "합니", "는데", "에요", "예요", "어서", "아서", "하고",
              "너무", "좋아", "그냥", "진짜", "정말", "생각", "보다", "구매", "제품", "상품", "있어", "같아", "없어",
              "했는", "는거", "해서", "하는", "이에", "에서", "으로", "지만", "하네", "하게", "되네", "되어", "있는"}


CATEGORY_RE = {c: re.compile(r"(?:^|\s)(?:" + "|".join(map(re.escape, kws)) + ")") for c, kws in CATEGORIES.items()}


def category_of(text):
    """Keywords must start a word, so '립' (lip) does not match 그립 or 조립."""
    hits = {c: len(rx.findall(text)) for c, rx in CATEGORY_RE.items()}
    best, n = max(hits.items(), key=lambda x: x[1])
    return best if n > 0 else "기타_other"


def stems(text):
    """Word-start chunks (2-4 syllables) of each word: keeps 케이스, 충전, 접착력; drops endings like 습니다."""
    out = []
    for w in re.findall(r"[가-힣]+", text):
        if w[:2] in STOP_STEMS:  # 별로네요, 너무너무, 좋아요 ... carry no topic
            continue
        for n in (2, 3, 4):
            if len(w) >= n:
                t = w[:n]
                if t not in STOP_STEMS:
                    out.append(t)
    return out


def is_gibberish(text):
    syll = len(HANGUL_SYLLABLE.findall(text))
    jamo = len(JAMO.findall(text))
    return syll < 3 or jamo > syll


def aspect_models(labels_path, out):
    lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    lab = lab[lab["text_sentiment"] != ""]
    models, quality = {}, {}
    for a in ASPECTS:
        y = lab[a].str.strip().str.lower().replace("", "none")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=5, max_iter=3000, class_weight="balanced")),
        ])
        cv = cross_val_predict(pipe, lab["text"], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        quality[a] = {
            "labeled_neg": int((y == "neg").sum()),
            "cv_macro_f1": round(f1_score(y, cv, average="macro"), 3),
            "cv_neg_f1": round(f1_score(y == "neg", cv == "neg"), 3),
        }
        models[a] = pipe.fit(lab["text"], y)
    (out / "aspect_model_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    print("Aspect model quality (5-fold CV on labeled sample):")
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    return models


def distinctive_terms(texts_in, texts_all_vec, vec, k=4):
    """Most distinctive Hangul character chunks of a cluster vs. its pool."""
    sub = vec.transform(texts_in)
    lift = np.asarray(sub.mean(axis=0)).ravel() - np.asarray(texts_all_vec.mean(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    terms = []
    for i in np.argsort(-lift):
        if lift[i] <= 0:
            break
        t = vocab[i]
        if t in STOP_TERMS or t in STOP_STEMS:
            continue
        clash = [s for s in terms if t.startswith(s) or s.startswith(t)]
        if clash:
            # keep the longer stem if it is nearly as distinctive (e.g. 케이스 over 케이)
            s = clash[0]
            if len(t) > len(s) and lift[i] >= 0.6 * lift[list(vocab).index(s)]:
                terms[terms.index(s)] = t
            continue
        terms.append(t)
        if len(terms) == k:
            break
    return terms


def cluster_pool(pool, depth=0, max_depth=1):
    """Cluster complaints; a broad cluster (few members share its key words) is split once more."""
    vec = TfidfVectorizer(analyzer=stems, min_df=3, sublinear_tf=True, max_features=50_000)
    X = vec.fit_transform(pool["text"])
    k = int(min(15, max(2, len(pool) // 300)))
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5, batch_size=2048).fit(X)
    dist = km.transform(X)
    found = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) < 10:
            continue
        members = pool.iloc[idx]
        terms = distinctive_terms(members["text"], X, vec)
        has_term = members["text"].map(lambda t: sum(term in t for term in terms))
        coverage = float((has_term > 0).mean())
        if coverage < 0.4 and len(members) >= 1000 and depth < max_depth:
            found += cluster_pool(members, depth + 1, max_depth)
            continue
        # Evidence order: mentions the issue's words, clearly negative text, not too short, closest to the centre.
        negativity = 1 - members["p_text_positive"].values
        order_key = (has_term.values > 0) * 2 + (negativity >= 0.5) + (members["text"].str.len().values >= 15) * 0.5
        order = np.lexsort((dist[idx, c], -order_key))
        found.append({"members": members, "terms": terms, "coverage": coverage,
                      "evidence": members.iloc[order[:25]], "examples": members.iloc[order[:3]]})
    return found


def discover(df, min_pool, out):
    issues, evidence = [], []
    trusted = df[df["trusted_complaint"]]
    # Complaints with fewer than 2 content stems ("별로네요") say nothing specific to fix.
    n_stems = trusted["text"].map(lambda t: len(set(stems(t))))
    vague = trusted[n_stems < 2]
    vague.groupby(["category", "aspect"]).size().rename("vague_complaints").to_csv(
        out / "vague_complaints.csv", encoding="utf-8-sig")
    print(f"Vague complaints set aside (no specific content): {len(vague)} of {len(trusted)}")
    for (cat, a), pool in trusted[n_stems >= 2].groupby(["category", "aspect"]):
        if len(pool) < min_pool:
            continue
        cat_total = int((df["category"] == cat).sum())
        for j, f in enumerate(cluster_pool(pool)):
            m = f["members"]
            sev = float((1 - m["p_text_positive"]).mean())
            broad = f["coverage"] < 0.4
            issue_id = f"{cat}|{a}|{j}"
            name = " / ".join(f["terms"]) + (" (broad)" if broad else "")
            issues.append({
                "issue_id": issue_id, "category": cat, "aspect": a, "name": name, "specific": not broad,
                "complaints": int(len(m)),
                "share_of_category_reviews": round(len(m) / cat_total, 4),
                "low_star_reviews": int((m["rating"] <= 2).sum()),
                "hidden_in_4_5_star": int((m["rating"] >= 4).sum()),
                "avg_rating": round(float(m["rating"].mean()), 2),
                "severity": round(sev, 3),
                "keyword_coverage": round(f["coverage"], 3),
                "priority": round(len(m) * sev, 1),
                "examples": " || ".join(f["examples"]["text"].str.slice(0, 80)),
            })
            evidence.append({
                "issue_id": issue_id, "category": cat, "aspect": a, "name": name, "specific": not broad,
                "complaints": int(len(m)), "low_star_reviews": int((m["rating"] <= 2).sum()),
                "hidden_in_4_5_star": int((m["rating"] >= 4).sum()),
                "reviews": [{"review_id": int(r.review_id), "rating": int(r.rating), "text": r.text}
                            for r in f["evidence"].itertuples()],
            })
    iss = pd.DataFrame(issues)
    # Specific issues first (a company can act on them), then by priority.
    iss = iss.sort_values(["category", "specific", "priority"], ascending=[True, False, False])
    iss["rank_in_category"] = iss.groupby("category").cumcount() + 1
    iss.to_csv(out / "issues.csv", index=False, encoding="utf-8-sig")
    ranks = dict(zip(iss["issue_id"], iss["rank_in_category"]))
    with open(out / "issue_evidence.jsonl", "w", encoding="utf-8") as fh:
        for e in sorted(evidence, key=lambda e: (e["category"], ranks[e["issue_id"]])):
            e["rank_in_category"] = int(ranks[e["issue_id"]])
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return iss


def rq3_filter_effect(iss, df, out):
    """What changes when a company trusts text instead of stars.

    Star-only view: read the 1-2 star reviews. Trusted view: complaints found in the text,
    from any star rating, minus low-star reviews whose text is positive.
    """
    res = {}
    for cat, g in iss.groupby("category"):
        low = df[(df["category"] == cat) & (df["rating"] <= 2) & ~df["gibberish"]]
        star_rank = g["low_star_reviews"].rank(ascending=False, method="min")
        trust_rank = g["complaints"].rank(ascending=False, method="min")
        g = g.assign(star_rank=star_rank, trust_rank=trust_rank,
                     hidden_share=g["hidden_in_4_5_star"] / g["complaints"])
        big = g[g["complaints"] >= 50]
        under = big.loc[big["hidden_share"].idxmax()] if len(big) else None
        res[cat] = {
            "issues": int(len(g)),
            "low_star_reviews": int(len(low)),
            "low_star_reviews_with_positive_text": int(low["positive_text_low_star"].sum()),
            "share_of_low_star_reviews_that_are_not_complaints": round(float(low["positive_text_low_star"].mean()), 3),
            "complaints_found_only_in_4_5_star_reviews": int(g["hidden_in_4_5_star"].sum()),
            "share_of_complaints_hidden_in_4_5_star": round(float(g["hidden_in_4_5_star"].sum() / g["complaints"].sum()), 3),
            "rank_correlation_star_vs_trusted": round(float(spearmanr(star_rank, trust_rank)[0]), 3) if len(g) > 2 else None,
            "issues_that_move_2plus_ranks": int(((star_rank - trust_rank).abs() >= 2).sum()),
            "most_underestimated_issue": None if under is None else {
                "name": under["name"], "aspect": under["aspect"],
                "hidden_share": round(float(under["hidden_share"]), 3),
                "rank_by_stars": int(under["star_rank"]), "rank_trusted": int(under["trust_rank"]),
            },
        }
    (out / "rq3_filter_effect.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def keyword_baseline(df, out):
    rows = []
    neg = df[(df["rating"] <= 2) & ~df["gibberish"]]
    for cat, g in neg.groupby("category"):
        for a, kws in ASPECT_KEYWORDS.items():
            rows.append({"category": cat, "aspect": a,
                         "low_star_reviews_with_keyword": int(g["text"].str.contains("|".join(kws)).sum())})
    kb = pd.DataFrame(rows).sort_values(["category", "low_star_reviews_with_keyword"], ascending=[True, False])
    kb.to_csv(out / "keyword_baseline.csv", index=False, encoding="utf-8-sig")
    return kb


def write_report(iss, rq3, filt, out, top=5):
    lines = ["# Issue report (automatic)", "",
             "Generated by `issues.py`. Issues are clusters of trusted negative aspect mentions.",
             "Priority = complaints x severity (mean negativity of the text). The dataset has no dates, so recent growth is not available.", "",
             "## Trust filter", ""]
    for k, v in filt.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Effect of the trust filter (RQ3)", "",
              "Star-only view = reading 1-2 star reviews. Trusted view = complaints in the text from any rating, "
              "minus low-star reviews whose text is positive.", "",
              "| Category | 1-2★ reviews that are not complaints | Complaints only in 4-5★ | Most underestimated issue (share hidden in 4-5★, rank stars → trusted) |",
              "| --- | --- | --- | --- |"]
    for c, r in rq3.items():
        u = r["most_underestimated_issue"]
        us = f"{u['name']} ({u['hidden_share']:.0%}, #{u['rank_by_stars']} → #{u['rank_trusted']})" if u else "-"
        lines.append(f"| {c} | {r['low_star_reviews_with_positive_text']} ({r['share_of_low_star_reviews_that_are_not_complaints']:.0%}) | "
                     f"{r['complaints_found_only_in_4_5_star_reviews']} ({r['share_of_complaints_hidden_in_4_5_star']:.0%}) | {us} |")
    for cat, g in iss.groupby("category"):
        lines += ["", f"## {cat}", "", "| Rank | Aspect | Issue (key words) | Complaints | In 4-5★ | Severity | Example |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in g.head(top).itertuples():
            ex = r.examples.split(" || ")[0].replace("|", "/")
            lines.append(f"| {r.rank_in_category} | {r.aspect} | {r.name} | {r.complaints} | {r.hidden_in_4_5_star} | {r.severity} | {ex} |")
    (out / "issue_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/all_scores.csv")
    ap.add_argument("--labels", default="annotation/claude_labels.csv")
    ap.add_argument("--out", default="results/issues")
    ap.add_argument("--min-pool", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    df["text"] = df["text"].astype(str)
    df["category"] = df["text"].map(category_of)
    df["gibberish"] = df["text"].map(is_gibberish)
    df["positive_text_low_star"] = (df["rating"] <= 2) & (df["p_text_positive"] >= 0.5)
    filt = {
        "reviews": int(len(df)),
        "gibberish_removed": int(df["gibberish"].sum()),
        "low_star_but_positive_text_removed": int((df["positive_text_low_star"] & ~df["gibberish"]).sum()),
    }
    print("Trust filter:", filt)
    print("Categories:", df["category"].value_counts().to_dict())

    models = aspect_models(args.labels, out)
    trusted = ~df["gibberish"] & ~df["positive_text_low_star"]
    aspect_neg = pd.DataFrame(index=df.index)
    for a, m in models.items():
        aspect_neg[a] = m.predict(df["text"]) == "neg"
    # One row per (review, negative aspect): a review can complain about two things.
    long = aspect_neg.stack()
    long = long[long].reset_index().rename(columns={"level_1": "aspect"})[["level_0", "aspect"]]
    comp = df.loc[long["level_0"]].reset_index(drop=True)
    comp["aspect"] = long["aspect"].values
    comp["trusted_complaint"] = trusted.loc[long["level_0"]].values
    filt["negative_aspect_mentions"] = int(len(comp))
    filt["trusted_negative_aspect_mentions"] = int(comp["trusted_complaint"].sum())

    iss = discover(comp, args.min_pool, out)
    rq3 = rq3_filter_effect(iss, df, out)
    keyword_baseline(df, out)
    write_report(iss, rq3, filt, out)
    df[["review_id", "category"]].to_csv(out / "review_category.csv", index=False)
    print(f"\n{len(iss)} issues across {iss['category'].nunique()} categories. See {out / 'issue_report.md'}")
    print(json.dumps(rq3, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

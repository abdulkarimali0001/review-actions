"""
Unit tests for the rules the paper's numbers depend on.
Run from the project folder:  python -m pytest -q
They use tiny made-up inputs, so they run in a few seconds and need no data download.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_reviews  # noqa: E402
from evaluate import mismatch_from, scores, weighted_rate  # noqa: E402
from issues import category_of, is_gibberish, stems  # noqa: E402
from llm_client import mock_answer, parse_json  # noqa: E402
from llm_label import fleiss_kappa  # noqa: E402
from sample_for_annotation import band  # noqa: E402


# ---------------------------------------------------------------- data loading
def test_load_reviews_cleans_dedupes_and_labels(tmp_path):
    f = tmp_path / "reviews.txt"
    f.write_text("5\t좋아요\n1\t별로예요\n5\t좋아요\n2\t   \n4\t배송 빨라요\n", encoding="utf-8")
    df = load_reviews(f)
    assert list(df["text"]) == ["좋아요", "별로예요", "배송 빨라요"]  # duplicate + empty removed
    assert list(df["label"]) == [1, 0, 1]                           # 4-5 stars = positive
    assert list(df["review_id"]) == [0, 1, 2]                       # ids follow the cleaned order


# ---------------------------------------------------------------- mismatch rule
def test_mismatch_rule():
    rating = pd.Series([5, 5, 1, 1, 4, 2, 5])
    text = pd.Series(["neg", "pos", "pos", "neg", "mixed", "neutral", "neutral"])
    # only high stars + negative text, or low stars + positive text, are mismatches
    assert list(mismatch_from(rating, text)) == [True, False, True, False, False, False, False]


def test_scores_perfect_and_empty():
    y = np.array([True, False, True])
    w = np.array([1.0, 2.0, 3.0])
    perfect = scores(y, y.copy(), w)
    assert perfect["sample"]["f1"] == 1.0 and perfect["weighted"]["f1"] == 1.0
    none = scores(y, np.zeros(3, dtype=bool), w)
    assert none["sample"]["recall"] == 0.0 and none["sample"]["precision"] == 0.0


def test_weighted_rate_uses_weights():
    flag = pd.Series([True, False])
    w = pd.Series([3.0, 1.0])
    assert weighted_rate(flag, w) == 0.75  # unweighted would be 0.5


# ---------------------------------------------------------------- sampling bands
def test_band_thresholds():
    assert band(0.95) == "high" and band(0.9) == "high"
    assert band(0.7) == "mid" and band(0.5) == "mid"
    assert band(0.49) == "low"


# ---------------------------------------------------------------- issues.py helpers
def test_category_keywords_must_start_a_word():
    assert category_of("립스틱 색이 예뻐요") == "뷰티_beauty"
    assert category_of("그립감이 좋아요") != "뷰티_beauty"  # '립' inside 그립 must not count
    assert category_of("맛있어요 또 먹을게요") == "식품_food"
    assert category_of("좋아요") == "기타_other"


def test_gibberish_detection():
    assert is_gibberish("ㅁㄴㅇㄹㅁㄴㅇㄹ")
    assert is_gibberish("굿")                   # fewer than 3 syllables
    assert not is_gibberish("배송이 빨라서 좋아요")


def test_stems_drop_filler_words():
    s = stems("사이즈가 너무 작아요")
    assert "사이즈" in s and "작아" in s
    assert not any(t.startswith("너무") for t in s)  # filler word removed


# ---------------------------------------------------------------- LLM helpers
def test_parse_json_strips_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('{"b": [1, 2]}') == {"b": [1, 2]}


def test_mock_label_answer_shape():
    ans = json.loads(mock_answer("TASK: LABEL_REVIEWS\n[1] 배송이 너무 늦어요 별로\n[2] 좋아요 최고"))
    ids = [x["id"] for x in ans["labels"]]
    assert ids == [1, 2]
    assert ans["labels"][0]["sentiment"] == "neg"
    assert set(ans["labels"][0]["aspects"]) == {"delivery", "price", "quality", "packaging", "service"}


def test_fleiss_kappa_perfect_agreement():
    table = [[3, 0], [0, 3], [3, 0]]  # 3 raters agree on every item
    assert abs(fleiss_kappa(table) - 1.0) < 1e-9


def test_prf_weighted():
    from analysis import prf
    y = np.array([True, True, False, False])
    pred = np.array([True, False, True, False])
    p, r, f = prf(y, pred, np.ones(4))
    assert (p, r, f) == (0.5, 0.5, 0.5)
    # weighting the missed mismatch more lowers recall
    p, r, f = prf(y, pred, np.array([1, 3, 1, 1]))
    assert p == 0.5 and r == 0.25


def test_cascade_extremes():
    from analysis import cascade
    df = pd.DataFrame({"mismatch": [True, False, True, False], "weight": [1.0] * 4,
                       "dis_cheap": [0.95, 0.6, 0.4, 0.05], "flag_gemini": [True, False, True, False]})
    t, _ = cascade(df, "cheap")
    # sending everything between 0.05 and 1.01 to the LLM reproduces the LLM exactly
    row = t[(t["low"] == 0.05) & (t["high"] == 1.01)].iloc[0]
    assert row["f1"] == 1.0 and row["share_sent_to_llm"] == 0.75


def test_label_tool_is_blind(tmp_path):
    """The labeling page must contain review text only: no star ratings or model scores."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"item": [1, 2], "pilot": ["True", "False"], "text": ["PILOTTEXT </script>", "HIDDENTEXT"],
                  "rating": [5, 1]}).to_csv(sheet, index=False)
    out = tmp_path / "tool.html"
    subprocess.run([sys.executable, str(root / "make_label_tool.py"), "--sheet", str(sheet), "--out", str(out)],
                   check=True, cwd=root, capture_output=True)
    html = out.read_text(encoding="utf-8")
    assert "PILOTTEXT" in html and "HIDDENTEXT" not in html          # pilot rows only
    assert '"rating"' not in html and "<\\/script>" in html  # no stars leak; text cannot break the page

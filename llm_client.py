"""
Small Gemini client shared by the LLM scripts.

- Reads the key from the GEMINI_API_KEY environment variable (free key: https://aistudio.google.com/apikey).
- Asks for JSON output at temperature 0 (less run-to-run variation).
- Retries with back-off on rate limits (free tier allows only a few requests per minute).
- Caches every answer in a .jsonl file, so an interrupted run resumes without paying twice.
- --mock mode returns fake but well-formed answers, to test the pipeline without a key.

Install:  pip install google-genai
Check current model IDs at https://ai.google.dev/gemini-api/docs/models
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

DEFAULT_LABEL_MODEL = "gemini-3.5-flash-lite"   # cheap and fast, for labeling 3,000 reviews
DEFAULT_REASON_MODEL = "gemini-3.8-flash"       # stronger, for recommendations and verification


class LLM:
    def __init__(self, model, cache_path, rpm=10, mock=False, temperature=0.0):
        self.model = model
        self.mock = mock
        self.temperature = temperature
        self.min_gap = 60.0 / max(rpm, 1)
        self.last_call = 0.0
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
        if not mock:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise SystemExit("Set GEMINI_API_KEY first (free key: https://aistudio.google.com/apikey), "
                                 "or run with --mock to test without a key.")
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=key)
            self.types = types

    def _key(self, prompt, tag):
        return hashlib.sha256(f"{self.model}|{self.temperature}|{tag}|{prompt}".encode()).hexdigest()

    def json(self, prompt, tag=""):
        """Send a prompt, return parsed JSON. `tag` separates repeated runs of the same prompt."""
        k = self._key(prompt, tag)
        if k in self.cache:
            return self.cache[k]
        text, usage = (mock_answer(prompt), None) if self.mock else self._call(prompt)
        data = parse_json(text)
        rec = {"key": k, "tag": tag, "model": self.model, "response": data}
        if usage:  # latency and token counts, used for the cost/speed table (not part of the answer)
            rec["usage"] = usage
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.cache[k] = data
        return data

    def _call(self, prompt, max_tries=8):
        for attempt in range(max_tries):
            wait = self.min_gap - (time.time() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.time()
            try:
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature, response_mime_type="application/json"),
                )
                um = getattr(resp, "usage_metadata", None)
                usage = {"seconds": round(time.time() - t0, 2),
                         "prompt_tokens": getattr(um, "prompt_token_count", None),
                         "output_tokens": getattr(um, "candidates_token_count", None)}
                return resp.text, usage
            except Exception as e:  # rate limit or temporary server error
                msg = str(e)
                if re.search(r"PerDay|per day", msg, re.I):  # daily free quota: waiting minutes will not help
                    raise SystemExit(f"Daily request limit reached for {self.model}. Answers so far are cached; "
                                     "run again after the limit resets (see the 'retry in' time in the error).\n"
                                     + msg[:300])
                if attempt == max_tries - 1 or not re.search(r"429|500|503|RESOURCE_EXHAUSTED|UNAVAILABLE|deadline", msg, re.I):
                    raise
                delay = min(120, 10 * 2 ** attempt)
                print(f"  API busy ({msg[:80]}...), retrying in {delay}s")
                time.sleep(delay)


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


# ----------------------------------------------------------------------------- mock answers
NEG_WORDS = ["별로", "실망", "불량", "안되", "느려", "늦", "최악", "비추", "작아", "아쉽", "냄새", "파손", "찢어", "깨져", "반품"]
POS_WORDS = ["좋아", "만족", "최고", "빠르", "예뻐", "맛있", "굿", "추천", "편해", "저렴"]


def _mock_sentiment(text):
    n = sum(w in text for w in NEG_WORDS)
    p = sum(w in text for w in POS_WORDS)
    if n and p:
        return "mixed"
    return "neg" if n else "pos" if p else "neutral"


def mock_answer(prompt):
    """Fake, deterministic answers in the same JSON shapes the real prompts ask for."""
    if "TASK: LABEL_REVIEWS" in prompt:
        items = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.M)
        out = []
        for i, t in items:
            s = _mock_sentiment(t)
            asp = {a: "none" for a in ["delivery", "price", "quality", "packaging", "service"]}
            if "배송" in t:
                asp["delivery"] = "neg" if any(w in t for w in ["느", "늦"]) else "pos"
            if s in ("neg", "mixed"):
                asp["quality"] = "neg"
            out.append({"id": int(i), "sentiment": s, "aspects": asp})
        return json.dumps({"labels": out})
    if "TASK: RATING_TEXT_CONSISTENCY" in prompt:
        items = re.findall(r"^\[(\d+)\] stars=(\d) \| (.*)$", prompt, re.M)
        out = []
        for i, r, t in items:
            s = _mock_sentiment(t)
            mm = (int(r) >= 4 and s == "neg") or (int(r) <= 2 and s == "pos")
            out.append({"id": int(i), "consistent": not mm})
        return json.dumps({"labels": out})
    if "TASK: GROUNDED_RECOMMENDATION" in prompt:
        ids = re.findall(r"^\[R(\d+)\]", prompt, re.M)
        if len(ids) < 5:
            return json.dumps({"abstain": True, "abstain_reason": "Too few reviews to support a recommendation.",
                               "problem_summary": "", "claims": [], "recommendation": "", "owner": "", "confidence": "low"})
        return json.dumps({
            "abstain": False, "abstain_reason": "",
            "problem_summary": "Customers report the same problem repeatedly (mock).",
            "claims": [{"text": "Several customers describe this problem (mock).", "review_ids": [int(x) for x in ids[:3]]}],
            "recommendation": "Check the most common failure and fix it at the source (mock).",
            "owner": "product", "confidence": "medium"})
    if "TASK: UNGROUNDED_RECOMMENDATION" in prompt:
        return json.dumps({
            "problem_summary": "Customers are unhappy with this aspect (mock).",
            "claims": [{"text": "Customers are unhappy with this aspect (mock)."},
                       {"text": "Competitors handle this better (mock, unsupported)."}],
            "recommendation": "Improve quality control and communication (mock).",
            "owner": "product", "confidence": "high"})
    if "TASK: VERIFY_CLAIM" in prompt:
        has_reviews = bool(re.search(r"^\[R\d+\]", prompt, re.M))
        unsupported = "unsupported" in prompt.split("CLAIM:")[-1].split("\n")[0]
        verdict = "not_supported" if unsupported or not has_reviews else "supported"
        ids = [int(x) for x in re.findall(r"^\[R(\d+)\]", prompt, re.M)]
        return json.dumps({"verdict": verdict, "relevant_review_ids": ids[:2], "reason": "mock"})
    raise ValueError("Unknown prompt type for mock")

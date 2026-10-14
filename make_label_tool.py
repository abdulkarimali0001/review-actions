"""
Build annotation/label_tool.html: an offline page for labeling the pilot reviews.

Each annotator opens the file in a browser, picks A or B, and labels one review at a time
(text only, never the stars). Progress is saved in the browser; "Export CSV" downloads
pilot_annotator_A.csv / _B.csv, which agreement.py reads directly (put it in annotation/).

Usage:  python make_label_tool.py            # pilot rows (items 1-200)
        python make_label_tool.py --all      # all 3,000 rows
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ASPECTS = [("배송_delivery", "Delivery 배송", "q"), ("가격_price", "Price 가격", "w"),
           ("품질_quality", "Quality 품질", "e"), ("포장_packaging", "Packaging 포장", "r"),
           ("서비스_service", "Service 서비스", "t"), ("기타_other", "Other 기타", "y")]

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review Labeling</title>
<style>
:root{--bg:#f7f6f3;--card:#fff;--ink:#1f1f1e;--muted:#6b6a64;--line:#dedcd5;--acc:#2a78d6;--pos:#1a8a5f;--neg:#c2410c;--mix:#8a5cf6;--neu:#6b6a64}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1e;--ink:#f0efec;--muted:#a9a89f;--line:#3a3935;--acc:#5598e7;--pos:#3fbf8a;--neg:#f07a43;--mix:#a78bfa;--neu:#a9a89f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;margin-bottom:14px}
h1{font-size:18px;margin:0}.muted{color:var(--muted);font-size:13px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:6px 0 16px}.bar>div{height:100%;background:var(--acc);width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
.review{font-size:20px;line-height:1.6;margin:6px 0 18px;word-break:keep-all;white-space:pre-wrap}
.row{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.on{color:#fff;border-color:transparent}
.s-pos.on{background:var(--pos)}.s-neg.on{background:var(--neg)}.s-mixed.on{background:var(--mix)}.s-neutral.on{background:var(--neu)}
.asp{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px}
.asp div{display:flex;align-items:center;justify-content:space-between;border:1px solid var(--line);border-radius:8px;padding:6px 8px}
.asp button{padding:4px 9px;font-size:13px}.a-pos.on{background:var(--pos)}.a-neg.on{background:var(--neg)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;color:var(--muted)}
textarea{width:100%;font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;min-height:44px}
.nav{display:flex;justify-content:space-between;align-items:center;margin-top:14px;gap:8px;flex-wrap:wrap}
.primary{background:var(--acc);color:#fff;border-color:transparent}
details{margin-top:16px}summary{cursor:pointer;color:var(--muted)}
.start{max-width:420px;margin:12vh auto;text-align:center}.start button{margin:6px;padding:12px 22px;font-size:16px}
.warn{color:var(--neg);font-size:13px;min-height:18px}
</style></head><body><main id="app"></main>
<script>
const ITEMS = __ITEMS__;
const ASPECTS = __ASPECTS__;
const SENT = [["pos","Positive 긍정","1"],["neg","Negative 부정","2"],["mixed","Mixed 혼합","3"],["neutral","Neutral 중립","4"]];
let who = null, i = 0, L = {};
const app = document.getElementById("app");
const key = () => "labels_" + who + "_" + ITEMS.length;
function store(){ try{ localStorage.setItem(key(), JSON.stringify({i, L})); }catch(e){} }
function restore(){ try{ const s = JSON.parse(localStorage.getItem(key()) || "null"); if(s){ L = s.L || {}; i = s.i || 0; } }catch(e){} }
const cur = () => (L[ITEMS[i].item] ||= {s:"", a:{}, u:false, n:""});
const done = () => ITEMS.filter(x => L[x.item] && (L[x.item].s || L[x.item].u)).length;

function start(){
  app.innerHTML = `<div class="start card"><h1>Review labeling</h1>
  <p class="muted">Label the text only. Do not look at the other annotator's labels.</p>
  <p>Which annotator are you?</p><button class="primary" onclick="pick('A')">Annotator A</button>
  <button class="primary" onclick="pick('B')">Annotator B</button>
  <p class="muted">Progress is saved in this browser. Use the same browser to continue later.</p></div>`;
}
function pick(w){ who = w; restore(); render(); }

function render(){
  const it = ITEMS[i], c = cur(), n = done();
  app.innerHTML = `<header><h1>Annotator ${who} · Review ${i+1} / ${ITEMS.length}</h1>
    <span class="muted">${n} labeled</span></header>
  <div class="bar"><div style="width:${100*n/ITEMS.length}%"></div></div>
  <div class="card">
    <div class="muted">Item ${it.item}</div>
    <div class="review"></div>
    <div class="muted">Overall sentiment of the text</div>
    <div class="row">${SENT.map(([v,l,k]) => `<button class="s-${v} ${c.s===v?"on":""}" onclick="setS('${v}')">${l} <kbd>${k}</kbd></button>`).join("")}</div>
    <div class="muted" style="margin-top:10px">Aspects mentioned (leave blank if not mentioned). Key cycles blank → pos → neg.</div>
    <div class="asp">${ASPECTS.map(([col,l,k]) => `<div><span>${l} <kbd>${k}</kbd></span><span>
      <button class="a-pos ${c.a[col]==="pos"?"on":""}" onclick="setA('${col}','pos')">pos</button>
      <button class="a-neg ${c.a[col]==="neg"?"on":""}" onclick="setA('${col}','neg')">neg</button></span></div>`).join("")}</div>
    <div class="row" style="margin-top:10px"><button class="${c.u?"on s-neutral":""}" onclick="toggleU()">Can't judge (unclear) <kbd>x</kbd></button></div>
    <textarea id="note" placeholder="Note for hard cases (optional)">${esc(c.n)}</textarea>
    <div class="warn" id="warn"></div>
    <div class="nav"><button onclick="go(-1)">← Back</button>
      <span class="muted"><kbd>Enter</kbd> next · <kbd>←</kbd> back · <kbd>j</kbd> jump to first unlabeled</span>
      <button class="primary" onclick="go(1)">Next →</button></div>
  </div>
  <div class="row" style="margin-top:14px"><button onclick="exportCSV()">Export CSV (${n}/${ITEMS.length})</button>
    <button onclick="switchWho()">Switch annotator</button></div>
  <details><summary>Labeling rules (short version)</summary><ul>
    <li><b>pos</b>: overall satisfied, even with a small reservation. <b>neg</b>: overall dissatisfied, even with a small positive.</li>
    <li><b>mixed</b>: clear praise and clear complaint with no obvious winner. <b>neutral</b>: no opinion (e.g. "잘 받았습니다").</li>
    <li>Judge the meaning, not keywords: "별로 안 좋아요" → neg, "나쁘지 않아요" → usually pos. Sarcasm counts as what is meant.</li>
    <li>Quality = the product itself (material, taste, size, function, design). Price includes 가성비.</li>
    <li>Full guide: annotation/LABELING_GUIDE.md</li></ul></details>`;
  app.querySelector(".review").textContent = it.text;
  document.getElementById("note").oninput = e => { cur().n = e.target.value; store(); };
}
function esc(s){ return (s||"").replace(/[&<>"]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[m])); }
function setS(v){ const c = cur(); c.s = c.s===v ? "" : v; if(c.s) c.u = false; store(); render(); }
function setA(col, v){ const c = cur(); c.a[col] = c.a[col]===v ? "" : v; store(); render(); }
function cycleA(col){ const c = cur(), o = ["", "pos", "neg"]; c.a[col] = o[(o.indexOf(c.a[col]||"")+1)%3]; store(); render(); }
function toggleU(){ const c = cur(); c.u = !c.u; if(c.u) c.s = ""; store(); render(); }
function go(d){
  const c = cur();
  if(d>0 && !c.s && !c.u){ document.getElementById("warn").textContent = "Choose a sentiment (or 'unclear') before moving on."; return; }
  i = Math.min(ITEMS.length-1, Math.max(0, i+d)); store(); render(); window.scrollTo(0,0);
}
function jump(){ const k = ITEMS.findIndex(x => !(L[x.item] && (L[x.item].s || L[x.item].u))); if(k>=0){ i = k; store(); render(); } }
function switchWho(){ who = null; i = 0; L = {}; start(); }
function exportCSV(){
  const cols = ["item","text","text_sentiment",...ASPECTS.map(a=>a[0]),"unclear","note"];
  const q = v => '"' + String(v ?? "").replace(/"/g,'""') + '"';
  const rows = ITEMS.map(it => { const c = L[it.item] || {s:"",a:{},u:false,n:""};
    return [it.item, it.text, c.s, ...ASPECTS.map(a => c.a[a[0]]||""), c.u?"1":"", c.n].map(q).join(","); });
  const blob = new Blob(["﻿" + [cols.join(","), ...rows].join("\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `pilot_annotator_${who}.csv`; a.click();
}
document.addEventListener("keydown", e => {
  if(!who || e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey) return;
  const s = SENT.find(x => x[2] === e.key); if(s){ setS(s[0]); return; }
  const a = ASPECTS.find(x => x[2] === e.key.toLowerCase()); if(a){ cycleA(a[0]); return; }
  if(e.key === "x") toggleU(); else if(e.key === "Enter" || e.key === "ArrowRight") go(1);
  else if(e.key === "ArrowLeft") go(-1); else if(e.key === "j") jump();
});
start();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="annotation/annotator_A.csv", help="blind sheet (item, pilot, text)")
    ap.add_argument("--all", action="store_true", help="include all rows, not only the pilot")
    ap.add_argument("--out", default="annotation/label_tool.html")
    args = ap.parse_args()

    df = pd.read_csv(args.sheet, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not args.all:
        df = df[df["pilot"].str.lower() == "true"]
    items = [{"item": int(r["item"]), "text": r["text"]} for _, r in df.iterrows()]
    page = (PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
                .replace("__ASPECTS__", json.dumps(ASPECTS, ensure_ascii=False)))
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"Saved {args.out} with {len(items)} reviews (text only, no star ratings)")


if __name__ == "__main__":
    main()

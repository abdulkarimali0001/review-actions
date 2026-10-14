# 리뷰 라벨링 가이드 / Review Labeling Guide

Two people label **every** review independently. Do not discuss labels until both are done.
You see the review text only, with no star rating. That is on purpose: judge the text alone.

Start with the **pilot** rows (`pilot = True`, items 1–200). After the pilot, run
`python agreement.py --dir annotation --pilot`. If kappa is below 0.6, discuss the
disagreements, update this guide, then continue.

## Easiest way to label: the labeling page

Open `annotation/label_tool.html` in Chrome or Safari (double-click it; no internet needed).
Pick **Annotator A** or **B**, then label one review at a time:

- `1` pos · `2` neg · `3` mixed · `4` neutral · `x` can't judge
- `q` delivery · `w` price · `e` quality · `r` packaging · `t` service · `y` other (each press: blank → pos → neg)
- `Enter` next · `←` back · `j` jump to the first unlabeled review

Progress is saved in that browser. When done, click **Export CSV** and put the downloaded
`pilot_annotator_A.csv` (or `_B.csv`) in the `annotation/` folder. `agreement.py` reads it directly.
The Excel sheets still work if you prefer them.

## Columns to fill

| Column | Allowed values | Meaning |
| --- | --- | --- |
| `text_sentiment` | `pos` / `neg` / `mixed` / `neutral` | Overall feeling of the text |
| `배송_delivery` | `pos` / `neg` / blank | Speed, arrival, missing items, courier |
| `가격_price` | `pos` / `neg` / blank | Price, value for money (가성비) |
| `품질_quality` | `pos` / `neg` / blank | The product itself: material, taste, size, function, design |
| `포장_packaging` | `pos` / `neg` / blank | Box, wrapping, damage in transit |
| `서비스_service` | `pos` / `neg` / blank | Seller response, exchange, refund, inquiry |
| `기타_other` | `pos` / `neg` / blank | A clear opinion about something not listed above |
| `unclear` | `1` or blank | You cannot judge (no meaning, foreign language, only symbols) |
| `note` | free text | Anything worth explaining, especially for hard cases |

Leave an aspect **blank** if the review does not mention it. Use lowercase English values exactly as written.

## Rules for `text_sentiment`

- **pos**: overall satisfied, even with a tiny reservation ("좋아요 근데 조금 작아요" → pos, quality neg).
- **neg**: overall dissatisfied, even with a small positive ("배송은 빠른데 제품이 별로" → neg).
- **mixed**: clear praise **and** clear complaint with no obvious winner ("제품은 좋은데 배송이 너무 느려요").
- **neutral**: no opinion ("아직 안 써봤어요", "잘 받았습니다" with nothing else).
- Judge the **meaning**, not keywords. "별로 안 좋아요" is neg; "나쁘지 않아요" is usually pos.
- ㅠㅠ alone does not make a review negative; read the words around it.
- Sarcasm counts as what the writer means ("참 빨리도 오네요 2주 걸림" → neg, delivery neg).

## Worked examples (real reviews from the dataset)

| Review | text_sentiment | Aspects |
| --- | --- | --- |
| 배공빠르고 굿 | pos | delivery pos |
| 택배가 엉망이네용 저희집 밑에층에 말도없이 놔두고가고 | neg | delivery neg |
| 아주좋아요 바지 정말 좋아서2개 더 구매했어요 이가격에 대박입니다. 바느질이 조금 엉성하긴 하지만 편하고 가성비 최고예요. | pos | quality pos, price pos |
| 선물용으로 빨리 받아서 전달했어야 하는 상품이었는데 머그컵만 와서 당황했습니다. ... 배송이 오래걸렸으면 사는거 다시 생각했을거같아요 아쉽네요.. | neg | delivery neg, service pos (seller responded quickly) |
| 민트색상 예뻐요. 옆 손잡이는 거는 용도로도 사용되네요 ㅎㅎ | pos | quality pos |
| 배송만 배송은 좋으나 포장은 별로.. 그냥 스크레치 안생길 정도의 포장이였음. 사용하기엔 나쁘지 않음 | mixed | delivery pos, packaging neg, quality pos |
| 포장이 다뜯긴상태로 왔어요 별로네요 | neg | packaging neg |
| 그저그래요 싼맛에 씀 | neutral | price pos (cheap is the only reason) |
| 다 안왔어요 ㅠㅠ 다시보내주세요 | neg | delivery neg |
| 생각했던 디자인과 다른듯해요ㅠ | neg | quality neg |

Hard cases: write your reasoning in `note`. The third person uses those notes to settle disagreements.

## After labeling

1. `python agreement.py --dir annotation` writes `disagreements.csv`.
2. A third person fills `final_sentiment` for each row and saves the file as `annotation/adjudicated.csv`.
3. Run `python agreement.py --dir annotation` again. It writes `gold_labels.csv` and `prevalence.json`.

**Never open or share `sample_key_DO_NOT_SHARE.csv` with annotators.** It contains the star ratings and model scores.

# Dataset review (`data/eval_raw.jsonl`)

## State
- **Rows:** 265 of the planned 1100 (ids 0–264). Ids 0–99 are **dev**, used for decoding tuning. Ids 100–264 are **eval**: 165 rows, never used for tuning.
- **Generation:** `qwen3:14b` via `sidm.gen_dataset`. Generation stopped when the session ended and can be resumed with the same command; rows that are already done are skipped.
- **Validation:** 256 rows passed on the first attempt. 9 needed a retry with error feedback.
- **Distribution:**
  - Category depth: L3 140, L2 93, L1 17, no category 15.
  - Filters per query: 0 → 38, 1 → 110, 2 → 85, 3 → 32.
  - 29 queries were asked to contain a typo. All 265 queries are distinct.

## How it was checked
1. **Automatic checks, on every row:**
   - Every word is covered by exactly one span, in order.
   - The category span is close to a synonym of the gold node, and doesn't match a different node.
   - The filter spans carry exactly the gold attribute=value pairs.
   - Width numbers fall inside their gold bucket.
   - Residual words contain no category or attribute term (the leak check).
2. **Manual reading:** ~60 random rows, all 22 flagged rows, and the first 12 rows in `data/preview.txt`.

## Findings
- **Labels are mostly right and follow the connecting-word rule.** Examples: "oak legs" is filter, "for my mom" is residual, "74 inches" is filter.
- **Fixed by hand (5 rows):** these rows had a connecting word or category word labeled residual. They are marked `manual_fix` in `flags`, and the original is kept in `eval_raw.before_review.jsonl`.
  - #34 "baby" in "baby crib"
  - #46 "room" in "for kids room"
  - #128 "with" in "with tufted"
  - #207 "style" in "bohemian style"
  - #248 "in"/"with" in "in teal" and "with pine", and the repeated "make up vanities"
- **Known gold noise, left as is:**
  - "bed for kids" is gold `beds`, but `kids_beds` is a defensible answer.
  - "furniture round tabke" is gold NONE, but it plausibly means tables.
  - "cabinet cupboard" repeats the category.
- **Realism issues** (the labels are correct, but the queries read less like real searches):
  - Word order is often stilted ("coffee table curved quality affordable wooden legs").
  - Some attribute combinations are implausible ("bench with reclining", "latex mattress with metal" in the early data). They come from attribute applicability being defined per L1 group.
  - The L1 wordings "sleep products" and "office furniture" are unnatural and rare.
  - No-category queries almost always use the word "furniture" (labeled residual).
- **Width:** after the fix, numbers vary and are validated against their bucket. The examples include inches, `"`, cm and ft.

## Implications for the eval
- Gold label noise is estimated at roughly 2–3% of rows.
- The queries are cleaner and more templated than real traffic, so these accuracy numbers are an upper bound on what to expect with real queries.

## Full generation (ids 265–1099)
- **Result:** 834 of 835 rows generated automatically, with 39 retries.
- **Row 1087 failed four times and was written by hand** (`flags: manual_row`): "acent cabinet with natural wood legs under 500 dollars".
  - Its intent had `leg_color=natural` and `leg_material=wood`. Both are naturally expressed by one phrase, "natural wood legs", but a span can carry only one attribute.
  - Split as: "with natural" → leg_color and "wood legs" → leg_material.
  - Every word in that phrase has the role `filter` either way, so the word-level gold is unambiguous.
- **Schema issue:** `leg_color=natural` is worded "natural wood legs" / "light wood legs", which overlaps `leg_material=wood`. See `backlog.md`.

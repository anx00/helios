# Papertrading Strategy Study - 2026-03-02

## Question

User concern:

- papertrader opened very few trades
- several trades looked unintuitive, especially cheap `NO` bets on low-probability buckets
- question: is that actually good technique, or is the policy over-trading tiny complement edges

## Short answer

The current policy is mathematically consistent, but strategically too permissive for a small bankroll.

The main issue is not "papertrading execution". The issue is that the selector strongly rewards cheap `NO` complements, while most protective logic is only applied to secondary `YES` trades.

That produces trades that can be locally EV-positive on paper, but are fragile to model calibration error, stale books, and late-day market weirdness.

## Method

This review combined:

- live code inspection in `core/autotrader.py` and `core/trading_signal.py`
- current board snapshots for `KLGA`, `KATL`, `KORD`, `KMIA`, `KDAL`, `LFPG`, `EGLC`
- local recordings in `data/recordings/date=*/ch=nowcast/events.ndjson`

## Findings

### 1. Candidate selection is asymmetric: `YES` gets extra guardrails, `NO` mostly does not

In `core/autotrader.py`, `_build_candidate_from_trade()` adds a special penalty only for secondary `YES` buckets:

- `secondary_yes_not_good_enough`

There is no matching penalty for "cheap `NO` against the favorite" or "cheap `NO` far from top bucket".

Relevant code:

- `core/autotrader.py:961-964`

The trade side is chosen first by raw edge:

- `edge_yes = fair_yes - yes_entry`
- `edge_no = (1 - fair_yes) - no_entry`
- if `edge_no > edge_yes`, the row becomes `NO`

Relevant code:

- `core/trading_signal.py:1121-1139`

### 2. Policy scoring amplifies complement trades

Allowed trades are scored as:

- `policy_score = edge_points * (0.7 + 0.3 * selected_fair)`

Relevant code:

- `core/trading_signal.py:906-910`

That means a row with:

- `fair_no ~= 0.998`
- `entry_no = 0.03`

gets a very large edge and also a high score multiplier.

Then `evaluate_trade_candidate()` sorts by `strategy_score`, `edge_points`, and `fair_price`, so these rows tend to win station selection.

Relevant code:

- `core/autotrader.py:1137-1145`

### 3. Current live board reproduces the exact behavior the user flagged

Current snapshot for `2026-03-02`:

- `KORD`
  - top `YES`: `48F or higher`
  - `fair_yes = 0.86043`
  - `entry_yes = 0.0015`
  - `policy_score = 82.296574`
  - top `NO`: `42-43F`
  - `fair_no = 0.997669`
  - `entry_no = 0.03`
  - `policy_score = 96.699231`
  - selected candidate: `BUY_NO 42-43F`

- `LFPG`
  - top `YES`: `11C or below`
  - `policy_score = 84.681001`
  - top `NO`: `17C`
  - `fair_no = 1.0`
  - `entry_no = 0.0005`
  - `policy_score = 99.95`
  - selected candidate: `BUY_NO 17C`

- `EGLC`
  - top `YES`: `10C or below`
  - `policy_score = 58.348495`
  - top `NO`: `16C`
  - `fair_no = 0.999991`
  - `entry_no = 0.0005`
  - `policy_score = 99.94883`
  - selected candidate: `BUY_NO 16C`

This is not random behavior. It is the direct output of the current ranking logic.

### 4. The thesis text itself is biased toward allowing complement trades

If a trade is `NO`, `_trade_policy_thesis()` says:

- `The market is overpaying for {label} versus the rest of the board.`

Relevant code:

- `core/trading_signal.py:795-799`

This is mathematically true, but it hides the operational problem:

- for a multi-bucket weather market, "the rest of the board" is not a clean single thesis
- for a small bankroll, these complement trades can dominate selection even when the model is not sharply concentrated
- a slight model miss or late settlement weirdness can wipe out the apparent edge

### 5. Local nowcast data says strong one-bucket dominance is not common enough to justify aggressive complement trading by default

Across local recordings:

- `53,228` snapshots
- `11` days
- `top1_mean = 0.4377`
- `top1_median = 0.4121`
- `top1_p75 = 0.5333`
- `gap_mean = 0.1376`
- `gap_median = 0.0898`
- `gap_p75 = 0.1818`

Only:

- `34.0%` of snapshots have `top1 >= 0.5`
- `5.6%` of snapshots have `top1 >= 0.7`

This matches the prior repo finding that HELIOS is often not saying "one bucket wins clearly". So a policy that keeps selecting complement `NO` trades as if the model were near-certain is too aggressive.

Supporting repo note:

- `docs/trading/AUTOTRADER_DATA_FINDINGS_2026-03-01.md:54-58`

### 6. A sizing safeguard for diffuse distributions appears inactive in the current payload path

`compute_trade_budget_usd()` tries to reduce budget when:

- `distribution_top_probability` is weak
- `distribution_top_gap` is weak

Relevant code:

- `core/autotrader.py:1259-1272`

But the candidate currently receives:

- `distribution_top_probability = null`
- `distribution_top_gap = null`
- `bucket_rank = null`

because `_payload_bucket_distribution()` only looks in:

- root `p_bucket`
- `nowcast.p_bucket`
- `data.p_bucket`
- `trading.p_bucket`

Relevant code:

- `core/autotrader.py:375-404`

The current dashboard payload does not expose those fields there, so `_distribution_stats_for_label()` returns `{}` and the multiplier falls back to `1.0`.

This means one of the intended protections for "flat distribution" is not helping in this runtime path.

## Strategic conclusion

Answering the user's question directly:

- buying only the least likely outcomes is not a good general technique
- buying `NO` on a low-probability bucket can be correct in principle
- but in this system it is currently over-favored relative to the quality of the signal

The present policy is too eager to convert "bucket probability is small" into "buy `NO` aggressively".

For a small account, the better default is:

- prefer the main scenario when the model is concentrated enough
- only allow complement `NO` trades under much stricter conditions
- avoid letting far-from-top `NO` rows dominate station ranking just because their complement fair is near `1.0`

## Recommendation

### Immediate operating recommendation

For papertrading, do not trust current `NO`-heavy behavior as evidence that the strategy is good.

If the goal is learning signal quality with low variance, the safer default is:

1. prefer `YES` on the forecast winner
2. heavily penalize or temporarily disable `NO` entries
3. only re-enable `NO` when the model is sharply dominant and calibration has been checked

### Concrete policy changes to implement next

1. Add an explicit `NO` penalty when the bucket is far from `forecast_winner`.
2. Require stronger dominance for `NO` trades, for example:
   - top bucket probability floor
   - top1-top2 gap floor
   - maximum allowed distance from top bucket
3. Prefer top-bucket `YES` over off-top `NO` unless the `NO` edge is overwhelmingly better after penalties.
4. Fix payload wiring so distribution-based budget reductions actually work.
5. Add papertrader diagnostics showing, per entry:
   - top `YES`
   - chosen trade
   - why chosen trade beat the top forecast bucket

## Bottom line

The odd trades in papertrading are not just bad luck.

They are consistent with the current policy design, and that design is too permissive with cheap complement `NO` trades.

For this bankroll and this model quality, I would tighten the strategy before drawing any conclusions from the papertrading results.

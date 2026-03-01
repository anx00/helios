# Autotrader Forensics - March 1, 2026

## Scope

This report reconstructs what the live autotrader did on **March 1, 2026** using the actual data left on the VM after the bot was stopped.

The goal is not to defend the current strategy. The goal is to identify:

- what the bot actually did
- why profitable or valid positions were targeted for exit
- where the execution/accounting path is broken
- what must change before the bot is ever turned back on

## Files Audited

### Local workspace files

- `.env`
- `core/autotrader.py`
- `market/polymarket_execution.py`
- `web_server.py`
- `data/AUTOTRADE_STOP`

### Remote VM files

- `/opt/helios/.env`
- `/opt/helios/data/autotrader_state.json`
- `/opt/helios/data/autotrader_runtime.json`
- `/opt/helios/data/AUTOTRADE_STOP`
- `/opt/helios/logs/autotrader_trades.jsonl`
- `/opt/helios/logs/helios.log`
- `/opt/helios/logs/helios-error.log`
- `/opt/helios/data/recordings/date=2026-03-01/ch=market/events.ndjson`
- `/opt/helios/data/recordings/date=2026-03-01/ch=nowcast/events.ndjson`
- `/opt/helios/data/recordings/date=2026-03-01/ch=pws/events.ndjson`
- `/opt/helios/data/recordings/date=2026-03-01/ch=l2_snap/events.ndjson`

## Executive Summary

The bot was not failing because of one single bad forecast. It was failing because the live execution and live position lifecycle are internally inconsistent.

The main forensic conclusion is:

1. The bot sends many orders that come back as `success=true, status=delayed`.
2. The runtime treats those orders as `matched_size=0` and therefore as "unfilled".
3. Because it thinks nothing filled, it keeps retrying the same entry or exit.
4. Later, positions disappear from the live portfolio and are marked closed by sync, but the bot never records a fill or an exit reason.
5. On top of that, the exit policy can trigger `policy_flip` even while the held position still has positive mark-to-market PnL and positive model edge.

That means the current system is not just "a bad strategy". It is a combination of:

- broken order lifecycle handling
- broken exit policy semantics
- broken audit trail
- excessive retry behavior after allowance/balance failures

## Hard Findings

### 1. `delayed` orders are being treated as unfilled

Code path:

- `market/polymarket_execution.py`
- `summarize_order_response(...)`

Current behavior:

- when order response status is `delayed`
- and no immediate fill size is returned
- the function sets `matched_size = 0.0`

That is fatal in live trading because a delayed order is not the same thing as a rejected order.

Observed counts from `/opt/helios/logs/autotrader_trades.jsonl`:

- `live_unfilled`: **84**
- `live_exit_unfilled`: **19**
- all sampled rows had:
  - `summary.status = delayed`
  - `raw.status = delayed`
  - `success = true`
  - `matched_size = 0.0`

So the bot repeatedly treated "accepted but delayed" as "did not happen".

### 2. The bot retried the same entries many times

This is direct evidence of duplicate order spam caused by the `delayed -> unfilled` mistake.

Top duplicated entry candidates:

- `EGLC|2026-03-01|14°C OR HIGHER|YES`: **20** delayed order attempts
- `KATL|2026-03-01|76°F OR HIGHER|NO`: **14**
- `KLGA|2026-03-01|40-41°F|NO`: **8**
- `KDAL|2026-02-28|88°F OR HIGHER|YES`: **7**
- `KATL|2026-02-28|68-69°F|NO`: **7**

Example:

- `EGLC|2026-03-01|14°C OR HIGHER|YES`
- multiple orders were sent between **2026-03-01 01:01:53 UTC** and **2026-03-01 01:02:27 UTC**
- same thesis
- same candidate
- same order mode
- same delayed response pattern

This means the bot did not have a pending-order state machine and did not de-duplicate by `order_id` or by `position_key`.

### 3. The bot attempted exits on profitable positions due to `policy_flip`

Code path:

- `core/autotrader.py`
- `_evaluate_exit_decision(...)`

For non-tactical positions, this condition exists:

- if `policy_now.allowed == false` and bid is above stop floor
- force exit with reason `policy_flip`

The problem is that `policy_flip` is binary, while the held position still has continuous information:

- `fair_now`
- `edge_now_points`
- current mark-to-market

Observed `policy_flip` exit attempts: **7**

Of those:

- `policy_flip` with **positive edge_now_points**: **6**
- `policy_flip` with **positive cash_pnl_usd**: **5**
- `policy_flip` targeting `external_live`: **1**

That is exactly the failure mode the user described: positions with remaining positive edge or positive unrealized PnL were still being pushed toward exit because the policy gate turned false.

### 4. At least one `external_live` position was targeted for exit

Observed from nested exit events:

- position: `EGLC|2026-03-01|14°C OR HIGHER|YES`
- strategy: `external_live`
- source: `polymarket_live`
- exit reason: `policy_flip`
- mark PnL at time of attempt: `cash_pnl_usd = +0.7627`
- `fair_now = 0.140157`
- `entry_price = 0.0737`
- `current_price = 0.0975`
- exit request came at **2026-03-01 01:59:52 UTC**

Important nuance:

- current code in `core/autotrader.py` only actively manages positions that `_is_autotrader_managed_position(position)` returns true for
- but the evidence proves at least one `external_live` position was still routed into the exit path historically

Most likely explanations:

- stale state created while live imported positions were being treated as managed
- or historical state/runtime mismatch during the period when `assume_live_positions_managed` semantics were changed

Either way, the live behavior was not safe.

### 5. Closed positions are not being reconciled correctly

Observed from `/opt/helios/data/autotrader_state.json`:

- total closed positions: **9**
- closed positions with `fills_count = 0`: **9**
- managed closed positions: **6**
- managed closed positions with `fills_count = 0`: **6**

Examples:

- `KLGA|2026-03-01|40-41°F|NO`
- `KLGA|2026-03-01|42-43°F|NO`
- `KLGA|2026-03-01|44-45°F|YES`
- `KORD|2026-03-01|34-35°F|YES`

These positions have:

- `status = CLOSED`
- no stored fills
- no `last_exit_reason`
- no `last_exit_utc`

That means they were not closed through `_persist_exit(...)`.
They were closed later by the live sync path because they were no longer present in the wallet snapshot.

This is the strongest evidence that delayed orders were later affecting the portfolio while the autotrader believed they had not filled.

### 6. Allowance/balance failures were spammed without proper cooldown

Observed counts:

- JSONL rows with `status = error`: **1410**
- `helios-error.log` lines containing `not enough balance / allowance`: **1427**

Typical error:

- `PolyApiException[status_code=400, error_message={'error': 'not enough balance / allowance'}]`

This means that after a delayed entry/exit or after balance got tied up in pending orders, the bot kept hammering the API.

That created:

- noisy logs
- unclear portfolio state
- repeated retries
- poor signal-to-noise for forensic analysis

### 7. The trade log itself is corrupted at least once

Observed malformed line:

```json
{"station_id": "LFPG", "status": "skip", "reasons": ["portfolio_full"], "exits": [], "ts_utc": "2026{"station_id": "KATL", "status": "skip", "reasons": ["portfolio_full"], "exits": [], "ts_utc": "2026-03-01T11:24:11.980080+00:00"}
```

Interpretation:

- two JSON objects were partially interleaved on one line
- there is at least one non-atomic concurrent append to `logs/autotrader_trades.jsonl`

So even the audit trail is not fully trustworthy in its current write model.

## Reconstructed Timeline

### 2026-03-01 01:01 UTC to 01:02 UTC

The bot sends repeated delayed entry orders for the same positions:

- `EGLC|2026-03-01|14°C OR HIGHER|YES`
- `KATL|2026-03-01|76°F OR HIGHER|NO`

The responses are `success=true, status=delayed`, but the bot marks them as `live_unfilled`.

### 2026-03-01 01:59:52 UTC

The bot attempts to exit:

- `EGLC|2026-03-01|14°C OR HIGHER|YES`

Exit type:

- `policy_flip`

This is notable because the position was profitable at mark:

- `cash_pnl_usd = +0.7627`

The exit response again comes back:

- `status = delayed`
- `success = true`

Then repeated allowance errors follow. Shortly after, the position disappears from live holdings and is later marked closed with no fill history.

### 2026-03-01 19:55:09 UTC

Position:

- `KLGA|2026-03-01|42-43°F|YES`

Exit attempted:

- `take_profit`

Response:

- `delayed`

Closed in state at:

- `2026-03-01 19:55:22 UTC`

But with:

- `fills_count = 0`
- no exit reason persisted

### 2026-03-01 20:09:49 UTC

Position:

- `KLGA|2026-03-01|40-41°F|NO`

Exit attempted:

- `policy_flip`

At that moment:

- `cash_pnl_usd = +1.2881`
- `fair_now = 0.644098`
- `entry_price = 0.3299`
- `best_bid = 0.37`
- `edge_now_points = +1.4098`

This is not a clean "model broke" case. It is a binary policy override fighting a still-positive edge.

### 2026-03-01 20:25:24 UTC

Same position:

- `KLGA|2026-03-01|40-41°F|NO`

Now an exit is attempted as:

- `take_profit`

Again the response is:

- `delayed`

Then allowance errors hit, and the position disappears from the wallet snapshot.

### 2026-03-01 20:53:19 UTC and 20:53:52 UTC

Position:

- `KLGA|2026-03-01|42-43°F|NO`

Two exit attempts are observed:

- first `take_profit`
- then `policy_flip`

Both are `delayed`. The position closes in state with:

- `fills_count = 0`
- no persisted exit metadata

### 2026-03-01 21:00:39 UTC

Position:

- `KLGA|2026-03-01|44-45°F|YES`

Exit attempted:

- `policy_flip`

At that moment:

- `cash_pnl_usd = +0.3499`
- `edge_now_points = +15.4318`

This is one of the clearest examples of an exit rule that is semantically wrong.

### 2026-03-01 21:03:43 UTC

Position:

- `KORD|2026-03-01|34-35°F|YES`

Exit attempted:

- `policy_flip`

At that moment:

- `fair_now = 0.169706`
- `entry_price = 0.04`
- `edge_now_points = +12.0706`

That should not be an urgent forced exit.

## Why The Current Strategy Is Not Safe

The current live system combines three ideas that should not be mixed this way:

1. **continuous valuation**
   - `fair_now`
   - `edge_now_points`
   - current mark-to-market

2. **binary gating**
   - `policy_now.allowed`

3. **fire-and-forget delayed exchange responses**
   - `status = delayed`

When those three interact badly, the result is:

- a position can still have positive model edge
- the gate can flip to "not allowed"
- the bot sends a delayed exit order
- it records no fill
- it retries or issues more orders
- later the position disappears from the wallet
- state becomes inconsistent

That is not a tolerable failure mode for live money.

## Replay / Recording Observations From Today

Recorded sizes for **2026-03-01** on the VM:

- `ch=l2_snap`: **552,091,648 bytes**
- `ch=pws`: **319,529,979 bytes**
- `ch=market`: **12,478,462 bytes**
- `ch=nowcast`: **13,230,571 bytes**
- `ch=features`: **3,310,729 bytes**
- `ch=event_window`: **1,071,941 bytes**
- `ch=world`: **121,277 bytes**

Interpretation:

- for autotrading forensics, `market + nowcast + world + event_window` are the highest-value compact channels
- `pws` is still too heavy because it is storing large `pws_readings` arrays repeatedly
- `l2_snap` is still very heavy, but at least now it is limited by the newer recorder policy

For strategy analysis, the replay of today suggests:

- the compact `market` channel is enough to reconstruct best bid / ask / spread / depth behavior
- the compact `nowcast` channel is enough to reconstruct fair-value drift
- full dense `pws` snapshots are not necessary at every write interval for trading forensics

## Root Causes

### Root Cause A - No pending order state machine

Missing concepts:

- pending entries by `candidate.position_key`
- pending exits by `position_key`
- `order_id -> reconciliation`
- exchange polling for delayed orders
- retry suppression while an order is unresolved

Without that state machine, `delayed` becomes catastrophic.

### Root Cause B - Exit policy is too eager and too binary

`policy_flip` should not force a live exit when:

- edge remains positive
- fair remains above entry minus a tolerance
- current position is still aligned with the model in value terms

Right now it can.

### Root Cause C - Management boundary is not strict enough

Live imported positions and bot-created positions must be separated by a hard boundary.

The live runtime should never infer management ownership from "account only" assumptions in a live wallet with real money.

### Root Cause D - Observability is not transaction-grade

The bot currently has:

- corrupted log lines
- closed positions with zero stored fills
- missing exit reasons for actual portfolio changes

That is not good enough for live execution.

## Recommended Redesign

### Phase 0 - Keep the bot off

Do not restart live autotrading until the following are done:

- delayed order reconciliation
- pending order dedupe
- exit rule rewrite
- strict managed-position ownership

### Phase 1 - Fix execution before strategy

Required changes:

1. Treat `status=delayed` as `PENDING`, not `UNFILLED`.
2. Store a pending order record keyed by:
   - `order_id`
   - `candidate.position_key` for entries
   - `position_key` for exits
3. Poll the exchange for final order state before retrying.
4. Block new entry retries while a pending order exists for the same thesis.
5. Block new exit retries while a pending exit exists for the same position.
6. Add cooldown after `not enough balance / allowance`.

### Phase 2 - Rewrite exits

Recommended live exit semantics:

- allow `take_profit`
- allow `stop_loss`
- allow `resolution/market_mature`
- allow tactical timeout only for explicitly tactical positions
- do **not** allow `policy_flip` as a hard forced exit by itself

Replace it with:

- soft downgrade
- no new adds
- only exit if:
  - edge is now non-positive
  - or fair is below entry minus tolerance
  - or stop is hit
  - or position has become stale after an official update window

### Phase 3 - Simplify strategy before adding learning

Recommended temporary live strategy:

- `terminal_value` only
- disable `tactical_reprice`
- one active position per station/day
- no same-day flip between opposite sides of the same bracket
- no auto-management of imported positions
- no multi-bucket laddering until execution is trustworthy

### Phase 4 - Learning only after auditability is fixed

The bot should learn from:

- entry snapshot
- exit snapshot
- final realized outcome
- pending order latency
- market microstructure at submit / confirmation / close

But that learning is useless if fills and exits are not recorded correctly.

## Concrete Files To Inspect Manually

If reviewing this by hand, start here:

### Most important

- `/opt/helios/logs/autotrader_trades.jsonl`
- `/opt/helios/data/autotrader_state.json`
- `/opt/helios/data/autotrader_runtime.json`
- `/opt/helios/logs/helios-error.log`

### Useful for replay validation

- `/opt/helios/data/recordings/date=2026-03-01/ch=market/events.ndjson`
- `/opt/helios/data/recordings/date=2026-03-01/ch=nowcast/events.ndjson`
- `/opt/helios/data/recordings/date=2026-03-01/ch=pws/events.ndjson`
- `/opt/helios/data/recordings/date=2026-03-01/ch=l2_snap/events.ndjson`

### Code paths that explain the behavior

- `core/autotrader.py`
- `market/polymarket_execution.py`
- `web_server.py`

## Bottom Line

The autotrader should remain stopped.

The evidence does **not** support the idea that the current losses are only "bad forecasting".
The stronger explanation is:

- duplicate entries caused by delayed-order misclassification
- exit attempts on positions that still had positive value edge
- missing reconciliation of actual fills
- retry loops after allowance failures
- incomplete ownership boundary for live positions

This must be fixed at the execution/state-machine layer before any new strategy or learning loop is trusted with live money.

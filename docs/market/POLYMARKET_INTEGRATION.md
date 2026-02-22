# Polymarket Integration Reference

This document describes how HELIOS integrates Polymarket market data today.

Scope:
- event discovery
- token ID extraction
- live orderbook ingestion
- local L2 mirror
- internal API endpoints used by UI/autotrader
- market-specific pitfalls (YES/NO, units, labels, maturity/target day)

Use this as the primary reference before touching:
- `market/`
- Polymarket-related routes in `web_server.py`
- autotrader market context logic

## 1. Integration architecture (critical design rule)

HELIOS splits Polymarket integration into two paths:

### A. Gamma API = metadata / discovery / macro snapshot
Used for:
- finding the correct event for a station/date
- reading bracket labels (`groupItemTitle`)
- reading snapshot probabilities (`outcomePrices`)
- extracting `clobTokenIds`
- reading volume / active / closed flags

### B. CLOB WebSocket = live market microstructure (hot path)
Used for:
- orderbook snapshots and deltas
- bid/ask spread
- live tape-like updates (`trade`, `last_trade_price`)
- local in-memory orderbook mirror for low-latency reads

This avoids using Gamma as a fake "realtime" feed.

Related architecture rationale:
- `docs/system/ARCHITECTURE_EVOLUTION.md`

## 2. External Polymarket endpoints used

### Gamma API (events)

Endpoint:
- `https://gamma-api.polymarket.com/events`

Typical HELIOS usage:

1. By slug (fallback / direct)
- `GET /events?slug=<event_slug>&_ts=<timestamp_ms>`

2. By date window + weather tag (robust discovery)
- `GET /events?tag_id=84&end_date_min=...&end_date_max=...&limit=200&_ts=<timestamp_ms>`

Notes:
- HELIOS adds `_ts` to reduce cache issues.
- Response is event-centric and includes `markets[]`.
- `outcomePrices` and `clobTokenIds` often arrive as JSON strings, not native arrays.

### CLOB Market WebSocket

Endpoint:
- `wss://ws-subscriptions-clob.polymarket.com/ws/market`

Used for live ingestion of:
- `book` snapshots / deltas
- `price_change`
- `trade`
- `last_trade_price`
- `best_bid_ask`

## 3. Core modules and responsibilities

### Discovery / event metadata
- `market/discovery.py`
  - station/date event resolution
  - token ID extraction (YES + NO)
  - robust parsing of `clobTokenIds`

### Market snapshot / maturity / target date logic
- `market/polymarket_checker.py`
  - slug builder
  - Gamma fallback fetch
  - parsing bracket labels and probabilities
  - maturity checks
  - `get_target_date()` rollover logic (today vs tomorrow market)

### Live market WebSocket client
- `market/polymarket_ws.py`
  - CLOB market channel connection
  - subscriptions
  - message parsing
  - local state (`LiveMarketState`)

### Local orderbook mirror (per token)
- `market/orderbook.py`
  - snapshots, deltas, cached best bid/ask
  - L2 top-N view for UI/backtest/replay

### Price normalization / quote selection policy
- `core/market_pricing.py`
  - robust probability selection from bid/ask/mid/reference
  - conservative behavior on wide spreads

### Label normalization (bucket names)
- `core/polymarket_labels.py`
  - normalize/parse/sort labels
  - map observed temp to bucket labels

## 4. Event discovery flow (station/date -> event -> token IDs)

This is the operational sequence used by HELIOS.

1. Build station/date slug prefix
- format: `highest-temperature-in-{city}-on-{month}-{day}`

2. Query Gamma weather events for the target date window
- `tag_id=84`
- `end_date_min` / `end_date_max`

3. Filter event list by slug prefix
- match station-specific city slug
- prefer most recently updated event if multiple exist

4. Extract market token IDs from `markets[]`
- parse `clobTokenIds`
- keep both outcomes:
  - index 0 = `YES`
  - index 1 = `NO`

5. Register token metadata locally
- HELIOS stores token metadata in string form:
  - `STATION|BRACKET|OUTCOME`

Fallbacks used in production:
- discovery by station/date
- fetch by slug
- parse raw event payload directly
- cached previous token set if discovery fails

## 5. Probability and bracket parsing (Gamma snapshots)

### Important semantics

- `outcomePrices = [YesPrice, NoPrice]`
- HELIOS uses `YesPrice` (index `0`) as the bracket probability for that market option

### Common payload gotcha

These fields may be strings containing JSON:
- `outcomePrices`
- `clobTokenIds`

Always parse defensively.

### Typical data extracted per bracket

- `name` (from `groupItemTitle`)
- `yes_price`
- `volume`
- `yes_token_id`
- `no_token_id`

## 6. Live CLOB WebSocket ingestion (hot path)

### Subscription payloads used by HELIOS

Initial subscription:

```json
{
  "type": "market",
  "assets_ids": ["token_yes", "token_no"],
  "custom_feature_enabled": true
}
```

Incremental subscription:

```json
{
  "operation": "subscribe",
  "assets_ids": ["token_yes_2"],
  "custom_feature_enabled": true
}
```

### Keepalive

HELIOS sends text `PING` periodically (around every 10s) in addition to library-level websocket ping/pong behavior.

### Message types handled

- `price_change`
- `trade`
- `last_trade_price`
- `book`
- `best_bid_ask`
- `subscribed`
- `error`

`book` messages can represent:
- full snapshot (with `bids/asks` or `buys/sells`)
- deltas (with `changes` / `price_changes`)

## 7. Local orderbook mirror (L2) design

HELIOS keeps an in-memory `LocalOrderBook` per token ID.

### Why this exists

- low-latency reads for UI and autotrader
- decouple fast WS ingestion from API reads
- preserve top-of-book and small L2 views without querying external APIs

### Key properties

- bids/asks stored as price-string -> size maps (precision-safe lookup)
- deltas applied in-memory
- periodic commits build a cached snapshot
- L2 snapshots return:
  - `best_bid`, `best_ask`, `spread`, `mid`
  - top `bids[]` / `asks[]`
  - `bid_depth`, `ask_depth`
  - `staleness_ms`

## 8. Internal HELIOS API endpoints for market data

These are the endpoints you should reuse from bots/tools/UI inside this project.

### `GET /api/stations`
Includes:
- station ID
- timezone
- `market_unit` (`F` or `C`)

Useful to prevent unit mismatches (US vs London/Ankara/Paris markets).

### `GET /api/market/{station_id}`
Simple Gamma-based snapshot endpoint:
- event title/slug/date
- sorted bracket probabilities
- total volume

Use when you only need snapshot probabilities and volumes.

### `GET /api/realtime/market/{station_id}`
Low-latency endpoint backed by local WS mirror:
- returns L2 or top-of-book per token
- includes `bracket`, `outcome`, `staleness_ms`
- can warm up subscriptions on-demand if station has no active books

Use when you need live orderbook state.

### `GET /api/polymarket/{station_id}`
Unified endpoint (best all-around market context):
- Gamma prices + volumes + token IDs
- WS orderbook enrichment for YES and NO books
- market maturity / closed flags
- crowd wisdom sentiment and shifts
- WS connection diagnostics

Use this for dashboards and trading context.

## 9. Market maturity and target day rollover

HELIOS distinguishes between:
- market maturity (probability concentration)
- actual rollover condition (today market closed/inactive/endDate passed)

Important behavior:
- it does **not** roll to tomorrow only because a bracket is near 100%
- it rolls when the today event is effectively closed/inactive
- it prefers tomorrow only when tomorrow market is discoverable

This avoids empty subscriptions and premature target switching.

## 10. YES/NO semantics, labels, and units

### YES/NO semantics

Each bracket is a binary market with two tokens:
- `YES` token
- `NO` token

HELIOS often computes strategy context on the YES side, but stores or exposes both sides in the unified endpoint.

### Label normalization

HELIOS normalizes bracket labels to canonical Polymarket-style strings:
- ranges (`33-34°F`)
- tails (`28°F or below`, `39°F or higher`)

This is necessary because:
- labels can vary in formatting
- mojibake/encoding artifacts can appear
- replay/backtest/nowcast comparisons require exact label matching

### Unit handling (F vs C)

Not all Polymarket temperature markets use Fahrenheit.
HELIOS tracks market unit per station via config (`F` or `C`) and converts labels when needed for internal comparisons (especially nowcast/autotrader logic).

## 11. Quote selection policy (important for trading)

Mid price is not always a good "probability" proxy.

HELIOS uses `core/market_pricing.py` to choose a robust price from:
- best bid
- best ask
- midpoint
- optional reference price (e.g., Gamma snapshot)

Policy summary:
- tight spread -> midpoint is acceptable
- wide spread -> prefer executable best bid (conservative)
- if a reference sits inside the quote band, it may be preferred

This avoids overestimating tradable edge in wide books.

## 12. Autotrader integration (current state)

### What is implemented

- Autotrader consumes market context from local WS mirror and normalized labels.
- Paper broker simulates execution using orderbook-derived prices and execution models.
- REST endpoints expose autotrader status, positions, orders, fills, and diagnostics.

### What is not implemented yet

- Real order submission to Polymarket CLOB is not fully wired.
- `core/autotrader/execution_adapter_live.py` is a feature-flagged stub.
- `py_clob_client` dependency exists, but live adapter integration is pending.

## 13. Operational pitfalls and regression checklist

When touching Polymarket code, check these first:

1. Do not use Gamma as a replacement for live orderbook data.
2. Parse `outcomePrices` and `clobTokenIds` defensively (string JSON vs array).
3. Preserve YES index = `0`, NO index = `1` assumption for binary brackets.
4. Subscribe both YES and NO tokens if UI/trader uses cross-side implied quotes.
5. Keep label normalization in one place (`core/polymarket_labels.py`).
6. Respect market units (`F` vs `C`) when comparing against nowcast.
7. Do not roll target day early just because market looks mature.

## 14. Related docs

- `docs/market/POLYMARKET_TOKEN_IDS.md`
- `docs/system/ARCHITECTURE_EVOLUTION.md`
- `docs/trading/AUTOTRADER.md`
- `docs/market/rollouts/`
- `docs/weather/`
- `docs/trading/`

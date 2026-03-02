# Multi-Strategy Research - 2026-03-02

## Objective

Answer a practical question:

- if HELIOS should run 4 strategies in parallel, which 4 actually make sense
- which parts of the repo already support that
- what external evidence says about where durable edge usually comes from in prediction markets

## Executive summary

Running 4 strategies in parallel can make sense here, but not if they are just 4 threshold variants of the same idea.

The repo today does **not** have a real multi-strategy runtime. The current live path selects one candidate from two entry logics:

- `terminal_value`
- `tactical_reprice`

The old 4-strategy world (`conservative_edge`, `aggressive_edge`, `fade_event_qc`, `maker_passive`) survives mainly in backtest/legacy artifacts, not in the current production path.

My recommendation is:

1. do **not** run `conservative` + `aggressive` + current strategy + maker as four separate live bots
2. do run 4 **orthogonal** strategies with shared portfolio risk
3. make the first three directional/informational and the fourth passive/experimental

Recommended stack:

1. `winner_value_core`
2. `tactical_reprice`
3. `event_fade_qc`
4. `passive_maker`

## What the repo supports today

### Current runtime

The production autotrader path is:

- `core/autotrader.py`
- `web_server.py`
- `templates/polymarket.html`

Current behavior:

- fetch one signal payload per station
- evaluate `terminal_value`
- evaluate `tactical_reprice`
- choose one candidate
- size it with shared portfolio caps
- execute with `limit_fak` or `market_fok`

Important consequence:

- the runtime is single-candidate, not multi-strategy
- it cannot yet run 4 strategies independently with attribution, conflict resolution, or budget allocation

### Legacy / backtest policy families

`core/backtest/policy.py` still defines 4 strategy families:

- `create_conservative_policy()`
- `create_aggressive_policy()`
- `create_fade_policy()`
- `create_maker_passive_policy()`

These are useful as research references, but they are not wired into the current live runtime.

### Current execution limitation

The current runtime supports:

- `FAK`
- `FOK`

It does not implement a true resting maker path in the live runtime.

That matters because one of the 4 candidate strategy families, `maker_passive`, only makes sense if we can actually post non-marketable resting quotes and track queue/fill behavior correctly.

## Local evidence from this repo

### 1. The current runtime is not a strategy portfolio engine

`evaluate_trade_candidate()` returns a single candidate after comparing `terminal_value` and `tactical_reprice`.

So the architecture today is closer to:

- "pick one best trade now"

than to:

- "run several independent strategies with shared capital"

### 2. The old 4-strategy evidence stored locally is weak

Local SQLite data in `data/autotrader.db` contains:

- `conservative_edge`
- `aggressive_edge`
- `fade_event_qc`
- `maker_passive`

But the sample is tiny and not convincing:

- only `40` decisions per strategy in the local table
- rewards are near zero or negative
- `maker_passive` is worst in this local sample
- orders, fills and positions are effectively empty in the current DB snapshot

This is not strong enough evidence to resurrect the old 4-strategy setup as-is.

### 3. The current policy has a structural bias we should not replicate into 4 bots

The active selector currently over-favors cheap complement `NO` trades.

So "run 4 strategies" should not mean:

- current `terminal_value`
- current `tactical_reprice`
- current `terminal_value` but looser
- current `terminal_value` but tighter

That would mostly multiply the same thesis and the same model error.

## What external sources say

### A. Prediction markets are not perfectly calibrated, especially far from expiration

Page and Clemen (2013) find a favorite-longshot bias in prediction markets:

- high-likelihood events tend to be underpriced
- low-likelihood events tend to be overpriced
- calibration is better closer to expiration

Practical implication for HELIOS:

- blindly preferring cheap longshot complements is not a robust default
- a strategy that leans more toward strong favorites near resolution is more defensible than a strategy that loves far-tail `NO`

Source:

- https://academic.oup.com/ej/article/123/568/491/5079498

### B. Winning accounts are often either informed traders or liquidity providers

Bossaerts et al. (2023) show:

- price-sensitive traders add information to prices
- non-price-sensitive traders look more like noise
- price-sensitive traders earn higher profits on average

Practical implication:

- if you are seeing "many winning accounts", some are likely not just directional bettors
- they may be informed traders reacting faster/better
- or makers earning spread/rebates for supplying good liquidity

Source:

- https://www.sciencedirect.com/science/article/pii/S1386418123000794

### C. Polymarket explicitly rewards passive liquidity

Polymarket documentation says liquidity rewards are designed to:

- encourage liquidity through a market's lifecycle
- reward two-sided depth
- reward tighter spread around midpoint

That means some apparently "winning" accounts may be monetizing:

- spread capture
- maker rebates / liquidity rewards

not just directional prediction skill.

Source:

- https://docs.polymarket.com/market-makers/liquidity-rewards

### D. True maker execution is now supported by Polymarket, but not by this runtime

Polymarket docs and changelog show:

- `post-only` orders exist
- post-only requires `GTC` or `GTD`
- `postOnly=true` cannot be used with `FAK` or `FOK`
- Polymarket changelog lists `Post Only Orders` on `2026-01-06`

Practical implication:

- a real maker strategy is now externally feasible
- but HELIOS would need a new execution path to use it correctly

Sources:

- https://docs.polymarket.com/trading/orders/overview
- https://docs.polymarket.com/changelog

### E. Multiple simultaneous edges need joint sizing, not isolated Kelly bets

Jacot and Mochkovitch (2023) explicitly study Kelly/fractional Kelly for non-mutually-exclusive bets.

Practical implication:

- if we run 4 strategies in parallel, we must not let each strategy size independently as if it were alone
- we need a shared allocator with correlation awareness, caps, and fractional Kelly

Source:

- https://www.degruyterbrill.com/document/doi/10.1515/jqas-2020-0122/pdf

### F. Forecast sharpness only matters after calibration

Gneiting, Balabdaoui and Raftery (2007) argue for maximizing sharpness subject to calibration.

Practical implication:

- strategy proliferation without forecast calibration is a mistake
- first fix probability quality and dominance signals
- then build specialized strategies on top

Source:

- https://academic.oup.com/jrsssb/article-abstract/69/2/243/7109375

## What 4 strategies should mean here

The right design is 4 different sources of edge, not 4 parameter sets.

### 1. `winner_value_core`

Purpose:

- trade the model's main scenario

Rules:

- mostly `YES` on `forecast_winner`
- optionally adjacent bucket only if dominance is strong
- no cheap far-tail `NO` by default
- strongest near expiration or when top1-gap is large

Why it makes sense:

- aligns with favorite-longshot evidence
- lower thesis complexity
- easier to diagnose and calibrate

This should be the portfolio core.

### 2. `tactical_reprice`

Purpose:

- short-horizon trades around the next official observation

Rules:

- only `target_day = 0`
- require valid next-observation window
- require directional pressure and repricing quality
- fast exits and tight timeout

Why it makes sense:

- genuinely different holding period
- monetizes intraday repricing, not just terminal mispricing

This already exists conceptually in the current runtime.

### 3. `event_fade_qc`

Purpose:

- mean reversion / bad-print fade around QC or event-window regimes

Rules:

- only active when event window or QC regime is on
- small sizing
- fast mean-reversion exits
- no use outside special conditions

Why it makes sense:

- orthogonal to "winner value"
- specifically targets transient dislocations rather than terminal belief

This family existed in legacy/backtest and is worth reviving as a specialized bot.

### 4. `passive_maker`

Purpose:

- earn spread and, when eligible, maker incentives

Rules:

- only with true post-only `GTC/GTD`
- only where spread is wide enough and depth is attractive
- preferably quote both sides or quote complement books coherently
- use queue/cancel management, stale quote protection, and inventory controls

Why it makes sense:

- explains some consistently profitable external accounts
- orthogonal to directional forecasting edge

Why it is not ready today:

- current HELIOS runtime is still `FAK/FOK`
- no real resting order lifecycle or queue management
- no maker attribution in the live runtime

This should be treated as experimental until implemented properly.

## What I would not run

### 1. `conservative` and `aggressive` as separate live alpha streams

Those are mostly threshold variants of the same edge logic.

They do not give enough diversification to justify separate live slots.

### 2. A standalone "cheap `NO` complement hunter"

Given both local findings and external favorite-longshot evidence, this is too dangerous as a default strategy family.

### 3. A maker strategy on top of `FAK/FOK`

That is not maker trading.

It is taker logic pretending to be maker.

### 4. Independent bankrolls per strategy

If 4 strategies all size in isolation, total leverage will be wrong.

All 4 need one portfolio allocator.

## Architecture required to support 4 strategies correctly

### 1. Strategy portfolio model

Need a real `StrategySpec` layer:

- name
- eligibility function
- candidate generator
- sizing modifier
- exit logic
- attribution hooks

### 2. Multi-candidate evaluation

Current `evaluate_trade_candidate()` returns one candidate.

We would need something like:

- `evaluate_trade_candidates(payload, config, state) -> list[candidate]`

Then rank, net, or co-execute those candidates under shared constraints.

### 3. Shared allocator

The allocator must:

- see all candidates at once
- cap total station exposure
- cap total market exposure
- downweight correlated candidates
- apply fractional Kelly at portfolio level

### 4. Conflict policy

Need rules for:

- same label, same side from multiple strategies
- same label, opposite side from different strategies
- same station but different buckets

Without this, attribution becomes fake and risk accounting breaks.

### 5. Per-strategy telemetry

Need dashboard metrics by strategy:

- entries
- exits
- realized pnl
- unrealized pnl
- fill ratio
- slippage
- reward / score

Otherwise you cannot tell which of the 4 is carrying the portfolio.

## Recommended implementation order

### Phase 1

Build the safe 3-strategy directional stack:

1. `winner_value_core`
2. `tactical_reprice`
3. `event_fade_qc`

Skip maker for the first iteration.

### Phase 2

Add:

- true post-only execution
- live resting-order management
- queue-aware maker telemetry

Then introduce `passive_maker`.

### Phase 3

Re-run replay/backtest with:

- per-strategy attribution
- joint portfolio sizing
- station-level and market-level caps

## Bottom line

Yes, 4 strategies in parallel can make sense.

But the correct interpretation is:

- 4 different edge families

not:

- 4 different knobs around the same edge

For HELIOS, the best candidate set is:

1. `winner_value_core`
2. `tactical_reprice`
3. `event_fade_qc`
4. `passive_maker` after true post-only support exists

If the goal is to copy the economic behavior of visibly winning accounts, the repo should assume that at least some of those accounts are:

- informed/fast traders
- liquidity providers with maker economics

not just better versions of the current single-shot directional bot.

# 2026-03-03 — Sanity check para precios NO anomalos

## Contexto

El paper trader registro un trade en KLGA 32-33°F [NO]:
- Compra NO a 0.6c, venta a 99.5c → +$61.53
- Pero YES estaba a <1%, lo que implica NO deberia costar ~99c, no 0.6c
- El 0.6c es el precio YES, no un precio NO realista
- Causa probable: CLOB de Polymarket para NO tokens en buckets extremos es ultra-thin con asks a precios absurdos que serian arbiados por bots reales

## Archivos modificados

### `core/autotrader.py`

**`_build_candidate_from_trade()`** — Nuevo sanity check:
- Si `side == "NO"` y `market_price < expected_no * 0.50` (donde `expected_no = 1 - yes_entry`), rechaza con `no_price_anomaly`
- Solo aplica cuando `expected_no > 0.10` (evita false positives en buckets cercanos a 50/50)

**`apply_orderbook_guardrails()`** — Segundo sanity check:
- Si `side == "NO"` y `ask < fair_price * 0.50`, rechaza con `no_ask_price_anomaly`
- Doble validacion a nivel del orderbook live

### `core/papertrader.py`

- Logging de anomalias de precio en el journal JSONL
- Evento `price_anomaly` con detalles del candidato rechazado
- Dos puntos de log: despues de `_evaluate_strategy_candidate` y despues de `apply_orderbook_guardrails`

## Lo que NO se toca

- Logica de YES entries (no tiene este problema)
- Logica de exits (usa book real correctamente)
- Token IDs o data WS (el routing es correcto)
- Trades ya registrados (no se modifican retroactivamente)

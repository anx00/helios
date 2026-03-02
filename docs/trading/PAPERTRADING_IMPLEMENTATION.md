# Papertrading Shadow Module — Documentación de Implementación

**Fecha:** 2026-03-02
**Estado:** Implementado y testado (52 tests, 0 fallos)
**Brief de origen:** `PAPERTRADING_SHADOW_MODULE_BRIEF.md`

---

## 1. Resumen

Se construyó un módulo de papertrading live completamente separado del autotrader real. Opera en paralelo, usa las mismas señales y la misma lógica de decisión que el autotrader, pero simula la ejecución de órdenes caminando el orderbook L2 live nivel a nivel. No envía ninguna orden al exchange.

El objetivo es tener un entorno serio de shadow trading para validar la estrategia antes de reactivar el bot real — especialmente después de las 9 posiciones `resolved_open_loser` que dejaron el wallet en pérdidas.

---

## 2. Archivos Creados

| Archivo | Propósito |
|---|---|
| `market/paper_execution.py` | Motor de simulación de fills sobre L2 |
| `core/papertrader.py` | Runtime de shadow trading |
| `tests/test_paper_execution.py` | 23 tests del simulador de fills |
| `tests/test_papertrader.py` | 29 tests del runtime |

### Archivos modificados

| Archivo | Cambios |
|---|---|
| `web_server.py` | +1 import, +5 endpoints `/api/papertrader/*`, inyección de status en el payload principal |
| `templates/polymarket.html` | +CSS panel `.pt-*`, +div `#pt-panel`, +JS `renderPaperTrader()` |

---

## 3. Arquitectura

```
Payload de señal (/api/polymarket/{station_id})
          │
          ▼
┌──────────────────────────────────┐
│  Funciones de decisión (compartidas, module-level en autotrader.py)
│  evaluate_trade_candidate()
│  compute_trade_budget_usd()
│  apply_orderbook_guardrails()
│  apply_exit_guardrails()
└─────────────┬────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                     ▼
AutoTrader            PaperTrader
(live/paper real)     (shadow, sin órdenes reales)
    │                     │
    ▼                     ▼
PolymarketExecution   PaperExecutionEngine
(CLOB API real)       (camina L2 nivel a nivel)
```

### Decisión clave de diseño: reutilización sin refactor

Las funciones de decisión del autotrader (`evaluate_trade_candidate`, `compute_trade_budget_usd`, `apply_orderbook_guardrails`, `apply_exit_guardrails`) son todas **module-level** en `core/autotrader.py` — no son métodos de clase. El papertrader las importa directamente sin duplicar lógica y sin tocar el autotrader.

La única excepción es `_evaluate_exit_decision`, que sí es un método de clase porque lee `self.config`. Se replicó íntegramente en `PaperTrader` (~120 líneas), con la misma lógica exacta.

---

## 4. `market/paper_execution.py` — Motor de Fill

### Algoritmo de fill (limit_fak BUY)

```
asks ordenadas por precio ascendente
para cada nivel ask donde nivel_precio <= limit_price:
    consumido = min(size_restante, size_nivel)
    notional_acumulado += consumido * nivel_precio
    size_restante -= consumido
    si size_restante <= 0: break

si filled > 0:
    avg_price = notional / filled
    slippage = avg_price - best_ask   ← siempre >= 0 para compras
    si size_restante <= 0: FILLED
    si no: PARTIAL
si filled == 0: REJECTED
```

Para **market_fok**: se aplica el mismo algoritmo de book-walking, pero si el resultado es PARTIAL se convierte en REJECTED (semántica FOK — todo o nada).

Para **sells**: se camina el bid side descendentemente.

### Tipos de datos principales

```python
@dataclass
class PaperFill:
    filled_size: float
    avg_price: float
    notional_usd: float
    slippage_vs_best: float   # en precio (no puntos base)
    levels_consumed: int
    per_level: List[dict]     # detalle nivel a nivel
    status: str               # FILLED / PARTIAL / REJECTED

@dataclass
class PaperOrder:
    order_id: str             # UUID corto
    kind: str                 # "entry" / "exit"
    station_id, target_date, label, side, token_id: str
    order_type: str           # "limit_fak" / "market_fok"
    direction: str            # "buy" / "sell"
    requested_size: float
    limit_price: float
    status: str               # NEW/WORKING/PARTIAL/FILLED/CANCELED/REJECTED
    filled_size, avg_fill_price, fill_notional_usd: float
    slippage_vs_best: float
    book_snapshot_submit: dict   # top-10 del libro en submit
    book_snapshot_fill: dict     # top-10 del libro en fill
    fills: List[dict]            # detalle por nivel
```

### API pública

```python
engine = PaperExecutionEngine(latency_ms=50.0)

# Stateless (no rastrean la orden en memoria)
engine.execute_limit_fak_buy(token_id, limit_price, size, book_snapshot) → PaperFill
engine.execute_limit_fak_sell(token_id, min_price, size, book_snapshot) → PaperFill
engine.execute_market_fok_buy(token_id, max_price, size, book_snapshot) → PaperFill
engine.execute_market_fok_sell(token_id, min_price, size, book_snapshot) → PaperFill

# Stateful (rastrean lifecycle en _orders dict)
engine.place_order(kind, station_id, ..., book_snapshot) → PaperOrder
engine.get_order(order_id) → PaperOrder
engine.cancel_order(order_id) → PaperOrder
engine.get_open_orders() → List[PaperOrder]
```

---

## 5. `core/papertrader.py` — Runtime Shadow

### `PaperTraderConfig`

Dataclass con los mismos parámetros de riesgo y señal que `AutoTraderConfig`, más:

| Parámetro | Default | Descripción |
|---|---|---|
| `initial_bankroll_usd` | 100.0 | Capital virtual de partida |
| `state_path` | `data/papertrader_state.json` | Estado del portfolio |
| `runtime_path` | `data/papertrader_runtime.json` | Overrides de UI/API |
| `log_path` | `logs/papertrader_trades.jsonl` | Diario de eventos |
| `latency_ms` | 50.0 | Latencia simulada (cosmética) |

Expone `to_autotrader_config() → AutoTraderConfig` para pasar a las funciones compartidas sin modificarlas.

### Loop principal

```
run_once()
  ├─ load_paper_state()
  ├─ _reset_daily_counters_if_needed()
  └─ para cada station_id:
       run_station_once(station_id)
         ├─ resolve_station_market_target()        ← compartida
         ├─ _fetch_signal_payload()                ← misma lógica que autotrader
         ├─ _manage_open_positions()               ← exits primero
         │    └─ para cada posición OPEN:
         │         _evaluate_exit_decision()       ← lógica idéntica al autotrader
         │         execute_limit_fak_sell()        ← fill sobre libro L2
         │         _persist_exit()
         │         _append_journal()
         ├─ evaluate_trade_candidate()             ← compartida
         ├─ _get_live_book_from_payload()          ← TopOfBook desde brackets WS
         ├─ compute_trade_budget_usd()             ← compartida (Kelly + guardrails de riesgo)
         ├─ apply_orderbook_guardrails()           ← compartida
         ├─ execute_limit_fak_buy() / market_fok   ← fill sobre libro L2
         ├─ _persist_trade()
         └─ _append_journal()
  └─ _mark_to_market_all()                        ← MTM con libros live
  └─ _save_state()
```

### Estado (`data/papertrader_state.json`)

```json
{
  "paper_equity_usd": 97.45,
  "initial_bankroll_usd": 100.0,
  "positions": {
    "KLGA|2026-03-05|33-34°F|YES": {
      "status": "OPEN",
      "mode": "paper_shadow",
      "strategy": "terminal_value",
      "entry_price": 0.36,
      "shares": 25.0,
      "shares_open": 25.0,
      "cost_basis_open_usd": 9.0,
      "current_value_usd": 10.5,
      "unrealized_pnl_usd": 1.5,
      "realized_pnl_usd": 0.0,
      "fills": []
    }
  },
  "trades_today": 1,
  "realized_pnl_today_usd": -0.50,
  "open_risk_usd": 10.5,
  "total_fills": 3,
  "total_partial_fills": 1,
  "total_rejections": 1,
  "total_slippage_usd": 0.025,
  "exits_by_reason": { "take_profit": 2, "stop_loss": 1 },
  "total_realized_pnl_usd": -0.50
}
```

### Diario (`logs/papertrader_trades.jsonl`)

Un evento por línea. Tipos de evento:

| `event` | Cuándo |
|---|---|
| `entry_fill` | Entrada llenada (FILLED o PARTIAL) |
| `entry_rejected` | No fill (REJECTED) |
| `exit_fill` | Salida ejecutada |
| `exit_rejected` | Intento de salida sin fill |
| `guardrail_skip` | Guardrail bloqueó la entrada |
| `budget_skip` | Budget demasiado pequeño |
| `fetch_error` | Error al obtener el payload |
| `tick` | Tick sin actividad relevante |

Cada evento incluye: `ts_utc`, `station_id`, `position_key`, `signal_snapshot`, `book_snapshot`, `fill`, `reason`.

### Mark to market

En cada `run_once()`, después de los ticks por estación, se actualiza `current_price`, `current_value_usd` y `unrealized_pnl_usd` de todas las posiciones abiertas usando el libro live del último payload cacheado. Si no hay libro disponible, el valor no se modifica.

### Lógica de salida (replicada del autotrader)

Mismos triggers, mismos umbrales:

| Trigger | Estrategia | Condición |
|---|---|---|
| `take_profit` | ambas | `bid >= fair - buffer` y `bid >= entry + tp_pts` |
| `stop_loss` | terminal | `bid <= entry - sl_pts` y `edge <= 0` |
| `stop_loss` | tactical | `bid <= entry - sl_pts` y `held >= 2 min` |
| `model_broke` | terminal | `fair < entry - buffer` |
| `timeout_exit` | tactical | `held >= tactical_timeout_minutes` |
| `post_official_reprice` | tactical | siguiente METAR ha pasado >5 min |
| `signal_flip` | tactical | política ya no permite |
| `next_official_against_position` | tactical | dirección siguiente METAR opuesta |

---

## 6. API Web

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/papertrader/status` | Snapshot completo del estado |
| `POST` | `/api/papertrader/config` | Actualizar stations, fecha, cap, bankroll |
| `POST` | `/api/papertrader/run-once` | Ejecutar un tick manual |
| `POST` | `/api/papertrader/start` | Arrancar el loop continuo |
| `POST` | `/api/papertrader/stop` | Parar el loop |

El status también se inyecta en el payload de `/api/polymarket/{station_id}` bajo la clave `papertrader`, junto al `autotrader`. Así el refresco de 5s de la UI obtiene todo en una sola petición.

### Payload de `/api/papertrader/config`

```json
{
  "station_ids": ["KLGA", "KORD"],
  "target_date": "2026-03-05",
  "max_total_exposure_usd": 25.0,
  "initial_bankroll_usd": 100.0
}
```

### Payload de respuesta de `/api/papertrader/status`

```json
{
  "ok": true,
  "status": {
    "running": false,
    "mode": "paper_shadow",
    "paper_equity_usd": 97.45,
    "initial_bankroll_usd": 100.0,
    "open_positions_count": 2,
    "closed_positions_count": 3,
    "open_risk_usd": 12.30,
    "realized_pnl_usd": -2.55,
    "unrealized_pnl_usd": 1.20,
    "realized_pnl_today_usd": -0.50,
    "total_entries": 5,
    "total_exits": 3,
    "total_fills": 5,
    "total_partial_fills": 1,
    "total_rejections": 2,
    "fill_ratio": 0.71,
    "hit_rate": 0.33,
    "avg_slippage_bps": 1.4,
    "total_slippage_usd": 0.035,
    "exits_by_reason": { "take_profit": 2, "stop_loss": 1 },
    "open_positions": [...],
    "recent_events": [...],
    "available_stations": [...],
    "last_stop_reason": null
  }
}
```

### Persistencia de configuración

Los overrides de UI se guardan en `data/papertrader_runtime.json` entre reinicios del servidor. Al arrancar, se cargan y se aplican sobre la configuración base. Los overrides que persisten: `station_ids`, `target_date`, `max_total_exposure_usd`, `initial_bankroll_usd`.

---

## 7. UI — Panel SHADOW

El panel se renderiza en `templates/polymarket.html` debajo del panel del autotrader, dentro de `#pm-trading-section`.

**Diferenciación visual:** acento morado `#8B5CF6` / `#A78BFA` en lugar del verde del autotrader, badge "SHADOW" junto al título.

### Secciones del panel

1. **Cabecera** — título "Paper Trader" + badge SHADOW + estado (`running dot` animado si está corriendo) + botones Save / Run once / Start / Stop

2. **Config** — pills de stations con checkbox, date picker para la fecha de mercado, cap de exposición en USD, bankroll inicial. Se guarda con "Save setup" o automáticamente al pulsar "Start".

3. **Grid de métricas (12 celdas)**

   | Métrica | Descripción |
   |---|---|
   | Paper equity | Capital virtual actual |
   | Total PnL | Realizado + no realizado + % sobre bankroll |
   | Realized PnL | PnL de posiciones cerradas |
   | Unrealized PnL | MTM de posiciones abiertas |
   | Open risk | Exposición abierta / cap |
   | Open positions | Abiertas + cerradas |
   | Today | Entradas hoy + PnL del día |
   | All time | Total entradas + salidas |
   | Fill ratio | Fills / intentos |
   | Hit rate | % posiciones cerradas con PnL > 0 |
   | Avg slippage | Deslizamiento medio en puntos base |
   | Exit breakdown | Conteo por razón de salida |

4. **Tabla de posiciones abiertas** — station, label, side, estrategia, precio de entrada, shares, coste, PnL no realizado, antigüedad

5. **Log de actividad reciente** — últimos 8 eventos del diario JSONL: tipo de evento, posición, fill size/precio, slippage, razón, timestamp

---

## 8. Tests

### `tests/test_paper_execution.py` — 23 tests

| Clase | Tests |
|---|---|
| `TestLimitFakBuy` | Camina múltiples niveles, fill parcial por falta de depth, rechazado si limit < best ask, fill exacto un nivel, libro vacío |
| `TestLimitFakSell` | Camina bids, fill parcial, rechazado si min > best bid |
| `TestMarketFokBuy` | Full fill con depth suficiente, rechazado si no hay suficiente depth |
| `TestMarketFokSell` | Full fill, rechazado |
| `TestOrderLifecycle` | FILLED, REJECTED, PARTIAL, tracking en engine, sell direction |
| `TestSlippage` | Zero slippage nivel único, slippage positivo multinivel, slippage en sell |
| `TestBookSnapshot` | `compact_snapshot`, `PaperOrder.to_dict()`, `PaperFill.to_dict()` |

### `tests/test_papertrader.py` — 29 tests

| Clase | Tests |
|---|---|
| `TestStatePersistence` | Estado default correcto, guardar/recargar, archivo faltante, JSON corrupto |
| `TestPersistTrade` | Equity decrementa, posición creada correctamente, contador trades, open_risk, slippage trackeado |
| `TestPersistExit` | PnL calculado, posición se cierra, equity aumenta, razón trackeada, salida parcial, ciclo entry→exit completo |
| `TestDailyReset` | Contadores se resetean en día nuevo, no se resetean en el mismo día |
| `TestRiskLimits` | Cap de exposición bloquea entradas, límite de pérdida diaria bloquea, max trades bloquea |
| `TestJournalEvents` | Evento escrito en JSONL, múltiples eventos se acumulan en orden |
| `TestMarkToMarket` | MTM actualiza current_price/value/unrealized_pnl desde payload, sin payload no cambia |
| `TestStatusSnapshot` | Estado inicial correcto, con posición abierta, fill ratio, hit rate, adaptador de config |

---

## 9. Separación de responsabilidades

Lo que el papertrader **comparte** con el autotrader:

- Funciones de selección de candidato (`evaluate_trade_candidate`)
- Sizing Kelly (`compute_trade_budget_usd`)
- Guardrails de entrada (`apply_orderbook_guardrails`)
- Guardrails de salida (`apply_exit_guardrails`)
- Resolución de fecha de mercado (`resolve_station_market_target`)
- Normalización de labels (`normalize_label`)

Lo que el papertrader **tiene propio**:

- Estado (`data/papertrader_state.json`)
- Logs (`logs/papertrader_trades.jsonl`)
- Config y runtime (`data/papertrader_runtime.json`)
- Motor de ejecución (`PaperExecutionEngine`)
- Endpoints `/api/papertrader/*`
- Panel UI con CSS prefijo `.pt-`
- Lógica de mark to market
- Contadores de métricas (fill ratio, hit rate, slippage)

---

## 10. Cómo usarlo

### Arranque manual (un tick)

```bash
# En la UI: panel SHADOW → Run once
# O via curl:
curl -X POST http://localhost:8000/api/papertrader/run-once
```

### Loop continuo

```bash
# En la UI: configurar stations + fecha → Start loop
# O via curl:
curl -X POST http://localhost:8000/api/papertrader/config \
  -H 'Content-Type: application/json' \
  -d '{"station_ids":["KLGA","KORD"],"target_date":"2026-03-05","max_total_exposure_usd":25,"initial_bankroll_usd":100}'

curl -X POST http://localhost:8000/api/papertrader/start
```

### Inspeccionar estado

```bash
curl http://localhost:8000/api/papertrader/status | python -m json.tool
```

### Ver el diario

```bash
tail -f logs/papertrader_trades.jsonl | python -m json.tool
```

### Variables de entorno opcionales

```bash
HELIOS_PAPERTRADER_ENABLED=true
HELIOS_PAPERTRADER_STATIONS=KLGA,KORD
HELIOS_PAPERTRADER_BANKROLL_USD=100
```

---

## 11. Qué falta (siguientes fases)

Según el brief original, queda pendiente para fases futuras:

- **Settlement automático:** cuando un mercado madura, marcar posiciones paper como `resolved` y calcular `expected_settlement_pnl`
- **Export/auditoría de sesiones:** exportar sesiones completas para comparar paper vs real vs settlement
- **`missed_alpha_usd`:** diferencia entre salida paper y settlement esperado
- **Comparativa paper vs autotrader real:** tabla lado a lado con las mismas estaciones/fechas

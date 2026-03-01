# Brief Para Claude: Modulo De Live Papertrading

Este documento es el contexto y la guia de implementacion para construir un modulo aparte de papertrading live en HELIOS.

No es un backtest.
No es un replay offline.
No debe reutilizar el `mode="paper"` actual del autotrader como solucion final.

La idea es tener un runtime paralelo, de shadow trading, que use la misma inteligencia de decision que el autotrader actual, pero ejecute ordenes simuladas contra el libro live de Polymarket usando L2 / WebSocket.

## 1. Contexto Real Del Proyecto

El autotrader actual ya tiene bastante cirugia hecha:

- lifecycle de ordenes pendientes arreglado
- `policy_flip` quitado como hard exit para terminal
- take profit endurecido para no salir demasiado pronto
- cooldown de errores de balance/allowance
- ownership live reparado para cuentas dedicadas al bot
- logging endurecido para no corromper `jsonl`
- cierre `CLOSED_UNRECONCILED` cuando una posicion live desaparece sin fill confirmado

El bot real sigue apagado.

Contexto operativo importante:

- `core/autotrader.py` es el runtime principal actual
- `market/polymarket_execution.py` es la capa de ejecucion live
- `market/polymarket_ws.py` mantiene el orderbook live en memoria
- `web_server.py` expone `/api/autotrader/*` y `/api/polymarket/{station_id}`

Hallazgo clave a fecha de hoy:

- Las `9` posiciones abiertas actuales del wallet salen como `resolved_open_loser` en el informe [AUTOTRADER_OPEN_POSITION_AUDIT_2026-03-01.md](/c:/Users/anxoo/Magic/Polymarket/helios/forense/AUTOTRADER_OPEN_POSITION_AUDIT_2026-03-01.md).
- Eso significa que el problema gordo no es solo reconciliacion; el sistema necesita un entorno serio de papertrading live antes de volver a encender el bot real.

## 2. Que Problema Hay Con El "Paper" Actual

Hoy el autotrader soporta `mode="paper"`, pero eso solo hace:

- `paper_fill`
- `paper_exit`
- `paper_reduce_only_exit`

sin simulacion realista del libro.

En la practica:

- decide la entrada
- aplica guardrails
- y persiste el trade como si hubiera llenado directamente

Eso sirve para probar logica de decision.
No sirve para medir ejecucion.

Quiero un modulo aparte que:

- use senales live reales
- use libro live real
- simule fills de forma razonablemente fiel
- no mande ordenes al exchange
- deje metricas y estado propios

## 3. Restricciones Y Filosofia De Implementacion

### 3.1 Debe ser un modulo aparte

No quiero que conviertas `core/autotrader.py` en una mezcla rara de:

- live real
- paper ingenuo
- paper con orderbook

Haz un runtime separado.

La recomendacion es algo de este estilo:

- `core/papertrader.py`
- `market/paper_execution.py`

Opcionalmente:

- `data/papertrader_state.json`
- `data/papertrader_runtime.json`
- `logs/papertrader_trades.jsonl`

### 3.2 Debe reutilizar la inteligencia de decision actual

No reescribas la estrategia desde cero.

Reutiliza tanto como puedas de:

- `evaluate_trade_candidate()`
- `compute_trade_budget_usd()`
- `apply_orderbook_guardrails()`
- `_evaluate_exit_decision()`
- guardrails de riesgo ya existentes

Si hace falta, extrae funciones compartidas a helpers neutrales en vez de duplicar logica.

### 3.3 No debe depender del replay para funcionar

El replay existe y es util, pero este modulo es para papertrading live.

Debe funcionar con:

- senal live
- WS live
- dashboard/trading payload live

### 3.4 Debe usar L2 live, pero sin reventar storage

Quiero L2 para simular fills.

No quiero volver a grabar full L2 continuo para todo.

Persistencia recomendada:

- guardar decision snapshot cuando se intenta una entrada o salida
- guardar excerpt del libro top `N` solo en:
  - submit
  - partial fill
  - fill
  - cancel / expire
- guardar `best_bid`, `best_ask`, `mid`, `spread`, `depth`, `top_n`
- no almacenar el stream completo de L2 por defecto

## 4. Fuentes Que Debes Reutilizar

Lee y reutiliza estas piezas antes de programar:

- [AUTOTRADER_CURRENT_BOTS.md](/c:/Users/anxoo/Magic/Polymarket/helios/docs/trading/AUTOTRADER_CURRENT_BOTS.md)
- [AUTOTRADER_FORENSICS_2026-03-01.md](/c:/Users/anxoo/Magic/Polymarket/helios/docs/trading/AUTOTRADER_FORENSICS_2026-03-01.md)
- [AUTOTRADER_DATA_FINDINGS_2026-03-01.md](/c:/Users/anxoo/Magic/Polymarket/helios/docs/trading/AUTOTRADER_DATA_FINDINGS_2026-03-01.md)
- [STORAGE_REPLAY_IMPLEMENTATION.md](/c:/Users/anxoo/Magic/Polymarket/helios/docs/trading/STORAGE_REPLAY_IMPLEMENTATION.md)
- [PREDICTION_SYSTEM.md](/c:/Users/anxoo/Magic/Polymarket/helios/docs/weather/PREDICTION_SYSTEM.md)
- [core/autotrader.py](/c:/Users/anxoo/Magic/Polymarket/helios/core/autotrader.py)
- [market/polymarket_execution.py](/c:/Users/anxoo/Magic/Polymarket/helios/market/polymarket_execution.py)
- [market/polymarket_ws.py](/c:/Users/anxoo/Magic/Polymarket/helios/market/polymarket_ws.py)
- [web_server.py](/c:/Users/anxoo/Magic/Polymarket/helios/web_server.py)

Puntos concretos ya disponibles en el repo:

- `market.polymarket_ws.get_ws_client()`
- `client.state.orderbooks[token_id]`
- `LocalOrderBook.get_l2_snapshot(top_n=...)`
- `/api/polymarket/{station_id}` con `target_date`

## 5. Objetivo Exacto Del Modulo

Construir un runtime de papertrading live que:

1. seleccione estaciones/fecha igual que el autotrader
2. reciba payloads de trading iguales o equivalentes
3. genere entradas y salidas con la misma logica de decision
4. simule la ejecucion sobre libro live WS
5. mantenga portfolio propio, sin tocar el wallet real
6. persista estado y journal de decisiones/fills
7. exponga API/UI propias para control y observabilidad

## 6. Arquitectura Recomendada

### 6.1 `core/papertrader.py`

Responsabilidad:

- orquestar el loop de shadow trading
- cargar config/runtime
- pedir signal payload
- mantener state de posiciones paper
- coordinar entries/exits
- reconciliar ordenes paper working / pending

Debe parecerse al autotrader, pero sin llamadas reales de compra/venta.

### 6.2 `market/paper_execution.py`

Responsabilidad:

- simular ordenes sobre L2
- soportar lifecycle de ordenes paper
- calcular fills, partial fills, expiraciones y cancels

Mi recomendacion es tener una interfaz tipo:

- `place_limit_buy(...)`
- `place_limit_sell(...)`
- `place_market_buy(...)`
- `place_market_sell(...)`
- `get_order(order_id)`
- `poll_open_orders(...)`

pero todo en paper.

### 6.3 Estado separado

No mezclar con `autotrader_state.json`.

Usa ficheros propios, por ejemplo:

- `data/papertrader_state.json`
- `data/papertrader_runtime.json`
- `logs/papertrader_trades.jsonl`

### 6.4 Endpoints separados

No reciclar `/api/autotrader/*`.

Crear endpoints propios, por ejemplo:

- `GET /api/papertrader/status`
- `POST /api/papertrader/config`
- `POST /api/papertrader/run-once`
- `POST /api/papertrader/start`
- `POST /api/papertrader/stop`

Si haces UI, que sea otra tarjeta o modulo separado en `templates/polymarket.html`.

## 7. Como Debe Simular La Ejecucion

## 7.1 Modos Minimos A Soportar

Soporta como minimo los mismos modos que hoy usa el autotrader real:

- `limit_fak`
- `market_fok`

No hace falta maker/post-only v1.

### 7.2 Semantica deseada

#### `limit_fak`

Para BUY:

- consumir asks desde el mejor ask hacia arriba
- solo hasta `limit_price`
- si no hay suficiente size hasta ese precio:
  - fill parcial o cero, segun el comportamiento que ya uses en el runtime paper

Para SELL:

- consumir bids desde el mejor bid hacia abajo
- solo hasta `min_price`

#### `market_fok`

- solo fill si existe liquidez suficiente para completar el size dentro del constraint de precio
- si no, no fill

### 7.3 Latencia y simplificaciones aceptables

Como estos mercados de weather suelen moverse despacio, acepto una v1 sin modelo complejo de queue position.

Simplificaciones aceptables:

- no modelar queue priority exacta
- no modelar competencia de otros fills entre ticks
- no modelar maker queue

Pero no acepto:

- fill instantaneo ficticio sin mirar depth
- salida/entrada al mid sin consumir libro

Si introduces una latencia fija pequena o configurable, mejor.

## 8. Estado De Ordenes Paper

Quiero que el papertrader tenga una maquina de estados seria.

Estados minimos:

- `NEW`
- `WORKING`
- `PARTIAL`
- `FILLED`
- `CANCELED`
- `EXPIRED`
- `REJECTED`

Cada orden paper debe guardar como minimo:

- `order_id`
- `kind`: `entry` / `exit`
- `station_id`
- `target_date`
- `label`
- `side`
- `token_id`
- `submitted_at_utc`
- `status`
- `requested_size`
- `filled_size`
- `remaining_size`
- `avg_fill_price`
- `limit_price` o `min_price` o `max_price`
- `book_snapshot_submit`
- `book_snapshot_last`

## 9. Portfolio Paper

Necesito un portfolio paper completo, separado del real.

Cada posicion paper debe tener:

- `position_key`
- `station_id`
- `target_date`
- `label`
- `side`
- `token_id`
- `strategy`
- `entry_price`
- `avg_price`
- `shares`
- `shares_open`
- `cost_basis_open_usd`
- `current_value_usd`
- `realized_pnl_usd`
- `unrealized_pnl_usd`
- `opened_at_utc`
- `closed_at_utc`
- `last_exit_reason`
- `fills`

## 10. Mark To Market Y Resolucion

El papertrader debe marcar posiciones de dos formas:

### 10.1 Intradia

Usando libro live:

- `best_bid`
- `best_ask`
- `mid`

Segun el lado de la posicion.

### 10.2 Settlement / resolucion

Cuando un mercado ya este maduro o practicamente resuelto segun HELIOS:

- usar `market_status.is_mature`
- usar `market_status.winning_outcome`
- si aplica, reflejar valor de settlement esperado

No cierres automaticamente por esto en v1 si no quieres.
Pero si debes:

- marcar posiciones paper como `resolved`
- exponer el winner esperado
- mostrar `expected_settlement_pnl`

## 11. Persistencia Minima Que Si Quiero

Quiero guardar suficiente informacion para aprender y depurar sin disparar el disco.

Persistir en `jsonl` por evento:

- decisiones de entrada
- submit de orden
- partial fill
- fill
- cancel / expiry
- decisiones de exit
- cierre de posicion

Cada evento debe incluir:

- snapshot de signal
- snapshot resumido del libro
- motivo de decision
- contexto de exit si aplica

No guardar:

- stream continuo completo de L2
- payloads gigantes duplicados si no cambian

## 12. Observabilidad Que Quiero Ver

En el status/API quiero ver:

- paper equity
- open risk
- posiciones abiertas
- pnl realizado
- pnl no realizado
- hit rate
- numero de fills
- partial fills
- fill ratio
- slippage vs best quote
- ordenes working
- ordenes expiradas
- exits por razon

Si puedes, anade tambien:

- `missed_alpha_usd`
- diferencia entre salida paper y settlement esperado

## 13. Lo Que No Quiero Que Hagas

- no mezclarlo con replay/backtest actual
- no convertir el autotrader real en un monstruo con mas modos
- no meter IA/bandits/auto-learning todavia
- no almacenar full L2 continuo
- no tocar maker/post-only en esta fase
- no cerrar el wallet real

## 14. Plan De Implementacion Recomendado

### Fase 1

- crear `paper_execution.py`
- crear `papertrader.py`
- portfolio/state separados
- entry/exit paper usando L2 actual

### Fase 2

- endpoints `/api/papertrader/*`
- status snapshot serio
- logs `papertrader_trades.jsonl`

### Fase 3

- UI minima en `polymarket.html`
- tabla de posiciones paper
- ordenes working
- metricas basicas

### Fase 4

- comparativa con settlement esperado
- export / auditoria de sesiones paper

## 15. Criterios De Aceptacion

Considera el trabajo bien hecho si al final existe todo esto:

1. un runtime separado de papertrading live
2. usa las senales actuales del autotrader
3. usa el orderbook WS actual para fills simulados
4. no manda ninguna orden real
5. persiste estado y journal propios
6. expone endpoints y snapshot propios
7. soporta `limit_fak` y `market_fok`
8. permite correr varias estaciones en shadow
9. no depende del replay para operar
10. tiene tests serios de fills, partials, no fills y exits

## 16. Tests Que Espero

Como minimo:

- buy `limit_fak` llena varias capas del ask
- buy `limit_fak` parcial por falta de depth
- sell `limit_fak` llena varias capas del bid
- `market_fok` no llena si no hay size suficiente
- lifecycle de orden `WORKING -> PARTIAL -> FILLED`
- posicion paper mark-to-market correcto
- bloqueos de riesgo iguales al autotrader
- endpoint status devuelve metricas coherentes

## 17. Mi Recomendacion Tecnica

Si tienes que elegir entre:

- copiar mucha logica del autotrader
- o extraer una capa compartida de decision

prefiero que extraigas una capa compartida pequena y limpia.

El papertrader debe compartir:

- seleccion de candidato
- sizing
- guardrails
- exits

pero no compartir:

- estado
- logs
- ejecucion
- runtime control

## 18. Resumen Ejecutivo Para Claude

Haz un modulo aparte de live papertrading para HELIOS.

Debe usar:

- mismas senales del autotrader
- libro live WS de Polymarket
- simulacion seria de fills sobre depth real

Debe evitar:

- tocar el wallet real
- depender del replay
- guardar L2 completo continuamente

Debe dejar:

- estado propio
- logs propios
- API propia
- tests propios

Y debe implementarse con la idea de que despues podamos comparar:

- papertrader live
- autotrader real
- settlement final

sin volver a correr riesgo real mientras iteramos estrategia.

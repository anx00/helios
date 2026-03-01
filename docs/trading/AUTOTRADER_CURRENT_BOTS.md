# Current Autotrading Bots

Documento canonico para entender que bots de autotrading existen hoy en este repo, cuales estan realmente en uso y cuales piezas son legacy o experimentales.

Objetivo:
- separar el runtime actual de la arquitectura historica
- explicar que hace cada "bot" o estrategia
- aclarar como ejecuta ordenes, como gestiona riesgo y donde guarda estado

## 1. Resumen rapido

Hoy hay un runtime principal de autotrading:

- `core/autotrader.py`
  Es el orquestador actual que usa la UI de Polymarket y los endpoints `/api/autotrader/*`.

Dentro de ese runtime hay dos estrategias de entrada:

- `terminal_value`
  Busca valor terminal en el bucket que HELIOS considera infrapreciado.

- `tactical_reprice`
  Busca repricing tactico antes de la siguiente observacion oficial, solo para `target_day = 0`.

Ademas, el runtime puede gestionar tres tipos de posicion:

- `terminal_value`
  Posiciones abiertas por la estrategia terminal.

- `tactical_reprice`
  Posiciones abiertas por la estrategia tactica.

- `managed_live`
  Posiciones detectadas en la cartera live de Polymarket e importadas como exposicion del bot cuando no vienen etiquetadas de otra forma.

- `external_live`
  Posiciones detectadas en la cartera live de Polymarket pero marcadas explicitamente como `managed_by_bot = False`. Se trackean para riesgo, pero no se consideran gestionadas por el bot.

## 2. Que es "el bot actual"

El bot actual es la combinacion de estas piezas:

- orquestador: `core/autotrader.py`
- ejecucion CLOB: `market/polymarket_execution.py`
- UI principal: `templates/polymarket.html`
- endpoints web: `web_server.py`

El flujo real es:

1. Carga config desde `.env` y/o overrides runtime.
2. Resuelve estacion y fecha de mercado.
3. Pide la senal a `get_polymarket_dashboard_data()` o al API interno.
4. Evalua candidatos `terminal_value` y `tactical_reprice`.
5. Aplica guardrails de riesgo y libro.
6. Decide si compra, mantiene o sale.
7. Persiste estado local y logs.
8. Sincroniza la cartera live para no ignorar exposicion externa.

## 3. Estrategias actuales

### 3.1 `terminal_value`

Fuente:
- `evaluate_trade_candidate()` en `core/autotrader.py`

Idea:
- tomar el bucket que el modelo ve mas infravalorado contra el precio de mercado
- priorizar el escenario principal (`forecast_winner`) y castigar buckets secundarios YES si no tienen edge extra

Uso tipico:
- entradas de valor cuando el fair del modelo supera claramente el precio de mercado

Salida:
- take profit
- stop loss
- `model_broke`
- `policy_flip`

### 3.2 `tactical_reprice`

Fuente:
- `evaluate_trade_candidate()` y `_evaluate_exit_decision()` en `core/autotrader.py`

Idea:
- capturar repricing a corto plazo alrededor de la siguiente observacion oficial
- exige ventana temporal valida, presion direccional y suficiente calidad del contexto tactico

Uso tipico:
- entradas intradia de corto plazo para `target_day = 0`

Salida:
- take profit
- timeout
- post official reprice
- signal flip
- next official against position
- stop loss

## 4. Como ejecuta ordenes de verdad

Actualmente existen dos modos de ejecucion configurables:

- `limit_fak`
- `market_fok`

Default en codigo:
- `order_mode = "limit_fak"` en `core/autotrader.py`

### 4.1 `limit_fak`

No es una orden limite resting en el libro.

En la practica hace esto:
- calcula un precio limite maximo para entrar o minimo para salir
- envia una orden limit `FAK` con `post_only = False`
- ejecuta de inmediato lo que pueda hasta ese precio
- cancela lo no llenado

Consecuencia:
- se comporta como taker con control de precio
- no deja una orden visible esperando en el book

### 4.2 `market_fok`

Usa orden de mercado con `FOK`.

Consecuencia:
- intenta llenarse completo de inmediato
- si no puede, no deberia dejar resto pendiente

### 4.3 Que no existe hoy

No existe un modo maker/post-only real en el runtime actual.

Si alguien espera ver ordenes limite resting publicadas, ese comportamiento no esta implementado en el bot actual.

## 5. Riesgo y sizing actuales

El sizing actual esta en `compute_trade_budget_usd()` en `core/autotrader.py`.

Hoy el presupuesto por trade sale de:

- Kelly fraccional:
  - `bankroll_usd`
  - `fractional_kelly`
  - edge entre `fair_price` y `market_price`

- caps globales:
  - `max_total_exposure_usd`
  - `max_station_exposure_usd`
  - `daily_loss_limit_usd`
  - `max_open_positions`
  - `max_station_positions`
  - `max_trades_per_day`
  - `cooldown_seconds`

- balance disponible:
  - `free_collateral_usd` live

### 5.1 Modo actual de "auto sizing"

La UI ya no pide `max_trade_usd`.

El runtime sigue soportando ese campo por compatibilidad, pero si vale `0` se interpreta como auto:

- no hay cap fijo por trade
- el riesgo disponible se reparte dentro del `portfolio cap`
- se usa tambien un reparto por slots restantes:
  - trades restantes del dia
  - posiciones abiertas restantes
  - slots restantes por estacion si el limite esta activado

Esto evita que el primer trade consuma demasiado riesgo solo porque Kelly salga grande.

## 6. Sincronizacion con la cartera live

Pieza clave:
- `_sync_live_positions_into_state()` en `core/autotrader.py`

Que hace:
- importa posiciones live de Polymarket al estado local
- las marca como `source = polymarket_live` o `autotrader_live`
- por defecto las trata como `managed_by_bot = True`
- solo quedan como tracking-only si llegan marcadas explicitamente con `managed_by_bot = False`

Consecuencia operativa:
- el bot si cuenta esas posiciones para riesgo total y exposicion
- y tambien las puede recortar o cerrar automaticamente, salvo que esten marcadas explicitamente como tracking-only

Esto es importante porque en snapshots reales puedes ver:
- `managed_open_positions_count = 0`
- `tracking_only_positions_count > 0`

Eso significa que el bot esta viendo riesgo real en la cuenta, pero no considera esas posiciones como suyas.

## 7. Estado, logs y runtime overrides

Ficheros principales:

- estado local: `data/autotrader_state.json`
- logs de eventos: `logs/autotrader_trades.jsonl`
- kill switch: `data/AUTOTRADE_STOP`
- runtime overrides UI/API: `data/autotrader_runtime.json`

El snapshot de control sale de:
- `get_status_snapshot()` en `core/autotrader.py`

Y expone, entre otras cosas:
- modo
- `order_mode`
- riesgo
- wallet live
- resumen de portfolio
- posiciones abiertas
- eventos recientes

## 8. Endpoints y control operativo

Endpoints principales en `web_server.py`:

- `GET /api/autotrader/status`
- `POST /api/autotrader/config`
- `POST /api/autotrader/run-once`
- `POST /api/autotrader/start`
- `POST /api/autotrader/stop`

La UI visible para este runtime es la tarjeta de autotrader dentro de:

- `templates/polymarket.html`

No es la UI antigua `templates/autotrader.html` de los documentos legacy.

## 9. Que es legacy o experimental, no el runtime principal

Todavia existen piezas de una arquitectura mas amplia de autotrading/learning:

- base SQLite: `data/autotrader.db`
- tablas:
  - `bandit_state`
  - `autotrader_decisions`
  - `autotrader_rewards`
  - `autotrader_orders`
  - `autotrader_fills`
  - `autotrader_positions`
  - `learning_runs`
  - `model_registry`

Esto indica que en el repo sigue habiendo infraestructura para:

- seleccion adaptativa de estrategias
- reward tracking
- aprendizaje offline
- comparacion de modelos/estrategias

Pero eso no significa automaticamente que ese loop este gobernando el runtime actual de `core/autotrader.py`.

La confusion principal historica del repo es esta:

- documentos legacy hablan de `linucb`, `maker_passive`, `service.py`, `risk.py`, `storage.py`
- el runtime actual visible y operativo hoy pasa por `core/autotrader.py`

## 10. Diferencia con la documentacion legacy

`docs/legacy/AUTOTRADER_EXPLICADO_FACIL.md` sigue siendo util como contexto historico, pero no debe tomarse como descripcion exacta del runtime actual.

Ejemplos de diferencias:

- la UI actual esta en `templates/polymarket.html`, no en `templates/autotrader.html`
- el runtime actual no usa un maker/post-only real
- el loop principal actual no esta organizado como el stack antiguo `service/strategy/risk/storage`
- el selector bandit y parte del learning existen como infraestructura, pero no son la mejor descripcion del camino de ejecucion actual

## 11. Regla practica para leer el sistema actual

Si quieres entender "que bot esta corriendo de verdad", sigue este orden:

1. `core/autotrader.py`
2. `market/polymarket_execution.py`
3. `web_server.py` (`/api/autotrader/*`)
4. `templates/polymarket.html`
5. `data/autotrader_state.json` y `logs/autotrader_trades.jsonl`

Si quieres entender la evolucion historica o la infraestructura de learning:

1. `data/autotrader.db`
2. `docs/legacy/AUTOTRADER_EXPLICADO_FACIL.md`
3. `docs/legacy/PROPUESTA-0702.md`

## 12. Estado conceptual actual

Resumen corto:

- si, hay autotrading actual y operativo
- no, no es un maker bot puro
- no, no deja ordenes limite resting hoy
- si, trackea posiciones live externas para riesgo
- no, eso no implica que las gestione como suyas
- si, existe infraestructura de learning/bandit en paralelo
- no, esa infraestructura no debe confundirse automaticamente con el runtime principal actual

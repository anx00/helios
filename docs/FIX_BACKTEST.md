**Contexto del problema (síntomas actuales)**
En la UI del Backtest, incluso con **MODE = Execution Aware**, aparecen:

1. **Un solo `predicted` por día** (un bucket final), cuando en live hay muchas predicciones intradía.
2. **Total fills = 0, PnL = 0** aunque se muestran buckets “predicted” diarios.
3. Métricas raras porque el sistema no deja claro “coverage”, “por qué no tradea”, o “por qué resume a 1 predicción”.

En el motor actual, `_run_day()`:

* itera un `timeline` a intervalos (60s),
* acumula `nowcast_sequence`,
* ejecuta la policy en cada timestep si hay `state.nowcast`,
* y luego calcula un único `predicted_winner` a partir de la secuencia. 
  Además, el simulador de ejecución (taker/maker) solo genera fills si hay signals y si las condiciones de fill se cumplen.

**Objetivo**
Quiero que el backtest (Execution Aware) se convierta en algo que:

* **evalúa predicciones intradía (time-series)**, no un único resumen diario,
* **simula muchas decisiones/trades por día** basadas en triggers (edge/confidence/régimen/QC),
* y da herramientas claras para entender:
  “no tradeó porque ___”, “no hubo fills porque ___”, “faltan market events porque ___”, etc.

---

# A) Diagnóstico rápido: confirmar qué datos hay realmente por día

### Paso A1 — Añadir endpoint de “Day Detail” (debug estructurado)

Implementar:

`GET /api/v5/backtest/day_detail/{station}?date=YYYY-MM-DD&mode=execution_aware`

Que devuelva un JSON con:

* `coverage`:

  * `has_label` (bool)
  * `nowcast_events_count` (raw)
  * `market_events_count` (raw)
  * `timeline_steps` (cuántos steps generó iterate_timeline)
  * `timesteps_with_nowcast`
  * `timesteps_with_market`
* `timeline_summary`:

  * primer/último timestamp utc/nyc
  * interval_seconds usado
* `reason_counters`:

  * `no_nowcast`
  * `no_market`
  * `stale_market`
  * `stale_nowcast`
  * `qc_blocked`
  * `edge_too_small`
  * `confidence_too_low`
  * `inventory_blocked`
  * `cooldown_blocked`
  * `maker_no_fill` (si aplica)
* `outputs`:

  * `predictions` (lista)  ← **esto es clave** (ver sección B)
  * `signals` (lista)      ← **esto es clave**
  * `orders` (lista)       ← maker placed/canceled/expired
  * `fills` (lista)        ← si no hay fills, que se vea claramente por qué

> Esto nos permite ver en 1 request si el problema es “no hay market data”, “policy no genera signals”, o “simulador nunca llena”.

---

### Paso A2 — Confirmar si Execution Aware se está quedando sin `market_state`

En `_run_day()` se construye `market_state = _build_market_state(state)` y se evalúa `policy.evaluate(...)`. 
Si `state.market` viene vacío o stale, es muy probable que:

* la policy decida **NO_TRADE** por falta de precio/liquidez,
* o directamente no pueda calcular edge.

Acción:

* Loggear (o incluir en Day Detail) **por timestep**:

  * `market_age_seconds`, `nowcast_age_seconds` (ya existen en TimelineState) 
  * `has_market`, `has_nowcast`
  * top-of-book (bid/ask) del bucket target si existe

Si `timesteps_with_market` es bajo → el problema no es “estrategia”, es “dataset/recording”.

---

# B) Problema 1: “Solo hay 1 predicted por día” (pero debería haber muchas predicciones)

### Qué está pasando ahora

El engine acumula `nowcast_sequence` y luego calcula un único `predicted_winner = _get_predicted_winner(nowcast_sequence)` y un único `predicted_tmax`. 
Eso **no es incorrecto** como “resumen diario”, pero **no es lo que necesitas** para análisis/trading, porque te falta la trayectoria.

### Solución: guardar “PredictionPoints” intradía

Crear un modelo `PredictionPoint` y rellenarlo en cada timestep donde exista nowcast:

Campos mínimos:

* `ts_utc`, `ts_nyc`
* `top_bucket` (argmax)
* `top_bucket_prob`
* `p_bucket` (solo cone; compactado)
* `tmax_mean`, `tmax_sigma` (si están en nowcast)
* `confidence`, `qc_state`
* `market_snapshot` mínimo (mid/spread/depth del bucket top y adyacentes)

Y añadirlo al resultado diario (DayResult) y al endpoint `day_detail`.

**Importante**: no dependas de “raw nowcast events” solamente. Necesitamos la versión “timeline step” para correlacionar con market state.

### UI/UX esperado

En Backtest UI, por día:

* Un chart de `top_bucket` vs tiempo
* Un chart de `P(top_bucket)` vs tiempo
* (Opcional) “cone heatmap” si quieres

Y además, el “Predicted (final)” puede seguir existiendo como resumen, pero **no como único output**.

---

# C) Problema 2: Execution Aware muestra 0 fills / 0 PnL (y tú esperas “muchas apuestas por día”)

Aquí hay 3 puntos que hay que comprobar de forma determinista:

## C1 — ¿La policy está generando signals?

En `_run_day()` se hace `signals = self._policy.evaluate(...)` por timestep, y luego se ejecutan. 
Pero ahora mismo tu UI solo enseña “Total fills”. Si signals = 0, nunca habrá fills.

Acciones:

1. Instrumentar la policy para que devuelva (además de `signals`) un set de **NoTradeReasons** por timestep:

   * `NO_MARKET_DATA`, `NO_EDGE`, `LOW_CONF`, `QC_BLOCK`, etc.
2. En `day_detail`, guarda:

   * `signals_by_time`: lista de (ts, signals_count, reasons)
3. Añadir un contador total correcto:

   * OJO: en el pseudo del engine, `signals_generated=len(signals)` al final del día parece depender de la variable `signals` del último timestep (bug de diseño/alcance). Haz un `total_signals += len(signals)` o guarda lista acumulada. 

**Meta**: que podamos ver claramente:

* “Se generaron 47 signals ese día, pero 0 fills (maker no llenó / taker no ejecutó por precio)”
  vs
* “Se generaron 0 signals porque edge nunca superó threshold”.

---

## C2 — Si hay signals, ¿el simulador realmente puede llenar?

El simulador:

* ejecuta taker inmediatamente a best bid/ask (con slippage) 
* o coloca maker orders que solo se llenan si:

  * el precio cruza, o
  * la cola baja + tiempo mínimo en precio. 

Si tienes 0 fills, puede ser por:

* señales siempre Maker y nunca se cumple condición de fill,
* `market_state` no tiene best bid/ask y el modelo cae en default,
* TTL o `min_time_at_price` demasiado restrictivo,
* `queue_consumption_rate` muy conservador (no consume cola nunca).

Acción:

* En `day_detail`, para cada order maker:

  * `placed_ts`, `limit_price`, `queue_pos_init`, `ttl`, `time_at_price`
  * `expired / canceled / filled`
  * razón de no-fill (no crossed, queue not cleared, ttl exceeded)

**Test controlado**: crea una policy “ToyPolicyDebug” que genere 1 señal taker por hora con tamaño pequeño para verificar que el pipeline produce fills. Si ni con eso hay fills → el problema es market_state/simulador.

---

## C3 — “Muchas apuestas por día” = necesitas una policy con “máquina de estados + triggers”

Ahora mismo, aunque el engine evalúa por timestep, tu policy puede estar pensada como “una entrada” por día o muy restrictiva.

Implementar explícitamente una policy de intradía:

* evalúa cada timestep y puede:

  * entrar,
  * escalar,
  * reducir,
  * cerrar,
  * flippear (con límites),
  * hacer short-term mean reversion (“fade”) si hay QC/regimen.

**Triggers mínimos recomendados (v0):**

* `edge = fair_prob - market_prob` por bucket
* `enter_long` si `edge > enter_threshold` y `confidence > min_conf`
* `exit_long` si `edge < exit_threshold` o `qc_state` empeora o `market_regime` cambia
* `cooldown_seconds` tras cada acción para evitar churn
* `max_flips_per_day`

**Resultado esperado**: en días normales, deberían aparecer múltiples signals (aunque luego fills dependan del modelo de ejecución).

---

# D) Cambios de diseño: backtest debe devolver (y UI debe mostrar) 3 capas

Tu UI ahora mismo está mostrando un mix de: resumen diario + calibración + trading, pero te falta el **blotter** y el **timeline**. Lo mínimo:

### D1 — Timeline de predicción

* `PredictionPoints[]` (por timestep con nowcast)

### D2 — Timeline de decisiones

* `DecisionPoints[]`:

  * ts
  * edge top N buckets
  * reason (trade/no-trade)
  * señal emitida (si aplica)

### D3 — Blotter de ejecución

* `Orders[]` (maker lifecycle)
* `Fills[]` (taker y maker)
* `Position timeline` (exposure por bucket)

Así, si sale 0 fills, lo verás en 10 segundos: “hubo 0 señales” o “hubo señales pero maker no llenó”.

---

# E) Cómo probarlo (plan de pruebas reproducible)

### E1 — Prueba 1: “Data availability”

Para cada día:

* `nowcast_events_count > 0`
* `market_events_count > 0` (si Execution Aware)
* `timesteps_with_nowcast / timeline_steps` razonable
* `timesteps_with_market / timeline_steps` razonable

Si falla aquí: arreglar recorder/compactor/dataset join, no strategy.

### E2 — Prueba 2: “Policy emits signals”

* Correr un día con `ToyPolicyDebug`:

  * 1 taker buy/hora al bucket del top prediction
* Esperado: `signals > 0` y `fills > 0`

Si `signals > 0` pero `fills = 0`: simulador/market_state.

### E3 — Prueba 3: “Conservative policy”

* Correr con tu policy real y loggear `NoTradeReasons` por timestep.
* Ajustar thresholds solo si el debug muestra que nunca supera edge/conf.

### E4 — Prueba 4: “UI correctness”

* En Execution Aware, la tabla diaria debe mostrar además:

  * `signals_generated_total`
  * `fills_total`
  * `days_with_labels/days_with_data`
* Y debe existir una vista “Day Drilldown” con:

  * chart predicción vs tiempo
  * chart market prob vs fair prob
  * blotter de órdenes y fills

---

# F) Entregable final (Definition of Done)

1. Backtest Execution Aware produce:

   * múltiples `PredictionPoints` por día
   * múltiples `DecisionPoints`/signals por día (si triggers se cumplen)
   * fills cuando hay signals (al menos con ToyPolicyDebug)
2. Si un día no tiene fills, el sistema explica por qué con reasons + evidencia.
3. UI deja de parecer “rara”:

   * ya no es “1 predicted/day” sin contexto, sino resumen + timeline.
4. Se puede auditar: “por qué tradeó/no tradeó” y “por qué llenó/no llenó”.

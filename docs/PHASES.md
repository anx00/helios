1. **Fuente equivocada para “tiempo real” (Gamma API)**
   En HELIOS V1 el *Market Engine* “lee precios y probabilidades” vía **Polymarket Gamma API**. Esa API es útil para *discovery/metadata* y para tener una foto “macro”, pero **no es el feed de microestructura** (L2/tape) y suele introducir **caching/agregación** y, por diseño, no te da la prioridad de “lo que está pasando ahora en el libro”. En tu propio brief, HELIOS V1 lista Gamma como “precios y probabilidades en tiempo real”, lo cual es el origen conceptual del desfase. 

2. **Cadencia y bloqueo del loop (scheduler + trabajos pesados + SQLite)**
   HELIOS V1 usa `schedule` con “ciclos de 30 min” y tareas nocturnas. Aunque la UI se vea “live”, si parte del pipeline se refresca por **polling** o por jobs que se pisan (y encima escribes en SQLite en el camino), es fácil que el sistema entre en **drift**: cada iteración tarda más que el intervalo, se acumula backlog y acabas viendo retrasos de “segundos → minutos”. 

### ¿Esto lo soluciona SEARA y Amaterasu?

Sí, porque ambos se apoyan en el principio clave que a HELIOS V1 le falta: **separar “metadata” de “hot path”** y usar el feed correcto.

* **SEARA** ya modela esto: Gamma para discovery, pero **CLOB WebSocket** como fuente de mercado en tiempo real, y además optimiza el “hot loop” y mide latencia E2E con objetivos sub-segundo. 
* **Amaterasu** formaliza la arquitectura de *Market Tape* (WSS CLOB) + *World Tape* (METAR/PWS/features), con estado operable y snapshots, y deja Gamma explícitamente fuera del hot path. 

---

## Plan por fases para refactorizar HELIOS V1 (sin “reinventar” el proyecto)

Objetivo: **mantener HELIOS V1 (su UI y estética) pero arreglar el “motor”** para que sea un sistema event-driven, con staleness controlado y listo para acoplar predicción/backtesting/trading.

### Fase 0 — Baseline de latencia y “staleness” (tangible en 1 paso)

**Qué haces**

* Añades instrumentación obligatoria por evento/fuente:

  * `ts_source` (si la fuente lo trae),
  * `ts_ingest_utc` (cuando tú lo recibes),
  * `ts_nyc` y `ts_es` derivados (America/New_York y Europe/Madrid),
  * `staleness_ms = now_utc - ts_ingest_utc`,
  * “event loop lag” y tiempos por etapa (fetch/parse/compute/db/ui).
* Estandarizas el reloj: **fuente de verdad `ts_utc` y derivados** (NYC/ES). 

**Tecnologías recomendadas**

* Python: `datetime` + `zoneinfo` (ya alineado con tu stack) 
* Logging: JSON estructurado (aunque sea simple al inicio).

**Entregable tangible**

* En la UI de HELIOS aparece un panel “Health” con:

  * staleness por feed (market/metar/pws/model),
  * latencia por tramo,
  * backlog/reconnects.

---

### Fase 1 — Arreglar “real time” de Polymarket (la mejora que más vas a notar)

**Qué haces**

* Sustituyes la ingesta live del mercado:

  * Gamma API queda **solo para discovery/metadata** (mapeos, token_id, etc.). 
  * El live se alimenta desde **CLOB WebSocket** (market + user) y creas un **mirror L2 top 10–20** (cone operativo) para lo que visualizas/tradeas. 
* Construyes un componente “Market Tape” separado (idealmente reutilizando piezas de SEARA):

  * conecta WSS,
  * mantiene orderbook en memoria,
  * emite `BookSnapshot` 1s + `Deltas` (si quieres),
  * expone un stream interno para la UI.

**Por qué esto elimina tu delay**

* Dejas de depender de polling/caching y pasas a **push updates** del matching venue.
* Separas el hot loop del resto del sistema, evitando que el motor de física o el storage “congelen” la UI.

**Tecnologías recomendadas (mínimo fricción)**

* Opción A (recomendada): **Rust para Market Tape** (tokio) y HELIOS lo consume. Amaterasu lo plantea exactamente así (Rust hot path). 
* IPC: NATS o Redis Streams; si quieres ultra-simple, WebSocket interno/UDS. 

**Entregable tangible**

* La UI de HELIOS pasa a mostrar:

  * ladder L2 (top N),
  * mid/spread/imbalance,
  * “last trade tape”,
    con staleness típicamente sub-segundo en LAN.

---

### Fase 2 — Refactor de ingesta “World” (METAR/WU/PWS) con QC fuerte

**Qué haces**

* Conviertes Collector en un sistema event-driven:

  * METAR parseado en cuanto llega (incluyendo grupo T si está),
  * PWS/Mesonets por API con caching inteligente,
  * WU como “judge alignment layer” (no como verdad física).
* Añades **QC** como módulo explícito antes de que nada afecte predicción o trading (outlier detection, consenso por clusters, etc.). Amaterasu ya fija umbrales base para arrancar. 

**Tecnologías recomendadas**

* Python async: `asyncio` + `httpx` (ya está en tu orientación de stack) 
* Modelo de datos: Pydantic/dataclasses para eventos normalizados.

**Entregable tangible**

* Panel “World Live”:

  * METAR raw + temp/dewpoint,
  * PWS cluster + mediana/MAD,
  * estado QC (“OK / outlier likely / degraded”),
  * staleness por fuente.

---

### Fase 3 — Separación estricta de “ingesta/UI” vs “compute” (evitar que el motor de física congele todo)

**Qué haces**

* Aíslas el *Synthesizer* (motor de física/ensemble) para que **no viva en el mismo loop** que la ingesta.
* Cambias el contrato de salida:

  * en vez de “una Tmax puntual”, produces **distribución** sobre buckets y una distribución simple de `t_peak` (bins por horas/2h), porque es lo que monetizas. (Esto ya está en la especificación de Amaterasu.) 
* El compute corre a cadencia 1m + event-driven, pero nunca bloquea el Market Tape.

**Tecnologías recomendadas**

* Python para compute (manteniendo HELIOS V1), pero en **proceso separado** o worker con cola.
* Cache de features y snapshots para reproducibilidad.

**Entregable tangible**

* UI “Nowcast”:

  * P(bucket) en tiempo real,
  * `t_peak` por bins,
  * confidence + QC gating,
    y lo ves reaccionar a observaciones sin lag de UI.

---

### Fase 4 — Storage y replay (sin matar rendimiento)

**Qué haces**

* Sacas SQLite del camino crítico:

  * buffer in-memory + batch flush,
  * o “append log” NDJSON/Parquet para lo crudo y luego ingest.
* Implementas “replay híbrido”: snapshots 1s siempre + tick/deltas solo en ventanas de evento (como plantea Amaterasu). 

**Tecnologías recomendadas**

* ClickHouse + Parquet (si vas en serio con análisis), tal como recomienda Amaterasu. 
* Si quieres ultra-simple primero: Parquet + DuckDB para análisis/backtest y más tarde ClickHouse.

**Entregable tangible**

* “Replay mode” en HELIOS: eliges un rango horario y la UI reproduce:

  * market ladder,
  * world obs,
  * output del modelo,
    como si fuera live.

---

### Fase 5 — Backtesting (equivalencia lógica, no bit-a-bit)

**Qué haces**

* Construyes un backtester que re-inyecta:

  * Market Tape (snapshots + event windows),
  * World Tape (obs + QC),
  * y recalcula decisiones del modelo.
* Evalúas: calibración por bucket, drift vs juez, sensibilidad a QC, y fricción (slippage proxy, fill rate).

**Tecnologías recomendadas**

* Python batch (pandas/duckdb) para research; motor de replay determinista.

**Entregable tangible**

* Report automático diario:

  * error de predicción,
  * “si hubiera operado esta policy” PnL simulado,
  * breakdown por régimen.

---

### Fase 6 — Trading (integración limpia con SEARA, sin volver a “sniping”)

**Qué haces**

* Adoptas el patrón “modelo produce **PolicyTable**, ejecución decide y dispara” (evitas desync y latencia IPC). 
* Reutilizas el Execution/Risk core de SEARA (o su enfoque) y lo alimentas con la policy de nowcast (no con el “dato antes que nadie”). SEARA ya tiene foco en hot loop y métricas E2E. 

**Entregable tangible**

* Paper trading live con:

  * límites de riesgo,
  * logs correlacionados (qué snapshot/policy disparó qué orden),
  * kill-switches.

---

## Qué puntos débiles de HELIOS V1 estás corrigiendo con este plan

* **“Real time” falso** (Gamma como feed): lo sustituyes por **CLOB WSS** y L2 operable.  
* **Loop bloqueado / drift** por mezclar ingesta + compute + storage: separas hot path y haces compute asíncrono/aislado. 
* **Falta de observabilidad de staleness**: introduces métricas y panel Health desde el día 1. 
* **No QC explícito antes de decisiones**: lo conviertes en gating formal. 
* **Backtesting poco reproducible**: pasas a replay híbrido y equivalencia lógica. 

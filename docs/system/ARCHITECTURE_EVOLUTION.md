# Architecture Evolution and Refactor Rationale

Este documento recoge el razonamiento de evolucion del sistema (market data, hot path,
loops event-driven, storage/replay y separacion de responsabilidades).

Incluye un plan por etapas, pero se conserva como referencia de decisiones de
implementacion, no como checklist de fase.

1. **Fuente equivocada para â€œtiempo realâ€ (Gamma API)**
   En HELIOS V1 el *Market Engine* â€œlee precios y probabilidadesâ€ vÃ­a **Polymarket Gamma API**. Esa API es Ãºtil para *discovery/metadata* y para tener una foto â€œmacroâ€, pero **no es el feed de microestructura** (L2/tape) y suele introducir **caching/agregaciÃ³n** y, por diseÃ±o, no te da la prioridad de â€œlo que estÃ¡ pasando ahora en el libroâ€. En tu propio brief, HELIOS V1 lista Gamma como â€œprecios y probabilidades en tiempo realâ€, lo cual es el origen conceptual del desfase. 

2. **Cadencia y bloqueo del loop (scheduler + trabajos pesados + SQLite)**
   HELIOS V1 usa `schedule` con â€œciclos de 30 minâ€ y tareas nocturnas. Aunque la UI se vea â€œliveâ€, si parte del pipeline se refresca por **polling** o por jobs que se pisan (y encima escribes en SQLite en el camino), es fÃ¡cil que el sistema entre en **drift**: cada iteraciÃ³n tarda mÃ¡s que el intervalo, se acumula backlog y acabas viendo retrasos de â€œsegundos â†’ minutosâ€. 

### Â¿Esto lo soluciona SEARA y Amaterasu?

SÃ­, porque ambos se apoyan en el principio clave que a HELIOS V1 le falta: **separar â€œmetadataâ€ de â€œhot pathâ€** y usar el feed correcto.

* **SEARA** ya modela esto: Gamma para discovery, pero **CLOB WebSocket** como fuente de mercado en tiempo real, y ademÃ¡s optimiza el â€œhot loopâ€ y mide latencia E2E con objetivos sub-segundo. 
* **Amaterasu** formaliza la arquitectura de *Market Tape* (WSS CLOB) + *World Tape* (METAR/PWS/features), con estado operable y snapshots, y deja Gamma explÃ­citamente fuera del hot path. 

---

## Plan de evolucion para refactorizar HELIOS V1 (sin â€œreinventarâ€ el proyecto)

Objetivo: **mantener HELIOS V1 (su UI y estÃ©tica) pero arreglar el â€œmotorâ€** para que sea un sistema event-driven, con staleness controlado y listo para acoplar predicciÃ³n/backtesting/trading.

### Fase 0 â€” Baseline de latencia y â€œstalenessâ€ (tangible en 1 paso)

**QuÃ© haces**

* AÃ±ades instrumentaciÃ³n obligatoria por evento/fuente:

  * `ts_source` (si la fuente lo trae),
  * `ts_ingest_utc` (cuando tÃº lo recibes),
  * `ts_nyc` y `ts_es` derivados (America/New_York y Europe/Madrid),
  * `staleness_ms = now_utc - ts_ingest_utc`,
  * â€œevent loop lagâ€ y tiempos por etapa (fetch/parse/compute/db/ui).
* Estandarizas el reloj: **fuente de verdad `ts_utc` y derivados** (NYC/ES). 

**TecnologÃ­as recomendadas**

* Python: `datetime` + `zoneinfo` (ya alineado con tu stack) 
* Logging: JSON estructurado (aunque sea simple al inicio).

**Entregable tangible**

* En la UI de HELIOS aparece un panel â€œHealthâ€ con:

  * staleness por feed (market/metar/pws/model),
  * latencia por tramo,
  * backlog/reconnects.

---

### Fase 1 â€” Arreglar â€œreal timeâ€ de Polymarket (la mejora que mÃ¡s vas a notar)

**QuÃ© haces**

* Sustituyes la ingesta live del mercado:

  * Gamma API queda **solo para discovery/metadata** (mapeos, token_id, etc.). 
  * El live se alimenta desde **CLOB WebSocket** (market + user) y creas un **mirror L2 top 10â€“20** (cone operativo) para lo que visualizas/tradeas. 
* Construyes un componente â€œMarket Tapeâ€ separado (idealmente reutilizando piezas de SEARA):

  * conecta WSS,
  * mantiene orderbook en memoria,
  * emite `BookSnapshot` 1s + `Deltas` (si quieres),
  * expone un stream interno para la UI.

**Por quÃ© esto elimina tu delay**

* Dejas de depender de polling/caching y pasas a **push updates** del matching venue.
* Separas el hot loop del resto del sistema, evitando que el motor de fÃ­sica o el storage â€œcongelenâ€ la UI.

**TecnologÃ­as recomendadas (mÃ­nimo fricciÃ³n)**

* OpciÃ³n A (recomendada): **Rust para Market Tape** (tokio) y HELIOS lo consume. Amaterasu lo plantea exactamente asÃ­ (Rust hot path). 
* IPC: NATS o Redis Streams; si quieres ultra-simple, WebSocket interno/UDS. 

**Entregable tangible**

* La UI de HELIOS pasa a mostrar:

  * ladder L2 (top N),
  * mid/spread/imbalance,
  * â€œlast trade tapeâ€,
    con staleness tÃ­picamente sub-segundo en LAN.

---

### Fase 2 â€” Refactor de ingesta â€œWorldâ€ (METAR/WU/PWS) con QC fuerte

**QuÃ© haces**

* Conviertes Collector en un sistema event-driven:

  * METAR parseado en cuanto llega (incluyendo grupo T si estÃ¡),
  * PWS/Mesonets por API con caching inteligente,
  * WU como â€œjudge alignment layerâ€ (no como verdad fÃ­sica).
* AÃ±ades **QC** como mÃ³dulo explÃ­cito antes de que nada afecte predicciÃ³n o trading (outlier detection, consenso por clusters, etc.). Amaterasu ya fija umbrales base para arrancar. 

**TecnologÃ­as recomendadas**

* Python async: `asyncio` + `httpx` (ya estÃ¡ en tu orientaciÃ³n de stack) 
* Modelo de datos: Pydantic/dataclasses para eventos normalizados.

**Entregable tangible**

* Panel â€œWorld Liveâ€:

  * METAR raw + temp/dewpoint,
  * PWS cluster + mediana/MAD,
  * estado QC (â€œOK / outlier likely / degradedâ€),
  * staleness por fuente.

---

### Fase 3 â€” SeparaciÃ³n estricta de â€œingesta/UIâ€ vs â€œcomputeâ€ (evitar que el motor de fÃ­sica congele todo)

**QuÃ© haces**

* AÃ­slas el *Synthesizer* (motor de fÃ­sica/ensemble) para que **no viva en el mismo loop** que la ingesta.
* Cambias el contrato de salida:

  * en vez de â€œuna Tmax puntualâ€, produces **distribuciÃ³n** sobre buckets y una distribuciÃ³n simple de `t_peak` (bins por horas/2h), porque es lo que monetizas. (Esto ya estÃ¡ en la especificaciÃ³n de Amaterasu.) 
* El compute corre a cadencia 1m + event-driven, pero nunca bloquea el Market Tape.

**TecnologÃ­as recomendadas**

* Python para compute (manteniendo HELIOS V1), pero en **proceso separado** o worker con cola.
* Cache de features y snapshots para reproducibilidad.

**Entregable tangible**

* UI â€œNowcastâ€:

  * P(bucket) en tiempo real,
  * `t_peak` por bins,
  * confidence + QC gating,
    y lo ves reaccionar a observaciones sin lag de UI.

---

### Fase 4 â€” Storage y replay (sin matar rendimiento)

**QuÃ© haces**

* Sacas SQLite del camino crÃ­tico:

  * buffer in-memory + batch flush,
  * o â€œappend logâ€ NDJSON/Parquet para lo crudo y luego ingest.
* Implementas â€œreplay hÃ­bridoâ€: snapshots 1s siempre + tick/deltas solo en ventanas de evento (como plantea Amaterasu). 

**TecnologÃ­as recomendadas**

* ClickHouse + Parquet (si vas en serio con anÃ¡lisis), tal como recomienda Amaterasu. 
* Si quieres ultra-simple primero: Parquet + DuckDB para anÃ¡lisis/backtest y mÃ¡s tarde ClickHouse.

**Entregable tangible**

* â€œReplay modeâ€ en HELIOS: eliges un rango horario y la UI reproduce:

  * market ladder,
  * world obs,
  * output del modelo,
    como si fuera live.

---

### Fase 5 â€” Backtesting (equivalencia lÃ³gica, no bit-a-bit)

**QuÃ© haces**

* Construyes un backtester que re-inyecta:

  * Market Tape (snapshots + event windows),
  * World Tape (obs + QC),
  * y recalcula decisiones del modelo.
* EvalÃºas: calibraciÃ³n por bucket, drift vs juez, sensibilidad a QC, y fricciÃ³n (slippage proxy, fill rate).

**TecnologÃ­as recomendadas**

* Python batch (pandas/duckdb) para research; motor de replay determinista.

**Entregable tangible**

* Report automÃ¡tico diario:

  * error de predicciÃ³n,
  * â€œsi hubiera operado esta policyâ€ PnL simulado,
  * breakdown por rÃ©gimen.

---

### Fase 6 â€” Trading (integraciÃ³n limpia con SEARA, sin volver a â€œsnipingâ€)

**QuÃ© haces**

* Adoptas el patrÃ³n â€œmodelo produce **PolicyTable**, ejecuciÃ³n decide y disparaâ€ (evitas desync y latencia IPC). 
* Reutilizas el Execution/Risk core de SEARA (o su enfoque) y lo alimentas con la policy de nowcast (no con el â€œdato antes que nadieâ€). SEARA ya tiene foco en hot loop y mÃ©tricas E2E. 

**Entregable tangible**

* Paper trading live con:

  * lÃ­mites de riesgo,
  * logs correlacionados (quÃ© snapshot/policy disparÃ³ quÃ© orden),
  * kill-switches.

---

## QuÃ© puntos dÃ©biles de HELIOS V1 estÃ¡s corrigiendo con este plan

* **â€œReal timeâ€ falso** (Gamma como feed): lo sustituyes por **CLOB WSS** y L2 operable.  
* **Loop bloqueado / drift** por mezclar ingesta + compute + storage: separas hot path y haces compute asÃ­ncrono/aislado. 
* **Falta de observabilidad de staleness**: introduces mÃ©tricas y panel Health desde el dÃ­a 1. 
* **No QC explÃ­cito antes de decisiones**: lo conviertes en gating formal. 
* **Backtesting poco reproducible**: pasas a replay hÃ­brido y equivalencia lÃ³gica. 


# Storage & Replay Implementation (hybrid)

En HELIOS V1 hoy el rol de persistencia lo cubre el **Registrar (SQLite)** y el anÃ¡lisis post-evento el **Auditor**. Eso vale para un prototipo, pero se queda corto para: **series densas (L2), ventanas de evento, replay fiel, y datasets batch**. 

Amaterasu ya define exactamente el enfoque correcto para esto: **captura hÃ­brida + NDJSON + Parquet (+ ClickHouse opcional)** y el concepto de â€œequivalencia lÃ³gicaâ€ en replay/backtest.  
Y SEARA ya tiene implementada la mentalidad de **Replay + NDJSON + Sweep**, aunque su foco sea arbitraje de latencia. 

---

## 1. Contexto

## 2. Objetivo operativo

Al completar esta parte de la implementacion, debes poder:

1. **Grabar** todo lo relevante del dÃ­a (mercado + mundo + features + outputs) sin ralentizar el live system.
2. **Reproducir** cualquier dÃ­a con un â€œreloj virtualâ€, alimentando la misma UI (mismos canales WS/SSE) como si fuese live.
3. Tener **datasets limpios** (Parquet) para que el backtesting sea trivial con DuckDB/polars.

Esto responde al problema tÃ­pico de HELIOS V1: â€œse ve bonito, pero no sÃ© por quÃ© cambiÃ³ / no puedo reproducirlo / no puedo medir fricciÃ³n y stalenessâ€. 

---

## 4.2 Principio de diseÃ±o: â€œcapture enough, but not everythingâ€

La polÃ­tica que debes aplicar (tal cual Amaterasu) es:

* **Snapshots L2 1s** (topN del cone) **siempre**
* **Tick/deltas solo en ventanas de evento** (no todo el dÃ­a)
* **Observaciones crudas mÃ­nimas** (METAR raw + PWS raw en puntos clave)
* **Features y outputs del modelo siempre** (1m suele bastar)  

La razÃ³n: el L2/npm de mensajes puede explotar si lo grabas todo; pero si grabas bien los puntos â€œinformativosâ€, luego puedes reconstruir *lo que importa*.

---

## 4.3 QuÃ© datos grabas exactamente (canales y cadencias)

### A) Market Tape (CLOB)

**1) `l2_snap_1s`** (siempre)

* Por token (solo cone operativo), guardas top **10â€“20 niveles** bid/ask y BBO.
* Incluye:

  * `ts_utc`, `ts_nyc`, `ts_es`
  * `seq`/`update_id` si existe
  * `token_id`, `market_id`
  * `bids[[p, s]]`, `asks[[p, s]]`
  * `spread`, `microprice`, `imbalance` (derivados Ãºtiles)

Esto es el backbone para replay y para backtests de ejecuciÃ³n.  

**2) `tape_trades`** (event-driven o cada trade si no es mucho)

* Trades reales del mercado (si el feed lo expone).
* Ãštil para estimar â€œqueue consumptionâ€ en backtest maker. 

**3) `l2_deltas_event_window`** (solo en ventanas)

* Guardas deltas raw (o comprimidos) cuando estÃ¡s en una ventana de evento.

### B) World Tape (observaciones)

**1) `metar_obs`** (cada nueva observaciÃ³n)

* Guardas:

  * `raw` METAR (texto)
  * parseado (temp, dewpoint, wind, etc.)
  * `obs_time_utc` (timestamp del METAR)
  * `ingest_time_utc` + NYC/ES
  * `judge_aligned_temp` (tu redondeo contractual)

**2) `pws_cluster`** (cada update / cada 1â€“5s)

* Guardas:

  * `median`, `MAD`, `support`
  * lista mÃ­nima de PWS usadas (ids) en los puntos clave
  * `qc_state`

### C) Feature Tape (contexto)

**`features_1m`** (cada minuto o cuando cambia el run)

* Vector compacto: viento, nubes, shortwave proxy, SST, AOD, upstream delta, etc.
* Siempre con staleness por feature.

### D) Model output (Fase 3)

**`nowcast_1m`**

* `tmax_mean`, `sigma`
* `P(bucket)` (cone)
* `t_peak_bins`
* `confidence`, `qc_state`

### E) Health / observabilidad

**`health_1s-5s`**

* latencias tramo-a-tramo (sourceâ†’ingest, ingestâ†’model, etc.)
* reconnects, gaps, resyncs
* staleness por fuente 

> Nota: estos canales estÃ¡n alineados con el protocolo WS que ya definiste en Amaterasu (l2/tape/world/features/health). HELIOS puede adoptar los mismos payloads para simplificar replay/UI.  

---

## 4.4 Ventanas de evento (la pieza que hace el storage â€œbaratoâ€ y Ãºtil)

DefiniciÃ³n (tal cual Amaterasu): una **ventana de evento** es cuando algo â€œinformativoâ€ ocurre y quieres capturar granularidad extra (ticks/deltas). 

### Triggers recomendados (MVP)

**World-driven**

* Entra nuevo METAR (siempre abre ventana 60â€“180s).
* QC cambia a `OUTLIER_LIKELY` (abre ventana 120s).

**Market-driven**

* Shock: `|mid(t) - mid(t-1s)| > X ticks`
* Cambio de rÃ©gimen de spread: LOCKED â†” normal â†” wide
* Depth cliff detectado (topN cae por debajo de umbral)

### QuÃ© se graba dentro de la ventana

* deltas L2 raw (o agregados a 50â€“200ms si hace falta)
* trades
* decisiones internas (si ya tienes â€œpolicy eventsâ€)

### CÃ³mo se implementa sin liarte

Un `EventWindowManager` con estado:

* `active: bool`
* `reason: METAR|QC|MID_SHOCK|SPREAD_REGIME...`
* `start_ts`, `end_ts`
* `window_id` (para correlaciÃ³n)

Y cada grabaciÃ³n dentro de ventana lleva `window_id`.

---

## 4.5 Formato y arquitectura de persistencia (sin matar el live loop)

### Regla crÃ­tica

**Nada de escribir a SQLite en el hot path.**
En HELIOS V1 el Registrar SQLite era el camino obvio, pero para L2+replay te va a meter fricciÃ³n y lock contention. 

### Arquitectura recomendada (MVP realista)

**(1) Ring buffers en memoria** para la UI live

* Ãšltimos 5â€“30 minutos por canal.

**(2) NDJSON append-only** como â€œjournalâ€

* Cada evento se escribe como una lÃ­nea JSON.
* Es lo mÃ¡s robusto para no perder datos si cae el proceso. 

**(3) Compaction batch a Parquet**

* Un job (cada X minutos o al final del dÃ­a) convierte NDJSON â†’ Parquet particionado. 

**(4) (Opcional) ClickHouse para queries rÃ¡pidas**

* Solo si quieres analÃ­tica interactiva rÃ¡pida (latencias, drift, etc.).
* Importante: **ClickHouse no es para alimentar la UI live**; es para anÃ¡lisis.  

Esto coincide con Amaterasu: NDJSON streaming + Parquet histÃ³rico + ClickHouse como almacÃ©n de series.  

---

## 4.6 Esquemas de datos y versionado (si no haces esto, te arrepientes en 2 semanas)

Amaterasu fija que **todo debe ir versionado** (`schema_version`) y con correlaciÃ³n (`event_id`, `correlation_id`).  

### Campos obligatorios en TODO evento

* `schema_version`
* `event_id` (uuid)
* `correlation_id` (para agrupar: â€œMETAR 17:55 updateâ€ + deltas + nowcast output)
* `ts_ingest_utc`, `ts_nyc`, `ts_es`
* `obs_time_utc` si aplica (METAR/PWS)
* `market_id`, `token_id` cuando aplica
* `station_id` cuando aplica

Ejemplo mÃ­nimo (conceptual) en NDJSON:
`{"schema_version":1,"ch":"world","event_id":"...","correlation_id":"...","station":"KLGA","obs_time_utc":"...","ts_ingest_utc":"...","ts_nyc":"...","ts_es":"...","src":"METAR","raw":"...","temp":26.5,"temp_aligned":27,"qc":"OK"}`

---

## 4.7 Particionado de Parquet (para que DuckDB vuele)

No compliques: particiona por *dÃ­a* y *market/station*.

Ejemplo de layout:

* `data/parquet/station=KLGA/date=2026-01-29/ch=world/part-0000.parquet`
* `data/parquet/station=KLGA/date=2026-01-29/ch=l2_snap_1s/part-0000.parquet`
* `data/parquet/station=KLGA/date=2026-01-29/ch=nowcast_1m/part-0000.parquet`
* `data/parquet/station=KLGA/date=2026-01-29/ch=event_windows/part-0000.parquet`

Esto estÃ¡ pensado para que en Fase 5 puedas hacer:

* `SELECT * FROM '.../ch=world/*.parquet' WHERE ts BETWEEN ...`
  sin montar infraestructura adicional (DuckDB sobre Parquet). 

---

## 4.8 Replay Engine (la parte â€œwowâ€ de tu UI)

AquÃ­ copias el patrÃ³n de SEARA (Replay) pero adaptado a tu web UI. 

### Concepto: â€œVirtual Clockâ€

* el replay engine carga un rango (ej. 08:00â€“20:00 NYC)
* define `t0` (primera marca de tiempo del dataset)
* emite eventos en orden segÃºn un multiplicador:

  * `speed = 1x` (real-time)
  * `speed = 10x`
  * `step_to_next_event_window`
  * `step_to_next_METAR`

### CÃ³mo se integra con tu UI (sin reescribir frontend)

Tu UI ya consume un WebSocket â€œliveâ€ con canales.
Replay debe **emitir exactamente los mismos canales/payloads** (`l2`, `tape`, `world`, `features`, `health`, y `nowcast` si lo aÃ±ades). Amaterasu ya define este protocolo como estÃ¡ndar. 

Resultado:

* un toggle â€œLIVE / REPLAYâ€
* la UI no sabe si el origen es live o replay; solo ve eventos.

### Funcionalidades mÃ­nimas del replay UI

* timeline con scrubber
* botones: `play/pause`, `1x/5x/10x`, `jump METAR`, `jump window`, `jump shock`
* panel â€œinputs del nowcastâ€: al pausar, muestra el `correlation_id` y quÃ© obs/features estaban activas (esto luego alimenta backtesting)

---

## 4.9 Integridad y detecciÃ³n de corrupciÃ³n del mirror (muy importante)

Si el orderbook mirror se desincroniza, tu replay/backtest se contamina.

Por eso:

* cada snapshot 1s guarda:

  * `seq` actual
  * `resync_epoch_id` (incrementa al pedir snapshot full)
* el recorder marca explÃ­citamente eventos:

  * `BOOK_RESYNC_START`, `BOOK_RESYNC_DONE`
  * `GAP_DETECTED`

Esto tambiÃ©n es clave para â€œequivalencia lÃ³gicaâ€: no necesitas replicar red, pero sÃ­ necesitas **saber** cuÃ¡ndo tu estado era confiable. 

---

## 4.10 RetenciÃ³n y coste (polÃ­tica recomendada)

MVP (simple y eficaz):

* NDJSON crudo: **7â€“14 dÃ­as** (debug rÃ¡pido)
* Parquet: **6â€“24 meses** (dataset histÃ³rico)
* ClickHouse (si lo usas): **30â€“90 dÃ­as** para queries rÃ¡pidas
* Event windows tick: conservar mÃ¡s (son baratos y valiosos)

Esto sigue el espÃ­ritu de â€œcaptura suficiente sin coste explosivoâ€. 

---

## 4.11 TecnologÃ­as recomendadas (alineadas con tus docs)

**Escritura NDJSON (Python)**

* `orjson` + append file + flush controlado (batch)
  **Parquet**
* `pyarrow` (writer)
* compactor job (cron/APScheduler si quieres, pero fuera del hot path)
  **Queries**
* DuckDB (sobre Parquet) para anÃ¡lisis y Fase 5 
  **Si metes ClickHouse**
* inserciÃ³n batch (HTTP) de `l2_snap_1s`, `health`, `nowcast_1m` 

---

# QuÃ© queda â€œDoneâ€ al final de Fase 4 (checklist real)

1. Existe un **Recorder** que no bloquea live y genera NDJSON por canales.
2. Existe un **Compactor** que produce Parquet particionado (por dÃ­a/estaciÃ³n/canal).
3. Existe un **Replay Server** que lee Parquet y emite eventos por WS con el mismo protocolo que live. 
4. Tu UI tiene modo **REPLAY** con timeline y saltos por eventos (METAR / ventanas).
5. Puedes seleccionar un dÃ­a y demostrar: â€œesto fue lo que vio HELIOS y asÃ­ reaccionÃ³ el modeloâ€.

---

## Referencias directas (para justificar el diseÃ±o)

* HELIOS V1 tiene **Registrar SQLite** + mÃ³dulos Collector/Synthesizer/Auditor: es el punto de partida, pero no estÃ¡ optimizado para L2+replay. 
* Amaterasu define explÃ­citamente **Storage & Replay hÃ­brido**: snapshots 1s + tick en ventanas + NDJSON + Parquet (+ ClickHouse).  
* SEARA ya incluye **Replay + NDJSON** como motor prÃ¡ctico (y â€œSweepâ€ para explorar parÃ¡metros). 




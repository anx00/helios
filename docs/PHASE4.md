En HELIOS V1 hoy el rol de persistencia lo cubre el **Registrar (SQLite)** y el análisis post-evento el **Auditor**. Eso vale para un prototipo, pero se queda corto para: **series densas (L2), ventanas de evento, replay fiel, y datasets batch**. 

Amaterasu ya define exactamente el enfoque correcto para esto: **captura híbrida + NDJSON + Parquet (+ ClickHouse opcional)** y el concepto de “equivalencia lógica” en replay/backtest.  
Y SEARA ya tiene implementada la mentalidad de **Replay + NDJSON + Sweep**, aunque su foco sea arbitraje de latencia. 

---

# Fase 4 — Storage & Replay (híbrido) a nivel “equipo de ingeniería”

## 4.1 Objetivo operativo de la fase

Al terminar Fase 4, debes poder:

1. **Grabar** todo lo relevante del día (mercado + mundo + features + outputs) sin ralentizar el live system.
2. **Reproducir** cualquier día con un “reloj virtual”, alimentando la misma UI (mismos canales WS/SSE) como si fuese live.
3. Tener **datasets limpios** (Parquet) para que Fase 5 (backtesting) sea trivial con DuckDB/polars.

Esto responde al problema típico de HELIOS V1: “se ve bonito, pero no sé por qué cambió / no puedo reproducirlo / no puedo medir fricción y staleness”. 

---

## 4.2 Principio de diseño: “capture enough, but not everything”

La política que debes aplicar (tal cual Amaterasu) es:

* **Snapshots L2 1s** (topN del cone) **siempre**
* **Tick/deltas solo en ventanas de evento** (no todo el día)
* **Observaciones crudas mínimas** (METAR raw + PWS raw en puntos clave)
* **Features y outputs del modelo siempre** (1m suele bastar)  

La razón: el L2/npm de mensajes puede explotar si lo grabas todo; pero si grabas bien los puntos “informativos”, luego puedes reconstruir *lo que importa*.

---

## 4.3 Qué datos grabas exactamente (canales y cadencias)

### A) Market Tape (CLOB)

**1) `l2_snap_1s`** (siempre)

* Por token (solo cone operativo), guardas top **10–20 niveles** bid/ask y BBO.
* Incluye:

  * `ts_utc`, `ts_nyc`, `ts_es`
  * `seq`/`update_id` si existe
  * `token_id`, `market_id`
  * `bids[[p, s]]`, `asks[[p, s]]`
  * `spread`, `microprice`, `imbalance` (derivados útiles)

Esto es el backbone para replay y para backtests de ejecución.  

**2) `tape_trades`** (event-driven o cada trade si no es mucho)

* Trades reales del mercado (si el feed lo expone).
* Útil para estimar “queue consumption” en backtest maker. 

**3) `l2_deltas_event_window`** (solo en ventanas)

* Guardas deltas raw (o comprimidos) cuando estás en una ventana de evento.

### B) World Tape (observaciones)

**1) `metar_obs`** (cada nueva observación)

* Guardas:

  * `raw` METAR (texto)
  * parseado (temp, dewpoint, wind, etc.)
  * `obs_time_utc` (timestamp del METAR)
  * `ingest_time_utc` + NYC/ES
  * `judge_aligned_temp` (tu redondeo contractual)

**2) `pws_cluster`** (cada update / cada 1–5s)

* Guardas:

  * `median`, `MAD`, `support`
  * lista mínima de PWS usadas (ids) en los puntos clave
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

* latencias tramo-a-tramo (source→ingest, ingest→model, etc.)
* reconnects, gaps, resyncs
* staleness por fuente 

> Nota: estos canales están alineados con el protocolo WS que ya definiste en Amaterasu (l2/tape/world/features/health). HELIOS puede adoptar los mismos payloads para simplificar replay/UI.  

---

## 4.4 Ventanas de evento (la pieza que hace el storage “barato” y útil)

Definición (tal cual Amaterasu): una **ventana de evento** es cuando algo “informativo” ocurre y quieres capturar granularidad extra (ticks/deltas). 

### Triggers recomendados (MVP)

**World-driven**

* Entra nuevo METAR (siempre abre ventana 60–180s).
* QC cambia a `OUTLIER_LIKELY` (abre ventana 120s).

**Market-driven**

* Shock: `|mid(t) - mid(t-1s)| > X ticks`
* Cambio de régimen de spread: LOCKED ↔ normal ↔ wide
* Depth cliff detectado (topN cae por debajo de umbral)

### Qué se graba dentro de la ventana

* deltas L2 raw (o agregados a 50–200ms si hace falta)
* trades
* decisiones internas (si ya tienes “policy events”)

### Cómo se implementa sin liarte

Un `EventWindowManager` con estado:

* `active: bool`
* `reason: METAR|QC|MID_SHOCK|SPREAD_REGIME...`
* `start_ts`, `end_ts`
* `window_id` (para correlación)

Y cada grabación dentro de ventana lleva `window_id`.

---

## 4.5 Formato y arquitectura de persistencia (sin matar el live loop)

### Regla crítica

**Nada de escribir a SQLite en el hot path.**
En HELIOS V1 el Registrar SQLite era el camino obvio, pero para L2+replay te va a meter fricción y lock contention. 

### Arquitectura recomendada (MVP realista)

**(1) Ring buffers en memoria** para la UI live

* Últimos 5–30 minutos por canal.

**(2) NDJSON append-only** como “journal”

* Cada evento se escribe como una línea JSON.
* Es lo más robusto para no perder datos si cae el proceso. 

**(3) Compaction batch a Parquet**

* Un job (cada X minutos o al final del día) convierte NDJSON → Parquet particionado. 

**(4) (Opcional) ClickHouse para queries rápidas**

* Solo si quieres analítica interactiva rápida (latencias, drift, etc.).
* Importante: **ClickHouse no es para alimentar la UI live**; es para análisis.  

Esto coincide con Amaterasu: NDJSON streaming + Parquet histórico + ClickHouse como almacén de series.  

---

## 4.6 Esquemas de datos y versionado (si no haces esto, te arrepientes en 2 semanas)

Amaterasu fija que **todo debe ir versionado** (`schema_version`) y con correlación (`event_id`, `correlation_id`).  

### Campos obligatorios en TODO evento

* `schema_version`
* `event_id` (uuid)
* `correlation_id` (para agrupar: “METAR 17:55 update” + deltas + nowcast output)
* `ts_ingest_utc`, `ts_nyc`, `ts_es`
* `obs_time_utc` si aplica (METAR/PWS)
* `market_id`, `token_id` cuando aplica
* `station_id` cuando aplica

Ejemplo mínimo (conceptual) en NDJSON:
`{"schema_version":1,"ch":"world","event_id":"...","correlation_id":"...","station":"KLGA","obs_time_utc":"...","ts_ingest_utc":"...","ts_nyc":"...","ts_es":"...","src":"METAR","raw":"...","temp":26.5,"temp_aligned":27,"qc":"OK"}`

---

## 4.7 Particionado de Parquet (para que DuckDB vuele)

No compliques: particiona por *día* y *market/station*.

Ejemplo de layout:

* `data/parquet/station=KLGA/date=2026-01-29/ch=world/part-0000.parquet`
* `data/parquet/station=KLGA/date=2026-01-29/ch=l2_snap_1s/part-0000.parquet`
* `data/parquet/station=KLGA/date=2026-01-29/ch=nowcast_1m/part-0000.parquet`
* `data/parquet/station=KLGA/date=2026-01-29/ch=event_windows/part-0000.parquet`

Esto está pensado para que en Fase 5 puedas hacer:

* `SELECT * FROM '.../ch=world/*.parquet' WHERE ts BETWEEN ...`
  sin montar infraestructura adicional (DuckDB sobre Parquet). 

---

## 4.8 Replay Engine (la parte “wow” de tu UI)

Aquí copias el patrón de SEARA (Replay) pero adaptado a tu web UI. 

### Concepto: “Virtual Clock”

* el replay engine carga un rango (ej. 08:00–20:00 NYC)
* define `t0` (primera marca de tiempo del dataset)
* emite eventos en orden según un multiplicador:

  * `speed = 1x` (real-time)
  * `speed = 10x`
  * `step_to_next_event_window`
  * `step_to_next_METAR`

### Cómo se integra con tu UI (sin reescribir frontend)

Tu UI ya consume un WebSocket “live” con canales.
Replay debe **emitir exactamente los mismos canales/payloads** (`l2`, `tape`, `world`, `features`, `health`, y `nowcast` si lo añades). Amaterasu ya define este protocolo como estándar. 

Resultado:

* un toggle “LIVE / REPLAY”
* la UI no sabe si el origen es live o replay; solo ve eventos.

### Funcionalidades mínimas del replay UI

* timeline con scrubber
* botones: `play/pause`, `1x/5x/10x`, `jump METAR`, `jump window`, `jump shock`
* panel “inputs del nowcast”: al pausar, muestra el `correlation_id` y qué obs/features estaban activas (esto luego alimenta backtesting)

---

## 4.9 Integridad y detección de corrupción del mirror (muy importante)

Si el orderbook mirror se desincroniza, tu replay/backtest se contamina.

Por eso:

* cada snapshot 1s guarda:

  * `seq` actual
  * `resync_epoch_id` (incrementa al pedir snapshot full)
* el recorder marca explícitamente eventos:

  * `BOOK_RESYNC_START`, `BOOK_RESYNC_DONE`
  * `GAP_DETECTED`

Esto también es clave para “equivalencia lógica”: no necesitas replicar red, pero sí necesitas **saber** cuándo tu estado era confiable. 

---

## 4.10 Retención y coste (política recomendada)

MVP (simple y eficaz):

* NDJSON crudo: **7–14 días** (debug rápido)
* Parquet: **6–24 meses** (dataset histórico)
* ClickHouse (si lo usas): **30–90 días** para queries rápidas
* Event windows tick: conservar más (son baratos y valiosos)

Esto sigue el espíritu de “captura suficiente sin coste explosivo”. 

---

## 4.11 Tecnologías recomendadas (alineadas con tus docs)

**Escritura NDJSON (Python)**

* `orjson` + append file + flush controlado (batch)
  **Parquet**
* `pyarrow` (writer)
* compactor job (cron/APScheduler si quieres, pero fuera del hot path)
  **Queries**
* DuckDB (sobre Parquet) para análisis y Fase 5 
  **Si metes ClickHouse**
* inserción batch (HTTP) de `l2_snap_1s`, `health`, `nowcast_1m` 

---

# Qué queda “Done” al final de Fase 4 (checklist real)

1. Existe un **Recorder** que no bloquea live y genera NDJSON por canales.
2. Existe un **Compactor** que produce Parquet particionado (por día/estación/canal).
3. Existe un **Replay Server** que lee Parquet y emite eventos por WS con el mismo protocolo que live. 
4. Tu UI tiene modo **REPLAY** con timeline y saltos por eventos (METAR / ventanas).
5. Puedes seleccionar un día y demostrar: “esto fue lo que vio HELIOS y así reaccionó el modelo”.

---

## Referencias directas (para justificar el diseño)

* HELIOS V1 tiene **Registrar SQLite** + módulos Collector/Synthesizer/Auditor: es el punto de partida, pero no está optimizado para L2+replay. 
* Amaterasu define explícitamente **Storage & Replay híbrido**: snapshots 1s + tick en ventanas + NDJSON + Parquet (+ ClickHouse).  
* SEARA ya incluye **Replay + NDJSON** como motor práctico (y “Sweep” para explorar parámetros). 

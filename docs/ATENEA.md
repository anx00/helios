# Fase auxiliar (HELIOS) — Atenea Button + Chat Copilot con evidencias

## 0) Objetivo

Añadir a HELIOS un “copiloto” accesible desde cualquier pantalla:

* El usuario pulsa **Atenea** (botón fijo).
* Se abre un **chat** (panel lateral o modal).
* Atenea responde preguntas sobre:

  * mercado (orderbook, shocks, spreads, cambios),
  * mundo (METAR, PWS, QC, features),
  * nowcast/predicción (por qué cambió, qué drivers, confianza),
  * backtests (por qué sale logloss raro, predicted vacío, labels faltantes),
  * salud del sistema (staleness, reconexiones, gaps),
* siempre con **evidencias** trazables (no opiniones).

> Atenea NO ejecuta trading ni cambia parámetros automáticamente. Es diagnóstico y explicación.

---

## 1) UX: Atenea como botón global + chat

### 1.1 Botón global

* Un botón “🜂 Atenea” fijo (navbar / esquina inferior).
* Visible en:

  * Market Live
  * World Live
  * Features
  * Nowcast
  * Replay
  * Backtest Lab
  * Health

### 1.2 Chat directo

* Chat persistente (mantiene contexto de la pantalla actual).
* Tiene:

  * input normal (“¿por qué…?”)
  * botón “Attach current context” (por defecto ON)
  * botón “Citar evidencias” (siempre ON y obligatorio)

### 1.3 “Contexto automático”

Cuando abres Atenea desde una pantalla, HELIOS le adjunta automáticamente:

* `screen`: Market/World/Nowcast/Backtest/Replay…
* `station_id` (ej KLGA)
* `time_range` actual (si estás en Replay, el timestamp del scrubber)
* `selected_token_ids` (si estás mirando un cone)
* `mode`: LIVE o REPLAY

Esto permite preguntas tipo:

* “Explícame este salto” (sin que tengas que copiar nada).

---

## 2) Requisito clave: respuestas con evidencias

Atenea **no puede** responder sin evidencias. Si no tiene datos, debe decir “no tengo evidencia suficiente” y explicar qué falta.

### 2.1 Formato de evidencias (estándar)

Cada respuesta de Atenea debe incluir una sección tipo:

**Evidence**

* `E1` (Market): `l2_snap_1s` @ `2026-01-30 17:55:02 NYC` — spread=…, mid=… (event_id=…, seq=…)
* `E2` (World): METAR raw “…” @ obs_time=… ingest_time=… (event_id=…)
* `E3` (Nowcast): output `nowcast_1m` @ … — P(<18)=… sigma=… (event_id=…)
* `E4` (Health): market_staleness_ms=… ws_reconnect_count=… (event_id=…)

**Nota:** evidencia = referencias a eventos reales de HELIOS (IDs/timestamps), más extractos (pequeños) del payload.

---

## 3) Cómo accede Atenea a la información (sin bloquear HELIOS)

Atenea necesita un “brazo” determinista que consulte datos de HELIOS. La solución es un **Atenea Data Access Layer**.

### 3.1 Dos fuentes de datos

**A) Live State (in-memory)**

* MarketState (orderbook mirror agregados)
* WorldState (última obs + QC + features)
* NowcastState (último output y drivers)
* HealthState (staleness/latency/gaps)
* ring buffers (últimos 5–30 min)

**B) Historical/Replay Store**

* NDJSON/Parquet (si ya lo tienes de Fase 4)
* Backtest results store (runs, params, coverage, etc.)

### 3.2 API interna de consulta (determinista)

Implementa endpoints internos (o funciones) tipo:

* `GET /athena/context/live?station=KLGA`
* `GET /athena/context/window?station=KLGA&from=...&to=...`
* `GET /athena/evidence/event?id=...`
* `GET /athena/backtest/run?id=...`

Esto lo usa Atenea para recuperar evidencia **antes** de llamar al LLM.

> Importante: el LLM no “busca” en tu data. Un módulo determinista le trae los datos y el LLM solo los interpreta/resume.

---

## 4) Flujo de una pregunta (pipeline completo)

### 4.1 Pasos

1. Usuario pregunta: “¿Por qué la predicción cambió tanto a las 17:55?”
2. **Atenea Router** clasifica intención:

   * nowcast explanation / QC / market shock / system health / backtest debug…
3. **Evidence Builder** decide qué evidencias necesita:

   * ventana `t-10min → t+2min`
   * eventos: METAR, QC changes, book shocks, nowcast updates, resyncs
4. Recupera evidencias (Live + Store).
5. Construye un “Context Pack” compacto:

   * snapshot “antes” y “después”
   * lista de eventos en medio
   * métricas clave
6. Llama a Gemini Flash (gemini-3-flash-preview) con:

   * instrucciones estrictas (“responde solo con evidencias, cita E1…En”)
7. Devuelve respuesta con:

   * explicación
   * hipótesis (si aplica) marcadas como hipótesis
   * evidencias enumeradas
   * “acciones sugeridas” (opcionales)

### 4.2 Regla de oro

Si Evidence Builder no encuentra evidencia suficiente:

* Atenea responde: “No tengo datos suficientes para afirmar X”
* y enseña qué falta (ej. “no hay nowcast_1m entre 17:40–18:00”)

---

## 5) Qué preguntas debe soportar desde el día 1 (use cases)

### 5.1 Diagnóstico de latencia (tu dolor actual)

* “¿Por qué el orderbook va con retraso?”
  Atenea debe mirar:
* market_staleness_ms
* ws reconnects/resyncs
* seq gaps
* event loop lag
  y responder con evidencias.

### 5.2 Explicación de nowcast

* “¿Qué cambió y por qué?”
* “¿Qué variables están empujando Tmax?”
  Debe citar:
* METAR/PWS/QC
* features (viento, radiación, upstream)
* diferencia base vs bias

### 5.3 QC/outliers

* “¿Esta lectura PWS es real?”
  Debe mostrar:
* cluster median/MAD/support
* comparación con METAR
* flags (rate-of-change)

### 5.4 Backtest Lab (lo que te pasó en la captura)

* “¿Por qué Predicted sale ‘–’?”
* “¿Por qué logloss es enorme?”
  Debe verificar:
* coverage (#días con predicción y label)
* si P(bucket) existe o es degenerate (0/1)
* si hay clamp epsilon
* si falta label en un día
  Con evidencias del run.

### 5.5 Replay

* “Resume esta ventana de evento”
  Debe:
* enumerar triggers
* mostrar cambios en market/world/nowcast
* y cerrar con “root cause probable” + evidencia

---

## 6) Guardrails para que no “alucine”

### 6.1 Prompt/contracto estricto

En cada llamada:

* “No inventes datos”
* “Cita evidencias E1..En”
* “Si no hay evidencia, dilo”

### 6.2 Validator de respuesta

Antes de mostrar:

* si Atenea menciona un valor sin `E#`, se marca como “no soportado” y se elimina o se pide reintento con evidencia.

### 6.3 Límites de autoridad

* Atenea nunca escribe configuración.
* Solo propone cambios (“Suggestion”), con evidencias.

---

## 7) Cost & Performance (para que no te queme créditos)

### 7.1 Triggers y límites

* 1 consulta “pesada” (ventana 10–30 min) máximo cada X segundos.
* caching de context packs recientes.
* compresión: no mandar L2 completo; mandar agregados + 2 snapshots representativos.

### 7.2 Modo “fast”

Para preguntas simples:

* responder solo con LiveState (sin store)
* sin ventana de 10 min

---

## 8) Entregables (Definition of Done)

1. Botón Atenea visible en todas las pantallas.
2. Chat funcional (con contexto automático por pantalla).
3. Router + Evidence Builder:

   * responde a 5 categorías (market/world/nowcast/health/backtest).
4. Respuestas siempre con sección **Evidence**:

   * IDs + timestamps + extractos.
5. Si faltan datos, Atenea lo reconoce y lo explica.
6. Logs de “Atenea sessions” (para auditar).

---

## 9) Implementación mínima recomendada (MVP en HELIOS)

Para empezar sin complicarte:

* **Atenea Service (Python)** dentro del mismo backend de HELIOS:

  * endpoint `POST /api/athena/chat`
  * usa Evidence Builder + llama a Gemini
* **Front**:

  * botón global + panel chat
  * render de evidencias con links “ver evento” (abre el snapshot exacto en la UI o replay)

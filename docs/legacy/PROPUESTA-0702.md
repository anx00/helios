Vale — con lo que has pasado, **HELIOS ya tiene la columna vertebral correcta** para llegar al bot “de verdad”. Lo que te falla ahora mismo (y por eso el backtest “no parece real” o la UI “no cuadra”) no es tanto el modelo, sino **la fidelidad del replay + el encaje entre “intraday” y lo que resume el frontend**.

Te lo dejo en 3 bloques: **qué está bien**, **qué no está bien / por qué te salen cosas raras**, y **siguientes pasos en orden** (para convertirlo en un autotrader paper serio).

---

## 1) Qué tienes bien (y es exactamente lo que necesitas)

### Predicción: el “output contract” es muy sólido

Tu `NowcastDistribution` está súper bien pensado: **mean/sigma**, **probabilidades por bucket**, **cumulativas por strike**, **distribución de t-peak**, **confidence + factores**, y sobre todo **explanations + inputs_used** para trazabilidad. Eso es oro para debugging y para justificar decisiones del bot.  

### Matemática del nowcast: simple, controlable, y debuggable

La parte clave (bias EMA + decay exponencial hasta el pico + penalizaciones en sigma + confidence) es el tipo de “math engine” que luego puedes **calibrar** sin romperlo todo. Y las constraints (floor + post-peak cap) están exactamente en la dirección correcta para evitar predicciones físicamente/temporalmente imposibles cuando ya estás “post-peak”.  

### Data sources: jerarquía y QC/health tracking bien planteados

* METAR como autoridad (y tu **3-way race**) te da robustez.
* PWS consensus con MAD/outlier filtering es justo lo que necesitas para “confirmar o dudar” sin casarte con un sensor.
* Health states (LIVE/OK/STALE/DEAD) y cómo impactan sigma/confidence es la forma correcta de convertir “calidad de datos” en “incertidumbre cuantificada”.   

### Arquitectura de ejecución: triggers + pipeline clara

Tienes la idea correcta de **loop periódico** + **updates event-driven** (METAR, PWS, forecast updates, QC state change…), y eso cuadra con un sistema “real-time” de verdad.  

### Autotrader: la estructura es pro

Lo que describes en el Autotrader es exactamente el “stack mínimo serio”:

* catálogo multi-estrategia
* selección adaptativa (LinUCB)
* risk gate
* paper broker con fees/slippage
* persistencia (SQLite) de decisiones/órdenes/fills
* y “offline learning” nocturno  

---

## 2) Qué no tienes bien (o por qué ahora “no se entiende”)

### (A) El backtest **no está usando market data real** en algunos días → por eso “execution-aware” es medio mentira

Tú mismo lo viste: `timesteps_with_market=0`. Eso significa que tu timeline tiene nowcast en cada step, pero **no tiene eventos de mercado** para construir el `market_state`. Resultado típico:

* el motor simula o deja vacío el market_state,
* los fills / señales no reflejan el orderbook real,
* y el PnL sale raro o no corresponde con lo que tú “ves” en vivo.

El documento de replay/backtest está diseñado precisamente para grabar y re-jugar “lo que pasó”, con recorder por canales y replay posterior. Si el canal de market (top-of-book / L2) **no está entrando al tape** en esas fechas, el replay pierde fidelidad.  

**Traducción:** ahora mismo estás backtesteando “modelo vs label” bastante bien… pero **no** “estrategia vs microestructura real” de forma consistente.

---

### (B) Tu UI está mostrando un **resumen diario**, pero tu sistema genera **predicciones intradía**

Esto es la confusión de “¿por qué solo hay un predicted por día si yo veo muchas predicciones?”:

* El motor sí genera intradía (y tú lo comprobaste con `predictions_count=786`, `decisions_count=786`).
* Lo que pasa es que el panel está enseñando algo tipo `predicted_winner` diario o un “snapshot representativo”, en vez de renderizar la **serie temporal** y el **blotter** (señales/fills).

O sea: **el backend ya tiene el dato**, pero el frontend está leyendo/mostrando la capa equivocada.  

---

### (C) Modelo de ejecución: está bien como MVP, pero todavía es “optimista” o inconsistente si no tienes L2 real

Tu Paper Broker está conceptualmente bien (fees + slippage + fills), pero su realismo depende de que le des:

* best bid/ask reales por timestamp (y mejor si hay profundidad)
* spreads reales en cambios de régimen

Sin market tape, el execution model se apoya en suposiciones… y por eso luego no te cuadra con lo que tú “sabes” que habría pasado mirando Polymarket.  

---

### (D) El requisito de **mínimo $1 por orden** tiene que vivir en 2 sitios

Lo mencionas como requisito del sistema final: perfecto. Solo ojo con esto: tienes que forzarlo:

1. en paper/live broker (para no mandar órdenes inválidas)
2. en backtest (para que el PnL no “haga trampas” con tamaños demasiado pequeños)

Si lo tienes solo en uno, el backtest te puede dar edges falsos. (El Autotrader está montado para poder hacerlo bien, pero hay que verificar que el backtest adapter lo respeta igual.)  

---

## 3) Siguientes pasos (el orden que te desbloquea todo)

### Paso 1 — Hacer el replay “de mercado” real (esto es el cuello de botella)

Objetivo: que un `day_detail` tenga:

* `timesteps_with_market > 0`
* `market_events_count > 0`
* y que puedas reconstruir best bid/ask por step

Acción concreta:

* Asegurar que el recorder graba **market events** (al menos top-of-book por bracket del cone; ideal: snapshots/deltas acotadas).
* Compactar eso en Parquet igual que haces con nowcast, para que el dataset builder pueda hacer **as-of joins** nowcast↔market.

Esto convierte tu backtest de “modelo vs label” a “estrategia vs mercado real”.   

---

### Paso 2 — Arreglar la UX del backtest (para que veas lo que *realmente* pasa intradía)

En vez de “una fila por día con un predicted”:

* gráfica intradía de `tmax_mean_f` + banda sigma
* gráfica intradía de `p_bucket` (solo top-K buckets o cone)
* overlay de señales y fills (marcas verticales)
* y “drilldown” por timestamp con: inputs_used + explanations + market snapshot del momento

Tu propio diseño de predicción ya trae `explanations`/`inputs_used`, así que esto es “solo” conectarlo a UI.  

---

### Paso 3 — Normalizar reglas de sizing (incluido el mínimo $1) + costes

Aquí cierras el gap “mi backtest dice X pero en real sería Y”:

* enforce min $1 en backtest + paper broker
* aplicar fees/slippage igual en ambos
* registrar PnL gross vs net y turnover (ya lo tienes encaminado con storage y métricas)  

---

### Paso 4 — “Aprendizaje” de verdad, pero con disciplina

Ahora sí tiene sentido lo de autoajuste:

* calibración del modelo (bias alpha, tau, sigma penalties, umbrales QC) por estación/mes
* calibración de políticas (edge_threshold, cooldowns, etc.) usando backtest walk-forward
* promoción de parámetros/modelos con registry (sin pisar producción)

Tu Autotrader ya contempla offline learning y model registry, así que el sitio donde poner esto existe.  

---

## Conclusión rápida

**Lo tienes bien en lo importante**: predicción trazable + math controlable + fuentes con QC + autotrader con broker/risk/learning.

Lo que te está rompiendo la cabeza ahora es:

1. **fidelidad del market replay** (sin market tape, “execution-aware” cojea)
2. **la UI está leyendo el resumen diario** en vez del timeline intradía que ya existe

Perfecto. Te dejo dos cosas: (1) un **checklist/runbook** para localizar *exactamente* por qué tu replay/backtest puede quedarse sin market (o usar market “sintético”) y (2) una explicación clara de **auto-aprendizaje**: qué tienes ya implementado y qué modelos algorítmicos encajan para que HELIOS se autoajuste sin volverse una caja negra.

---

## Checklist “Market Tape Integrity” (para que Execution-Aware sea real)

La idea del sistema está bien montada: **Recorder → Compactor (Parquet) → HybridReader → Dataset (as-of joins) → BacktestEngine/Simulator**.  
Lo que te está rompiendo el realismo es cuando el dataset no puede construir `market_state` por timestep.

### A) ¿Se está grabando market realmente? (lo primero)

1. **Comprueba en disco que existe el canal de mercado para ese día y estación**
   En Parquet, el diseño esperado incluye algo como `ch=l2_snap/` dentro de `data/parquet/station=KLGA/date=YYYY-MM-DD/…` 
   Si para `2026-01-30` no existe `ch=l2_snap` (o el canal equivalente de mercado que uses), ya tienes el motivo de `timesteps_with_market=0`.

2. Si no está compactado a Parquet, busca el NDJSON del día en `data/recordings/...`
   La gracia del **HybridReader** es que intenta NDJSON primero y si no, Parquet. 
   Si no hay ni NDJSON ni Parquet de mercado, entonces **no hay market tape** (punto).

3. “Sanity count” por canal
   Para la fecha: cuenta eventos por canal (aunque sea con un script rápido o endpoint debug):

* `world`
* `nowcast`
* `l2_snap` (o como se llame tu mercado)
  Si `l2_snap == 0`, no tiene sentido que haya fills en execution-aware (eso es bug lógico).

---

### B) ¿Se están grabando los timestamps correctos? (el bug silencioso típico)

Tu replay ordena eventos por `ts_ingest_utc` (según guía). 
Si el canal de mercado está usando otro campo, o se guarda con un timestamp vacío/malformado, puede “quedarse fuera del timeline”.

Checklist:
4) En un evento de mercado, valida que existan siempre:

* `ts_ingest_utc` (cuando lo capturaste)
* y si tienes `ts_event_utc` / `obs_time` (momento de mercado) mejor, pero el replay se apoya en ingest.
  Si faltan o vienen nulos, el DatasetBuilder no puede hacer as-of join.

5. Comprueba el timezone del particionado (NYC vs UTC)
   Si particionas por “fecha NYC” pero guardas por “fecha UTC”, puedes tener market en `2026-01-31` UTC y verlo como `2026-01-30` NYC, y al revés. Resultado: “hay datos pero en otro día”.

---

### C) ¿El Dataset está haciendo el join del mercado o filtrando mal?

6. Verifica que el dataset realmente carga el canal market al construir `TimelineState`
   Si el HybridReader devuelve mercado pero `timesteps_with_market=0`, entonces el bug está en **dataset.py**: filtrado por estación, canal, o shape del evento.

7. Revisa el “mapping” bucket → token en replay
   Tú ya tienes mapping en SEARA/HELIOS; el error típico es que en replay el bucket está normalizado distinto (`"30-31"` vs `"30-31°F"` vs `"30–31"`).
   En tu autotrader ya aparece que el `market_state` se indexa por **label normalizado** y solo incluye tokens YES. Si esa normalización no coincide con lo grabado en tape, el join no casa. 

8. Prueba de oro: para un timestamp concreto del día (ej. 15:20 NYC), pide al dataset:

* nowcast (existe)
* market snapshot (debería existir)
  Si market snapshot sale `None` pero hay eventos l2_snap cercanos en tiempo, el as-of join está mal (tolerancia, orden, o campo timestamp).

---

### D) “Hard gate” obligatorio: si no hay market, no puede haber fills (corrige el bug que te confundía)

9. En `execution_aware`, impón regla:

* si `market_state` es None en el timestep → **no se pueden emitir órdenes ni fills**
* si `timesteps_with_market == 0` en el día → `status=NO_MARKET_DATA`, `signals_total=0`, `fills=0`, `pnl=0` y un reason counter claro.
  Esto evita backtests “bonitos” con ejecución ficticia.

---

### E) UI / performance (esto ya es secundario, pero fácil)

Tu día tiene 786 steps (perfecto, 1-min). El backend ya devuelve arrays completos.
10) Añade `downsample` y `limit`, pero con una regla: **fills nunca se downsamplean**

* Overview: `downsample=5`
* Drilldown: `downsample=1&from=...&to=...`
  Esto no cambia la verdad; solo hace la UX fluida.

---

## Auto-aprendizaje: qué tienes ya y qué te falta

Aquí viene lo importante: **sí tienes ya un módulo de learning**. No es humo.

### Lo que YA tienes implementado (y está bien diseñado)

* Existe un pipeline de **offline learning** con walk-forward: train D-60..D-8 y validación D-7..D-1, grid search (hasta `max_combinations`) y promoción a `model_registry` si cumple umbral.  
* Existe un loop nocturno que lo ejecuta automático (3:15 AM NYC) y persiste runs en SQLite + artifacts JSON. 
* Existe selección adaptativa de estrategias tipo **multi-policy + LinUCB** (bandit) y un adapter para backtest comparativo baseline vs candidate.  

Eso ya es “autoajuste” real… pero está más enfocado a **parámetros de estrategia / selección** que a corregir el **modelo de predicción**.

---

## Qué modelos algorítmicos usaría un sistema serio para “aprender” aquí

Piensa en 3 capas, de la más segura a la más ambiciosa:

### 1) Auto-calibración del nowcast (lo más rentable y menos arriesgado)

Tu nowcast es matemático y explicable (bias decay, sigma penalties, caps…).  
Eso es perfecto para **aprender parámetros**, no para meter una red neuronal.

Qué aprender (ejemplos muy concretos):

* `tau` de decay del bias (ahora mismo fijo)
* penalizaciones de sigma por “STALE/DEAD sources” (tú ya defines reglas tipo +0.3°F; esto es calibrable por estación/mes) 
* umbrales QC de PWS (MAD z-score, soporte mínimo, etc.) 
* margen del post-peak cap y su shrink (para estaciones como KLGA esto cambia por estación/mes) 

**Técnica recomendada:** grid search / Bayesian optimization sobre esos parámetros usando métricas de calibración (log loss/Brier) + MAE de Tmax + “reliability”. Tu backtest ya contempla calibration/metrics como módulo formal. 

👉 Esto te da “autoaprendizaje” sin ML opaco: solo “HELIOS ajusta sus knobs”.

---

### 2) Capa de calibración probabilística (probabilidades que se ajustan a realidad)

Aunque tu `p_bucket` salga de una Normal con sigma, casi siempre se puede mejorar con calibración:

* **Temperature scaling / isotonic regression** sobre `P(bucket)` para que cuando dices “70%” realmente se cumpla ~70% en histórico.
* Esto es especialmente útil si tu sigma tiende a estar sistemáticamente subestimada o sobreestimada.

Beneficio real: el trading se basa en **edge = P_model − P_market**. Si tu P está mal calibrada, tu edge es fake.

---

### 3) Modelo residual interpretable (para capturar patrones tipo “no es X, es Y”)

Esto es exactamente lo que describes: “una variable suma X pero en realidad debería ser Y”.

Hazlo así (modo ingeniero, no magia):

* Define el error: `err = actual_tmax_aligned − predicted_tmax_aligned`
* Entrena un modelo simple que prediga `err` desde features:

  * drift PWS-METAR, MAD, soporte
  * viento (dir/speed), upstream delta, SST onshore flag
  * radiación/sky cover si lo tienes
  * hour-of-day y hours_to_peak
  * health states / staleness

Modelos que encajan muy bien:

* **Ridge/Lasso** (lineal regularizado): te da coeficientes interpretables (“+0.6°F cuando onshore + SST fría”)
* **Gradient Boosting** (XGBoost/LightGBM) solo si ya tienes suficiente data y quieres capturar no linealidades, pero con SHAP para explicación.

Luego:

* `tmax_mean_final = tmax_mean_base + residual_correction`
* y ajustas sigma si detectas que el residual model no es fiable.

Esto mantiene explicabilidad y además encaja con tu estructura de `explanations`/`inputs_used`. 

---

## Auto-aprendizaje aplicado al trading (sin fliparse)

Tu autotrader ya contempla:

* catálogo de estrategias
* bandit LinUCB para seleccionar
* y learning offline con promoción.  

El “siguiente salto” aquí sería:

* que el grid search no solo toque parámetros de estrategia, sino también:

  * thresholds de edge por régimen
  * cooldowns
  * sizing mínimo (tu constraint de $1)
  * condiciones de “range capture” vs “fade”
* y que el criterio de promoción use **PnL neto + drawdown + turnover** además de métricas de predicción (multi-objetivo con mínimos).  


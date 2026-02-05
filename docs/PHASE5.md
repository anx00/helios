# Fase 5 — Backtesting + Calibration Loop (signal + execution)

## 5.0 Objetivo real de la fase

Al terminar Fase 5 debes poder responder, con datos:

1. **¿Mi nowcast está bien calibrado para el juez?**
   No “si acierto Tmax”, sino si **P(bucket)** coincide con frecuencias reales.

2. **¿Hay edge económico neto tras fricción?**
   Señal buena ≠ estrategia rentable si pagas spread/slippage o te comen en maker.

3. **¿Cuándo funciona y cuándo muere?**
   Por estación, por mes, por régimen (onshore wind, advección fuerte, nubes…).

4. **¿Qué parte del sistema está fallando cuando falla?**
   Feed staleness, QC, bias, features, o ejecución.

En la práctica: Fase 5 te produce un “laboratorio científico” para iterar.

---

## 5.1 “Verdad” del backtest: labels y targets correctos

Antes de backtestear, define *qué es verdad*.

### 5.1.1 Target principal (mercado)

* `y_bucket_winner`: el bucket ganador del día (según regla del mercado / juez).
* `y_tmax_aligned`: Tmax alineada al juez (redondeo contractual).
* `y_t_peak_bin` (opcional): bin horario donde ocurrió el máximo (si lo puedes derivar de observaciones).

**Importante:** el sistema debe etiquetar con `America/New_York` para el “día” del mercado. Si el label está corrido, todo se rompe.

### 5.1.2 “Ground truth” meteorológica vs “judge truth”

Tú ya lo tienes claro: el trading optimiza **judge truth**. La meteorología física sirve solo para predecir al juez.

En backtesting, por tanto, separa:

* métricas “contra juez” (las que importan para PnL),
* métricas “contra físico” (solo diagnóstico).

---

## 5.2 Dataset: qué se necesita para backtestear bien

Fase 4 ya te dejó Parquet y replay. Para Fase 5 construyes un “dataset de día” por mercado, con estas tablas (o equivalentes):

### 5.2.1 Tablas mínimas

1. `world_obs`
   METAR, PWS consensus, upstream, con QC y timestamps.

2. `features`
   HRRR/GFS/NBM/LAMP agregados + SST/AOD + derivados (advección, radiación proxy, cloud, etc).

3. `nowcast`
   outputs: `tmax_mean`, `sigma`, `P(bucket)`, `t_peak_bins`, `confidence`, `qc_state`.

4. `market`
   snapshots L2 1s/5s (cone operativo) + best bid/ask + mid + spread + depth.

5. `events`
   ventanas de evento, resyncs, gaps, staleness spikes, etc.

6. `labels`
   winner del día + max aligned + hora/ventana de pico si la derivas.

### 5.2.2 Unificación (lo que hace que el backtest sea fácil)

Construye un “**timeline join**” que, para cada timestamp `t`, te permite recuperar:

* estado del mundo en `t` (último obs válido + QC),
* estado de features vigente,
* output del nowcast en `t`,
* mercado en `t` (bid/ask, depth, etc).

Esto se hace bien con:

* Parquet + DuckDB (joins por asof / last observation carried forward),
* o polars con `join_asof`.

---

## 5.3 Dos backtests distintos: “signal-only” y “execution-aware”

No intentes hacerlo todo a la vez; separa para no engañarte.

### 5.3.1 Backtest A: Signal-only (¿mi modelo está bien?)

Ignora ejecución y simula “podría comprar al mid” o “precio justo”.

Sirve para:

* calibración probabilística,
* ranking de features,
* detectar drift por estación/mes.

### 5.3.2 Backtest B: Execution-aware (¿puedo monetizarlo?)

Incluye:

* spread,
* slippage,
* fill rates (taker vs maker),
* y constraints de inventory/riesgo.

Este es el que decide si hay dinero.

---

## 5.4 Métricas de predicción (las que importan de verdad)

HELIOS V1 se centraba en “acertar” con ajustes físicos. Aquí lo medimos de forma estadística.

### 5.4.1 Calibración de probabilidades (P(bucket))

* **Brier Score** por bucket y global.
* **Log Loss** (si tienes P para todos los buckets).
* **Reliability / calibration curve**: cuando dices 70%, ¿ocurre 70%?
* **ECE (Expected Calibration Error)** para cuantificar descalibración.
* **Sharpness**: no basta calibrar; quieres distribuciones informativas (no siempre 10% todo).

### 5.4.2 Métricas de “tmax_mean”

* MAE/RMSE contra `tmax_aligned` (diagnóstico).
* Error por régimen (viento onshore, high AOD, etc).

### 5.4.3 Métricas de `t_peak_bins`

* Accuracy de bin (si lo etiquetas).
* “mass near truth”: probabilidad asignada al bin correcto ±1 bin.

### 5.4.4 Métricas de estabilidad intradía

Tu problema: “cambia muchísimo a lo largo del día”.
Mide:

* **prediction churn**: suma de |ΔP(bucket)| por hora.
* **flip count**: cuántas veces cambia el bucket con mayor probabilidad.
* relación churn vs resultado (si churn alto suele ser días malos, puedes usarlo como riesgo).

---

## 5.5 Métricas de mercado y microestructura (para explicar PnL)

Si quieres “ser mejor que 95%”, no basta el modelo.

Mide y guarda por día:

* liquidez media en topN,
* spread por régimen,
* “fragility”: cuánto se mueve el mid por poco size,
* volatilidad del price por hora,
* eventos: shocks, locked markets, wide spreads.

Esto te permite segmentar: días donde el mercado es “tradeable” vs no.

---

## 5.6 El núcleo: Policy Backtester (simular decisiones)

En Fase 3 tu modelo produce un `NowcastDistribution`. En Fase 5 defines una **Policy** que decide qué hacer con eso.

### 5.6.1 PolicyTable (lo correcto)

No “compra ya”. La policy produce:

* niveles de valor por bucket (precio máximo para comprar YES, mínimo para vender, etc),
* tamaño (sizing) según confidence, churn, QC, liquidez,
* y régimen maker/taker.

### 5.6.2 Máquina de estados (para evitar overtrading)

Define estados por mercado:

* `NEUTRAL`
* `BUILD_POSITION`
* `HOLD`
* `REDUCE`
* `FADE_EVENT` (solo si QC lo permite)
* `RISK_OFF` (si staleness o condiciones malas)

Esto evita que el bot “persiga” ruido intradía.

### 5.6.3 Inventario y riesgo (desde backtest)

Incluso en sim:

* límites de exposición por bucket,
* max daily loss,
* cooldowns,
* cap de órdenes por minuto.

El objetivo es que el backtest refleje operación realista.

---

## 5.7 Simulación de ejecución (la parte que más engaña si la haces mal)

Aquí es donde la mayoría se autoengaña: “mi señal da dinero” → luego en live no.

### 5.7.1 Modelo Taker (fácil y útil)

Si decides ejecutar como taker:

* compras a best ask (YES) y vendes a best bid (YES),
* incluyes fees y un slippage fijo o proporcional a profundidad.

Esto ya te da un baseline “honesto”.

### 5.7.2 Modelo Maker (aproximado pero no naive)

Maker es más complejo. Para backtest “práctico” necesitas:

* L2 snapshots,
* y si tienes trades, mejor.

Modelo maker MVP:

* si pones una orden a precio P:

  * estimas cuánta cola hay delante (`queue_size_at_P` desde L2),
  * estimas consumo de cola con trades (si hay tape) o con cambios en size en snapshots,
  * fill ocurre si el “consumo acumulado” supera tu cola antes de que canceles/expire.
* Si no tienes tape fiable, usa una aproximación:

  * “si el best bid/ask cruza tu precio” + “la liquidez delante cae por debajo de X” → fill probable.

No es perfecto, pero es mejor que “asumir fill siempre”.

### 5.7.3 Adverse selection proxy

Maker sufre de “te llenan justo antes de moverse en tu contra”.
Mide:

* retorno del precio 1–5–30s después del fill simulado.
  Si es consistentemente negativo, tu maker está siendo “comida”.

---

## 5.8 “Fade trades” con QC: backtest específico

Tú quieres la táctica:

* el mercado se mueve por un print,
* tu sistema cree que ese print no es real,
* entras corto (buy/sell) esperando revert.

Esto solo se puede backtestear con:

* QC flags,
* ventanas de evento,
* y microestructura.

Backtest del “fade module”:

* detecta eventos (METAR/PWS outlier, shock),
* valida que QC realmente marcó outlier,
* simula trade con TTL corto (p.ej., 2–10 min),
* evalúa:

  * hit rate,
  * promedio por trade,
  * worst-case excursion (para stops),
  * dependencia del régimen de liquidez.

Si el QC no es muy bueno, este módulo pierde dinero. Fase 5 lo prueba sin arruinarte.

---

## 5.9 Walk-forward y anti-leakage (para que no te autoengañes)

Si calibras parámetros con el mismo periodo que evalúas, te engañas.

### 5.9.1 Split temporal recomendado

* Entrenamiento/calibración: meses anteriores
* Validación: mes siguiente
* Repite en rolling window

### 5.9.2 No uses features “del futuro”

Parece obvio, pero en meteorología pasa:

* usar HRRR run que salió después del tiempo simulado,
* usar “max del día” cuando aún no ocurrió,
* o usar market states más recientes de lo que el bot habría visto (staleness).

En replay/backtest todo debe respetar `ingest_time_utc` y staleness.

---

## 5.10 Calibration loop (lo que cierra el círculo)

Fase 5 no es solo evaluar; es **ajustar** y re-evaluar.

Parámetros típicos a calibrar (con métodos simples al inicio):

* `bias_alpha` (EMA)
* `decay_horizon_hours` (cuánto dura el bias)
* `sigma_base` y cómo crece con QC/staleness
* peso de advección/upstream
* thresholds QC (MAD multiplier, rate-of-change)
* gating de “fade”
* sizing vs confidence/churn

Método MVP:

* grid search por día/mes,
* optimizando una función objetivo:

  * “calibración + PnL simulado penalizado por drawdown”.

SEARA hablaba de “sweep”; esa idea aplicada aquí es perfecta, pero con datasets Parquet. (No necesitas Rust para esto.)

---

## 5.11 Entregables tangibles de Fase 5 (lo que el equipo debe producir)

1. **CLI de backtesting**
   `backtest --station KLGA --date 2026-01-14 --mode execution-aware --policy v0`

2. **Reporte automático por día** (HTML/MD)

* calibration (Brier/ECE)
* estabilidad (churn)
* resumen de trades sim (PnL, DD, slippage)
* breakdown por régimen

3. **Leaderboard de policies** (por periodo)

* policy v0 vs v1 vs v2
* ranking por robustez (no solo PnL)

4. **Suite de tests de integridad**

* no-leakage checks
* replay equivalence checks
* staleness sanity checks

5. **Panel en tu UI (Amaterasu Console/HELIOS UI)**

* seleccionas un día y ves:

  * timeline,
  * señales,
  * decisiones,
  * fills sim,
    como un “trading simulator”.

---

## 5.12 Definition of Done (cuándo está realmente “hecha”)

Fase 5 está terminada si puedes:

* correr 30–90 días históricos,
* obtener métricas de calibración estables,
* simular ejecución con fricción razonable,
* identificar claramente qué subset de días te da edge,
* y producir un reporte que un tercero pueda auditar.

---

## Stack recomendado (con tu enfoque actual)

Sin cambiarte a Rust:

* **Parquet + DuckDB** para queries rápidas (join asof, slicing).
* **polars** para pipelines de features y métricas.
* **Python** para policy engine y simulación.
* El “execution core real” vendrá después (Fase 6+), pero aquí ya lo simulas.

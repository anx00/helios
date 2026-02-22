# Backtesting + Calibration Implementation (signal + execution)

## 1. Objetivo real de implementacion

Al completar esta parte de la implementacion debes poder responder, con datos:

1. **Â¿Mi nowcast estÃ¡ bien calibrado para el juez?**
   No â€œsi acierto Tmaxâ€, sino si **P(bucket)** coincide con frecuencias reales.

2. **Â¿Hay edge econÃ³mico neto tras fricciÃ³n?**
   SeÃ±al buena â‰  estrategia rentable si pagas spread/slippage o te comen en maker.

3. **Â¿CuÃ¡ndo funciona y cuÃ¡ndo muere?**
   Por estaciÃ³n, por mes, por rÃ©gimen (onshore wind, advecciÃ³n fuerte, nubesâ€¦).

4. **Â¿QuÃ© parte del sistema estÃ¡ fallando cuando falla?**
   Feed staleness, QC, bias, features, o ejecuciÃ³n.

En la prÃ¡ctica: esto te produce un â€œlaboratorio cientÃ­ficoâ€ para iterar.

---

## 5.1 â€œVerdadâ€ del backtest: labels y targets correctos

Antes de backtestear, define *quÃ© es verdad*.

### 5.1.1 Target principal (mercado)

* `y_bucket_winner`: el bucket ganador del dÃ­a (segÃºn regla del mercado / juez).
* `y_tmax_aligned`: Tmax alineada al juez (redondeo contractual).
* `y_t_peak_bin` (opcional): bin horario donde ocurriÃ³ el mÃ¡ximo (si lo puedes derivar de observaciones).

**Importante:** el sistema debe etiquetar con `America/New_York` para el â€œdÃ­aâ€ del mercado. Si el label estÃ¡ corrido, todo se rompe.

### 5.1.2 â€œGround truthâ€ meteorolÃ³gica vs â€œjudge truthâ€

TÃº ya lo tienes claro: el trading optimiza **judge truth**. La meteorologÃ­a fÃ­sica sirve solo para predecir al juez.

En backtesting, por tanto, separa:

* mÃ©tricas â€œcontra juezâ€ (las que importan para PnL),
* mÃ©tricas â€œcontra fÃ­sicoâ€ (solo diagnÃ³stico).

---

## 5.2 Dataset: quÃ© se necesita para backtestear bien

Fase 4 ya te dejÃ³ Parquet y replay. Para Fase 5 construyes un â€œdataset de dÃ­aâ€ por mercado, con estas tablas (o equivalentes):

### 5.2.1 Tablas mÃ­nimas

1. `world_obs`
   METAR, PWS consensus, upstream, con QC y timestamps.

2. `features`
   HRRR/GFS/NBM/LAMP agregados + SST/AOD + derivados (advecciÃ³n, radiaciÃ³n proxy, cloud, etc).

3. `nowcast`
   outputs: `tmax_mean`, `sigma`, `P(bucket)`, `t_peak_bins`, `confidence`, `qc_state`.

4. `market`
   snapshots L2 1s/5s (cone operativo) + best bid/ask + mid + spread + depth.

5. `events`
   ventanas de evento, resyncs, gaps, staleness spikes, etc.

6. `labels`
   winner del dÃ­a + max aligned + hora/ventana de pico si la derivas.

### 5.2.2 UnificaciÃ³n (lo que hace que el backtest sea fÃ¡cil)

Construye un â€œ**timeline join**â€ que, para cada timestamp `t`, te permite recuperar:

* estado del mundo en `t` (Ãºltimo obs vÃ¡lido + QC),
* estado de features vigente,
* output del nowcast en `t`,
* mercado en `t` (bid/ask, depth, etc).

Esto se hace bien con:

* Parquet + DuckDB (joins por asof / last observation carried forward),
* o polars con `join_asof`.

---

## 5.3 Dos backtests distintos: â€œsignal-onlyâ€ y â€œexecution-awareâ€

No intentes hacerlo todo a la vez; separa para no engaÃ±arte.

### 5.3.1 Backtest A: Signal-only (Â¿mi modelo estÃ¡ bien?)

Ignora ejecuciÃ³n y simula â€œpodrÃ­a comprar al midâ€ o â€œprecio justoâ€.

Sirve para:

* calibraciÃ³n probabilÃ­stica,
* ranking de features,
* detectar drift por estaciÃ³n/mes.

### 5.3.2 Backtest B: Execution-aware (Â¿puedo monetizarlo?)

Incluye:

* spread,
* slippage,
* fill rates (taker vs maker),
* y constraints de inventory/riesgo.

Este es el que decide si hay dinero.

---

## 5.4 MÃ©tricas de predicciÃ³n (las que importan de verdad)

HELIOS V1 se centraba en â€œacertarâ€ con ajustes fÃ­sicos. AquÃ­ lo medimos de forma estadÃ­stica.

### 5.4.1 CalibraciÃ³n de probabilidades (P(bucket))

* **Brier Score** por bucket y global.
* **Log Loss** (si tienes P para todos los buckets).
* **Reliability / calibration curve**: cuando dices 70%, Â¿ocurre 70%?
* **ECE (Expected Calibration Error)** para cuantificar descalibraciÃ³n.
* **Sharpness**: no basta calibrar; quieres distribuciones informativas (no siempre 10% todo).

### 5.4.2 MÃ©tricas de â€œtmax_meanâ€

* MAE/RMSE contra `tmax_aligned` (diagnÃ³stico).
* Error por rÃ©gimen (viento onshore, high AOD, etc).

### 5.4.3 MÃ©tricas de `t_peak_bins`

* Accuracy de bin (si lo etiquetas).
* â€œmass near truthâ€: probabilidad asignada al bin correcto Â±1 bin.

### 5.4.4 MÃ©tricas de estabilidad intradÃ­a

Tu problema: â€œcambia muchÃ­simo a lo largo del dÃ­aâ€.
Mide:

* **prediction churn**: suma de |Î”P(bucket)| por hora.
* **flip count**: cuÃ¡ntas veces cambia el bucket con mayor probabilidad.
* relaciÃ³n churn vs resultado (si churn alto suele ser dÃ­as malos, puedes usarlo como riesgo).

---

## 5.5 MÃ©tricas de mercado y microestructura (para explicar PnL)

Si quieres â€œser mejor que 95%â€, no basta el modelo.

Mide y guarda por dÃ­a:

* liquidez media en topN,
* spread por rÃ©gimen,
* â€œfragilityâ€: cuÃ¡nto se mueve el mid por poco size,
* volatilidad del price por hora,
* eventos: shocks, locked markets, wide spreads.

Esto te permite segmentar: dÃ­as donde el mercado es â€œtradeableâ€ vs no.

---

## 5.6 El nÃºcleo: Policy Backtester (simular decisiones)

En Fase 3 tu modelo produce un `NowcastDistribution`. En Fase 5 defines una **Policy** que decide quÃ© hacer con eso.

### 5.6.1 PolicyTable (lo correcto)

No â€œcompra yaâ€. La policy produce:

* niveles de valor por bucket (precio mÃ¡ximo para comprar YES, mÃ­nimo para vender, etc),
* tamaÃ±o (sizing) segÃºn confidence, churn, QC, liquidez,
* y rÃ©gimen maker/taker.

### 5.6.2 MÃ¡quina de estados (para evitar overtrading)

Define estados por mercado:

* `NEUTRAL`
* `BUILD_POSITION`
* `HOLD`
* `REDUCE`
* `FADE_EVENT` (solo si QC lo permite)
* `RISK_OFF` (si staleness o condiciones malas)

Esto evita que el bot â€œpersigaâ€ ruido intradÃ­a.

### 5.6.3 Inventario y riesgo (desde backtest)

Incluso en sim:

* lÃ­mites de exposiciÃ³n por bucket,
* max daily loss,
* cooldowns,
* cap de Ã³rdenes por minuto.

El objetivo es que el backtest refleje operaciÃ³n realista.

---

## 5.7 SimulaciÃ³n de ejecuciÃ³n (la parte que mÃ¡s engaÃ±a si la haces mal)

AquÃ­ es donde la mayorÃ­a se autoengaÃ±a: â€œmi seÃ±al da dineroâ€ â†’ luego en live no.

### 5.7.1 Modelo Taker (fÃ¡cil y Ãºtil)

Si decides ejecutar como taker:

* compras a best ask (YES) y vendes a best bid (YES),
* incluyes fees y un slippage fijo o proporcional a profundidad.

Esto ya te da un baseline â€œhonestoâ€.

### 5.7.2 Modelo Maker (aproximado pero no naive)

Maker es mÃ¡s complejo. Para backtest â€œprÃ¡cticoâ€ necesitas:

* L2 snapshots,
* y si tienes trades, mejor.

Modelo maker MVP:

* si pones una orden a precio P:

  * estimas cuÃ¡nta cola hay delante (`queue_size_at_P` desde L2),
  * estimas consumo de cola con trades (si hay tape) o con cambios en size en snapshots,
  * fill ocurre si el â€œconsumo acumuladoâ€ supera tu cola antes de que canceles/expire.
* Si no tienes tape fiable, usa una aproximaciÃ³n:

  * â€œsi el best bid/ask cruza tu precioâ€ + â€œla liquidez delante cae por debajo de Xâ€ â†’ fill probable.

No es perfecto, pero es mejor que â€œasumir fill siempreâ€.

### 5.7.3 Adverse selection proxy

Maker sufre de â€œte llenan justo antes de moverse en tu contraâ€.
Mide:

* retorno del precio 1â€“5â€“30s despuÃ©s del fill simulado.
  Si es consistentemente negativo, tu maker estÃ¡ siendo â€œcomidaâ€.

---

## 5.8 â€œFade tradesâ€ con QC: backtest especÃ­fico

TÃº quieres la tÃ¡ctica:

* el mercado se mueve por un print,
* tu sistema cree que ese print no es real,
* entras corto (buy/sell) esperando revert.

Esto solo se puede backtestear con:

* QC flags,
* ventanas de evento,
* y microestructura.

Backtest del â€œfade moduleâ€:

* detecta eventos (METAR/PWS outlier, shock),
* valida que QC realmente marcÃ³ outlier,
* simula trade con TTL corto (p.ej., 2â€“10 min),
* evalÃºa:

  * hit rate,
  * promedio por trade,
  * worst-case excursion (para stops),
  * dependencia del rÃ©gimen de liquidez.

Si el QC no es muy bueno, este mÃ³dulo pierde dinero. Fase 5 lo prueba sin arruinarte.

---

## 5.9 Walk-forward y anti-leakage (para que no te autoengaÃ±es)

Si calibras parÃ¡metros con el mismo periodo que evalÃºas, te engaÃ±as.

### 5.9.1 Split temporal recomendado

* Entrenamiento/calibraciÃ³n: meses anteriores
* ValidaciÃ³n: mes siguiente
* Repite en rolling window

### 5.9.2 No uses features â€œdel futuroâ€

Parece obvio, pero en meteorologÃ­a pasa:

* usar HRRR run que saliÃ³ despuÃ©s del tiempo simulado,
* usar â€œmax del dÃ­aâ€ cuando aÃºn no ocurriÃ³,
* o usar market states mÃ¡s recientes de lo que el bot habrÃ­a visto (staleness).

En replay/backtest todo debe respetar `ingest_time_utc` y staleness.

---

## 5.10 Calibration loop (lo que cierra el cÃ­rculo)

Fase 5 no es solo evaluar; es **ajustar** y re-evaluar.

ParÃ¡metros tÃ­picos a calibrar (con mÃ©todos simples al inicio):

* `bias_alpha` (EMA)
* `decay_horizon_hours` (cuÃ¡nto dura el bias)
* `sigma_base` y cÃ³mo crece con QC/staleness
* peso de advecciÃ³n/upstream
* thresholds QC (MAD multiplier, rate-of-change)
* gating de â€œfadeâ€
* sizing vs confidence/churn

MÃ©todo MVP:

* grid search por dÃ­a/mes,
* optimizando una funciÃ³n objetivo:

  * â€œcalibraciÃ³n + PnL simulado penalizado por drawdownâ€.

SEARA hablaba de â€œsweepâ€; esa idea aplicada aquÃ­ es perfecta, pero con datasets Parquet. (No necesitas Rust para esto.)

---

## 5.11 Entregables tangibles de Fase 5 (lo que el equipo debe producir)

1. **CLI de backtesting**
   `backtest --station KLGA --date 2026-01-14 --mode execution-aware --policy v0`

2. **Reporte automÃ¡tico por dÃ­a** (HTML/MD)

* calibration (Brier/ECE)
* estabilidad (churn)
* resumen de trades sim (PnL, DD, slippage)
* breakdown por rÃ©gimen

3. **Leaderboard de policies** (por periodo)

* policy v0 vs v1 vs v2
* ranking por robustez (no solo PnL)

4. **Suite de tests de integridad**

* no-leakage checks
* replay equivalence checks
* staleness sanity checks

5. **Panel en tu UI (Amaterasu Console/HELIOS UI)**

* seleccionas un dÃ­a y ves:

  * timeline,
  * seÃ±ales,
  * decisiones,
  * fills sim,
    como un â€œtrading simulatorâ€.

---

## 5.12 Definition of Done (cuÃ¡ndo estÃ¡ realmente â€œhechaâ€)

Fase 5 estÃ¡ terminada si puedes:

* correr 30â€“90 dÃ­as histÃ³ricos,
* obtener mÃ©tricas de calibraciÃ³n estables,
* simular ejecuciÃ³n con fricciÃ³n razonable,
* identificar claramente quÃ© subset de dÃ­as te da edge,
* y producir un reporte que un tercero pueda auditar.

---

## Stack recomendado (con tu enfoque actual)

Sin cambiarte a Rust:

* **Parquet + DuckDB** para queries rÃ¡pidas (join asof, slicing).
* **polars** para pipelines de features y mÃ©tricas.
* **Python** para policy engine y simulaciÃ³n.
* El â€œexecution core realâ€ vendrÃ¡ despuÃ©s (Fase 6+), pero aquÃ­ ya lo simulas.




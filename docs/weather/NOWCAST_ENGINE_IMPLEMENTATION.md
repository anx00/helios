# Nowcast Engine Implementation (v0)

## 1. Objetivo de implementacion (que debe quedar operativo)

Al completar esta parte de la implementacion, HELIOS V1 debe poder:

1. Tomar en tiempo real:

   * `OfficialObs` (METAR alineado al juez)
   * `AuxObs` (PWS consenso + upstream)
   * `EnvFeatures` (HRRR/GFS/etc + SST/AOD/cloud/radiation proxies)
   * `QCState`
   * `MarketState` (del orderbook mirror)

2. Producir en tiempo real:

   * **P(bucket)** en el *cone operativo* (Â±2 del bracket objetivo)
   * **P(â‰¥strike)** (si prefieres â€œcurva acumuladaâ€)
   * **t_peak_distribution** (bins por hora o por ventanas de 2h)
   * **confidence** y **valid_until**
   * **explicabilidad mÃ­nima** (quÃ© inputs han movido la predicciÃ³n)

3. Mostrarlo en la UI con latencia baja:

   * series de evoluciÃ³n de P(bucket)
   * evoluciÃ³n del **bias** (delta HRRR vs realidad)
   * estado de QC y por quÃ© afecta a la confianza

> Resultado tangible: puedes mirar la UI y ver cÃ³mo la distribuciÃ³n se reajusta con cada METAR y con cambios fÃ­sicos (viento/advecciÃ³n/etc.) sin â€œsaltos absurdosâ€.

---

## 3.1 Cambio clave: dejar de â€œpredecir Tmaxâ€ y pasar a â€œpredecir bucketsâ€

HELIOS V1 tiene una predicciÃ³n tipo **Tmax puntual** con ajustes fÃ­sicos y un *reality floor* basado en el mercado. Eso estÃ¡ bien como â€œcerebroâ€, pero para operar mercados por intervalos, el output correcto es:

* **DistribuciÃ³n de Tmax** (en Â°C/Â°F) con incertidumbre
  â†’ y de esa distribuciÃ³n derivar:
* **P(bucket)**, que es lo que realmente se monetiza.

### ImplementaciÃ³n prÃ¡ctica (v0)

* MantÃ©n un estimador central: `Tmax_mean` (tu mejor estimaciÃ³n)
* MantÃ©n una incertidumbre: `sigma` (dinÃ¡mica)
* Asume distribuciÃ³n simple (normal truncada o logÃ­stica), al menos en v0.

Luego:

* `P(bucket_i)` = integral de la densidad en el rango del bucket
* `P(â‰¥strike)` = cola derecha

Esto te permite:

* calibrar con Brier score en Fase 4/5,
* dimensionar (sizing) y diseÃ±ar policies mÃ¡s adelante.

---

## 3.2 Arquitectura interna del Nowcast Engine (v0)

### 3.2.1 Estado interno (â€œDailyStateâ€)

El motor mantiene un `DailyState` por mercado (KLGA), con:

* `max_so_far_aligned` (mÃ¡ximo ya observado alineado al juez)
* `last_obs_aligned` (Ãºltima temp alineada y su timestamp)
* `bias_state` (error del modelo base vs realidad)
* `trend_state` (tasa de cambio, suavizada)
* `t_peak_state` (distribuciÃ³n por bins)
* `uncertainty_state` (`sigma`, aumenta con QC/ruido, baja con consistencia)
* `market_constraints` (reality floor y posibles â€œimpossiblesâ€)
* `health` (staleness de fuentes, flags QC)

**Regla de oro**: `DailyState` se actualiza **event-driven**. No es un â€œbatch recomputeâ€ que recalcula todo cada ciclo.

---

### 3.2.2 Dos motores, no uno (para estabilidad)

Para que no te pase lo de â€œla predicciÃ³n cambia muchÃ­simo a lo largo del dÃ­aâ€ (lo que tÃº ya has visto), separa:

1. **Base Forecast Layer** (lento y suave)

* HRRR (y/o ensemble) define una curva â€œde fondoâ€ para el dÃ­a.
* Se actualiza cuando llega nuevo HRRR (o cada X tiempo).
* Es estable.

2. **Nowcast Adjustment Layer** (rÃ¡pido, reacciona a observaciones)

* Bias + advecciÃ³n + features intradÃ­a ajustan la salida.
* Se actualiza con cada METAR/PWS/feature relevante.

Esto evita que el motor â€œboteâ€ cada vez que una fuente se mueve.

---

## 3.3 ConstrucciÃ³n del â€œBase Forecastâ€ (reutilizando HELIOS V1)

HELIOS ya usa Open-Meteo HRRR/GFS/NBM/LAMP. En Fase 3 no hace falta inventar mÃ¡s: solo cambia el **contrato** y el **cÃ³mo lo usas**.

### BaseForecast v0

* `T(t)` para las prÃ³ximas 24h (o hasta el cierre del mercado)
* Extraer:

  * `Tmax_base` = mÃ¡ximo de `T(t)`
  * `t_peak_base` = argmax(t)
  * features asociadas al pico: radiaciÃ³n, nubes, viento, humedad suelo

### Mejoras de ingenierÃ­a

* Cachea el `BaseForecast` con:

  * `model_run_time` (timestamp del HRRR run)
  * `ingest_time_utc`
  * `valid_window`
* No recalcules BaseForecast en el hot loop.

---

## 3.4 Bias State (la pieza crÃ­tica que ya tienes, pero hay que â€œendurecerâ€)

TÃº describes: â€œHRRR dice 50Â°F, realidad 52Â°F â†’ Delta +2Â°F; decaimiento temporalâ€.

En Fase 3 esto se convierte en un â€œestadoâ€ con filtro:

### Bias v0 recomendado

* `bias = EMA(bias, obs - model_now)`
* `alpha` dinÃ¡mico:

  * si QC es OK â†’ alpha mÃ¡s alto (reaccionas)
  * si QC es incierto â†’ alpha bajo (no te dejas engaÃ±ar)

### Decaimiento a futuro (â€œpropagar biasâ€)

En vez de meter fÃ­sica compleja:

* `T_adj(t) = T_base(t) + bias * decay(t)`
* `decay(t)` cae a 0 en 4â€“8 horas (configurable)

Esto es muy parecido a lo que ya habÃ­as definido en discusiones previas para â€œno complicarse con matemÃ¡ticasâ€.

---

## 3.5 t_peak_distribution (para tu caso â€œpico por la nocheâ€)

Tu ejemplo es clave: a veces el mÃ¡ximo no es al mediodÃ­a. Por eso `t_peak` no puede ser un nÃºmero fijo.

### t_peak v0 (simple y Ãºtil)

* Bins por hora o por ventanas de 2h (p. ej. 00â€“02, 02â€“04â€¦)
* Asigna probabilidad en base a:

  * curva ajustada `T_adj(t)` (mÃ¡s peso donde la temperatura sea alta)
  * advecciÃ³n (si viene masa de aire cÃ¡lida a Ãºltima hora, mueve masa de probabilidad hacia tarde)
  * rÃ©gimen de nubes/radiaciÃ³n (si hay clearing tardÃ­o, desplaza)
* La salida es:

  * `P(t_peak in bin_k)`

### Para quÃ© sirve

* â€œReachabilityâ€: si faltan 2 horas y tu modelo sugiere +4Â°C aÃºn, baja probabilidad.
* Trading posterior: ayuda a decidir si un movimiento del mercado es prematuro.

---

## 3.6 IntegraciÃ³n con â€œReality Floorâ€ (mercado) sin contaminar el modelo

HELIOS tiene un concepto bueno: â€œsi el mercado ya pagÃ³ un bracket, bloquea rangosâ€.

En Fase 3 lo harÃ­a mÃ¡s limpio:

* `market_floor`: rangos imposibles por observaciÃ³n ya alcanzada (no por precio)
* `market_sanity`: el precio del mercado no debe alterar tu predicciÃ³n fÃ­sica, solo tu **confidence/sizing** mÃ¡s adelante.

**Regla para v0**:

* El mercado *no* cambia `Tmax_mean`.
* El mercado puede afectar:

  * `confidence` (si estÃ¡s muy divergente)
  * el â€œconeâ€ que decides mirar
  * (en Fase 5+) ejecuciÃ³n/sizing

Esto evita que el modelo se vuelva circular (â€œme creo al mercado porque el mercado diceâ€¦â€).

---

## 3.7 Reglas de actualizaciÃ³n (cadencia)

El motor se actualiza con dos triggers:

### Trigger A â€” periÃ³dico (cada 60s)

* recalcula outputs (P(bucket), t_peak dist, confidence) desde el estado actual
* sin hacer heavy fetch

### Trigger B â€” event-driven inmediato

* nuevo METAR (oficial)
* QC outlier o cambio de estado
* cambio fuerte de microestructura (mid shock / spread regime)
* llegada de HRRR run nuevo (actualiza BaseForecast)

Esto hace que reacciones cuando importa, sin churn.

---

## 3.8 Output del motor (contrato de salida)

Define un `NowcastDistribution` con:

* `ts_generated_utc`, `ts_nyc`, `ts_es`
* `market_id`, `station_id`
* `tmax_mean`, `tmax_sigma`
* `p_bucket`: lista por bucket (solo cone)
* `p_ge_strike`: opcional
* `t_peak_bins`: lista {bin_start, bin_end, prob}
* `confidence`: 0â€“1
* `explanations`: top 3 features que mÃ¡s han movido (bias, advecciÃ³n, cloud, SST, etc.)
* `valid_until_utc`: TTL

Esta salida se manda por WS/SSE a tu UI y se persiste (asÃ­ncrono) para replay.

---

## 3.9 UX: quÃ© cambia en la interfaz en Fase 3

AÃ±ades dos pantallas (o dos paneles):

### A) Nowcast Live

* GrÃ¡fico de P(bucket) en el cone
* Serie temporal de `tmax_mean` y `sigma`
* Serie de `bias`
* t_peak_distribution como histogram

### B) â€œModel Debugâ€

* Ãºltima observaciÃ³n oficial (METAR) vs base forecast
* componentes: `T_base`, `bias_adjustment`, `advection_adjustment`, etc.
* QC flags y cÃ³mo afectan alpha/sigma

Esto es lo que hace que HELIOS pase de â€œbonitoâ€ a â€œdepurableâ€.

---

## 3.10 Criterios de salida (Definition of Done)

Para dar por terminada Fase 3:

1. **Estabilidad**

* La distribuciÃ³n cambia de forma suave salvo eventos reales (METAR, shock).
* No hay jitter continuo.

2. **Coherencia con juez**

* `max_so_far_aligned` nunca decrece.
* `P(bucket < max_so_far)` tiende a 0.

3. **Robustez**

* Si se cae PWS o AOD, el motor sigue (confidence baja, pero no muere).
* Si METAR estÃ¡ stale, se refleja.

4. **Trazabilidad**

* Cada output tiene: inputs usados + timestamps + QC state.

5. **UX**

* Panel Nowcast responde en real-time (push) sin depender de queries pesadas.

---

## 3.11 QuÃ© NO harÃ­a todavÃ­a (para no liarla)

* ML complejo (XGBoost/NN) antes de tener replay/backtest serio.
* Ajustes fÃ­sicos hiperdetallados en caliente que bloqueen el loop.
* Trading real (eso viene despuÃ©s, cuando tengas validaciÃ³n).

---

## Referencias de tus proyectos / componentes que ya lo inspiran

* **HELIOS V1** ya contiene:

  * Bias/Deviation concept + advecciÃ³n + SST/AOD/upstream (lo reutilizas).
* **AMATERATSU** formaliza:

  * salida como distribuciÃ³n P(bucket) + t_peak bins + confidence + event-driven updates.
* **SEARA** aporta:

  * disciplina del hot-path (estado soberano, feed correcto, instrumentos de latencia) que debes mantener fuera del compute.




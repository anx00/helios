  # Fase 3 — Nowcasting Engine v0 (predicción operable, sin trading)

## 3.0 Objetivo de la fase (qué tiene que quedar “vivo” al final)

Al terminar Fase 3, HELIOS V1 debe poder:

1. Tomar en tiempo real:

   * `OfficialObs` (METAR alineado al juez)
   * `AuxObs` (PWS consenso + upstream)
   * `EnvFeatures` (HRRR/GFS/etc + SST/AOD/cloud/radiation proxies)
   * `QCState`
   * `MarketState` (del orderbook mirror)

2. Producir en tiempo real:

   * **P(bucket)** en el *cone operativo* (±2 del bracket objetivo)
   * **P(≥strike)** (si prefieres “curva acumulada”)
   * **t_peak_distribution** (bins por hora o por ventanas de 2h)
   * **confidence** y **valid_until**
   * **explicabilidad mínima** (qué inputs han movido la predicción)

3. Mostrarlo en la UI con latencia baja:

   * series de evolución de P(bucket)
   * evolución del **bias** (delta HRRR vs realidad)
   * estado de QC y por qué afecta a la confianza

> Resultado tangible: puedes mirar la UI y ver cómo la distribución se reajusta con cada METAR y con cambios físicos (viento/advección/etc.) sin “saltos absurdos”.

---

## 3.1 Cambio clave: dejar de “predecir Tmax” y pasar a “predecir buckets”

HELIOS V1 tiene una predicción tipo **Tmax puntual** con ajustes físicos y un *reality floor* basado en el mercado. Eso está bien como “cerebro”, pero para operar mercados por intervalos, el output correcto es:

* **Distribución de Tmax** (en °C/°F) con incertidumbre
  → y de esa distribución derivar:
* **P(bucket)**, que es lo que realmente se monetiza.

### Implementación práctica (v0)

* Mantén un estimador central: `Tmax_mean` (tu mejor estimación)
* Mantén una incertidumbre: `sigma` (dinámica)
* Asume distribución simple (normal truncada o logística), al menos en v0.

Luego:

* `P(bucket_i)` = integral de la densidad en el rango del bucket
* `P(≥strike)` = cola derecha

Esto te permite:

* calibrar con Brier score en Fase 4/5,
* dimensionar (sizing) y diseñar policies más adelante.

---

## 3.2 Arquitectura interna del Nowcast Engine (v0)

### 3.2.1 Estado interno (“DailyState”)

El motor mantiene un `DailyState` por mercado (KLGA), con:

* `max_so_far_aligned` (máximo ya observado alineado al juez)
* `last_obs_aligned` (última temp alineada y su timestamp)
* `bias_state` (error del modelo base vs realidad)
* `trend_state` (tasa de cambio, suavizada)
* `t_peak_state` (distribución por bins)
* `uncertainty_state` (`sigma`, aumenta con QC/ruido, baja con consistencia)
* `market_constraints` (reality floor y posibles “impossibles”)
* `health` (staleness de fuentes, flags QC)

**Regla de oro**: `DailyState` se actualiza **event-driven**. No es un “batch recompute” que recalcula todo cada ciclo.

---

### 3.2.2 Dos motores, no uno (para estabilidad)

Para que no te pase lo de “la predicción cambia muchísimo a lo largo del día” (lo que tú ya has visto), separa:

1. **Base Forecast Layer** (lento y suave)

* HRRR (y/o ensemble) define una curva “de fondo” para el día.
* Se actualiza cuando llega nuevo HRRR (o cada X tiempo).
* Es estable.

2. **Nowcast Adjustment Layer** (rápido, reacciona a observaciones)

* Bias + advección + features intradía ajustan la salida.
* Se actualiza con cada METAR/PWS/feature relevante.

Esto evita que el motor “bote” cada vez que una fuente se mueve.

---

## 3.3 Construcción del “Base Forecast” (reutilizando HELIOS V1)

HELIOS ya usa Open-Meteo HRRR/GFS/NBM/LAMP. En Fase 3 no hace falta inventar más: solo cambia el **contrato** y el **cómo lo usas**.

### BaseForecast v0

* `T(t)` para las próximas 24h (o hasta el cierre del mercado)
* Extraer:

  * `Tmax_base` = máximo de `T(t)`
  * `t_peak_base` = argmax(t)
  * features asociadas al pico: radiación, nubes, viento, humedad suelo

### Mejoras de ingeniería

* Cachea el `BaseForecast` con:

  * `model_run_time` (timestamp del HRRR run)
  * `ingest_time_utc`
  * `valid_window`
* No recalcules BaseForecast en el hot loop.

---

## 3.4 Bias State (la pieza crítica que ya tienes, pero hay que “endurecer”)

Tú describes: “HRRR dice 50°F, realidad 52°F → Delta +2°F; decaimiento temporal”.

En Fase 3 esto se convierte en un “estado” con filtro:

### Bias v0 recomendado

* `bias = EMA(bias, obs - model_now)`
* `alpha` dinámico:

  * si QC es OK → alpha más alto (reaccionas)
  * si QC es incierto → alpha bajo (no te dejas engañar)

### Decaimiento a futuro (“propagar bias”)

En vez de meter física compleja:

* `T_adj(t) = T_base(t) + bias * decay(t)`
* `decay(t)` cae a 0 en 4–8 horas (configurable)

Esto es muy parecido a lo que ya habías definido en discusiones previas para “no complicarse con matemáticas”.

---

## 3.5 t_peak_distribution (para tu caso “pico por la noche”)

Tu ejemplo es clave: a veces el máximo no es al mediodía. Por eso `t_peak` no puede ser un número fijo.

### t_peak v0 (simple y útil)

* Bins por hora o por ventanas de 2h (p. ej. 00–02, 02–04…)
* Asigna probabilidad en base a:

  * curva ajustada `T_adj(t)` (más peso donde la temperatura sea alta)
  * advección (si viene masa de aire cálida a última hora, mueve masa de probabilidad hacia tarde)
  * régimen de nubes/radiación (si hay clearing tardío, desplaza)
* La salida es:

  * `P(t_peak in bin_k)`

### Para qué sirve

* “Reachability”: si faltan 2 horas y tu modelo sugiere +4°C aún, baja probabilidad.
* Trading posterior: ayuda a decidir si un movimiento del mercado es prematuro.

---

## 3.6 Integración con “Reality Floor” (mercado) sin contaminar el modelo

HELIOS tiene un concepto bueno: “si el mercado ya pagó un bracket, bloquea rangos”.

En Fase 3 lo haría más limpio:

* `market_floor`: rangos imposibles por observación ya alcanzada (no por precio)
* `market_sanity`: el precio del mercado no debe alterar tu predicción física, solo tu **confidence/sizing** más adelante.

**Regla para v0**:

* El mercado *no* cambia `Tmax_mean`.
* El mercado puede afectar:

  * `confidence` (si estás muy divergente)
  * el “cone” que decides mirar
  * (en Fase 5+) ejecución/sizing

Esto evita que el modelo se vuelva circular (“me creo al mercado porque el mercado dice…”).

---

## 3.7 Reglas de actualización (cadencia)

El motor se actualiza con dos triggers:

### Trigger A — periódico (cada 60s)

* recalcula outputs (P(bucket), t_peak dist, confidence) desde el estado actual
* sin hacer heavy fetch

### Trigger B — event-driven inmediato

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
* `confidence`: 0–1
* `explanations`: top 3 features que más han movido (bias, advección, cloud, SST, etc.)
* `valid_until_utc`: TTL

Esta salida se manda por WS/SSE a tu UI y se persiste (asíncrono) para replay.

---

## 3.9 UX: qué cambia en la interfaz en Fase 3

Añades dos pantallas (o dos paneles):

### A) Nowcast Live

* Gráfico de P(bucket) en el cone
* Serie temporal de `tmax_mean` y `sigma`
* Serie de `bias`
* t_peak_distribution como histogram

### B) “Model Debug”

* última observación oficial (METAR) vs base forecast
* componentes: `T_base`, `bias_adjustment`, `advection_adjustment`, etc.
* QC flags y cómo afectan alpha/sigma

Esto es lo que hace que HELIOS pase de “bonito” a “depurable”.

---

## 3.10 Criterios de salida (Definition of Done)

Para dar por terminada Fase 3:

1. **Estabilidad**

* La distribución cambia de forma suave salvo eventos reales (METAR, shock).
* No hay jitter continuo.

2. **Coherencia con juez**

* `max_so_far_aligned` nunca decrece.
* `P(bucket < max_so_far)` tiende a 0.

3. **Robustez**

* Si se cae PWS o AOD, el motor sigue (confidence baja, pero no muere).
* Si METAR está stale, se refleja.

4. **Trazabilidad**

* Cada output tiene: inputs usados + timestamps + QC state.

5. **UX**

* Panel Nowcast responde en real-time (push) sin depender de queries pesadas.

---

## 3.11 Qué NO haría todavía (para no liarla)

* ML complejo (XGBoost/NN) antes de tener replay/backtest serio.
* Ajustes físicos hiperdetallados en caliente que bloqueen el loop.
* Trading real (eso viene después, cuando tengas validación).

---

## Referencias de tus proyectos / componentes que ya lo inspiran

* **HELIOS V1** ya contiene:

  * Bias/Deviation concept + advección + SST/AOD/upstream (lo reutilizas).
* **AMATERATSU** formaliza:

  * salida como distribución P(bucket) + t_peak bins + confidence + event-driven updates.
* **SEARA** aporta:

  * disciplina del hot-path (estado soberano, feed correcto, instrumentos de latencia) que debes mantener fuera del compute.

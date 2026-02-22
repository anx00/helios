# Resumen de mejoras Nowcast (HELIOS)

Fecha: 2026-02-01

## Objetivo
Hacer que el Nowcast sea consistente con HRRR y con el juez (settlement), evitando desalineaciones horarias, floors incorrectos y estados diarios mezclados.

## Cambios clave
- **BaseForecast HRRR conectado**: el nowcast recibe la curva horaria diaria en cada ciclo de predicción y en el endpoint manual.
- **Target day correcto**: el `DailyState` se crea usando el día de settlement de la observación (evita saltos a medianoche).
- **Timezone por estación**: peak hour, sigma y t_peak usan la zona real de la estación.
- **Raw vs aligned**: se guardan máximos y últimas obs en raw y judge‑aligned; los buckets usan aligned.
- **Staleness real**: penaliza sigma si METAR está viejo y marca la salud de la fuente.
- **Distribución consistente**: P(T>=strike) respeta `distribution_type`.
- **Validación BaseForecast**: se ignora si está stale o corresponde a otro día.
- **Clamps físicos**: asegura límites mínimos y máximos configurados.
- **HRRR horario corregido**: usa la hora local del payload para indexar current/soil/radiation.
- **Compatibilidad**: alias `fetch_metar_race` para endpoints existentes.

## Archivos tocados
- `helios-temperature/core/nowcast_engine.py`
- `helios-temperature/core/nowcast_models.py`
- `helios-temperature/core/nowcast_integration.py`
- `helios-temperature/collector/hrrr_fetcher.py`
- `helios-temperature/collector/metar_fetcher.py`
- `helios-temperature/main.py`
- `helios-temperature/web_server.py`

## Riesgos / Consideraciones
- Las etiquetas de `t_peak_bins` siguen en NYC/ES; si habilitas estaciones fuera de NYC conviene ajustar labels.
- Si Polymarket define settlement fuera de NYC para mercados no‑US, habría que alinear `target_date` a la zona local.

## Verificación rápida
1) Llama `/api/v3/nowcast/{station_id}/update` y verifica que `base_forecast_source` sea HRRR.
2) Revisa `/api/v3/nowcast/{station_id}/state` y confirma `max_so_far_f` vs `max_so_far_raw_f`.
3) Observa la estabilidad de tmax/sigma en `/nowcast` durante 2–3 METARs consecutivos.

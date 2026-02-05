# Nowcast Checklist (HELIOS)

## Implementado
- [x] Feed de BaseForecast HRRR hacia Nowcast (update en `collect_and_predict` y endpoint manual).
- [x] Alineación de `target_date` por observación (evita rollover incorrecto al cruzar medianoche).
- [x] Uso de timezone de estación en cálculos de peak, sigma y t_peak.
- [x] Separación de temps **raw** vs **judge‑aligned** para floor y buckets.
- [x] Penalización de staleness METAR + actualización de `sources_health`.
- [x] Respeto de `distribution_type` en probabilidades acumuladas.
- [x] BaseForecast válido solo si coincide `target_date` y no está stale.
- [x] Clamps físicos (min/max) en el tmax ajustado.
- [x] Corrección de indexado horario HRRR (timezone local).
- [x] Alias `fetch_metar_race` para compatibilidad.

## Verificación recomendada
- [ ] Ejecutar `web_server.py` y abrir `/nowcast`.
- [ ] Llamar a `/api/v3/nowcast/status` y confirmar trigger `hrrr` > 0.
- [ ] Llamar a `/api/v3/nowcast/{station_id}/update` y validar:
      - `base_forecast_source` = HRRR
      - `tmax_mean_f` coherente con HRRR
- [ ] Revisar `/api/v3/nowcast/{station_id}/state`:
      - `max_so_far_f` (aligned) y `max_so_far_raw_f` (decimales)
      - `staleness_seconds` aumenta si no hay METAR
- [ ] Comparar la curva HRRR vs nowcast (en panel debug) tras 1–2 ciclos de METAR.

## Pendiente (si habilitas estaciones no‑NYC)
- [ ] Etiquetas de `t_peak_bins` en timezone local (hoy se muestran NYC/ES).
- [ ] Confirmar la definición de settlement day si Polymarket usa zona local distinta.

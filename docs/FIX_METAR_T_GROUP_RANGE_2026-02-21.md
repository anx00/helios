# Fix METAR: T-group y rango de settlement (2026-02-21)

## Contexto
Se detecto un riesgo operativo en METAR decodificado:
- Si una observacion no trae `T group` (precision decimal), no se debe tratar como valor puntual exacto.
- Debe tratarse como rango posible.
- Nunca depender del `F` "decodificado" por endpoints de terceros.

## Que hacia HELIOS antes
- HELIOS ya calculaba Fahrenheit desde Celsius.
- Pero en observaciones sin `T group`, se estaba usando un punto unico (falsa precision), en vez de rango.

## Evidencia en datos historicos
Analizando `performance_logs`:
- `rows_with_racing = 81158`
- `rows_with_pair_disagreement_ge1 = 623`

Esto confirma que tuvimos casos con diferencias de `>=1F` entre fuentes NOAA del race.

## Cambios aplicados

1. Parser unificado de temperatura METAR desde `raw`:
- Archivo: `collector/metar/temperature_parser.py`
- Regla:
  - Si hay `T group`, usar precision decimal exacta.
  - Si no hay `T group`, usar el grupo principal (`TT/DD`) y generar rango:
    - `temp_c_low = n - 0.5`
    - `temp_c_high = n + 0.4`
  - Convertir a `temp_f_low/high` y `settlement_f_low/high`.

2. Todas las rutas NOAA usan el mismo parser:
- `collector/metar_fetcher.py` (JSON API)
- `collector/metar/tds_fetcher.py` (XML)
- `collector/metar/tgftp_fetcher.py` (TXT)

3. Propagacion de rango en el modelo y estado:
- `MetarData` ahora incluye:
  - `temp_f_low`, `temp_f_high`
  - `settlement_f_low`, `settlement_f_high`
  - `has_t_group`
- `OfficialObs` y `WorldState` exponen esos campos para SSE/API:
  - `core/models.py`
  - `core/world.py`
  - `core/nowcast_integration.py`

4. QC para evitar precision falsa:
- En `collector/metar_fetcher.py`:
  - Si `settlement_f_low != settlement_f_high`, se marca flag:
    - `TEMP_RANGE_<low>-<high>F_NO_T_GROUP`
  - Estado QC pasa a `UNCERTAIN` (si estaba en `OK`).

5. API realtime con rango:
- `web_server.py` endpoint `/api/realtime/{station_id}` ahora devuelve:
  - `noaa_low_f`, `noaa_high_f`
  - `settlement_low_f`, `settlement_high_f`
  - `has_t_group`

## Validacion ejecutada
- Test nuevo:
  - `tests/test_metar_temperature_parser.py`
- Resultado:
  - `pytest -q tests/test_metar_temperature_parser.py` -> `4 passed`
  - `pytest -q tests/test_nowcast_engine_pws.py` -> `5 passed`

## Ejemplo actual en vivo
- `EGLC`: `has_t_group=False`, `range settlement=47-49`
- `LTAC`: `has_t_group=False`, `range settlement=29-31`
- `KLGA`/`KATL`: con `T group`, rango puntual (`low==high`).

## Impacto operativo
- Menos riesgo de sobreconfianza en lecturas sin precision decimal.
- Mayor trazabilidad para auditoria y settlement.
- Misma logica entre JSON/XML/TXT, reduciendo discrepancias de parseo interno.


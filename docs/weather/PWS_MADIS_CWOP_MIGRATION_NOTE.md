# FIX_PWS_0802: Migración PWS a MADIS/CWOP (NOAA)

## Summary

Este documento describe la incidencia y el fix aplicado el **8 de febrero de 2026** para reemplazar la dependencia de Synoptic/Open-Meteo en el pipeline PWS por una integración robusta con **MADIS/CWOP**.

Objetivo del fix:
1. Eliminar dependencia operativa de `SYNOPTIC_API_TOKEN` en el flujo principal.
2. Dejar de usar Open-Meteo como pseudo-PWS (baja fidelidad en el caso KLGA).
3. Mantener contrato interno existente (`PWSConsensus` + `AuxObs`) para no romper la arquitectura actual.
4. Preservar QC robusto (hard rules + MAD spatial outlier).

---

## Problema Detectado

### Síntomas

- El flujo PWS dependía de:
  - Synoptic API (token, account tier, límites de trial).
  - Fallback Open-Meteo con grid points.
- Cuando Synoptic dejó de estar disponible, el sistema caía a Open-Meteo.
- En práctica, Open-Meteo mostraba desviaciones frecuentes frente a METAR real (especialmente en KLGA), degradando drift y soporte operativo para decisión intradía.

### Root Cause

- El adaptador `collector/pws_fetcher.py` estaba diseñado con dos fuentes (`SYNOPTIC` + `OPEN_METEO`) y no tenía un backend PWS NOAA nativo.
- Open-Meteo es modelo/grid, no red observacional PWS.

---

## Solución Implementada

Se reescribió el pipeline PWS para usar **MADIS Public Surface CGI** con proveedores de red observacional:

- **CWOP** (`APRSWXNET`) como fuente principal.
- **MesoWest** (vía MADIS) como complemento de cobertura.

### Endpoint utilizado

- `https://madis-data.ncep.noaa.gov/madisPublic/cgi-bin/madisXmlPublicDir`

### Estrategia de consulta

Por cada estación:
1. Bounding box por radio configurable (`PWS_SEARCH_CONFIG.radius_km`).
2. Ventana temporal backward (`PWS_MAX_AGE_MINUTES`).
3. Selección de variable `T` (temperatura).
4. Filtro QC MADIS (`qcsel`, `qctype`) + filtro local por `QCD`.
5. Parse XML -> normalización -> consenso robusto.

### Normalización y QC

1. **Conversión de unidades**:
   - MADIS devuelve `data_value` en Kelvin.
   - Se convierte a Celsius (`K - 273.15`).

2. **Filtros primarios**:
   - Parse seguro de `ObTime` (UTC).
   - Filtro de edad (`age_minutes <= PWS_MAX_AGE_MINUTES`).
   - Filtro de descriptor QC (`QCD` permitido: `C`, `S`, `V`).
   - Dedupe por `(provider, station_id)` conservando lectura más reciente.

3. **QC estadístico existente (HELIOS)**:
   - Hard rules (`core.qc.QualityControl.check_hard_rules`).
   - Spatial MAD outlier (`|T - median| > 3 * MAD`).
   - Consenso final por mediana.

---

## Cambios por Archivo

## 1) `collector/pws_fetcher.py`

Reemplazo completo del pipeline de fuentes:

- Eliminado:
  - Fetch Synoptic.
  - Fallback Open-Meteo grid.

- Añadido:
  - Builder de query MADIS:
    - `_build_madis_query_params(...)`
  - Parser XML robusto:
    - `_parse_madis_xml_records(...)`
  - Fetch MADIS:
    - `_fetch_madis_cluster(...)`
  - Cálculo de distancia geográfica:
    - `_haversine_km(...)`
  - Resolución de etiqueta de fuente:
    - `_resolve_station_source(...)`

- Conservado:
  - Contrato `PWSConsensus`.
  - Publicación `AuxObs` en WorldState.
  - Cálculo `drift` vs METAR oficial.
  - Generación de `pws_details` para UI.

## 2) `config.py`

Nueva configuración MADIS:

- `MADIS_BASE_URL`
- `MADIS_CGI_URL`
- `MADIS_TIMEOUT_SECONDS`
- `MADIS_QC_LEVEL`
- `MADIS_QC_TYPE`
- `MADIS_ACCEPT_QCD`
- `MADIS_RECWIN`
- `MADIS_PROVIDER_CONFIG` (por estación; incluye `APRSWXNET` + `MesoWest`)

Se mantiene configuración previa (`SYNOPTIC_*`) por compatibilidad, pero deja de ser el path principal en el fetcher nuevo.

## 3) `templates/world.html`

Actualización de etiquetado de fuentes en tabla PWS:

- Soporte explícito para:
  - `MADIS_APRSWXNET` -> badge `CWOP`
  - `MADIS_*` -> badge `MADIS`
- Agregación de recuentos por fuente real.
- Tooltips ajustados para evitar confusión con Open-Meteo/Synoptic cuando no aplican.

## 4) `tests/test_pws_madis.py`

Nuevos tests unitarios:

- Parse MADIS:
  - filtros de QC
  - filtros de antigüedad
  - dedupe
  - conversión Kelvin -> Celsius
- Builder de query:
  - presencia de campos clave (`nvars`, `qcsel`, `pvd`, etc.)

---

## Validación Ejecutada

Se validó localmente:

1. Compilación:
   - `python -m py_compile collector\pws_fetcher.py config.py`

2. Tests nuevos:
   - `python -m pytest -q tests\test_pws_madis.py`
   - Resultado: `2 passed`

3. Prueba live MADIS (KLGA):
   - `fetch_pws_cluster("KLGA")` devuelve consenso no nulo.
   - Soporte y outliers coherentes con datos reales de CWOP/MADIS.

4. Smoke multi-estación:
   - KLGA / KATL / EGLC devuelven consenso vía MADIS.

Nota: la suite global del repositorio tiene fallos preexistentes en módulos no relacionados (`test_market_discovery`, `test_orderbook`), no introducidos por este fix.

---

## Impacto Operativo

Mejoras:
- Menor dependencia de vendors privados para PWS.
- Datos observacionales reales (CWOP/MADIS) en lugar de grid model para el cluster PWS.
- Mayor trazabilidad de fuente en UI y en logs.

Riesgos/consideraciones:
- MADIS Public CGI puede tener latencia variable.
- Cobertura por estación depende de densidad CWOP local.
- Si quieres operación estrictamente CWOP pura, configura `MADIS_PROVIDER_CONFIG["KLGA"] = ["APRSWXNET"]`.

---

## Configuración Recomendada (KLGA)

Para priorizar calidad en NYC:

- `MADIS_PROVIDER_CONFIG["KLGA"] = ["APRSWXNET", "MesoWest"]` (default actual)
- `PWS_SEARCH_CONFIG["KLGA"]["radius_km"] = 25` (ajustable 20-35 según soporte diario)
- `PWS_MAX_AGE_MINUTES = 90`
- `MADIS_QC_LEVEL = 2` (balance entre limpieza y cobertura)

---

## Resultado Final

HELIOS deja de depender de Synoptic/Open-Meteo para el pipeline PWS principal y pasa a un flujo **MADIS/CWOP productivo**, manteniendo la arquitectura y contratos existentes del proyecto.

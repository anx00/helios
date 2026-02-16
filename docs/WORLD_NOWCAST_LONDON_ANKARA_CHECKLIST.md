# Integracion London + Ankara (World y Nowcast)

Fecha: 2026-02-16
Alcance: Solo `World` y `Nowcast` (sin trading, sin autotrader, sin backtest, sin analytics historico).

## 1) Estado actual (base NYC/Atlanta)

- `KLGA` y `KATL` estan activos en `config.py`.
- `EGLC` (London) existe en `config.py`, pero esta `enabled=False`.
- Ankara aun no existe en `STATIONS`.
- `World` y `Nowcast` se inicializan con estaciones activas (`get_active_stations()`).

## 2) Diferencias clave vs NYC/Atlanta

- Timezone:
  - NYC y Atlanta usan la misma timezone (`America/New_York`).
  - London usa `Europe/London` (DST europeo, cambios no sincronizados con US).
  - Ankara usa `Europe/Istanbul` (UTC+3 fijo, sin DST).
- Definicion de dia objetivo (`target_date`):
  - Parte del sistema sigue semantica NYC en textos/hardcodes.
  - `NowcastEngine` ya usa timezone local para estaciones no-NY en `_get_target_date`.
- Unidades y labels:
  - El pipeline de nowcast trabaja internamente en `degF`.
  - Hay componentes con labels fuertemente NYC/ES y `degF` hardcodeados.
- Cobertura de fuentes:
  - London ya tiene parte de mapeos WU/PWS.
  - Ankara no tiene mapeos WU ni coordenadas/config PWS en el codigo actual.
- Disponibilidad de mercado:
  - London ya tiene slug configurado.
  - Ankara no tiene slug configurado.
  - Si no hay evento de mercado, el sistema debe degradar de forma segura sin romper `World/Nowcast`.

## 3) Checklist de implementacion (World + Nowcast)

### A. Configuracion de estaciones

- [ ] Agregar Ankara a `STATIONS` en `config.py` (recomendado ICAO `LTAC`, timezone `Europe/Istanbul`).
- [ ] Habilitar `EGLC` (`enabled=True`) cuando se quiera entrar en produccion.
- [ ] Definir `characteristics` de microclima para Ankara (continental/meseta, amplitud termica).

### B. World (METAR + PWS + UI)

- [ ] Confirmar METAR NOAA para ambas estaciones en runtime (`EGLC`, `LTAC`).
- [ ] Agregar Ankara en `MADIS_PROVIDER_CONFIG` y `PWS_SEARCH_CONFIG` (`config.py`).
- [ ] Agregar Ankara en `_STATION_COORDS` de `collector/pws_fetcher.py`.
- [ ] Agregar Ankara en `_GRID_OFFSETS` de `collector/pws_fetcher.py`.
- [ ] Agregar Ankara en `_STATION_STATE` de `collector/pws_fetcher.py` (no-US -> `""` si aplica).
- [ ] Si se usa Wunderground PWS, crear candidatos de estaciones para Ankara (registry y/o fallback IDs).
- [ ] Actualizar etiquetas de timezone en UI (`templates/world.html`, `TZ_LABELS`) para Ankara.
- [ ] Verificar que `World` no dependa de supuestos NYC en textos o labels visibles.

### C. Nowcast (base forecast + target_date + UI)

- [ ] Mantener `target_date` por timezone de estacion para London/Ankara (ya soportado en engine, validar end-to-end).
- [ ] Agregar Ankara en `collector/wunderground_fetcher.py` (`STATION_WU_MAP`).
- [ ] Generalizar `country_code` WU: hoy el codigo solo distingue `GB` y `US`; Ankara requiere `TR`.
- [ ] Revisar URLs WU no-US/no-GB para que no caigan en ruta US por defecto.
- [ ] Actualizar `templates/nowcast.html` para no asumir NYC por defecto en textos (`t_peak`, reloj, settlement labels).
- [ ] Agregar Ankara a `TZ_LABELS` en `templates/nowcast.html`.
- [ ] Corregir fallback de estacion por defecto en nowcast UI: si `KLGA` no esta activa, seleccionar la primera activa (como en World).
- [ ] Revisar `t_peak_bins` en `core/nowcast_models.py`: labels actuales son NYC/ES hardcodeados; mover a etiqueta local de estacion.

### D. Mapeo de mercado (necesario para convivencia con nowcast, aunque no sea scope de trading)

- [ ] Agregar Ankara en `POLYMARKET_CITY_SLUGS` y `POLYMARKET_CITY_NAMES_ES` (`config.py`).
- [ ] Agregar Ankara en `CITY_SLUGS` y `CITY_NAMES_SPANISH` (`market/polymarket_checker.py`).
- [ ] Verificar slug real de Polymarket para Ankara antes de activar.
- [ ] Validar unidad de brackets (`degF` vs `degC`) por evento real antes de conectar decisiones de nowcast con mercado.

## 4) Checklist especifica por ciudad

### London (EGLC)

- [ ] Activar estacion (`enabled=True`) en `config.py`.
- [ ] Verificar refresh de base forecast WU para `EGLC`.
- [ ] Validar PWS support real en London (si soporte bajo, ajustar radio/min_support).
- [ ] QA de timezone y DST en UI World/Nowcast.

### Ankara (LTAC)

- [ ] Crear estacion en `config.py`.
- [ ] Anadir toda la cadena PWS (coords, grid, provider/radius).
- [ ] Anadir mapping WU y resolver `country_code=TR`.
- [ ] Anadir slug de mercado `ankara` (si corresponde al naming real del evento).
- [ ] Validar estabilidad de fuentes (METAR/WU/PWS) durante 24h antes de activar trading.

## 5) Pruebas minimas de aceptacion (DoD)

### World

- [ ] `/api/stations` incluye London y Ankara cuando estan activas.
- [ ] `/api/v2/world/metar_history/{station}` devuelve observaciones del dia local correcto.
- [ ] `/api/v2/world/snapshot` contiene `official_obs`, `pws_agg`, `qc_status` para ambas.
- [ ] UI World permite cambiar entre estaciones y muestra timezone local correcta.

### Nowcast

- [ ] `/api/v3/nowcast/{station}` responde sin error y con `target_date` correcto.
- [ ] `/api/v3/nowcast/{station}/state` muestra estado diario separado por estacion.
- [ ] `/api/v3/nowcast/{station}/update` regenera base/nowcast correctamente.
- [ ] UI Nowcast cambia estacion sin fallback roto (sin depender de `KLGA`).
- [ ] Labels horarios y texto de `t_peak` no quedan hardcodeados a NYC para London/Ankara.

## 6) Riesgos concretos detectados en el codigo

- `templates/nowcast.html`: default `currentStation='KLGA'` sin fallback automatico a primera estacion activa.
- `templates/nowcast.html` y `templates/world.html`: `TZ_LABELS` incompleto (sin Ankara).
- `core/nowcast_models.py`: `TPeakBin` etiquetado NYC/ES hardcodeado.
- `core/nowcast_engine.py`: timestamps de salida `ts_nyc`/`ts_es` fijos (no local de estacion).
- `collector/wunderground_fetcher.py`: logica de `country_code` no preparada para `TR`.
- `collector/pws_fetcher.py`: Ankara ausente en coordenadas y configuracion de fuentes.

## 7) Recomendacion de orden de ejecucion

- [ ] Paso 1: Activar London (EGLC) y cerrar gaps UI/timezone.
- [ ] Paso 2: Incorporar Ankara (LTAC) en config + PWS + WU.
- [ ] Paso 3: Resolver hardcodes de tiempo (NYC/ES) en Nowcast.
- [ ] Paso 4: Validacion funcional 24h por ciudad antes de habilitar capa de mercado/trading.

---

Nota operativa:
- En una comprobacion rapida (2026-02-16), NOAA METAR responde para `EGLC` y `LTAC`.
- London aparece con mercado historico/actual en Polymarket; Ankara debe validarse por slug real antes de activacion.

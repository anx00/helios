# Nuevos mercados: Chicago, Miami y Dallas (2026-02-22)

## Objetivo
Dar de alta los mercados de temperatura para:
- Chicago
- Miami
- Dallas

Con configuracion completa en HELIOS:
- ICAO de resolucion
- slug de Polymarket
- unidad de mercado
- integracion Wunderground / METAR / PWS

## Investigacion y decisiones

### 1) Fuente oficial de resolucion en Polymarket
Mercados revisados en Polymarket (reglas de mercado):
- Chicago: resolucion por **Weather Underground station KORD**
- Miami: resolucion por **Weather Underground station KMIA**
- Dallas: resolucion por **Weather Underground station KDAL**

Referencias:
- https://polymarket.com/event/highest-temperature-in-chicago-on-february-4
- https://polymarket.com/event/highest-temperature-in-miami-on-february-4
- https://polymarket.com/event/highest-temperature-in-dallas-on-february-4

### 2) ICAO/metadatos base (NOAA METAR)
Verificado con endpoint NOAA Aviation Weather JSON:
- `KORD` -> Chicago O'Hare
- `KMIA` -> Miami Intl
- `KDAL` -> Dallas Love Field

Endpoint:
- `https://aviationweather.gov/api/data/metar?ids=KORD,KMIA,KDAL&format=json`

### 3) Slugs Polymarket
- `KORD` -> `chicago`
- `KMIA` -> `miami`
- `KDAL` -> `dallas`

### 4) Unidad de mercado
Los tres mercados son US:
- `KORD`: `F`
- `KMIA`: `F`
- `KDAL`: `F`

### 5) Descubrimiento PWS (Weather.com / WU)
Comandos usados:
- `python discover_wu_pws.py --station KORD --lat 41.9602 --lon -87.9316 --limit 60 ...`
- `python discover_wu_pws.py --station KMIA --lat 25.7881 --lon -80.3169 --limit 60 ...`
- `python discover_wu_pws.py --station KDAL --lat 32.8384 --lon -96.8358 --limit 60 ...`

Resultados:
- `KORD`: 9 estaciones validas
- `KMIA`: 10 estaciones validas
- `KDAL`: 10 estaciones validas

Registro actualizado:
- `data/wu_pws_station_registry.json`

### 6) Nota complementaria SST/NDBC
Se reviso NDBC (NOAA) para completar la capa de features ambientales:
- `KMIA` se mapea a `VAKF1` (Virginia Key, FL) como primario y `41122` como fallback
  en `collector/ndbc_fetcher.py`
- `KORD` y `KDAL` quedan sin mapping SST activo por ahora (no se eligio una boya
  cercana/estable con `WTMP` util para este flujo)

Adicional:
- El parser NDBC ahora busca la primera fila reciente con `WTMP` valido, para evitar
  filas mas nuevas con `WTMP=MM`.
- Se agrega filtro de frescura (`NDBC_MAX_AGE_HOURS=12`) para no usar SST antigua.
- Se definio rango onshore para KMIA: `40-170 deg`.

## Cambios implementados

- Config global de estaciones/slugs/unidades/PWS:
  - `config.py`
- Slugs/nombres en checker de mercado:
  - `market/polymarket_checker.py`
- Mapeo Wunderground por estacion:
  - `collector/wunderground_fetcher.py`
- Pipeline PWS completo (fallback IDs, coords, grid, estado):
  - `collector/pws_fetcher.py`
- NBM/LAMP fallback para nuevas estaciones:
  - `collector/nbm_fetcher.py`
  - `collector/lamp_fetcher.py`
- Mensaje de arranque con estaciones activas reales:
  - `main.py`
- Capa SST/NDBC ampliada para Miami:
  - `collector/ndbc_fetcher.py`
- Ajustes de labels/selector para nuevas estaciones:
  - `templates/world.html`
  - `templates/nowcast.html`
  - `templates/replay.html`
  - `templates/autotrader.html`
  - `static/app.js`
- Deploy WU discovery actualizado a todas las estaciones:
  - `deploy/systemd/helios-wu-discover.service`
- Documentacion actualizada:
  - `docs/operations/WUNDERGROUND_PWS.md`
  - `docs/operations/DEPLOY_VPS_LINUX.md`

## Estado
Alta funcional completa para `KORD`, `KMIA`, `KDAL` en:
- market discovery
- fuente oficial METAR/WU
- PWS consensus
- NBM/LAMP fallback
- capa SST para mercado costero (KMIA)

# Wunderground (Weather.com) PWS Integration

This document explains how HELIOS integrates **Wunderground Personal Weather Stations (PWS)** via
**weather.com** endpoints, how discovery works, and how to configure the API key.

## Why This Exists

- **METAR** can be hourly. Some PWS update every few minutes.
- PWS is **auxiliary**: it does not override settlement logic, but it improves nowcast continuity and diagnostics.

HELIOS uses multiple PWS-like sources:
- **Synoptic** (real stations, requires `SYNOPTIC_API_TOKEN`)
- **NOAA MADIS** (CWOP/APRSWXNET + optional providers)
- **Open-Meteo** (grid pseudo-stations; low trust, used as fallback/diversity)
- **Wunderground / Weather.com PWS** (real PWS stations)

## API Key (Moved To `.env`)

HELIOS reads the Weather.com key from `.env` (loaded by `web_server.py`).

Supported variable names:
- `WUNDERGROUND_API_KEY` (preferred)
- `WU_API_KEY` (alias)

Example `.env`:
```bash
# Weather.com / Wunderground API Key
WUNDERGROUND_API_KEY=YOUR_KEY_HERE
```

### Public Key Used For Development

Historically, a widely-used public/shared key has been observed to work with some weather.com endpoints:

```text
e1f10a1e78da46f5b10a1e78da96f525
```

Notes:
- Treat it as **unstable**: it can be rate-limited, revoked, or changed at any time.
- For production, use your own key/terms with The Weather Company.

## Endpoints Used

Discovery (find nearby PWS station IDs):
```text
GET https://api.weather.com/v3/location/near
  ?geocode=<lat>,<lon>
  &product=pws
  &format=json
  &apiKey=<KEY>
```

Observation (current conditions per stationId):
```text
GET https://api.weather.com/v2/pws/observations/current
  ?stationId=<PWS_ID>
  &format=json
  &units=e
  &apiKey=<KEY>
```

Important behavior (tested):
- `v2/pws/...` does **not** accept `geocode` directly (you must discover IDs first).
- Passing multiple station IDs in one request often returns `204` (so HELIOS fetches per-station).

## Discovery Script: `discover_wu_pws.py`

The script:
1. Calls `v3/location/near` for the target station geocode.
2. Validates each station via `v2/pws/observations/current` (must return `200` + temperature).
3. Writes a reusable registry JSON.

Run discovery for all active stations:
```bash
python discover_wu_pws.py --stations KLGA,KATL,KORD,KMIA,KDAL,LFPG,EGLC,LTAC --limit 50 --out data/wu_pws_station_registry.json
```

Registry file:
- Default: `data/wu_pws_station_registry.json`
- It is **local runtime data** and should **not** be committed.

## Known Station IDs (Fallback Lists)

HELIOS includes small fallback station lists in `collector/pws_fetcher.py` so WU can work even if the
registry JSON is missing. These lists can go stale, so discovery refresh is still recommended.

KLGA fallback station IDs:
```text
KNYNEWYO1591
KNYNEWYO1552
KNYNEWYO1974
KNYNEWYO1771
KNYTHEBR7
KNYNEWYO1958
KNYNEWYO1620
KNYNEWYO1906
KNYNEWYO1800
KNYNEWYO1824
```

KATL fallback station IDs:
```text
KGAATLAN557
KGAHAPEV1
KGAEASTP2
KGAFORES13
KGAATLAN378
KGARIVER19
KGAATLAN1046
KGAFORES15
KGAATLAN972
KGACONLE4
```

KORD fallback station IDs:
```text
KILBENSE15
KILBENSE14
KILBENSE13
KILBENSE12
KILELMHU61
KILWOODD21
KILWOODD9
KILELMHU75
KILWOODD12
```

KMIA fallback station IDs:
```text
KFLMIAMI1030
KFLMIAMI69
KFLMIAMI454
KFLMIAMI1010
KFLMIAMI448
KFLMIAMI684
KFLMIAMI578
KFLMIAMI706
KFLMIAMI1095
```

KDAL fallback station IDs:
```text
KTXDALLA960
KTXDALLA1254
KTXDALLA703
KTXDALLA1236
KTXDALLA1224
KTXDALLA102
KTXDALLA1247
KTXDALLA1075
KTXDALLA843
```

LFPG fallback station IDs:
```text
IROISS4
IGONES3
IVMARS5
IFONTE105
IMOUSS19
IGONES2
IMOUSS10
IMITRY1
```

## HELIOS Integration Points

### PWS Cluster (`collector/pws_fetcher.py`)

- Wunderground readings are merged into the same PWS consensus pipeline.
- Individual station rows are stored in `world.pws_details[station]` with:
  - `source: "WUNDERGROUND"`
  - `station_id: <PWS_ID>`

Fallback behavior:
- If the registry JSON is missing, HELIOS has small built-in fallback lists for
  `KLGA`, `KATL`, `KORD`, `KMIA`, `KDAL`, `LFPG`, `EGLC`, and `LTAC`.
- The registry is still recommended so the list stays current.

### World UI (`templates/world.html`)

- Adds a filter chip: **Wunderground**
- Rows show a **WU** badge.
- Filtered stats (mean/median/support/drift) recompute based on active source filters.

### Nowcast Trust (`core/nowcast_engine.py`)

The PWS soft-anchor trust score recognizes `WUNDERGROUND` in the source mix.

## Legacy WU Fetcher (`collector/wunderground_fetcher.py`)

HELIOS also contains a separate WU fetcher used for "observed max" verification logic.
It uses the same `WUNDERGROUND_API_KEY` for some weather.com endpoints when available, and falls back to
HTML scraping if the key is missing/invalid.

## Deployment: Auto-Refresh With systemd Timer (Recommended)

Templates:
- `deploy/systemd/helios-wu-discover.service`
- `deploy/systemd/helios-wu-discover.timer`

Install on the VM:
```bash
sudo cp /opt/helios/deploy/systemd/helios-wu-discover.service /etc/systemd/system/
sudo cp /opt/helios/deploy/systemd/helios-wu-discover.timer /etc/systemd/system/
sudo systemctl daemon-reload

# Run once now
sudo systemctl start helios-wu-discover.service

# Enable periodic refresh (boot + every 6h)
sudo systemctl enable --now helios-wu-discover.timer

tail -f /opt/helios/logs/wu_discover.log
tail -f /opt/helios/logs/wu_discover_error.log
```

## Troubleshooting

- `401 Missing apiKey`: set `WUNDERGROUND_API_KEY` in `/opt/helios/.env` and restart service.
- `204 No Content`: stationId exists but is not returning current obs right now; discovery script filters these out.
- If WU disappears from the PWS table, rerun discovery (or check the timer logs).

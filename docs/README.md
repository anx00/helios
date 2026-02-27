# HELIOS Documentation

Documentacion organizada por partes del sistema (implementacion), no por fases.

Objetivo:
- encontrar rapido donde esta implementado algo
- mantener juntas las notas tecnicas y migraciones del mismo subsistema
- separar la referencia actual del material legado

## Estructura por dominio

- `docs/system/`: vision general, arquitectura y decisiones de evolucion
- `docs/weather/`: METAR, PWS, fuentes, nowcast, prediccion y matematicas
- `docs/market/`: integracion con Polymarket y notas de alta de mercados
- `docs/trading/`: replay, backtest y notas de ejecucion
- `docs/ai/`: Atenea
- `docs/operations/`: despliegue y runbooks operativos
- `docs/engineering/`: journal cronologico de cambios
- `docs/legacy/`: documentos antiguos/duplicados que se conservan por contexto

## Punto de entrada recomendado

### Sistema
- `docs/system/OVERVIEW.md`
- `docs/system/ARCHITECTURE.md`

### Weather / nowcast
- `docs/weather/DATA_SOURCES.md`
- `docs/weather/PREDICTION_SYSTEM.md`
- `docs/weather/MATH_REFERENCE.md`
- `docs/weather/NOWCAST_ENGINE_IMPLEMENTATION.md`

### Mercado (Polymarket)
- `docs/market/POLYMARKET_INTEGRATION.md`
- `docs/market/POLYMARKET_TOKEN_IDS.md`

### Trading
- `docs/trading/BACKTEST_REPLAY.md`
- `docs/trading/STORAGE_REPLAY_IMPLEMENTATION.md`

### Operaciones
- `docs/operations/DEPLOY_VPS_LINUX.md`
- `docs/operations/DEPLOY_VM_DUBLIN_FROM_ZERO.md`

## Notas de implementacion por subsistema

### Weather
- `docs/weather/METAR_RACING.md`
- `docs/weather/METAR_SETTLEMENT_RANGE_NOTE.md`
- `docs/weather/PWS_WUNDERGROUND.md`
- `docs/weather/PWS_MADIS_CWOP_MIGRATION_NOTE.md`
- `docs/weather/LONDON_ANKARA_INTEGRATION_NOTE.md`

### Market rollouts
- `docs/market/rollouts/CHICAGO_MIAMI_DALLAS_2026-02-22.md`
- `docs/market/rollouts/PARIS_2026-02-22.md`

### Trading notes
- `docs/trading/BACKTEST_DEBUG_NOTES.md`
- `docs/trading/REPLAY_BACKTEST_FIDELITY_NOTES.md`
- `docs/trading/BACKTEST_CALIBRATION_IMPLEMENTATION.md`

### Journal
- `docs/engineering/CHANGE_JOURNAL.md`

## Mapa de migracion (rutas antiguas -> actuales)

- `docs/HELIOS_ARCHITECTURE.md` -> `docs/system/ARCHITECTURE.md`
- `docs/HELIOS_DATA_SOURCES.md` -> `docs/weather/DATA_SOURCES.md`
- `docs/HELIOS_PREDICTIONS.md` -> `docs/weather/PREDICTION_SYSTEM.md`
- `docs/HELIOS_MATH.md` -> `docs/weather/MATH_REFERENCE.md`
- `docs/NOAA.md` -> `docs/weather/METAR_RACING.md`
- `docs/WUNDERGROUND_PWS.md` -> `docs/weather/PWS_WUNDERGROUND.md`
- `docs/TOKEN_ID.md` -> `docs/market/POLYMARKET_TOKEN_IDS.md`
- `docs/BACKTEST_REPLAY_GUIDE.md` -> `docs/trading/BACKTEST_REPLAY.md`
- `docs/ATENEA.md` -> `docs/ai/ATENEA.md`
- `docs/PHASES.md` -> `docs/system/ARCHITECTURE_EVOLUTION.md`
- `docs/PHASE3.md` -> `docs/weather/NOWCAST_ENGINE_IMPLEMENTATION.md`
- `docs/PHASE4.md` -> `docs/trading/STORAGE_REPLAY_IMPLEMENTATION.md`
- `docs/PHASE5.md` -> `docs/trading/BACKTEST_CALIBRATION_IMPLEMENTATION.md`
- `docs/progress.md` -> `docs/engineering/CHANGE_JOURNAL.md`

## Regla para nuevos documentos

1. Documenta por subsistema, no por fase.
2. Si es una migracion/fix, guardalo junto al subsistema afectado.
3. Mantén un documento canonico por tema y mueve versiones viejas a `docs/legacy/`.
4. Anota cambios cronologicos en `docs/engineering/CHANGE_JOURNAL.md`.
5. No crear notas sueltas en la raiz de `docs/`.

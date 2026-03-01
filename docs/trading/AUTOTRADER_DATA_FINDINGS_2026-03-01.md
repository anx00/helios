# Autotrader Data Findings (2026-03-01)

Resumen de lo que sale de los datos locales de esta maquina y de las decisiones aplicadas en el runtime actual.

## 1. Fuentes revisadas

- `data/autotrader.db`
- `logs/autotrader_trades.jsonl`
- `data/recordings/date=*/ch=nowcast/events.ndjson`

## 2. Hallazgos duros

### 2.1 Bandit / rewards historicos

Conteo local en `data/autotrader.db`:

- `autotrader_decisions`: 160
- `autotrader_rewards`: 40

Reward agregado por estrategia:

- `fade_event_qc`: `0.00`
- `conservative_edge`: `-0.01`
- `aggressive_edge`: `-0.02`
- `maker_passive`: `-0.03`

Lectura operativa:

- no hay ninguna evidencia local de que el enfoque maker haya sido mejor
- `maker_passive` es la peor estrategia del historico guardado en esta VM
- el enfoque conservador pierde menos que el agresivo

### 2.2 Distribucion real de `p_bucket`

Muestra local analizada:

- `8885` snapshots de nowcast
- estaciones: `KLGA`, `KATL`, `EGLC`

Estadisticas principales:

- `confidence` media: `0.8125`
- `tmax_sigma_f` media: `2.0805`
- probabilidad del bucket top (`top1`) media: `0.3151`
- gap entre top1 y top2 media: `0.0557`

Percentiles relevantes:

- `top1` mediana: `0.2727`
- `top1` p75: `0.3823`
- `gap` mediana: `0.0215`
- `gap` p75: `0.0767`

Lectura operativa:

- muchas veces HELIOS no esta diciendo "este bucket gana clarisimo"
- la distribucion suele estar bastante repartida
- por tanto, con cuentas pequenas, tratar cualquier edge como una apuesta fuerte en un unico bucket mete demasiada varianza

## 3. Decision estrategica tomada

En el runtime actual del autotrader:

- no priorizar maker por defecto
- penalizar sizing cuando `confidence` baja
- penalizar sizing cuando `tmax_sigma_f` sube
- penalizar sizing cuando la distribucion `p_bucket` esta difusa
- usar equity efectiva live para sizing y no solo el bankroll configurado
- asumir que las posiciones live de esta cuenta son del bot salvo marca explicita de tracking-only

## 4. Cambios trasladados a codigo

Aplicados en `core/autotrader.py`:

- gating por `min_signal_confidence`
- multiplicador de budget por `confidence`
- multiplicador de budget por `sigma`
- multiplicador de budget por dominancia de `p_bucket`
- `effective_bankroll_usd` usando cash libre + riesgo mark-to-market live
- sincronizacion live con `assume_live_positions_managed = true` por defecto

## 5. Conclusiones

La prioridad no es "hacer el bot mas activo".

La prioridad correcta, viendo estos datos, es:

1. menos tamano cuando la senal es mediocre
2. menos concentracion cuando la distribucion esta plana
3. no tratar posiciones live del bot como si fueran externas
4. no meter maker hasta que haya evidencia local de fills netos mejores

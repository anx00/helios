# 2026-03-08 - Revision integral de Probability Lab

## Contexto

Se reviso `Probability Lab` de extremo a extremo tras detectar incoherencias intradia en la vista de Paris:

- la card podia mostrar una recomendacion que no cuadraba con la realidad ya observada
- la ladder de detalle podia no coincidir con la recomendacion visible en la card
- la UI mostraba contexto de realidad separado del contexto real usado para construir la senal

Objetivo de esta pasada:

1. verificar si la arquitectura general tiene sentido
2. corregir incoherencias funcionales reales
3. dejar documentados riesgos residuales

## Evaluacion general

La arquitectura base si tiene sentido:

- `day-ahead` usa una fusion de fuentes (`WUNDERGROUND`, `NBM`, `OPEN_METEO`, `LAMP`) con calibracion historica
- `intraday` delega en `build_trading_signal()` / `evaluate_bracket_market()` y usa nowcast, oficial y presion tactica de PWS
- `Probability Lab` actua como una capa de presentacion/inspeccion sobre esa senal, no como un motor distinto

El problema no era el concepto general. El problema era de coherencia entre capas.

## Hallazgos relevantes

### 1. Intradia podia arrastrar un nowcast atrasado frente a una observacion oficial mas fresca

Sintoma:

- al final del dia, con maximo oficial ya claramente por debajo del bracket que lideraba el mercado, la ladder aun podia dejar masa material en ese bracket si el nowcast intradia venia atrasado o mal reconciliado

Impacto:

- recomendaciones tipo `BUY_NO 16C` podian aparecer con una explicacion/modelo visualmente inconsistente

Estado:

- corregido

Cambios:

- `core/trading_signal.py`
  - nuevo reconciliador intradia `_reconcile_intraday_terminal_model()`
  - si hay observacion oficial fresca y el maximo diario ya observado contradice al nowcast al final del dia, se ajustan `market_floor`, `market_ceiling` y `mean_market` antes de construir probabilidades
- `core/trading_bracket_market.py`
  - la reconciliacion se aplica antes de construir `fair_rows` y `fair_map`

Resultado:

- los buckets ya imposibles pasan a `fair_yes = 0`
- la recomendacion y la distribucion vuelven a ser fisicamente coherentes con el maximo observado

### 2. Probability Lab reconstruia la realidad intradia por separado del payload que construia la senal

Sintoma:

- `get_polymarket_dashboard_data()` ya obtenia `official` y `pws` para construir `trading`
- despues `Probability Lab` hacia otra lectura aparte de `world` para poblar el bloque `reality`
- si esa segunda lectura no tenia datos o iba desfasada, la card podia enseñar una senal construida con una realidad y un panel `Reality` vacio o distinto

Impacto:

- incoherencia entre la senal efectiva y lo que la UI explica

Estado:

- corregido

Cambios:

- `web_server.py`
  - `get_polymarket_dashboard_data()` ahora devuelve `intraday_context` con la observacion oficial usada en esa construccion
  - `_build_probability_lab_station_payload()` reutiliza primero ese `intraday_context` antes de volver a consultar `world`

Resultado:

- `Probability Lab` ya no depende de dos lecturas distintas para la misma historia intradia
- la senal y el bloque `Reality` quedan alineados dentro de la misma respuesta

### 3. La card podia usar la vista tactica, pero la ladder de detalle seguia pintando la terminal

Sintoma:

- para `target_day = 0`, `build_probability_lab_card()` puede elegir `best_tactical_trade`
- pero `build_probability_lab_station_detail()` y la tabla del drawer seguian mostrando `recommendation`, `selected_side`, `selected_entry` y `policy_reason` terminales

Impacto:

- card y detalle podian discrepar aunque ambos vinieran del mismo payload

Estado:

- corregido

Cambios:

- `core/probability_model.py`
  - `build_probability_lab_station_detail()` ahora expone campos `active_*` alineados con el modo realmente activo
- `templates/probability_lab.html`
  - la ladder usa `active_selected_side`, `active_selected_entry`, `active_edge_points`, `active_recommendation` y `active_policy_reason`

Resultado:

- el drawer ya explica la misma recomendacion que ve el usuario en la card

## Riesgos residuales

### 1. Intradia sigue degradando a nowcast-only si falta oficial fresca

Esto es aceptable como fallback, pero la UI deberia señalarlo mejor.

Sugerencia futura:

- badge o chip explicito tipo `Reality stale` / `Official missing`

### 2. El board sigue montando estaciones una a una

No es un bug de logica, pero si un coste de latencia.

Sugerencia futura:

- paralelizar la construccion del board o cachear parte del payload intradia

## Tests y validacion

Ejecutado:

- `pytest tests/test_trading_signal.py tests/test_probability_model.py tests/test_probability_lab_api.py -q`
- verificacion de plantilla con Jinja: `TEMPLATE_OK`

Resultado:

- `19 passed`

## Archivos tocados en esta pasada

- `core/trading_signal.py`
- `core/trading_bracket_market.py`
- `core/probability_model.py`
- `web_server.py`
- `templates/probability_lab.html`
- `tests/test_trading_signal.py`
- `tests/test_probability_model.py`
- `tests/test_probability_lab_api.py`

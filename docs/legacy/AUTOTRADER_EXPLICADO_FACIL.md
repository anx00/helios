# Autotrader explicado facil (para no traders)

Archivo orientado a usuarios que quieren entender la pantalla `/autotrader` sin conocimientos de trading.

## 1) Que es Autotrader

Autotrader es un modulo que revisa datos de HELIOS + mercado Polymarket cada cierto tiempo (normalmente cada 60s), decide si operar o no, y guarda todo para auditoria.

Importante:
- Por defecto trabaja en modo `paper` (simulado). No mueve dinero real.
- Cada decision pasa por reglas de riesgo antes de ejecutar.

## 2) Flujo simple de como funciona

En cada ciclo:
1. Lee nowcast de HELIOS (probabilidades por rango de temperatura, confianza, calidad).
2. Lee mercado en vivo (bid/ask, spread, profundidad).
3. Evalua varias estrategias.
4. Selecciona una estrategia (normalmente `linucb`, que va aprendiendo).
5. Pasa por el filtro de riesgo.
6. Si riesgo permite, crea ordenes y posibles ejecuciones.
7. Guarda decisiones, ordenes, fills, posiciones y reward.

## 3) Diferencia clave: Signal vs Order vs Fill

- `Signal`:
  - Es la intencion de trading (ej: BUY YES en un bracket).
  - Se ve en `Signal Timeline`.
  - Puede terminar en no operacion si riesgo bloquea o si no hay edge suficiente.

- `Order`:
  - Es la orden enviada por el broker de simulacion.
  - Puede ser tipo `taker` (agresiva) o `maker` (pasiva/limit).

- `Fill`:
  - Es ejecucion real de una orden (simulada en paper mode).
  - Solo cuando hay fill impacta posiciones y PnL.

Regla mental rapida:
`Signal` (idea) -> `Order` (intento) -> `Fill` (ejecutado)

## 4) Como leer cada bloque de la pantalla

## 4.1 Estado arriba (Running / Paused / Risk OFF)

- `Running`: motor activo.
- `Paused`: motor parado temporalmente.
- `Risk OFF`: bloqueo manual de trading.

## 4.2 KPI cards

- `Total PnL`: resultado total = realizado + mark-to-market.
- `Max Drawdown`: peor caida desde un maximo previo.
- `Orders / Fills`: cuantas ordenes se crearon y cuantas se ejecutaron.
- `Last Decision`: hora y estrategia de la ultima decision.

## 4.3 Signal Timeline (panel principal)

Cada fila es un ciclo de decision.

Columnas:
- `Time`: timestamp de la decision.
- `Strategy`: estrategia seleccionada en ese ciclo.
- `Signal`: acciones propuestas (BUY/SELL YES/NO, bracket, coste estimado).
- `Risk`: resultado del filtro de riesgo.
- `Reward`: score interno del ciclo.
- `Fills`: cantidad de ejecuciones logradas en ese ciclo.

Estados en `Risk`:
- `pass`: riesgo permite operar.
- `blocked`: riesgo bloqueo la operacion.
- `no_trade`: no hubo accion valida (aunque no estuviera bloqueado).

## 4.4 Execution Tape

Tiene 2 subpaneles:

- `Fills`:
  - Lo que realmente se ejecuto.
  - Muestra lado (BUY/SELL YES/NO), cantidad, notional `$`, precio, fees y slippage.

- `Orders`:
  - Ordenes creadas recientemente.
  - Muestra tipo de orden y estado.

Estados comunes de orden:
- `pending`: pendiente.
- `partial`: parcial.
- `filled`: completada.
- `expired`: vencida por tiempo.
- `cancelled`: cancelada.

Tipos comunes de fill:
- `taker_buy`, `taker_sell`, `maker_buy`, `maker_sell`, `simulated`.

## 4.5 Runtime Status

Resumen tecnico del motor:
- `Mode`: normalmente `paper`.
- `Selection`: `linucb` o `static`.
- `Interval`: segundos entre decisiones.
- `Market Buckets`: cuantos brackets de mercado estan disponibles y cuantos coinciden con nowcast.

## 4.6 Decision Context

Variables que explican el contexto de esa decision:
- `Confidence`: confianza del nowcast.
- `Sigma F`: incertidumbre del pronostico.
- `Nowcast Age` / `Market Age`: antiguedad de datos.
- `Spread Top`: diferencia bid/ask en el bucket principal.
- `Depth Imbalance`: desequilibrio de profundidad entre bid y ask.
- `Volatility`: movimiento reciente.
- `QC`: estado de calidad de datos.

## 4.7 Strategies

Muestra:
- cuantas veces se selecciono cada estrategia,
- peso relativo de seleccion,
- reward acumulado.

No es PnL directo. Es metrica interna para comparar estrategias.

## 4.8 Open Positions

Inventario abierto por bucket (posiciones activas):
- `Size`: tamano de posicion.
- `Avg Price`: precio medio.
- `Unrealized`: PnL no realizado.

## 4.9 Helios Top Buckets

Top probabilidades del modelo HELIOS (la "opinion" del modelo).

## 4.10 Polymarket Live Prices

Libro de mercado por bracket:
- precio YES/NO,
- bid/ask,
- spread,
- fuente de datos (`ws`, `gamma`, `derived`).

## 4.11 Controles (botones superiores)

Botones principales:
- `Start`: enciende el motor.
- `Pause`: pausa decisiones nuevas.
- `Resume`: reanuda tras pausa.
- `Risk OFF`: bloquea trading manualmente.
- `Risk ON`: quita bloqueo manual.
- `Refresh`: recarga paneles.

Selector de estacion:
- Cambia entre estaciones (ej: KLGA, KATL, EGLC).
- Al cambiar, reinicia estado en memoria del servicio para esa estacion.

## 4.12 Selector de mercado (Today/Tomorrow)

- `Today` = `D+0`: mercado del dia local de la estacion.
- `Tomorrow` = `D+1`: mercado del siguiente dia local.

La UI ensena tiempos en dos zonas:
- zona de la estacion (NYC/ATL/LON),
- hora de Espana (ES),
para evitar confusiones operativas.

## 5) Estrategias disponibles (explicado facil)

- `conservative_edge`:
  - mas conservadora,
  - exige mas ventaja antes de operar,
  - menos frecuencia.

- `aggressive_edge`:
  - mas activa,
  - entra con umbrales mas bajos,
  - mayor frecuencia.

- `fade_event_qc`:
  - solo opera si hay "event window" activo,
  - orientada a escenarios especiales/QC.

- `maker_passive`:
  - prioriza ordenes tipo maker (pasivas),
  - busca mejor precio, puede tardar mas en llenar.

## 6) Diccionario rapido de motivos de bloqueo (`blocked`)

Codigos mas comunes que puedes ver:
- `daily_loss_limit`: se alcanzo perdida diaria maxima.
- `max_total_exposure`: exposicion total demasiado alta.
- `max_position_per_bucket:<bucket>`: posicion demasiado grande en ese bucket.
- `max_orders_per_hour`: demasiadas ordenes en 1 hora.
- `cooldown_blocked`: aun en periodo de enfriamiento.
- `stale_nowcast`: nowcast demasiado viejo.
- `stale_market`: mercado demasiado viejo.
- `qc_blocked`: calidad de datos no apta.
- `manual_risk_off`: bloqueo manual activado.

## 7) Diccionario rapido de `no_trade`

Cuando sale `no_trade`, suele ser por uno de estos motivos:
- `no_nowcast`
- `no_market_data`
- `stale_nowcast`
- `stale_market`
- `low_confidence`
- `qc_blocked`
- `edge_too_small`
- `daily_loss_limit`
- `cooldown_blocked`
- `inventory_blocked`
- `size_too_small`
- `event_window_inactive`

## 8) Que significa Reward (sin formulas complejas)

`Reward` es una puntuacion interna por ciclo para ayudar al selector de estrategias.
No es igual a PnL puro.

Incluye:
- cambio de PnL,
- penalizacion por drawdown,
- penalizacion por exceso de rotacion.

Sirve para "aprender" que estrategia funciona mejor segun contexto.

## 9) Consejos practicos para usuarios no traders

Si quieres una lectura simple del estado:
1. Mira `Running/Paused/Risk OFF`.
2. Mira `Total PnL` y `Max Drawdown`.
3. En `Signal Timeline`, revisa si hay muchos `blocked`.
4. Si hay `blocked`, lee los `risk reasons`.
5. Verifica que `Nowcast Age` y `Market Age` no esten altos.
6. Mira en `Execution Tape` si hay fills reales o solo ordenes pendientes.

## 10) Donde mirar en codigo (por si quieres mas detalle)

- Orquestador: `core/autotrader/service.py`
- Estrategias: `core/autotrader/strategy.py`
- Riesgo: `core/autotrader/risk.py`
- Broker paper: `core/autotrader/paper_broker.py`
- Persistencia: `core/autotrader/storage.py`
- UI: `templates/autotrader.html`
- Tutorial tecnico existente: `templates/autotrader_tutorial.html`

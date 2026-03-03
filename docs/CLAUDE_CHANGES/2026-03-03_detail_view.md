# 2026-03-03 — Vista detalle de estrategia papertrader

## Contexto
Las tarjetas de Paper Arena solo mostraban 3 posiciones abiertas, 4 eventos y 4 resultados cerrados. "Compras y ventas" vs "Resultados cerrados" era confuso. Los timestamps solo mostraban hora ES. Se necesitaba una vista completa con toda la informacion.

## Archivos nuevos

### `templates/strategy_detail.html`
Pagina completa de detalle para cada estrategia papertrader. Accesible via `/strategy/{strategy_id}`.

Componentes:
- **Header**: nombre, badge, descripcion, equity actual
- **Grid de metricas**: Inicio, Riesgo, PnL cerrado/abierto, Acierto, Fills, Rechazos, Salidas por razon
- **Posiciones abiertas con barra visual**: barra de color (verde YES, naranja NO) que muestra `shares_open / shares` — se reduce cuando se venden shares
- **Historial unificado**: reemplaza "Compras y ventas" + "Resultados cerrados" en una sola timeline cronologica
- **Timestamps duales**: hora ES + hora local de la estacion (ej: "14:30 ES / 08:30 NYC")
- Auto-refresh cada 30 segundos

## Archivos modificados

### `web_server.py`
- `GET /api/papertrader/strategy/{strategy_id}/detail` — devuelve TODAS las posiciones y eventos sin limites (open, closed, events, metricas, station_timezones)
- `GET /strategy/{strategy_id}` — ruta HTML que renderiza la nueva template

### `templates/polymarket.html`
- Boton expandir (icono ⛶) en cada tarjeta de estrategia → navega a `/strategy/{strategy_id}`
- CSS para `.pt-lane-expand-btn`
- CSS para `.strat-badge.micro` (nueva estrategia micro_value_ensemble)
- Badge mapping para `micro_value_ensemble` en `_stratBadge()`

## Lo que NO se toca
- Las 3 secciones existentes en las tarjetas (posiciones, compras, resultados) se mantienen identicas
- Todos los datos del despliegue remoto se preservan
- El endpoint `/api/papertrader/status` no se modifica

## Verificacion
1. `/strategy/winner_value_core` carga la pagina de detalle
2. Timestamps muestran dual: ES + estacion
3. Barras de posicion se renderizan con ancho proporcional
4. Timeline unificado muestra compras + ventas juntas
5. Boton ⛶ visible en cada tarjeta de Paper Arena

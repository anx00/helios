# Gestión de Token IDs en Polymarket

Este documento explica el proceso técnico exhaustivo que sigue HELIOS para identificar, capturar y utilizar los `token_id` (o `clobTokenId`) necesarios para monitorear los mercados de temperatura en tiempo real.

## 1. Contexto: ¿Qué es un Token ID?

En Polymarket, cada mercado (por ejemplo, "¿Llegará la temperatura en NYC a 48-49°F?") se descompone en resultados binarios (SÍ/NO). Cada uno de estos resultados es técnicamente un **Token ERC-1155** en la red Polygon. Para obtener el precio exacto del Order Book (CLOB), necesitamos el identificador único de ese token, no solo el nombre del mercado.

---

## 2. Proceso de Obtención (Paso a Paso)

El flujo de descubrimiento de tokens se encuentra implementado principalmente en `market/polymarket_ws.py` y `market/polymarket_checker.py`.

### Paso A: Construcción del Slug del Evento
HELIOS genera dinámicamente el `slug` del evento de la NOAA basándose en la estación y la fecha objetivo.
*   **Función**: `build_event_slug(station_code, target_date)`
*   **Lógica**: `highest-temperature-in-{ciudad}-on-{mes}-{día}`
*   **Ejemplo**: `highest-temperature-in-nyc-on-january-16`

### Paso B: Consulta a la Gamma API (Discovery)
Con el slug, se realiza una petición a la API de metadatos de Polymarket (Gamma API).
*   **Endpoint**: `https://gamma-api.polymarket.com/events?slug={slug}`

### Paso C: Extracción de `clobTokenIds`
Dentro de la respuesta JSON de la API, cada evento contiene una lista de `markets`. Cada `market` representa un rango de temperatura (bracket).

```python
# Ejemplo de estructura de mercado en la API
{
    "groupItemTitle": "48-49°F",
    "clobTokenIds": "[\"231...456\", \"789...012\"]",
    "active": true
}
```

HELIOS parsea el campo `clobTokenIds` (que viene como un string JSON):
1.  **Primer ID (índice 0)**: Es el Token ID del resultado **YES** (el que nos interesa).
2.  **Segundo ID (índice 1)**: Es el Token ID del resultado **NO**.

### Paso D: Mapeo y Registro
Una vez obtenidos, los IDs se mapean a sus respectivos nombres de brackets en el cliente WebSocket:
```python
# market/polymarket_ws.py
self.market_info[token_id] = bracket  # Ejemplo: {"231...456": "48-49°F"}
```

---

## 3. Uso de los IDs en Tiempo Real

Una vez que HELIOS tiene la lista de Token IDs de los mercados activos para el día:

1.  **Suscripción por WebSocket**: Se envía un mensaje de suscripción al servidor CLOB (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) incluyendo todos los `assets_ids`.
2.  **Filtrado de Mensajes**: El sistema escucha eventos de tipo `trade` o `price_change`. Cuando llega un mensaje, el sistema usa el `asset_id` del mensaje para identificar a qué bracket pertenece gracias al mapeo previo.
3.  **Cálculo de Promedio Ponderado**: Con los precios (probabilidades) de cada Token ID, HELIOS calcula el "Market Average" que se muestra en el dashboard.

---

## 4. Resumen Técnico

*   **Fuente Primaria**: Gamma API (`gamma-api.polymarket.com`).
*   **Campo Clave**: `clobTokenIds`[0].
*   **Fichero de Referencia**: `market/polymarket_ws.py` -> `fetch_market_token_ids()`.
*   **Propósito**: Identificar unívocamente cada opción de apuesta para suscribirse a sus cambios de precio de milisegundo en el CLOB.

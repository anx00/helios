# Replay Endpoint Improvements

## Task: Mejorar /replay (4 issues)

### Issue 4: Estaciones hardcoded con estaciones incorrectas
- [x] Reemplazar dropdown hardcoded (KLGA/KJFK/KEWR) por loop Jinja2 con `active_stations`
- File: `templates/replay.html` (lineas 445-450)

### Issue 1: Scrubber bar no se puede arrastrar
- [x] Reemplazar `scrubber.onclick` por sistema drag completo (mouse + touch)
- [x] Eliminar CSS transition durante drag para fluidez
- [x] Evitar que polling sobreescriba posicion visual durante drag (`isDragging` guard)
- [x] Fix scope: mover `isDragging` al scope IIFE (no dentro de setupEventListeners)
- File: `templates/replay.html`

### Issue 2: Velocidad (x10, x50) no funciona
- [x] **Bug 2A**: `session.play(speed)` ignora cambios cuando state=PLAYING. Fix: nuevo `VirtualClock.set_speed()` + `ReplaySession.set_speed()` + actualizar endpoint
- [x] **Bug 2B**: Playback loop procesa 1 evento/iteracion. Fix: batch processing (hasta 50 eventos por ciclo), cap reducido de 2s a 0.5s
- [x] **Bug 2C**: Frontend polling fijo 500ms. Fix: `adjustPollInterval()` escala segun velocidad
- Files: `core/replay_engine.py`, `web_server.py`, `templates/replay.html`

### Issue 3: Event stream en tarjetas por categoria
- [x] Backend: `get_category_summary()` en ReplaySession
- [x] API: nuevo endpoint `GET /api/v4/replay/session/{id}/categories`
- [x] Frontend: grid de 6 tarjetas (METAR, Nowcast, PWS, Features, Event Windows, Health)
- [x] CSS responsive (3 cols > 2 cols > 1 col)
- [x] Event feed existente renombrado a "Combined Timeline"
- Files: `core/replay_engine.py`, `web_server.py`, `templates/replay.html`

## Review
- Bug principal encontrado: `session.play(speed)` hacia early return cuando state=PLAYING, ignorando silenciosamente cambios de velocidad
- Bug de scope detectado post-implementacion: `isDragging` declarado dentro de `setupEventListeners()` no era accesible desde `updateState()` - corregido moviendo al scope IIFE
- Pendiente: verificacion manual arrancando el servidor

---

## Task: Despliegue en VPS Linux (AWS Dublin)

- [x] Regenerar `requirements.txt` completo (~100 paquetes pinned, excluido helios2)
- [x] Crear `docs/DEPLOY.md` con guia completa:
  - Python 3.12 via deadsnakes PPA
  - Venv + pip install
  - Configuracion `.env` (6 variables, sin valores reales)
  - Directorios runtime (`logs/`, `data/`)
  - Servicio systemd (auto-start/restart)
  - Nginx reverse proxy (80 → 8000, SSE + WebSocket)
  - AWS Security Group (puertos 22, 80, 443)
  - SSL opcional con Let's Encrypt
- Files: `requirements.txt`, `docs/DEPLOY.md`

---

## Task: Polymarket Dashboard - Pagina y endpoint dedicado

- [x] Endpoint unificado `GET /api/polymarket/{station_id}` en `web_server.py`
  - Combina Gamma API (precios, volumen) + WS orderbook (bids/asks, spread) + crowd wisdom (sentimiento) + maturity
  - Join por `token_id` (clobTokenIds) entre Gamma y WS
- [x] Ruta pagina `GET /polymarket` en `web_server.py`
- [x] Nav link en `templates/base.html` (icono candlestick-chart)
- [x] Template `templates/polymarket.html` con:
  - Header con selector estacion + toggle Today/Tomorrow + badge WS
  - Overview card (titulo evento, volumen, status, sentimiento)
  - Grid de brackets: YES/NO prices, WS mid con delta, mini depth bars, spread, volumen
  - Panel orderbook detallado expandible (click en bracket)
  - Tabla market shifts (ultimos 60min)
  - WS diagnostics colapsable
  - Auto-refresh cada 5s + visibility handler
- Files: `web_server.py`, `templates/base.html`, `templates/polymarket.html`

---

## Task: Polymarket en Replay - Datos de mercado en replay

- [x] Añadir `l2_snap` al array de channels en `get_category_summary()` (era el unico canal no mostrado)
- [x] Tarjeta Polymarket (7a) en category grid de replay (icono candlestick-chart, color naranja)
- [x] Panel "Market State" full-width con grid de brackets:
  - Nombre bracket, mid price como %, best bid/ask, spread, depth bars
  - Bracket leading resaltado
  - Se actualiza con cada evento l2_snap durante replay
- [x] `formatEventContent()` para l2_snap: muestra top 3 brackets con mid price
- [x] CSS responsive para panel de mercado (4 > 3 > 2 cols)
- Files: `core/replay_engine.py`, `templates/replay.html`

---

## Task: Atenea AI Overhaul - De FAQ Bot a Experto HELIOS

**Objetivo**: Transformar Atenea de chatbot FAQ restrictivo a asistente experto en HELIOS, forecasting, trading y Polymarket.

### Cambios realizados:

- [x] **System Prompt** (`core/atenea/chat.py`):
  - Reescrito completamente de "EVIDENCE ONLY" a personalidad de experto
  - Nuevo rol: meteorologo senior + data scientist + quant trader
  - Capacidades: analizar condiciones, evaluar predicciones, interpretar mercado, correlacionar datos, dar opiniones
  - Ejemplos de respuestas expertas incluidos
  - Elimina restriccion "NO HALLUCINATION" y requisito de citas E1/E2

- [x] **Validacion simplificada** (`_validate_response()`):
  - Ya no requiere seccion "Evidence" en respuestas
  - Ya no falla por "claims without evidence"
  - Solo valida errores de generacion (respuesta vacia, error API)

- [x] **Prompt generation simplificado** (`_generate_response()`):
  - Elimina instrucciones de citar evidencia
  - Nuevo prompt: "Provide helpful, expert response... Reason through the data"

- [x] **Recoleccion de evidencia ampliada** (`_gather_evidence()`):
  - Ahora SIEMPRE recoge TODOS los datos disponibles, no solo los del intent
  - World + Nowcast + Market (WS) + Health + Polymarket API

- [x] **Nuevo metodo `_get_polymarket_api_evidence()`**:
  - Obtiene precios YES/NO de Gamma API
  - Incluye sentiment de crowd_wisdom
  - Incluye market shifts
  - Complementa datos de WS orderbook

- [x] **Formato LLM mejorado** (`core/atenea/context.py: to_llm_prompt()`):
  - Agrupa datos por tipo (WORLD, MARKET, NOWCAST, HEALTH)
  - Formato mas claro con Markdown headers
  - Incluye campos importantes de cada evidence (temp_f, tmax_mean, spread, etc.)

- [x] **Preguntas sugeridas** (`templates/atenea.html`):
  - Header: "Pregunta al experto" (antes: "Suggested Questions")
  - 5 nuevas preguntas que demuestran capacidades de experto:
    1. ¿Crees que la prediccion de HELIOS es correcta?
    2. ¿Que probabilidad real le das al bracket lider?
    3. Analiza la discrepancia entre HELIOS y Polymarket
    4. ¿Hay señales de que el mercado sabe algo que HELIOS no?
    5. ¿Que riesgos ves en la prediccion actual?

- [x] **Mock response mejorado** (para cuando no hay API key):
  - Muestra resumen organizado por tipo de datos
  - Iconos para cada categoria (📡 📊 🎯 💚)
  - Mensaje claro sobre modo demo

### Files modificados:
- `core/atenea/chat.py` - system prompt, validation, evidence gathering, nuevo metodo
- `core/atenea/context.py` - formato LLM mejorado
- `templates/atenea.html` - nuevas preguntas sugeridas

---

## Task: Documentación Técnica para Atenea — Conocimiento Profundo de HELIOS

**Objetivo**: Crear documentación técnica completa que Atenea pueda cargar dinámicamente para responder preguntas sobre cómo funciona HELIOS internamente.

### Documentos creados:

- [x] **`docs/HELIOS_ARCHITECTURE.md`** - Visión general del sistema
  - Arquitectura de dos capas (Base Forecast + Nowcast Adjustment)
  - Pipeline de datos completo (inputs → WorldState → NowcastEngine → Output)
  - Diagrama de flujo del sistema
  - Roles de cada componente principal

- [x] **`docs/HELIOS_MATH.md`** - Matemáticas y fórmulas
  - Fórmula de Tmax Ajustado: `T_adj = T_base + bias × exp(-h/τ)`
  - Bias EMA: `bias_new = α×δ + (1-α)×bias_old`
  - Decay exponencial: `decay(h) = exp(-h/τ)`, τ = 2.0h
  - CDF Logística y Normal
  - Probabilidad de bucket: `P(bucket) = CDF(high) - CDF(low)`
  - Sigma dinámico con penalties
  - Post-peak cap: `margin = 0.5 + 1.5×(h_sunset/span)`
  - Confidence calculation

- [x] **`docs/HELIOS_DATA_SOURCES.md`** - Fuentes de datos
  - METAR: 3-way race protocol, campos, QC
  - PWS: Algoritmo MAD, drift y support explicados
  - SST: NDBC buoys, ajuste por brisa marina
  - AOD: CAMS/OpenAQ, correlación PM2.5 → AOD
  - Modelos NWP: HRRR, GFS, LAMP, NBM
  - Health tracking estados

- [x] **`docs/HELIOS_PREDICTIONS.md`** - Sistema de predicciones
  - Cálculo paso a paso con ejemplo numérico completo
  - Constraints (floor, post-peak cap)
  - Bucket probability generation
  - Estructura de NowcastDistribution

### Integración con Atenea:

- [x] **`core/atenea/evidence.py`** - Nuevo método `get_technical_docs_evidence()`
  - Detecta keywords técnicos en la pregunta
  - Carga documentos relevantes como evidencia
  - Keywords para arquitectura, matemáticas, datos, predicciones

- [x] **`core/atenea/chat.py`** - Modificado `_gather_evidence()`
  - Ahora recibe `query` como parámetro
  - Llama a `get_technical_docs_evidence()` automáticamente

### Files creados/modificados:
- `docs/HELIOS_ARCHITECTURE.md` - CREADO
- `docs/HELIOS_MATH.md` - CREADO
- `docs/HELIOS_DATA_SOURCES.md` - CREADO
- `docs/HELIOS_PREDICTIONS.md` - CREADO
- `core/atenea/evidence.py` - MODIFICADO (nuevo método)
- `core/atenea/chat.py` - MODIFICADO (pasa query, llama docs)

---

## Task: Unificar estilos de headers de páginas

**Objetivo**: Hacer consistentes los headers de todas las páginas de HELIOS (especialmente /world y /polymarket).

### Cambios realizados:

- [x] **polymarket.html**: Eliminadas clases CSS custom `.pm-page-header` y `.pm-page-title`
- [x] **polymarket.html**: Usar clases estándar `.page-header` y `.page-title` de base.html
- [x] **polymarket.html**: Añadido estilo inline al icono Lucide (24px, accent-blue)
- [x] **world.html**: Cambiado de `<h1>` a `<div>` para `.page-title`
- [x] **world.html**: Añadido icono Lucide "globe" con estilo consistente

### Formato unificado:

```html
<div class="page-header">
    <div class="page-title">
        <i data-lucide="icon-name" style="width:24px;height:24px;color:var(--accent-blue);"></i>
        Page Title
    </div>
    <!-- controls/breadcrumbs -->
</div>
```

### Files modificados:
- `templates/polymarket.html`
- `templates/world.html`

---

## Task: Arreglar dropdowns de /backtest

**Objetivo**: Igualar el estilo de los dropdowns de backtest con los de replay.

### Cambios realizados:

- [x] **CSS**: Añadido estilos para `select.config-input`:
  - `appearance: none` para remover estilo nativo
  - Flecha SVG custom con `background-image`
  - Colores de option matching replay (`#1a1f2e`, `#e2e8f0`)
- [x] **HTML**: Reemplazado dropdown de estaciones hardcoded por Jinja2 loop:
  - Usa `active_stations` y `all_stations` del contexto global
  - Formato: `{{ sid }} - {{ stn.name }}`

### Files modificados:
- `templates/backtest.html`

---

## Task: Unificar etiquetas Polymarket en todo HELIOS

**Objetivo**: Usar las mismas etiquetas que Polymarket en nowcast, backtest y labels para que el trading y la evaluación estén alineados.

### Plan
- [x] Crear utilidades compartidas para formatear/normalizar etiquetas Polymarket (rangos, "or below", "or higher").
- [x] Normalizar etiquetas en nowcast/backtest (p_bucket y market snapshots) y derivar labels con brackets reales cuando existan.
- [x] Ajustar metricas/tests para aceptar etiquetas Polymarket y actualizar `tasks/lessons.md` con la correccion.

## Review
- Etiquetas Polymarket normalizadas y consistentes en nowcast/backtest/labels.
- Tests: `python -m pytest tests/test_backtest.py -q` (30 passed).

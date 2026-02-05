# Lessons Learned

## Session 2026-02-04

### 1. Siempre seguir CLAUDE.md antes de empezar
- **Error**: No cree `tasks/todo.md` ni `tasks/lessons.md` antes de implementar
- **Regla**: Al inicio de cada tarea, leer CLAUDE.md y crear/actualizar los ficheros de tracking ANTES de escribir codigo

### 2. Scope de variables en closures JS (IIFE pattern)
- **Error**: Declare `let isDragging` dentro de `setupEventListeners()` pero la use en `updateState()` - ambas funciones del mismo IIFE pero `isDragging` no era accesible fuera de `setupEventListeners()`
- **Regla**: Variables compartidas entre funciones de un IIFE deben declararse al nivel del IIFE, junto a las demas variables de estado (`sessionId`, `isPlaying`, etc.)
- **Pattern**: Si una variable se usa en mas de una funcion, declararla en el scope comun mas cercano

### 3. Early return silencioso = bug invisible
- **Contexto**: `ReplaySession.play(speed)` hacia `if self.state not in [READY, PAUSED]: return` - cuando state=PLAYING, el cambio de velocidad se ignoraba sin error ni warning
- **Regla**: Los early returns silenciosos en metodos que cambian estado son peligrosos. Mejor: crear metodos especificos (`set_speed`) en vez de reusar metodos existentes (`play`) para funcionalidad diferente
- **Pattern**: Un metodo debe hacer UNA cosa. Si necesitas cambiar velocidad durante playback, no reutilices `play()` - crea `set_speed()`

## Session 2026-02-05

### 1. Etiquetas Polymarket deben ser la fuente de verdad
- **Error**: Propuse normalizar los datos de mercado hacia etiquetas internas de HELIOS.
- **Regla**: En un sistema enfocado a trading en Polymarket, las etiquetas deben seguir SIEMPRE el formato de Polymarket y el resto del pipeline debe adaptarse.

### 2. Siempre verificar que subsistema sirve al UI antes de fixear bugs
- **Error recurrente**: Se arreglo el physics engine cuando el dashboard usa el Nowcast Engine. El bug persistio.
- **Regla**: Antes de tocar codigo, trazar el flujo completo: UI → rendering → data source → engine. Confirmar CUAL componente produce el output erroneo.
- **Pattern**: Usar subagents paralelos para investigar cada subsistema candidato si hay duda.

### 3. Auditorias de spec requieren SIEMPRE dos pasadas
- **Error recurrente**: Primera pasada de auditoria contra spec pierde requisitos (PWS fetcher, NDJSON tape, METAR dedupe, dual timestamps).
- **Regla**: Nunca marcar auditoria como completa tras una sola pasada. Segunda pasada = releer spec linea por linea y grep cada requisito contra el codigo.
- **Pattern**: Usar `/audit` skill para forzar el proceso de dos pasadas.

### 4. Displays de tiempo: nunca redondear a horas decimales
- **Error**: Se mostro "1.0h ago" en vez de minutos exactos para datos METAR.
- **Regla**: Bajo 2h → minutos exactos ("47m ago"). Sobre 2h → horas+minutos ("1h 23m ago"). Nunca decimales.

### 5. No mezclar bug fixes con features en la misma fase
- **Error recurrente**: Sessions terminan incompletas por intentar bug fix + feature + docs en paralelo.
- **Regla**: Priorizar: bugs → features → docs. No cambiar de contexto hasta que el usuario confirme que el fix funciona.

# HELIOS: Guía Completa de los Módulos de Backtest y Replay

Este documento describe en detalle la arquitectura, implementación y uso de los sistemas de **Replay** (Fase 4) y **Backtest** (Fase 5) del proyecto HELIOS Weather Lab.

---

## Tabla de Contenidos

1. [Visión General y Arquitectura](#1-visión-general-y-arquitectura)
2. [Sistema de Replay](#2-sistema-de-replay)
   - [Recorder: Grabación de Eventos](#21-recorder-grabación-de-eventos)
   - [Formato NDJSON y Esquema de Eventos](#22-formato-ndjson-y-esquema-de-eventos)
   - [Compactor: Conversión a Parquet](#23-compactor-conversión-a-parquet)
   - [ReplayEngine: Motor de Reproducción](#24-replayengine-motor-de-reproducción)
3. [Sistema de Backtest](#3-sistema-de-backtest)
   - [Labels: Ground Truth y Alineación con Juez](#31-labels-ground-truth-y-alineación-con-juez)
   - [Dataset: Timeline State y As-Of Joins](#32-dataset-timeline-state-y-as-of-joins)
   - [Metrics: Métricas de Calibración](#33-metrics-métricas-de-calibración)
   - [Policy: Máquina de Estados de Trading](#34-policy-máquina-de-estados-de-trading)
   - [Simulator: Modelos de Ejecución](#35-simulator-modelos-de-ejecución)
   - [Engine: Motor de Backtest](#36-engine-motor-de-backtest)
   - [Calibration: Optimización de Parámetros](#37-calibration-optimización-de-parámetros)
4. [Flujo de Datos End-to-End](#4-flujo-de-datos-end-to-end)
5. [Ejemplos de Uso](#5-ejemplos-de-uso)
6. [Referencia de API](#6-referencia-de-api)

---

## 1. Visión General y Arquitectura

### 1.1 Propósito

El sistema HELIOS necesita dos capacidades fundamentales para operar con confianza:

1. **Replay**: Reproducir sesiones pasadas exactamente como ocurrieron, permitiendo debug, análisis post-mortem, y desarrollo sin necesidad de datos en vivo.

2. **Backtest**: Evaluar la calidad de las predicciones del modelo y simular estrategias de trading sobre datos históricos.

### 1.2 Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SISTEMA LIVE                                   │
│                                                                          │
│  METAR ──► Features ──► Nowcast ──► Market                              │
│    │          │           │           │                                  │
│    └──────────┴───────────┴───────────┘                                  │
│                      │                                                   │
│              ┌───────▼───────┐                                           │
│              │   RECORDER    │  ◄─── Grabación async sin bloqueo        │
│              │ (7 canales)   │                                           │
│              └───────┬───────┘                                           │
│                      │                                                   │
└──────────────────────┼───────────────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    Ring Buffer    NDJSON Files   Live UI
    (in-memory)    (persistente)  (WebSocket)
         │             │
         │             │  ◄─── Batch job diario
         │             ▼
         │      ┌────────────┐
         │      │ COMPACTOR  │  ◄─── NDJSON → Parquet
         │      └─────┬──────┘
         │            │
         │            ▼
         │     Parquet Files
         │     (comprimidos)
         │            │
         └────────────┼────────────┐
                      │            │
              ┌───────▼───────┐    │
              │ HYBRID READER │◄───┘
              │ (NDJSON+Parq) │
              └───────┬───────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│  REPLAY ENGINE  │      │ BACKTEST ENGINE │
│                 │      │                 │
│ - Virtual Clock │      │ - Labels        │
│ - Sessions      │      │ - Metrics       │
│ - Playback      │      │ - Policy        │
│                 │      │ - Simulator     │
└────────┬────────┘      └────────┬────────┘
         │                        │
         ▼                        ▼
    Replay UI              Backtest Results
    (WebSocket)            (JSON + Reports)
```

### 1.3 Archivos Principales

| Módulo | Archivo | Líneas | Propósito |
|--------|---------|--------|-----------|
| Recorder | `core/recorder.py` | 644 | Grabación async de eventos |
| Compactor | `core/compactor.py` | 771 | Conversión NDJSON→Parquet |
| ReplayEngine | `core/replay_engine.py` | 633 | Reproducción con reloj virtual |
| Labels | `core/backtest/labels.py` | ~320 | Ground truth management |
| Dataset | `core/backtest/dataset.py` | ~350 | Timeline state y joins |
| Metrics | `core/backtest/metrics.py` | ~450 | Métricas de calibración |
| Policy | `core/backtest/policy.py` | ~420 | Estado de trading |
| Simulator | `core/backtest/simulator.py` | ~400 | Ejecución Taker/Maker |
| Engine | `core/backtest/engine.py` | ~450 | Orquestación principal |
| Calibration | `core/backtest/calibration.py` | ~380 | Optimización de parámetros |

---

## 2. Sistema de Replay

El sistema de Replay permite grabar todos los eventos del sistema en tiempo real y reproducirlos posteriormente con control total sobre la velocidad y posición.

### 2.1 Recorder: Grabación de Eventos

**Archivo**: `core/recorder.py`

El Recorder es el componente central para capturar eventos sin bloquear el hot path del sistema live.

#### 2.1.1 Arquitectura del Recorder

```
┌─────────────────────────────────────────────────────────────┐
│                        RECORDER                              │
│                                                              │
│  ┌──────────────────┐    ┌───────────────────────────────┐  │
│  │   Ring Buffers   │    │      NDJSON Writers           │  │
│  │   (por canal)    │    │      (por canal)              │  │
│  │                  │    │                               │  │
│  │  world: [...]    │    │  world: async writer          │  │
│  │  pws: [...]      │    │  pws: async writer            │  │
│  │  features: [...]│    │  features: async writer       │  │
│  │  nowcast: [...]  │    │  nowcast: async writer        │  │
│  │  l2_snap: [...]  │    │  l2_snap: async writer        │  │
│  │  health: [...]   │    │  health: async writer         │  │
│  │  event_window: []│    │  event_window: async writer   │  │
│  └──────────────────┘    └───────────────────────────────┘  │
│                                                              │
│  record_*() ──► Ring Buffer (inmediato) + Writer Queue      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.2 Clase RingBuffer (líneas 46-76)

Buffer circular en memoria para servir datos recientes a la UI sin acceso a disco.

```python
class RingBuffer:
    def __init__(self, max_size: int = 1000):
        self._buffer: Deque[Dict] = deque(maxlen=max_size)

    def append(self, event: Dict):
        """O(1) append al buffer circular"""
        self._buffer.append(event)

    def get_recent(self, n: int = 100) -> List[Dict]:
        """Retorna los últimos N eventos"""
        return list(self._buffer)[-n:]

    def get_since(self, ts_utc: datetime) -> List[Dict]:
        """Filtra eventos por timestamp"""
        return [e for e in self._buffer
                if e.get("ts_ingest_utc", "") >= ts_utc.isoformat()]
```

**Propiedades**:
- Tamaño fijo (default: 1000 eventos)
- Evicción FIFO automática
- Memoria acotada (~1-10 MB por canal)

#### 2.1.3 Clase NDJSONWriter (líneas 78-205)

Escritor async con batching para evitar bloquear el path crítico.

```python
class NDJSONWriter:
    def __init__(
        self,
        base_path: Path,
        channel: str,
        flush_interval_seconds: float = 5.0,
        max_batch_size: int = 100
    ):
        self._batch: List[str] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
```

**Métodos Principales**:

| Método | Línea | Propósito |
|--------|-------|-----------|
| `write()` | 133 | Encola evento para escritura (no bloqueante) |
| `_flush_batch()` | 143 | Escribe batch completo a disco |
| `_flush_loop()` | 164 | Tarea background de flush periódico |
| `start()` | 176 | Inicia la tarea de flush |
| `stop()` | 181 | Detiene y hace flush final |

**Flujo de Escritura**:
```
1. write(event) llamado
   │
2. async with _lock:
   │   _batch.append(json.dumps(event))
   │
3. Si len(_batch) >= max_batch_size:
   │   await _flush_batch()
   │
4. Alternativamente, cada flush_interval_seconds:
       _flush_loop() llama _flush_batch()
```

#### 2.1.4 Los 7 Canales de Grabación

El sistema graba eventos en 7 canales especializados:

| Canal | Cadencia | Contenido | Uso Principal |
|-------|----------|-----------|---------------|
| `world` | Por METAR | Observaciones oficiales | Stream de temperatura verdadera |
| `pws` | 1-5s | Consenso de estaciones PWS | Señal crowdsourced |
| `features` | 1m | Features de ambiente | Contexto del modelo |
| `nowcast` | 1m | Predicciones del modelo | Probabilidades por bucket |
| `l2_snap` | 1s | Snapshots del orderbook | Estado del mercado |
| `health` | 1-5s | Métricas de salud | Latencias, reconexiones |
| `event_window` | Event-driven | Marcadores START/END | Correlación de eventos |

#### 2.1.5 Métodos de Grabación Especializados

El Recorder provee métodos de conveniencia para cada tipo de evento:

**record_metar()** (líneas 366-401):
```python
async def record_metar(
    self,
    station_id: str,
    raw: str,                    # METAR crudo
    temp_c: float,               # Temperatura Celsius
    temp_f: float,               # Temperatura Fahrenheit
    temp_aligned: float,         # Redondeado contractual
    obs_time_utc: datetime,
    dewpoint_c: Optional[float],
    wind_dir: Optional[int],
    wind_speed: Optional[float],
    sky_condition: Optional[str],
    qc_state: str = "OK",
    source: str = "METAR",
    correlation_id: Optional[str] = None
)
```

**record_nowcast()** (líneas 449-476):
```python
async def record_nowcast(
    self,
    station_id: str,
    tmax_mean_f: float,          # Media de Tmax
    tmax_sigma_f: float,         # Desviación estándar
    p_bucket: List[Dict],        # Probabilidades por bucket
    t_peak_bins: List[Dict],     # Distribución de hora pico
    confidence: float,           # Confianza del modelo
    qc_state: str,
    bias_f: float,               # Sesgo acumulado
    correlation_id: Optional[str] = None
)
```

**record_event_window()** (líneas 515-535):
```python
async def record_event_window(
    self,
    window_id: str,
    action: str,                 # "START" o "END"
    reason: str,                 # Razón del evento
    station_id: Optional[str] = None,
    correlation_id: Optional[str] = None
)
```

### 2.2 Formato NDJSON y Esquema de Eventos

#### 2.2.1 Estructura de Directorio

```
data/recordings/
├── date=2026-01-28/
│   ├── ch=world/
│   │   └── events.ndjson
│   ├── ch=pws/
│   │   └── events.ndjson
│   ├── ch=features/
│   │   └── events.ndjson
│   ├── ch=nowcast/
│   │   └── events.ndjson
│   ├── ch=l2_snap/
│   │   └── events.ndjson
│   ├── ch=health/
│   │   └── events.ndjson
│   └── ch=event_window/
│       └── events.ndjson
└── date=2026-01-29/
    └── ...
```

**Convención de Nombres**:
- Particionamiento estilo Hive: `date=YYYY-MM-DD/ch={canal}/`
- Un archivo por canal por día
- Sin rotación intra-día (append-only)

#### 2.2.2 Esquema de Evento

Cada evento NDJSON tiene la siguiente estructura (líneas 272-321):

```json
{
  "schema_version": 1,
  "event_id": "uuid-único",
  "correlation_id": "uuid-correlación",
  "ch": "world",
  "ts_ingest_utc": "2026-01-29T14:32:15.123456+00:00",
  "ts_nyc": "2026-01-29 09:32:15",
  "ts_es": "2026-01-29 15:32:15",
  "obs_time_utc": "2026-01-29T14:30:00+00:00",
  "station_id": "KLGA",
  "market_id": "optional",
  "window_id": "optional",
  "data": {
    "src": "METAR",
    "raw": "KLGA 291432Z 31008KT...",
    "temp_c": -2.8,
    "temp_f": 26.9,
    "temp_aligned": 27.0,
    "qc": "OK"
  }
}
```

**Campos Obligatorios**:
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `schema_version` | int | Versión del esquema (actualmente 1) |
| `event_id` | UUID | Identificador único del evento |
| `ch` | string | Nombre del canal |
| `ts_ingest_utc` | ISO 8601 | Timestamp de ingestión (UTC) |
| `data` | object | Payload específico del canal |

**Campos Opcionales**:
| Campo | Descripción |
|-------|-------------|
| `correlation_id` | Agrupa eventos relacionados |
| `ts_nyc` | Timestamp en America/New_York |
| `ts_es` | Timestamp en Europe/Madrid |
| `obs_time_utc` | Tiempo de observación (para METAR) |
| `station_id` | Estación meteorológica |
| `market_id` | ID del mercado Polymarket |
| `window_id` | ID de ventana de evento |

#### 2.2.3 Ejemplo de Líneas NDJSON

```jsonl
{"schema_version":1,"event_id":"a1b2c3d4","ch":"world","ts_ingest_utc":"2026-01-29T14:32:15+00:00","station_id":"KLGA","data":{"src":"METAR","temp_f":26.9,"temp_aligned":27.0,"qc":"OK"}}
{"schema_version":1,"event_id":"e5f6g7h8","ch":"nowcast","ts_ingest_utc":"2026-01-29T14:33:00+00:00","station_id":"KLGA","data":{"tmax_mean_f":42.3,"tmax_sigma_f":1.2,"p_bucket":[{"bucket":"44°F","prob":0.15}],"confidence":0.92}}
```

**Ventajas del formato NDJSON**:
- Un evento por línea (fácil de procesar)
- Append-only (robusto ante crashes)
- Human-readable para debug
- Splittable para procesamiento paralelo
- Sin necesidad de parsear archivo completo

### 2.3 Compactor: Conversión a Parquet

**Archivo**: `core/compactor.py`

El Compactor convierte archivos NDJSON a Parquet para almacenamiento eficiente a largo plazo.

#### 2.3.1 Pipeline de Compactación

```
NDJSON (crudo)
     │
     ▼
read_ndjson_file()     ◄── Generator line-by-line
     │
     ▼
flatten_event()        ◄── Aplana "data" con prefijo "d_"
     │
     ▼
events_to_table()      ◄── Crea PyArrow Table
     │
     ▼
pq.write_table()       ◄── Escribe Parquet con Snappy
     │
     ▼
Parquet (comprimido)
```

#### 2.3.2 Función flatten_event() (líneas 49-70)

Aplana la estructura anidada para compatibilidad con Parquet:

```python
def flatten_event(event: Dict) -> Dict:
    """
    Mueve campos de 'data' al nivel superior con prefijo 'd_'.
    Serializa tipos complejos (dict/list) como JSON strings.
    """
    flat = {}
    for key, value in event.items():
        if key == "data" and isinstance(value, dict):
            for dk, dv in value.items():
                if isinstance(dv, (dict, list)):
                    flat[f"d_{dk}"] = json.dumps(dv)
                else:
                    flat[f"d_{dk}"] = dv
        else:
            flat[key] = value
    return flat
```

**Transformación de Ejemplo**:
```python
# Input
{"ch": "nowcast", "data": {"temp_f": 42.3, "p_bucket": [...]}}

# Output
{"ch": "nowcast", "d_temp_f": 42.3, "d_p_bucket": "[...]"}
```

#### 2.3.3 Clase Compactor (líneas 102-375)

```python
class Compactor:
    def __init__(
        self,
        ndjson_base: str = "data/recordings",
        parquet_base: str = "data/parquet",
        delete_after_compact: bool = False,
        retention_days: int = 14
    ):
```

**Métodos Principales**:

| Método | Línea | Propósito |
|--------|-------|-----------|
| `compact_channel()` | 190 | Compacta un canal para una fecha |
| `compact_date()` | 269 | Compacta todos los canales de una fecha |
| `compact_all_pending()` | 295 | Compacta todas las fechas pendientes (excepto hoy) |
| `cleanup_old_ndjson()` | 343 | Elimina NDJSON más antiguos que retention_days |

#### 2.3.4 Estructura de Parquet

```
data/parquet/
├── station=KLGA/
│   └── date=2026-01-28/
│       ├── ch=world/
│       │   └── part-0000.parquet
│       ├── ch=nowcast/
│       │   └── part-0000.parquet
│       └── ch=l2_snap/
│           └── part-0000.parquet
└── station=KATL/
    └── date=2026-01-28/
        └── ...
```

**Beneficios del Particionamiento**:
- **Pruning**: DuckDB/Pandas pueden saltar directorios completos
- **Paralelismo**: Cada partición se procesa independientemente
- **Escalabilidad**: Añadir estaciones sin reorganizar

**Compresión**:
- Formato: Parquet con compresión Snappy
- Ratio típico: **10:1** (20 MB NDJSON → 2 MB Parquet)

#### 2.3.5 HybridReader: Lectura Unificada (líneas 575-731)

El HybridReader provee una interfaz unificada para leer tanto NDJSON (reciente) como Parquet (compactado):

```python
class HybridReader:
    def read_channel(self, date_str, channel, station_id=None):
        """Lee canal: NDJSON primero, fallback a Parquet"""
        events = self.read_channel_ndjson(date_str, channel)
        if events:
            return events
        return self._parquet_reader.read_channel(date_str, channel, station_id)

    def get_events_sorted(self, date_str, station_id=None, channels=None):
        """Eventos ordenados por timestamp - método principal para replay"""
        all_data = self.read_all_channels(date_str, station_id)
        # Filtra canales si especificado
        # Aplana a lista única
        # Ordena por ts_ingest_utc
        return sorted_events
```

### 2.4 ReplayEngine: Motor de Reproducción

**Archivo**: `core/replay_engine.py`

El ReplayEngine permite reproducir sesiones grabadas con control total sobre velocidad y posición.

#### 2.4.1 Estados del Replay

```python
class ReplayState(Enum):
    IDLE = "idle"           # Sin sesión cargada
    LOADING = "loading"     # Cargando eventos
    READY = "ready"         # Listo para reproducir
    PLAYING = "playing"     # Reproduciendo
    PAUSED = "paused"       # Pausado
    FINISHED = "finished"   # Reproducción completada
```

**Transiciones de Estado**:
```
IDLE ──► LOADING ──► READY ◄──► PLAYING ◄──► PAUSED ──► FINISHED
                        ▲_________________________▲
```

#### 2.4.2 Velocidades de Reproducción

```python
class ReplaySpeed(Enum):
    REALTIME = 1.0      # Tiempo real
    FAST_2X = 2.0       # 2x más rápido
    FAST_5X = 5.0       # 5x más rápido
    FAST_10X = 10.0     # 10x más rápido
    FAST_50X = 50.0     # 50x más rápido
    INSTANT = 0.0       # Sin delay entre eventos
```

#### 2.4.3 VirtualClock: Reloj Virtual (líneas 49-163)

Maneja la escala de tiempo durante el replay:

```python
class VirtualClock:
    def initialize(self, start_time: datetime, end_time: datetime):
        """Inicializa con límites de sesión"""
        self._session_start = start_time
        self._session_end = end_time
        self._current_time = start_time
        self._is_paused = True

    def now(self) -> Optional[datetime]:
        """Calcula tiempo virtual actual"""
        if self._is_paused:
            return self._paused_at or self._current_time

        # Tiempo real transcurrido
        real_elapsed = (datetime.now(UTC) - self._real_start).total_seconds()

        # Aplicar multiplicador de velocidad
        virtual_elapsed = real_elapsed * self._speed

        # Calcular tiempo virtual
        virtual_time = self._current_time + timedelta(seconds=virtual_elapsed)

        # Limitar a fin de sesión
        if self._session_end and virtual_time > self._session_end:
            return self._session_end

        return virtual_time

    def get_progress_percent(self) -> float:
        """Progreso como porcentaje 0-100"""
        duration = (self._session_end - self._session_start).total_seconds()
        elapsed = (self.now() - self._session_start).total_seconds()
        return (elapsed / duration) * 100.0
```

**Fórmula de Escalado**:
```
tiempo_virtual = tiempo_actual + (tiempo_real_transcurrido × velocidad)
```

#### 2.4.4 ReplaySession: Sesión de Replay (líneas 165-452)

```python
class ReplaySession:
    def __init__(self, session_id, date_str, station_id=None, channels=None):
        self.session_id = session_id
        self.date_str = date_str
        self.station_id = station_id
        self.channels = channels

        self.state = ReplayState.IDLE
        self.clock = VirtualClock()

        # Datos
        self._events: List[Dict] = []
        self._event_index: int = 0
        self._metar_indices: List[int] = []    # Para navegación rápida
        self._window_indices: List[int] = []
```

**Carga de Eventos** (línea 200):
```python
async def load(self) -> bool:
    self.state = ReplayState.LOADING

    # Leer eventos ordenados
    self._events = self._reader.get_events_sorted(
        self.date_str, self.station_id, self.channels
    )

    # Construir índices de eventos especiales
    self._build_indices()  # METAR y ventanas de evento

    # Inicializar reloj
    first_ts = self._parse_timestamp(self._events[0])
    last_ts = self._parse_timestamp(self._events[-1])
    self.clock.initialize(first_ts, last_ts)

    self.state = ReplayState.READY
    return True
```

**Loop de Reproducción** (línea 366):
```python
async def _playback_loop(self):
    while self.state == ReplayState.PLAYING:
        if self._event_index >= len(self._events):
            self.state = ReplayState.FINISHED
            break

        event = self._events[self._event_index]
        event_ts = self._parse_timestamp(event)

        if event_ts:
            current = self.clock.now()
            if event_ts > current:
                # Calcular tiempo de espera
                wait_seconds = (event_ts - current).total_seconds()
                wait_seconds = wait_seconds / self.clock._speed  # Aplicar velocidad
                wait_seconds = min(wait_seconds, 2.0)  # Cap máximo

                if wait_seconds > 0.01:
                    await asyncio.sleep(wait_seconds)

        # Emitir evento
        if self._event_callback:
            self._event_callback(event)

        self._event_index += 1
```

**Navegación** (líneas 312-364):
```python
def seek_time(self, target_time: datetime):
    """Saltar a tiempo específico"""
    self.clock.seek(target_time)
    # Encontrar índice del evento
    for i, event in enumerate(self._events):
        if event.get("ts_ingest_utc") >= target_time.isoformat():
            self._event_index = i
            break

def jump_to_next_metar(self) -> bool:
    """Saltar al siguiente METAR"""
    for idx in self._metar_indices:
        if idx > self._event_index:
            self._event_index = idx
            event_ts = self._parse_timestamp(self._events[idx])
            self.clock.seek(event_ts)
            return True
    return False
```

#### 2.4.5 ReplayEngine: Gestor Global (líneas 454-618)

```python
class ReplayEngine:
    _instance = None  # Singleton

    def __init__(self):
        self._sessions: Dict[str, ReplaySession] = {}
        self._active_session_id: Optional[str] = None
        self._broadcast_callback: Optional[Callable] = None

    async def create_session(self, date_str, station_id=None, channels=None):
        """Crea y carga nueva sesión"""
        session = ReplaySession(...)
        session.set_event_callback(self._on_event)
        await session.load()
        self._sessions[session.session_id] = session
        return session

    def _on_event(self, event: Dict):
        """Transforma evento para WebSocket"""
        if self._broadcast_callback:
            ws_event = {
                "type": event.get("ch", "unknown"),
                "replay": True,
                "data": event
            }
            self._broadcast_callback(ws_event)
```

---

## 3. Sistema de Backtest

El sistema de Backtest evalúa la calidad de las predicciones y simula estrategias de trading sobre datos históricos.

### 3.1 Labels: Ground Truth y Alineación con Juez

**Archivo**: `core/backtest/labels.py`

#### 3.1.1 DayLabel: Verdad de un Día (líneas 26-99)

```python
@dataclass
class DayLabel:
    # Identificación
    station_id: str
    market_date: str            # YYYY-MM-DD
    market_id: Optional[str]

    # Verdad del juez (contractual)
    y_bucket_winner: str        # ej. "21-22"
    y_tmax_aligned: float       # Tmax redondeado (°F)
    y_bucket_index: int         # Índice del bucket ganador

    # Verdad física
    y_tmax_physical: float      # Tmax real observado
    y_t_peak_hour_nyc: int      # Hora del pico (0-23 NYC)
    y_t_peak_bin: str           # ej. "14-16"

    # Metadata
    source: str                 # "WU", "METAR", etc.
    confidence: float           # Confianza en el label
    notes: Optional[str]
    observations_count: int
```

#### 3.1.2 Definición de Buckets (líneas 113-120)

El sistema usa 24 buckets de temperatura alineados con los mercados de Polymarket:

```python
DEFAULT_BUCKETS = [
    ("<18", -999, 18),
    ("18-19", 18, 19),
    ("19-20", 19, 20),
    # ... buckets intermedios ...
    ("38-39", 38, 39),
    ("39-40", 39, 40),
    ("≥40", 40, 999),
]

# T-peak bins (ventanas de 2 horas)
DEFAULT_TPEAK_BINS = [
    "00-02", "02-04", "04-06", "06-08", "08-10", "10-12",
    "12-14", "14-16", "16-18", "18-20", "20-22", "22-24"
]
```

#### 3.1.3 Alineación con el Juez (línea 173)

```python
def align_to_judge(self, temp_f: float) -> float:
    """
    Redondeo "half-away-from-zero" para alineación con el juez.

    Ejemplos:
    - 26.4 → 26
    - 26.5 → 27  (half-up)
    - 26.51 → 27
    - -0.5 → -1  (half-away)
    - -1.5 → -2
    """
    if temp_f >= 0:
        return math.floor(temp_f + 0.5)
    else:
        return math.ceil(temp_f - 0.5)
```

#### 3.1.4 Derivación de Labels (línea 219)

```python
def derive_label_from_observations(
    self,
    station_id: str,
    market_date: str,
    observations: List[Dict]
) -> Optional[DayLabel]:
    """
    Deriva el label de ground truth desde observaciones METAR.

    1. Filtra observaciones del día de mercado (NYC)
    2. Encuentra Tmax observado
    3. Aplica alineación del juez
    4. Determina bucket ganador
    5. Calcula hora pico y bin
    """
    # Filtrar por día de mercado
    market_obs = [o for o in observations
                  if self._is_in_market_day(o, market_date)]

    if not market_obs:
        return None

    # Encontrar máximo
    max_obs = max(market_obs, key=lambda o: o.get("temp_f", -999))
    tmax_physical = max_obs["temp_f"]
    tmax_aligned = self.align_to_judge(tmax_physical)

    # Determinar bucket
    bucket_label, bucket_idx = self.get_bucket_for_temp(tmax_aligned)

    # Hora pico
    peak_hour = self._extract_hour_nyc(max_obs["obs_time_utc"])
    peak_bin = self.get_tpeak_bin(peak_hour)

    return DayLabel(
        station_id=station_id,
        market_date=market_date,
        y_bucket_winner=bucket_label,
        y_tmax_aligned=tmax_aligned,
        y_bucket_index=bucket_idx,
        y_tmax_physical=tmax_physical,
        y_t_peak_hour_nyc=peak_hour,
        y_t_peak_bin=peak_bin,
        ...
    )
```

### 3.2 Dataset: Timeline State y As-Of Joins

**Archivo**: `core/backtest/dataset.py`

#### 3.2.1 TimelineState: Estado en un Instante (líneas 29-94)

```python
@dataclass
class TimelineState:
    """Snapshot del estado del mundo en un timestamp específico"""

    # Timestamps
    timestamp_utc: datetime
    timestamp_nyc: datetime

    # Canales de datos (último valor válido)
    last_metar: Optional[Dict] = None
    last_pws: Optional[Dict] = None
    last_upstream: Optional[Dict] = None
    features: Optional[Dict] = None
    nowcast: Optional[Dict] = None
    market: Optional[Dict] = None

    # Edad de los datos (segundos desde última actualización)
    metar_age_seconds: float = float('inf')
    nowcast_age_seconds: float = float('inf')
    market_age_seconds: float = float('inf')

    # Estado acumulativo
    max_observed_f: Optional[float] = None
    max_observed_at: Optional[datetime] = None

    # QC
    qc_flags: List[str] = field(default_factory=list)
    active_window: Optional[str] = None

    def is_stale(self, threshold_seconds: float = 600) -> bool:
        """Verifica si datos críticos están obsoletos"""
        return (
            self.metar_age_seconds > threshold_seconds or
            self.nowcast_age_seconds > threshold_seconds
        )
```

#### 3.2.2 BacktestDataset: Dataset de un Día (líneas 97-266)

```python
@dataclass
class BacktestDataset:
    """Dataset completo para un día de mercado"""

    station_id: str
    market_date: str

    # Eventos por canal
    world_events: List[Dict]      # METAR
    pws_events: List[Dict]        # PWS cluster
    features_events: List[Dict]   # Features ambiente
    nowcast_events: List[Dict]    # Predicciones
    market_events: List[Dict]     # L2 snapshots
    event_windows: List[Dict]     # Ventanas

    # Timeline combinada
    all_events: List[Dict]        # Todos ordenados por timestamp

    # Ground truth
    label: Optional[DayLabel] = None
```

#### 3.2.3 iterate_timeline(): As-Of Join (línea 146)

El método central que itera sobre el tiempo manteniendo el estado actual:

```python
def iterate_timeline(
    self,
    interval_seconds: int = 60
) -> Generator[TimelineState, None, None]:
    """
    Itera sobre el timeline produciendo TimelineState cada intervalo.

    Usa patrón "as-of join": para cada timestamp, retorna el último
    valor válido de cada canal.
    """
    if not self.all_events:
        return

    # Determinar rango de tiempo
    first_ts = self._parse_ts(self.all_events[0])
    last_ts = self._parse_ts(self.all_events[-1])

    # Estado acumulativo
    current = TimelineState(
        timestamp_utc=first_ts,
        timestamp_nyc=first_ts.astimezone(NYC)
    )

    event_idx = 0
    current_time = first_ts

    while current_time <= last_ts:
        # Procesar todos los eventos hasta current_time
        while event_idx < len(self.all_events):
            event = self.all_events[event_idx]
            event_ts = self._parse_ts(event)

            if event_ts > current_time:
                break

            # Actualizar estado según canal
            ch = event.get("ch")
            if ch == "world":
                current.last_metar = event
                current.metar_age_seconds = 0
                # Actualizar max observado
                temp_f = event.get("data", {}).get("temp_f")
                if temp_f and (current.max_observed_f is None
                              or temp_f > current.max_observed_f):
                    current.max_observed_f = temp_f
                    current.max_observed_at = event_ts
            elif ch == "nowcast":
                current.nowcast = event
                current.nowcast_age_seconds = 0
            elif ch == "pws":
                current.last_pws = event
            # ... otros canales

            event_idx += 1

        # Actualizar edades
        if current.last_metar:
            current.metar_age_seconds = (
                current_time - self._parse_ts(current.last_metar)
            ).total_seconds()

        # Yield estado actual
        current.timestamp_utc = current_time
        current.timestamp_nyc = current_time.astimezone(NYC)
        yield current

        # Avanzar tiempo
        current_time += timedelta(seconds=interval_seconds)
```

**Diagrama del As-Of Join**:
```
Tiempo:     T1      T2      T3      T4      T5
            │       │       │       │       │
METAR:      M1──────────────M2──────────────────
PWS:        ──P1────────P2──────────P3──────────
Nowcast:    ────────N1──────────N2──────────────
            │       │       │       │       │
State@T3:   M1      P2      N1    (as-of join)
```

#### 3.2.4 DatasetBuilder (líneas 269-449)

```python
class DatasetBuilder:
    def __init__(self, reader: HybridReader = None):
        self._reader = reader or get_hybrid_reader()
        self._label_manager = get_label_manager()

    def build_dataset(
        self,
        station_id: str,
        market_date: str
    ) -> Optional[BacktestDataset]:
        """Construye dataset para un día"""

        # Leer eventos ordenados
        all_events = self._reader.get_events_sorted(
            market_date,
            station_id=station_id
        )

        if not all_events:
            return None

        # Categorizar por canal
        world_events = []
        pws_events = []
        nowcast_events = []
        # ...

        for event in all_events:
            ch = event.get("ch")
            if ch == "world":
                world_events.append(event)
            elif ch == "pws":
                pws_events.append(event)
            # ...

        # Cargar o derivar label
        label = self._label_manager.get_label(station_id, market_date)
        if not label and world_events:
            label = self._label_manager.derive_label_from_observations(
                station_id, market_date, world_events
            )

        return BacktestDataset(
            station_id=station_id,
            market_date=market_date,
            world_events=world_events,
            pws_events=pws_events,
            nowcast_events=nowcast_events,
            # ...
            all_events=all_events,
            label=label
        )
```

### 3.3 Metrics: Métricas de Calibración

**Archivo**: `core/backtest/metrics.py`

#### 3.3.1 Métricas de Calibración

**Brier Score** (línea 26):
```python
def brier_score(predicted_prob: float, actual: int) -> float:
    """
    Brier Score para predicción binaria.

    BS = (p - y)²

    Rango: [0, 1], menor es mejor.
    - BS = 0: predicción perfecta
    - BS = 1: predicción completamente incorrecta
    """
    return (predicted_prob - actual) ** 2
```

**Brier Score Multi-clase** (línea 42):
```python
def brier_score_multi(
    predicted_probs: List[float],  # Probabilidades por bucket
    actual_idx: int                 # Índice del bucket ganador
) -> float:
    """
    Brier Score promedio sobre todos los buckets.

    BS_multi = (1/K) * Σ (p_k - y_k)²

    donde y_k = 1 si k == actual_idx, else 0
    """
    total = 0.0
    for i, p in enumerate(predicted_probs):
        y = 1.0 if i == actual_idx else 0.0
        total += (p - y) ** 2
    return total / len(predicted_probs)
```

**Expected Calibration Error (ECE)** (línea 105):
```python
def expected_calibration_error(
    predictions: List[Tuple[float, int]],  # (prob, actual)
    n_bins: int = 10
) -> float:
    """
    Error de Calibración Esperado.

    ECE = Σ (|B_m| / n) * |accuracy(B_m) - confidence(B_m)|

    Mide si las probabilidades predichas coinciden con
    las frecuencias observadas.

    Ejemplo: Si predices 70% para eventos que ocurren 70% del tiempo,
    estás bien calibrado.
    """
    bins = [[] for _ in range(n_bins)]

    for prob, actual in predictions:
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, actual))

    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue

        avg_confidence = sum(p for p, _ in bin_items) / len(bin_items)
        avg_accuracy = sum(a for _, a in bin_items) / len(bin_items)

        ece += (len(bin_items) / len(predictions)) * abs(avg_accuracy - avg_confidence)

    return ece
```

**Sharpness** (línea 198):
```python
def sharpness(probs: List[float]) -> float:
    """
    Mide qué tan "puntiaguda" es la distribución.

    Sharpness = 1 - (H(p) / H_max)

    donde H(p) = -Σ p_i * log(p_i) es la entropía.

    - Sharpness = 1: distribución degenerada (todo en un bucket)
    - Sharpness = 0: distribución uniforme
    """
    # Normalizar
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]

    # Calcular entropía
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    # Entropía máxima (uniforme)
    max_entropy = math.log(len(probs))

    if max_entropy == 0:
        return 1.0

    return 1.0 - (entropy / max_entropy)
```

#### 3.3.2 Métricas de Predicción Puntual

```python
def mae(predictions: List[Tuple[float, float]]) -> float:
    """Mean Absolute Error: (1/n) Σ |pred - actual|"""
    return sum(abs(p - a) for p, a in predictions) / len(predictions)

def rmse(predictions: List[Tuple[float, float]]) -> float:
    """Root Mean Squared Error: sqrt((1/n) Σ (pred - actual)²)"""
    mse = sum((p - a) ** 2 for p, a in predictions) / len(predictions)
    return math.sqrt(mse)

def bias(predictions: List[Tuple[float, float]]) -> float:
    """Sesgo medio: (1/n) Σ (pred - actual). Positivo = sobreestima."""
    return sum(p - a for p, a in predictions) / len(predictions)
```

#### 3.3.3 Métricas de Estabilidad

```python
def prediction_churn(
    prob_sequence: List[List[float]]  # Secuencia de distribuciones
) -> float:
    """
    Suma de cambios absolutos en probabilidades.

    Churn = Σ_t Σ_k |P_k(t) - P_k(t-1)|

    Alto churn indica predicciones inestables.
    """
    if len(prob_sequence) < 2:
        return 0.0

    total_churn = 0.0
    for i in range(1, len(prob_sequence)):
        prev = prob_sequence[i-1]
        curr = prob_sequence[i]
        for p, c in zip(prev, curr):
            total_churn += abs(c - p)

    return total_churn

def flip_count(winner_sequence: List[int]) -> int:
    """Cuenta cambios en el bucket predicho ganador."""
    flips = 0
    for i in range(1, len(winner_sequence)):
        if winner_sequence[i] != winner_sequence[i-1]:
            flips += 1
    return flips
```

#### 3.3.4 CalibrationMetrics y MetricsCalculator

```python
@dataclass
class CalibrationMetrics:
    """Agregación de todas las métricas"""

    # Calibración
    brier_global: float
    brier_by_bucket: Dict[str, float]
    log_loss_global: float
    ece: float
    sharpness: float

    # Predicción puntual
    tmax_mae: float
    tmax_rmse: float
    tmax_bias: float

    # Estabilidad
    avg_churn: float
    avg_flips: float

    # T-Peak
    tpeak_accuracy: float
    tpeak_mass_near: float

class MetricsCalculator:
    """Acumula predicciones y calcula métricas agregadas"""

    def add_day(
        self,
        nowcast_sequence: List[Dict],  # Secuencia de nowcasts del día
        label: DayLabel                 # Ground truth
    ):
        """Añade un día al acumulador"""
        # Extrae probabilidades, ganadores, tmax predichos
        # Los añade a acumuladores internos

    def compute(self) -> CalibrationMetrics:
        """Calcula métricas finales sobre todos los días acumulados"""
        return CalibrationMetrics(
            brier_global=brier_score_multi(self._all_probs, self._all_winners),
            ece=expected_calibration_error(self._bucket_predictions),
            tmax_mae=mae(self._tmax_predictions),
            # ...
        )
```

### 3.4 Policy: Máquina de Estados de Trading

**Archivo**: `core/backtest/policy.py`

#### 3.4.1 Estados de la Policy

```python
class PolicyState(Enum):
    NEUTRAL = "neutral"           # Sin posición, esperando
    BUILD_POSITION = "building"   # Construyendo posición
    HOLD = "holding"              # Posición tomada, manteniendo
    REDUCE = "reducing"           # Reduciendo posición
    FADE_EVENT = "fading"         # Fade de evento a corto plazo
    RISK_OFF = "risk_off"         # Problemas de QC, no operar
```

**Transiciones de Estado**:
```
                    ┌──────────────────┐
                    │                  │
        ┌───────────▼───────────┐      │
        │      NEUTRAL          │      │
        │   (sin posición)      │      │
        └───────────┬───────────┘      │
                    │ edge > min_edge  │
                    ▼                  │
        ┌───────────────────────┐      │
        │   BUILD_POSITION      │      │
        │  (construyendo)       │      │
        └───────────┬───────────┘      │
                    │ size alcanzado   │
                    ▼                  │
        ┌───────────────────────┐      │
        │       HOLD            │◄─────┘
        │   (manteniendo)       │
        └───────────┬───────────┘
                    │ edge < exit_edge
                    ▼
        ┌───────────────────────┐
        │      REDUCE           │
        │   (reduciendo)        │
        └───────────┬───────────┘
                    │ posición cerrada
                    ▼
               NEUTRAL

    *** RISK_OFF sobrescribe cualquier estado si QC falla ***
```

#### 3.4.2 PolicyTable: Parámetros de Trading (líneas 121-202)

```python
@dataclass
class PolicyTable:
    """Configura umbrales y límites de la policy"""

    # Umbrales de edge
    min_edge_to_enter: float = 0.05   # 5% mínimo para entrar
    min_edge_to_add: float = 0.08     # 8% para añadir
    edge_to_fade: float = 0.10        # 10% para fades de evento

    # Sizing
    max_position_per_bucket: float = 1.0
    max_total_exposure: float = 3.0
    size_per_confidence: float = 0.5

    # Risk limits
    max_daily_loss: float = 0.20      # 20% max pérdida diaria
    position_decay_rate: float = 0.10
    min_confidence_to_trade: float = 0.50
    max_staleness_seconds: float = 600

    def calculate_edge(
        self,
        model_prob: float,
        market_price: float,
        side: Side
    ) -> float:
        """
        Calcula edge:
        - BUY edge = model_prob - market_price
        - SELL edge = market_price - model_prob

        Ejemplo:
        - Modelo dice 60%, mercado cotiza 50% → BUY edge = 10%
        - Modelo dice 40%, mercado cotiza 50% → SELL edge = 10%
        """
        if side == Side.BUY:
            return model_prob - market_price
        else:
            return market_price - model_prob

    def calculate_size(
        self,
        edge: float,
        confidence: float,
        current_position: float
    ) -> float:
        """
        Sizing estilo Kelly simplificado:

        size = confidence * size_per_confidence * edge_multiplier

        Limitado por max_position_per_bucket
        """
        if edge <= 0 or confidence < self.min_confidence_to_trade:
            return 0.0

        edge_mult = min(edge / self.min_edge_to_enter, 2.0)
        raw_size = confidence * self.size_per_confidence * edge_mult

        # Limitar por posición máxima
        available = self.max_position_per_bucket - abs(current_position)
        return min(raw_size, available)
```

#### 3.4.3 PolicySignal: Señal de Trading (líneas 52-101)

```python
@dataclass
class PolicySignal:
    """Señal de trading generada por la policy"""

    timestamp_utc: datetime
    bucket: str                  # ej. "21-22"
    side: Side                   # BUY o SELL
    order_type: OrderType        # TAKER o MAKER

    target_size: float           # Tamaño deseado
    max_price: Optional[float]   # Límite superior
    min_price: Optional[float]   # Límite inferior

    confidence: float            # Confianza del modelo
    urgency: float               # 0-1, urgencia de ejecución

    stop_loss: Optional[float]
    take_profit: Optional[float]
    ttl_seconds: int             # Time-to-live

    reason: str                  # Explicación
    policy_state: PolicyState    # Estado actual
```

#### 3.4.4 Policy.evaluate() (línea 261)

```python
class Policy:
    def evaluate(
        self,
        nowcast: Dict,           # Predicción actual
        market_state: Dict,      # Estado del mercado
        qc_flags: List[str],     # Flags de QC
        timestamp: datetime
    ) -> List[PolicySignal]:
        """
        Evalúa condiciones y genera señales de trading.
        """
        signals = []

        # 1. QC Gating
        if any(flag in qc_flags for flag in ["SEVERE_OUTLIER", "NO_DATA"]):
            self._state = PolicyState.RISK_OFF
            return []

        # 2. Daily loss check
        if self._daily_pnl < -self._table.max_daily_loss:
            self._state = PolicyState.RISK_OFF
            return []

        # 3. Cooldown check
        if self._last_trade_time:
            elapsed = (timestamp - self._last_trade_time).total_seconds()
            if elapsed < 60:  # 1 minuto cooldown
                return []

        # 4. Evaluar cada bucket
        p_bucket = nowcast.get("p_bucket", [])
        confidence = nowcast.get("confidence", 0.5)

        for bucket_info in p_bucket:
            bucket = bucket_info.get("bucket")
            model_prob = bucket_info.get("prob", 0)

            signal = self._evaluate_bucket(
                bucket=bucket,
                model_prob=model_prob,
                confidence=confidence,
                market_state=market_state,
                timestamp=timestamp
            )

            if signal:
                signals.append(signal)

        # 5. Actualizar estado
        if signals:
            self._state = PolicyState.BUILD_POSITION

        return signals

    def _evaluate_bucket(self, bucket, model_prob, confidence, market_state, timestamp):
        """Evalúa un bucket individual"""

        # Obtener precio de mercado (o usar prob del modelo como default)
        market_price = self._get_market_price(bucket, market_state)
        if market_price is None:
            market_price = model_prob  # Sin mercado, no hay edge

        # Determinar lado (BUY si modelo > mercado)
        if model_prob > market_price:
            side = Side.BUY
        else:
            side = Side.SELL

        # Calcular edge
        edge = self._table.calculate_edge(model_prob, market_price, side)

        # Verificar umbral
        if edge < self._table.min_edge_to_enter:
            return None

        # Calcular tamaño
        current_pos = self._positions.get(bucket, Position(bucket)).size
        size = self._table.calculate_size(edge, confidence, current_pos)

        if size <= 0:
            return None

        return PolicySignal(
            timestamp_utc=timestamp,
            bucket=bucket,
            side=side,
            order_type=OrderType.TAKER,  # Default a taker
            target_size=size,
            confidence=confidence,
            urgency=min(edge / 0.10, 1.0),  # Mayor edge = mayor urgencia
            reason=f"Edge={edge:.1%}, Conf={confidence:.1%}",
            policy_state=self._state
        )
```

### 3.5 Simulator: Modelos de Ejecución

**Archivo**: `core/backtest/simulator.py`

#### 3.5.1 TakerModel: Ejecución Inmediata (líneas 126-211)

```python
class TakerModel:
    """
    Ejecuta inmediatamente al mercado con slippage.

    Precio efectivo = mejor_precio ± slippage
    """

    def __init__(
        self,
        fee_rate: float = 0.02,          # 2% comisión
        base_slippage: float = 0.005,    # 0.5% slippage base
        size_impact: float = 0.01        # Impacto adicional por unidad
    ):
        self.fee_rate = fee_rate
        self.base_slippage = base_slippage
        self.size_impact = size_impact

    def execute(
        self,
        signal: PolicySignal,
        market_state: Dict
    ) -> Optional[Fill]:
        """Ejecuta señal como taker"""

        # Obtener mejor precio
        if signal.side == Side.BUY:
            best_price = self._get_best_ask(signal.bucket, market_state)
            if best_price is None:
                best_price = 0.5  # Default mid
        else:
            best_price = self._get_best_bid(signal.bucket, market_state)
            if best_price is None:
                best_price = 0.5

        # Calcular slippage (depende del tamaño)
        slippage = self.base_slippage + self.size_impact * signal.target_size

        # Calcular precio de ejecución
        if signal.side == Side.BUY:
            exec_price = best_price * (1 + slippage)
        else:
            exec_price = best_price * (1 - slippage)

        # Calcular fees
        fees = exec_price * signal.target_size * self.fee_rate

        return Fill(
            timestamp_utc=signal.timestamp_utc,
            bucket=signal.bucket,
            side=signal.side,
            size=signal.target_size,
            price=exec_price,
            fill_type=FillType.TAKER_BUY if signal.side == Side.BUY else FillType.TAKER_SELL,
            fees=fees,
            slippage=slippage * exec_price * signal.target_size
        )
```

#### 3.5.2 MakerModel: Órdenes Límite (líneas 214-421)

```python
class MakerModel:
    """
    Simula ejecución de órdenes límite con cola.

    Las órdenes se llenan basándose en:
    1. Cruce de precio
    2. Posición en la cola
    3. Tiempo mínimo en precio
    """

    def __init__(
        self,
        fee_rate: float = 0.0,           # Sin fee (rebate maker)
        queue_consumption_rate: float = 0.1,  # 10% cola consumida por update
        min_time_at_price: float = 5.0,  # 5s mínimo
        adverse_selection_prob: float = 0.3  # 30% selección adversa
    ):
        self._pending_orders: Dict[str, Order] = {}

    def place_order(
        self,
        signal: PolicySignal,
        market_state: Dict
    ) -> Order:
        """Coloca orden límite"""

        order = Order(
            order_id=str(uuid.uuid4()),
            timestamp_utc=signal.timestamp_utc,
            bucket=signal.bucket,
            side=signal.side,
            size=signal.target_size,
            limit_price=signal.max_price or signal.min_price,
            order_type=OrderType.MAKER,
            queue_position=self._estimate_queue(signal.bucket, market_state),
            ttl_seconds=signal.ttl_seconds
        )

        self._pending_orders[order.order_id] = order
        return order

    def update(
        self,
        timestamp: datetime,
        market_state: Dict
    ) -> List[Fill]:
        """
        Actualiza órdenes pendientes.
        Llamar cada timestep del backtest.
        """
        fills = []
        to_remove = []

        for order_id, order in self._pending_orders.items():
            # Verificar expiración
            elapsed = (timestamp - order.timestamp_utc).total_seconds()
            if elapsed > order.ttl_seconds:
                order.status = "expired"
                to_remove.append(order_id)
                continue

            # Actualizar posición en cola
            order.queue_position *= (1 - self.queue_consumption_rate)
            order.time_at_price += 1  # Simplificado

            # Verificar condiciones de fill
            current_price = self._get_current_price(order.bucket, market_state)

            should_fill = False

            # Condición 1: Precio cruzado (fill garantizado)
            if order.side == Side.BUY and current_price <= order.limit_price:
                should_fill = True
            elif order.side == Side.SELL and current_price >= order.limit_price:
                should_fill = True

            # Condición 2: Cola limpiada + tiempo mínimo
            if (order.queue_position < 1.0 and
                order.time_at_price >= self.min_time_at_price):
                should_fill = True

            if should_fill:
                fill = Fill(
                    timestamp_utc=timestamp,
                    bucket=order.bucket,
                    side=order.side,
                    size=order.size,
                    price=order.limit_price,
                    fill_type=FillType.MAKER_BUY if order.side == Side.BUY else FillType.MAKER_SELL,
                    fees=0.0
                )
                fills.append(fill)
                order.status = "filled"
                to_remove.append(order_id)

        for order_id in to_remove:
            del self._pending_orders[order_id]

        return fills
```

#### 3.5.3 ExecutionSimulator: Combinación (líneas 423-609)

```python
class ExecutionSimulator:
    """Combina modelos taker y maker"""

    def __init__(self, taker: TakerModel = None, maker: MakerModel = None):
        self._taker = taker or TakerModel()
        self._maker = maker or MakerModel()
        self._fills: List[Fill] = []

    def execute_signal(
        self,
        signal: PolicySignal,
        market_state: Dict
    ) -> Optional[Fill]:
        """Ejecuta señal según order_type"""

        if signal.order_type == OrderType.TAKER:
            fill = self._taker.execute(signal, market_state)
        else:
            order = self._maker.place_order(signal, market_state)
            fill = None  # Maker fills vienen en update()

        if fill:
            self._fills.append(fill)

        return fill

    def update(self, timestamp: datetime, market_state: Dict) -> List[Fill]:
        """Actualiza maker orders"""
        fills = self._maker.update(timestamp, market_state)
        self._fills.extend(fills)
        return fills

    def get_stats(self) -> Dict:
        """Estadísticas de ejecución"""
        if not self._fills:
            return {"total_fills": 0}

        taker_fills = [f for f in self._fills if f.fill_type in [FillType.TAKER_BUY, FillType.TAKER_SELL]]
        maker_fills = [f for f in self._fills if f.fill_type in [FillType.MAKER_BUY, FillType.MAKER_SELL]]

        return {
            "total_fills": len(self._fills),
            "taker_fills": len(taker_fills),
            "maker_fills": len(maker_fills),
            "total_fees": sum(f.fees for f in self._fills),
            "total_slippage": sum(f.slippage for f in self._fills),
            "avg_fill_price": sum(f.price for f in self._fills) / len(self._fills)
        }
```

### 3.6 Engine: Motor de Backtest

**Archivo**: `core/backtest/engine.py`

#### 3.6.1 BacktestMode

```python
class BacktestMode(Enum):
    SIGNAL_ONLY = "signal_only"         # Solo evalúa predicciones
    EXECUTION_AWARE = "execution_aware"  # Simula trading completo
```

#### 3.6.2 BacktestResult (líneas 107-219)

```python
@dataclass
class BacktestResult:
    """Resultados agregados del backtest"""

    # Configuración
    station_id: str
    start_date: str
    end_date: str
    mode: BacktestMode
    policy_name: str

    # Timing
    run_started: datetime
    run_completed: datetime
    run_duration_seconds: float

    # Resultados por día
    day_results: List[DayResult]

    # Métricas agregadas
    aggregated_metrics: CalibrationMetrics

    # Trading (solo EXECUTION_AWARE)
    total_pnl_gross: float
    total_pnl_net: float
    total_fills: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float

    # Cobertura
    days_with_data: int
    days_with_labels: int

    def summary(self) -> str:
        """Resumen textual"""
        return f"""
        === BACKTEST RESULTS ===
        Station: {self.station_id}
        Period: {self.start_date} to {self.end_date}
        Mode: {self.mode.value}

        CALIBRATION:
        - Brier Score: {self.aggregated_metrics.brier_global:.4f}
        - ECE: {self.aggregated_metrics.ece:.4f}
        - Sharpness: {self.aggregated_metrics.sharpness:.4f}

        POINT PREDICTION:
        - Tmax MAE: {self.aggregated_metrics.tmax_mae:.2f}°F
        - Tmax Bias: {self.aggregated_metrics.tmax_bias:.2f}°F

        TRADING:
        - Total PnL: ${self.total_pnl_net:.2f}
        - Win Rate: {self.win_rate:.1%}
        - Sharpe: {self.sharpe_ratio:.2f}
        - Max DD: {self.max_drawdown:.1%}

        COVERAGE: {self.days_with_labels}/{self.days_with_data} days with labels
        """
```

#### 3.6.3 BacktestEngine.run() (línea 258)

```python
class BacktestEngine:
    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        policy: Policy,
        simulator: ExecutionSimulator,
        mode: BacktestMode = BacktestMode.SIGNAL_ONLY
    ):
        self._builder = dataset_builder
        self._policy = policy
        self._simulator = simulator
        self._mode = mode
        self._metrics_calc = MetricsCalculator()

    async def run(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Ejecuta backtest sobre rango de fechas.
        """
        run_started = datetime.now(UTC)
        day_results = []

        # Obtener fechas disponibles
        available_dates = self._builder.list_available_dates()
        dates_to_process = [
            d for d in available_dates
            if start_date <= d <= end_date
        ]

        for i, date_str in enumerate(dates_to_process):
            if progress_callback:
                progress_callback(i, len(dates_to_process), date_str)

            day_result = await self._run_day(station_id, date_str)
            if day_result:
                day_results.append(day_result)

        # Agregar métricas
        aggregated = self._metrics_calc.compute()

        # Calcular estadísticas de trading
        total_pnl_gross = sum(d.pnl_gross for d in day_results)
        total_pnl_net = sum(d.pnl_net for d in day_results)
        total_fills = sum(d.fills for d in day_results)

        # Win rate
        winning_days = sum(1 for d in day_results if d.pnl_net > 0)
        win_rate = winning_days / len(day_results) if day_results else 0

        # Sharpe ratio (simplificado)
        daily_returns = [d.pnl_net for d in day_results]
        if len(daily_returns) > 1:
            mean_ret = sum(daily_returns) / len(daily_returns)
            std_ret = (sum((r - mean_ret)**2 for r in daily_returns) / len(daily_returns)) ** 0.5
            sharpe_ratio = (mean_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for d in day_results:
            cumulative += d.pnl_net
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return BacktestResult(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            mode=self._mode,
            day_results=day_results,
            aggregated_metrics=aggregated,
            total_pnl_gross=total_pnl_gross,
            total_pnl_net=total_pnl_net,
            total_fills=total_fills,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            days_with_data=len(dates_to_process),
            days_with_labels=sum(1 for d in day_results if d.actual_winner),
            run_started=run_started,
            run_completed=datetime.now(UTC),
            run_duration_seconds=(datetime.now(UTC) - run_started).total_seconds()
        )
```

#### 3.6.4 _run_day(): Lógica de un Día (línea 362)

```python
async def _run_day(
    self,
    station_id: str,
    market_date: str
) -> Optional[DayResult]:
    """Ejecuta backtest para un día"""

    # 1. Construir dataset
    dataset = self._builder.build_dataset(station_id, market_date)
    if not dataset:
        return None

    # 2. Reset policy y simulator
    self._policy.reset()
    self._simulator.reset()

    # 3. Recopilar secuencia de nowcasts
    nowcast_sequence = []

    # 4. Iterar sobre timeline
    staleness_sum = 0
    staleness_count = 0
    fills = []

    for state in dataset.iterate_timeline(interval_seconds=60):
        # Track staleness
        staleness_sum += state.metar_age_seconds
        staleness_count += 1

        # Recopilar nowcast
        if state.nowcast:
            nowcast_sequence.append(state.nowcast)

        # En modo EXECUTION_AWARE, ejecutar trading
        if self._mode == BacktestMode.EXECUTION_AWARE and state.nowcast:
            # Construir market state (simulado o real)
            market_state = self._build_market_state(state)

            # Evaluar policy
            signals = self._policy.evaluate(
                nowcast=state.nowcast,
                market_state=market_state,
                qc_flags=state.qc_flags,
                timestamp=state.timestamp_utc
            )

            # Ejecutar señales
            for signal in signals:
                fill = self._simulator.execute_signal(signal, market_state)
                if fill:
                    fills.append(fill)

            # Actualizar maker orders
            maker_fills = self._simulator.update(
                state.timestamp_utc,
                market_state
            )
            fills.extend(maker_fills)

    # 5. Calcular PnL al settlement
    pnl_gross = 0.0
    pnl_net = 0.0

    if dataset.label and fills:
        for fill in fills:
            # Settlement: 1 si bucket ganó, 0 si no
            settlement = 1.0 if fill.bucket == dataset.label.y_bucket_winner else 0.0

            if fill.side == Side.BUY:
                profit = (settlement - fill.price) * fill.size
            else:
                profit = (fill.price - settlement) * fill.size

            pnl_gross += profit
            pnl_net += profit - fill.fees - fill.slippage

    # 6. Añadir a metrics calculator
    if dataset.label and nowcast_sequence:
        self._metrics_calc.add_day(nowcast_sequence, dataset.label)

    # 7. Calcular métricas del día
    day_metrics = self._metrics_calc.compute_for_day(nowcast_sequence, dataset.label)

    return DayResult(
        station_id=station_id,
        market_date=market_date,
        metrics=day_metrics,
        signals_generated=len(signals) if self._mode == BacktestMode.EXECUTION_AWARE else 0,
        fills=len(fills),
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        actual_winner=dataset.label.y_bucket_winner if dataset.label else None,
        actual_tmax=dataset.label.y_tmax_aligned if dataset.label else None,
        predicted_winner=self._get_predicted_winner(nowcast_sequence),
        predicted_tmax=self._get_predicted_tmax(nowcast_sequence),
        total_events=len(dataset.all_events),
        nowcast_count=len(nowcast_sequence),
        metar_count=len(dataset.world_events),
        staleness_avg=staleness_sum / staleness_count if staleness_count > 0 else float('inf')
    )
```

### 3.7 Calibration: Optimización de Parámetros

**Archivo**: `core/backtest/calibration.py`

#### 3.7.1 ParameterSet (líneas 37-65)

```python
@dataclass
class ParameterSet:
    """Define un parámetro a calibrar"""

    name: str                    # Nombre del parámetro
    current_value: float         # Valor actual
    min_value: float             # Mínimo para grid
    max_value: float             # Máximo para grid
    step: float                  # Paso del grid
    param_type: str = "float"    # Tipo (float, int)

    def get_grid_values(self) -> List[float]:
        """Genera valores del grid"""
        values = []
        v = self.min_value
        while v <= self.max_value:
            values.append(v)
            v += self.step
        return values
```

#### 3.7.2 ObjectiveFunction (líneas 133-185)

```python
@dataclass
class ObjectiveFunction:
    """Combina métricas en score único para optimización"""

    # Pesos (negativos porque minimizamos errores)
    brier_weight: float = 1.0
    ece_weight: float = 0.5
    pnl_weight: float = 0.3
    drawdown_penalty: float = 0.5
    stability_weight: float = 0.2

    def compute(self, metrics: CalibrationMetrics, result: BacktestResult) -> float:
        """
        Calcula score de optimización.

        Score = -brier_weight*brier
                - ece_weight*ece
                - stability_weight*churn
                + pnl_weight*pnl
                - drawdown_penalty*drawdown

        Mayor score es mejor.
        """
        score = 0.0

        # Penalizar errores
        score -= self.brier_weight * metrics.brier_global
        score -= self.ece_weight * metrics.ece
        score -= self.stability_weight * metrics.avg_churn

        # Premiar PnL
        score += self.pnl_weight * result.total_pnl_net

        # Penalizar drawdown
        score -= self.drawdown_penalty * result.max_drawdown

        return score
```

#### 3.7.3 CalibrationLoop.grid_search() (línea 225)

```python
class CalibrationLoop:
    def __init__(
        self,
        engine: BacktestEngine,
        parameters: List[ParameterSet],
        objective: ObjectiveFunction
    ):
        self._engine = engine
        self._parameters = parameters
        self._objective = objective

    async def grid_search(
        self,
        station_id: str,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        max_combinations: int = 100
    ) -> CalibrationResult:
        """
        Grid search sobre parámetros.
        """
        # Generar todas las combinaciones
        grids = [p.get_grid_values() for p in self._parameters]
        combinations = list(itertools.product(*grids))

        # Samplear si hay demasiadas
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)

        best_params = None
        best_train_score = float('-inf')
        best_val_score = float('-inf')
        all_results = []

        for combo in combinations:
            # Crear dict de parámetros
            params = {
                p.name: v
                for p, v in zip(self._parameters, combo)
            }

            # Ejecutar backtest con estos parámetros
            train_result = await self._run_with_params(
                station_id, train_start, train_end, params
            )
            val_result = await self._run_with_params(
                station_id, val_start, val_end, params
            )

            # Calcular scores
            train_score = self._objective.compute(
                train_result.aggregated_metrics, train_result
            )
            val_score = self._objective.compute(
                val_result.aggregated_metrics, val_result
            )

            all_results.append({
                "params": params,
                "train_score": train_score,
                "val_score": val_score
            })

            # Track best
            if val_score > best_val_score:
                best_params = params
                best_train_score = train_score
                best_val_score = val_score

        return CalibrationResult(
            station_id=station_id,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            best_params=best_params,
            best_train_score=best_train_score,
            best_val_score=best_val_score,
            all_results=all_results,
            total_combinations=len(grids),
            combinations_tested=len(combinations)
        )
```

#### 3.7.4 Walk-Forward Validation (línea 353)

```python
async def walk_forward(
    self,
    station_id: str,
    start_date: str,
    end_date: str,
    train_days: int = 14,
    val_days: int = 7
) -> List[CalibrationResult]:
    """
    Calibración walk-forward con ventanas deslizantes.

    ┌─────────────────┬───────┐
    │     Train       │  Val  │
    └─────────────────┴───────┘
              ┌─────────────────┬───────┐
              │     Train       │  Val  │
              └─────────────────┴───────┘
                        ┌─────────────────┬───────┐
                        │     Train       │  Val  │
                        └─────────────────┴───────┘
    """
    results = []

    current_start = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    while True:
        # Calcular ventanas
        train_end = current_start + timedelta(days=train_days)
        val_start = train_end + timedelta(days=1)
        val_end = val_start + timedelta(days=val_days)

        if val_end > end_dt:
            break

        # Ejecutar grid search para esta ventana
        result = await self.grid_search(
            station_id=station_id,
            train_start=current_start.date().isoformat(),
            train_end=train_end.date().isoformat(),
            val_start=val_start.date().isoformat(),
            val_end=val_end.date().isoformat()
        )

        results.append(result)

        # Mover ventana
        current_start = val_start

    return results
```

---

## 4. Flujo de Datos End-to-End

### 4.1 Pipeline de Grabación a Replay

```
SISTEMA LIVE
     │
     ▼
┌─────────────────────────────────────────────┐
│ 1. GRABACIÓN (Recorder)                      │
│                                              │
│    METAR ──► record_metar() ──► world.ndjson │
│    PWS ────► record_pws() ────► pws.ndjson   │
│    Model ──► record_nowcast() ► nowcast.ndjson│
│                                              │
│    + Ring Buffer para UI live                │
└─────────────────────────────────────────────┘
                    │
                    │ (batch job nocturno)
                    ▼
┌─────────────────────────────────────────────┐
│ 2. COMPACTACIÓN (Compactor)                  │
│                                              │
│    NDJSON ──► flatten ──► Parquet (Snappy)  │
│                                              │
│    Estructura:                               │
│    station=KLGA/date=2026-01-28/ch=world/   │
└─────────────────────────────────────────────┘
                    │
                    │ (on-demand)
                    ▼
┌─────────────────────────────────────────────┐
│ 3. REPLAY (ReplayEngine)                     │
│                                              │
│    HybridReader ──► ReplaySession ──► WS    │
│                                              │
│    Controles:                                │
│    - Play/Pause/Seek                         │
│    - Velocidad 1x-50x                        │
│    - Saltar a METAR/Evento                   │
└─────────────────────────────────────────────┘
```

### 4.2 Pipeline de Backtest

```
DATOS HISTÓRICOS (HybridReader)
     │
     ▼
┌─────────────────────────────────────────────┐
│ 1. CONSTRUCCIÓN DE DATASET                   │
│                                              │
│    DatasetBuilder.build_dataset()            │
│         │                                    │
│         ├── Leer eventos por fecha/estación  │
│         ├── Categorizar por canal            │
│         ├── Cargar/derivar label             │
│         └── Crear BacktestDataset            │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 2. ITERACIÓN DE TIMELINE                     │
│                                              │
│    dataset.iterate_timeline(60s)             │
│         │                                    │
│         ├── As-of join por canal             │
│         ├── Track max observado              │
│         ├── Calcular staleness               │
│         └── Yield TimelineState              │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 3. EVALUACIÓN DE POLICY                      │
│                                              │
│    Para cada TimelineState:                  │
│         │                                    │
│         ├── QC gating                        │
│         ├── Calcular edge por bucket         │
│         ├── Determinar sizing                │
│         └── Generar PolicySignal             │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 4. SIMULACIÓN DE EJECUCIÓN                   │
│                                              │
│    ExecutionSimulator.execute_signal()       │
│         │                                    │
│         ├── Taker: ejecución inmediata      │
│         └── Maker: orden límite + cola       │
│                                              │
│    → Fill (precio, fees, slippage)          │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 5. CÁLCULO DE MÉTRICAS                       │
│                                              │
│    MetricsCalculator.add_day()               │
│         │                                    │
│         ├── Acumular predicciones vs labels  │
│         ├── Calcular Brier, ECE, Log Loss   │
│         ├── Calcular MAE, RMSE, Bias        │
│         └── Calcular Churn, Flips           │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 6. AGREGACIÓN Y REPORTE                      │
│                                              │
│    BacktestResult                            │
│         │                                    │
│         ├── Métricas agregadas              │
│         ├── PnL, Sharpe, Drawdown           │
│         ├── Win rate                         │
│         └── Cobertura                        │
└─────────────────────────────────────────────┘
```

---

## 5. Ejemplos de Uso

### 5.1 Grabación de una Sesión

```python
from core.recorder import initialize_recorder, get_recorder

# Inicializar
recorder = await initialize_recorder("data/recordings")

# Correlacionar eventos relacionados
correlation_id = recorder.start_correlation(reason="METAR 14:30 UTC")

# Grabar METAR
await recorder.record_metar(
    station_id="KLGA",
    raw="KLGA 291432Z 31008KT 10SM FEW250 M03/M06...",
    temp_c=-2.8,
    temp_f=26.9,
    temp_aligned=27.0,
    obs_time_utc=datetime(2026, 1, 29, 14, 30, tzinfo=UTC),
    qc_state="OK",
    correlation_id=correlation_id
)

# Grabar nowcast
await recorder.record_nowcast(
    station_id="KLGA",
    tmax_mean_f=42.3,
    tmax_sigma_f=1.2,
    p_bucket=[
        {"bucket": "42-43", "prob": 0.25},
        {"bucket": "43-44", "prob": 0.35},
        {"bucket": "44-45", "prob": 0.20}
    ],
    t_peak_bins=[{"hour": 14, "prob": 0.45}],
    confidence=0.92,
    qc_state="OK",
    bias_f=-0.5,
    correlation_id=correlation_id
)

recorder.end_correlation()

# Shutdown
await recorder.stop()
```

### 5.2 Compactación de Registros

```python
from core.compactor import Compactor

# Inicializar
compactor = Compactor(
    ndjson_base="data/recordings",
    parquet_base="data/parquet",
    delete_after_compact=True,
    retention_days=14
)

# Compactar una fecha específica
result = compactor.compact_date("2026-01-28")
print(f"Compacted {result['total_events']} events")

# O compactar todo lo pendiente
compactor.compact_all_pending()

# Limpiar NDJSON antiguos
deleted = compactor.cleanup_old_ndjson()
print(f"Deleted {deleted} old NDJSON directories")
```

### 5.3 Reproducción de una Sesión

```python
from core.replay_engine import get_replay_engine

# Obtener engine
engine = get_replay_engine()

# Configurar callback para WebSocket
def broadcast_to_ws(event):
    asyncio.create_task(ws_manager.broadcast(event))

engine.set_broadcast_callback(broadcast_to_ws)

# Crear sesión
session = await engine.create_session(
    date_str="2026-01-29",
    station_id="KLGA",
    channels=["world", "nowcast"]
)

print(session.get_state())
# Output:
# {
#     "session_id": "a1b2c3d4",
#     "state": "ready",
#     "total_events": 2400,
#     "metar_count": 8,
#     "progress_percent": 0.0
# }

# Reproducir a 5x
await engine.play(speed=5.0)

# Pausar
await engine.pause()

# Saltar al siguiente METAR
engine.jump_next_metar()

# Seek a 50%
engine.seek_percent(50.0)

# Cerrar sesión
await engine.close_session(session.session_id)
```

### 5.4 Ejecutar Backtest

```python
from core.backtest.engine import BacktestEngine, BacktestMode
from core.backtest.dataset import DatasetBuilder
from core.backtest.policy import Policy, create_conservative_policy
from core.backtest.simulator import ExecutionSimulator

# Crear componentes
builder = DatasetBuilder()
policy = create_conservative_policy()
simulator = ExecutionSimulator()

# Crear engine
engine = BacktestEngine(
    dataset_builder=builder,
    policy=policy,
    simulator=simulator,
    mode=BacktestMode.EXECUTION_AWARE
)

# Ejecutar backtest
result = await engine.run(
    station_id="KLGA",
    start_date="2026-01-01",
    end_date="2026-01-30",
    progress_callback=lambda i, n, d: print(f"Processing {d} ({i+1}/{n})")
)

# Ver resumen
print(result.summary())

# Guardar resultados
result.save("backtest_results/klga_jan_2026.json")
```

### 5.5 Calibración de Parámetros

```python
from core.backtest.calibration import CalibrationLoop, ParameterSet, ObjectiveFunction

# Definir parámetros a calibrar
parameters = [
    ParameterSet(
        name="bias_alpha",
        current_value=0.1,
        min_value=0.05,
        max_value=0.30,
        step=0.05
    ),
    ParameterSet(
        name="min_edge_to_enter",
        current_value=0.05,
        min_value=0.02,
        max_value=0.10,
        step=0.02
    )
]

# Definir función objetivo
objective = ObjectiveFunction(
    brier_weight=1.0,
    ece_weight=0.5,
    pnl_weight=0.3
)

# Crear loop de calibración
loop = CalibrationLoop(engine, parameters, objective)

# Grid search
result = await loop.grid_search(
    station_id="KLGA",
    train_start="2026-01-01",
    train_end="2026-01-20",
    val_start="2026-01-21",
    val_end="2026-01-28"
)

print(f"Best params: {result.best_params}")
print(f"Train score: {result.best_train_score:.4f}")
print(f"Val score: {result.best_val_score:.4f}")

# Walk-forward
results = await loop.walk_forward(
    station_id="KLGA",
    start_date="2026-01-01",
    end_date="2026-02-28",
    train_days=14,
    val_days=7
)
```

---

## 6. Referencia de API

### 6.1 Endpoints de Replay

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/v4/recorder/stats` | GET | Estadísticas de grabación |
| `/api/v4/replay/dates` | GET | Fechas disponibles |
| `/api/v4/replay/dates/{date}/channels` | GET | Canales para una fecha |
| `/api/v4/replay/start` | POST | Iniciar sesión de replay |
| `/api/v4/replay/pause` | POST | Pausar replay |
| `/api/v4/replay/resume` | POST | Reanudar replay |
| `/api/v4/replay/seek` | POST | Seek a timestamp |
| `/api/v4/replay/events` | GET | Obtener eventos (polling) |
| `/api/v4/replay/stream` | GET | SSE stream de eventos |

### 6.2 Endpoints de Backtest

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/v5/backtest/dates` | GET | Fechas con datos grabados |
| `/api/v5/backtest/run` | POST | Ejecutar backtest |
| `/api/v5/backtest/labels/{station_id}` | GET | Labels de una estación |
| `/api/v5/backtest/metrics` | GET | Definiciones de métricas |
| `/api/v5/backtest/policies` | GET | Policies disponibles |

### 6.3 CLI de Backtest

```bash
# Ejecutar backtest
python backtest_cli.py run --station KLGA --start 2026-01-01 --end 2026-01-30

# Calibrar parámetros
python backtest_cli.py calibrate --station KLGA --param bias_alpha

# Generar reporte
python backtest_cli.py report --input results.json --format html

# Listar fechas disponibles
python backtest_cli.py list --station KLGA

# Validar no-leakage
python backtest_cli.py validate --station KLGA --date 2026-01-15
```

---

## Glosario

| Término | Definición |
|---------|------------|
| **As-Of Join** | Patrón de join que retorna el último valor válido para un timestamp dado |
| **Brier Score** | Métrica de calibración: (p - y)², menor es mejor |
| **Churn** | Suma de cambios absolutos en probabilidades entre timesteps |
| **ECE** | Expected Calibration Error: mide si probabilidades predichas coinciden con frecuencias observadas |
| **Edge** | Diferencia entre probabilidad del modelo y precio de mercado |
| **Fill** | Ejecución de una orden |
| **Ground Truth** | Valor real/verdadero usado para evaluar predicciones |
| **HybridReader** | Lector que combina NDJSON (reciente) y Parquet (compactado) |
| **NDJSON** | Newline-Delimited JSON: un objeto JSON por línea |
| **Policy** | Máquina de estados que decide cuándo y cómo operar |
| **Sharpness** | Qué tan "puntiaguda" es una distribución de probabilidad |
| **Staleness** | Edad de los datos (tiempo desde última actualización) |
| **TimelineState** | Snapshot del estado del mundo en un instante |
| **Walk-Forward** | Validación con ventanas deslizantes de train/test |

---

*Documento generado automáticamente basado en el código fuente de HELIOS v13.x*

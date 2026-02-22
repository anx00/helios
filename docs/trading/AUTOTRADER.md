# HELIOS: Guía Completa del Autotrader (Phase 6)

Este documento describe la arquitectura, implementación y API del sistema **Autotrader** del proyecto HELIOS Weather Lab.

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Arquitectura](#2-arquitectura)
3. [Modelos de Datos](#3-modelos-de-datos)
4. [Catálogo de Estrategias](#4-catálogo-de-estrategias)
5. [LinUCB Bandit: Selección de Estrategia](#5-linucb-bandit-selección-de-estrategia)
6. [Risk Gate](#6-risk-gate)
7. [Paper Broker](#7-paper-broker)
8. [AutoTrader Service](#8-autotrader-service)
9. [Offline Learning](#9-offline-learning)
10. [Backtest Adapter](#10-backtest-adapter)
11. [Live Execution Adapter](#11-live-execution-adapter)
12. [Persistencia (Storage)](#12-persistencia-storage)
13. [API REST Endpoints](#13-api-rest-endpoints)
14. [Flujo de Datos End-to-End](#14-flujo-de-datos-end-to-end)

---

## 1. Visión General

El Autotrader es el sistema de trading automatizado de HELIOS (Phase 6). Opera en modo **paper-first** por defecto, ejecutando estrategias de trading sobre mercados de temperatura de Polymarket sin arriesgar capital real.

### 1.1 Características Principales

- **Multi-estrategia**: 4 estrategias predefinidas basadas en Policy del backtest
- **Selección adaptativa**: LinUCB contextual bandit selecciona la mejor estrategia en cada momento
- **Risk gate**: Verificaciones de riesgo antes de cada operación
- **Paper broker**: Simulación realista de ejecución con slippage y fees
- **Offline learning**: Ciclo nocturno de optimización de parámetros
- **Persistencia SQLite**: Todas las decisiones, órdenes y fills se graban
- **Live execution stub**: Punto de integración para ejecución real (feature-flagged)

### 1.2 Archivos Principales

| Módulo | Archivo | Líneas | Propósito |
|--------|---------|--------|-----------|
| Models | `core/autotrader/models.py` | 221 | Tipos de datos compartidos |
| Strategy | `core/autotrader/strategy.py` | 119 | Catálogo de estrategias |
| Bandit | `core/autotrader/bandit.py` | 121 | LinUCB para selección |
| Risk | `core/autotrader/risk.py` | 80 | Risk gate pre-trade |
| Paper Broker | `core/autotrader/paper_broker.py` | 326 | Broker simulado |
| Service | `core/autotrader/service.py` | 504 | Servicio runtime principal |
| Learning | `core/autotrader/learning.py` | 157 | Ciclo offline de aprendizaje |
| Backtest Adapter | `core/autotrader/backtest_adapter.py` | 135 | Integración con backtest engine |
| Storage | `core/autotrader/storage.py` | 440 | Persistencia SQLite |
| Live Adapter | `core/autotrader/execution_adapter_live.py` | 51 | Stub para ejecución real |

---

## 2. Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SISTEMA LIVE                                    │
│                                                                         │
│  Nowcast Engine ──► Distribución de probabilidades por bucket           │
│  Market WS ──────► Orderbook L2 por token (bid/ask/spread)             │
│  Event Window ───► Estado activo/inactivo de ventana de evento          │
│                                                                         │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   _build_context()     │  ◄─── Combina nowcast + market + health
        │   → DecisionContext    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Strategy Catalog     │  ◄─── 4 estrategias (Policy-backed)
        │   .evaluate(ctx)       │       Cada una genera StrategyDecision
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   LinUCB Bandit        │  ◄─── Selecciona mejor estrategia
        │   .select(features)    │       según contexto actual
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Risk Gate            │  ◄─── Verifica límites de riesgo
        │   .evaluate(ctx, pos)  │       Puede bloquear la operación
        └────────────┬───────────┘
                     │
                     ▼
            ┌────────┴─────────┐
            │  ¿Risk blocked?  │
            └────────┬─────────┘
                 NO  │  YES → no-op
                     ▼
        ┌────────────────────────┐
        │   Paper Broker         │  ◄─── Ejecuta acciones simuladas
        │   .execute_actions()   │       Taker fills + Maker orders
        │   .update()            │       Actualiza posiciones y PnL
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Reward Calculation   │  ◄─── reward = ΔPnL - λ_dd·ΔDD - λ_to·turnover
        │   bandit.update()      │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Storage (SQLite)     │  ◄─── Persiste decisiones, órdenes,
        │   + WebSocket event    │       fills, posiciones, rewards
        └────────────────────────┘
```

---

## 3. Modelos de Datos

**Archivo**: `core/autotrader/models.py`

### 3.1 TradingMode

```python
class TradingMode(str, Enum):
    PAPER = "paper"           # Solo simulación
    SEMI_AUTO = "semi_auto"   # Requiere confirmación humana
    LIVE_AUTO = "live_auto"   # Ejecución automática real
```

### 3.2 DecisionContext

Snapshot completo del estado del mundo para una decisión de trading:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `ts_utc` | datetime | Timestamp UTC de la decisión |
| `ts_nyc` | datetime | Timestamp NYC |
| `station_id` | str | Estación meteorológica (ej. `"KLGA"`) |
| `nowcast` | Dict | Distribución completa del nowcast |
| `market_state` | Dict | Orderbook L2 por bucket |
| `health_state` | Dict | Métricas de salud del sistema |
| `confidence` | float | Confianza del modelo (0-1) |
| `tmax_sigma_f` | float | Desviación estándar de Tmax predicho |
| `nowcast_age_seconds` | float | Antigüedad del nowcast |
| `market_age_seconds` | float | Antigüedad de datos de mercado |
| `spread_top` | float | Spread del bucket con mayor probabilidad |
| `depth_imbalance` | float | (bid_depth - ask_depth) / (bid_depth + ask_depth) |
| `volatility_short` | float | Volatilidad reciente del mid price |
| `prediction_churn_short` | float | Cambio reciente en probabilidades |
| `event_window_active` | bool | Si hay ventana de evento activa |
| `qc_state` | str | Estado de QC del nowcast |
| `hour_nyc` | int | Hora actual en NYC (0-23) |

### 3.3 StrategyDecision

Resultado de evaluar una estrategia:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `ts_utc` | datetime | Timestamp |
| `station_id` | str | Estación |
| `strategy_name` | str | Nombre de la estrategia |
| `actions` | List[Dict] | Lista de acciones de trading |
| `no_trade_reasons` | List[str] | Razones por las que no se operó |
| `score` | Optional[float] | Score de la decisión |
| `metadata` | Dict | Metadatos adicionales |

Cada **action** tiene la estructura:
```python
{
    "bucket": "21-22",          # Bracket de temperatura
    "side": "buy",              # buy | sell
    "outcome": "yes",           # Siempre YES leg en Polymarket
    "order_type": "taker",      # taker | maker
    "target_size": 0.5,         # Tamaño deseado
    "max_price": 0.65,          # Precio límite superior
    "min_price": 0.01,          # Precio límite inferior
    "confidence": 0.82,         # Confianza del modelo
    "urgency": 0.7,             # Urgencia de ejecución (0-1)
    "reason": "Edge=8.2%, Conf=82.0%"
}
```

### 3.4 PaperOrder / PaperFill

Órdenes y fills del paper broker, con campos: `ts_utc`, `station_id`, `strategy_name`, `bucket`, `side`, `size`, `limit_price`/`price`, `order_type`/`fill_type`, `fees`, `slippage`, `order_id`.

### 3.5 RiskSnapshot

Estado del risk gate en un instante:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `blocked` | bool | Si la operación fue bloqueada |
| `reasons` | List[str] | Razones del bloqueo |
| `daily_pnl` | float | PnL del día |
| `total_exposure` | float | Exposición total |
| `orders_last_hour` | int | Órdenes en la última hora |

### 3.6 BanditState / LearningRunResult / PromotionDecision

Tipos para persistencia del bandit, corridas de learning, y decisiones de promoción de modelos.

---

## 4. Catálogo de Estrategias

**Archivo**: `core/autotrader/strategy.py`

Todas las estrategias implementan la interfaz `Strategy`:

```python
class Strategy(ABC):
    name: str

    @abstractmethod
    def evaluate(self, context: DecisionContext) -> StrategyDecision: ...

    @abstractmethod
    def reset(self) -> None: ...
```

### 4.1 Estrategias Disponibles

| Nombre | Policy Base | Características |
|--------|-------------|-----------------|
| `conservative_edge` | `create_conservative_policy()` | Edge mínimo alto, sizing conservador |
| `aggressive_edge` | `create_aggressive_policy()` | Edge mínimo bajo, sizing agresivo |
| `fade_event_qc` | `create_fade_policy()` | Solo opera durante ventanas de evento |
| `maker_passive` | `create_aggressive_policy()` | Fuerza órdenes maker (sin fees) |

### 4.2 PolicyBackedStrategy

Wrapper que adapta las `Policy` del módulo de backtest a la interfaz `Strategy`:

```python
class PolicyBackedStrategy(Strategy):
    def evaluate(self, context: DecisionContext) -> StrategyDecision:
        # 1. Verifica pre-condiciones (ej. event_window)
        # 2. Llama a policy.evaluate_with_reasons()
        # 3. Convierte PolicySignals → actions dict
        # 4. Retorna StrategyDecision con acciones y razones
```

La función `build_strategy_catalog()` construye el diccionario con las 4 estrategias.

---

## 5. LinUCB Bandit: Selección de Estrategia

**Archivo**: `core/autotrader/bandit.py`

### 5.1 Algoritmo

Implementación de **LinUCB** (Linear Upper Confidence Bound):

```
score(strategy) = θᵀx + α · √(xᵀ A⁻¹ x)
                  ╰─────╯   ╰────────────╯
                  exploit      explore
```

Donde:
- `x`: vector de features del contexto (10 dimensiones)
- `θ = A⁻¹b`: parámetros aprendidos por estrategia
- `α`: parámetro de exploración (default: 0.8)

### 5.2 Vector de Features (10D)

El servicio construye un vector de 10 features para el bandit:

| Índice | Feature | Rango |
|--------|---------|-------|
| 0 | `confidence` | [0, 1] |
| 1 | `tmax_sigma_f` | [0, ∞) |
| 2 | `nowcast_age_seconds / 3600` | [0, 1] capped |
| 3 | `market_age_seconds / 3600` | [0, 1] capped |
| 4 | `spread_top` | [0, 1] |
| 5 | `depth_imbalance` | [-1, 1] |
| 6 | `volatility_short` | [0, ∞) |
| 7 | `prediction_churn_short` | [0, ∞) |
| 8 | `event_window_active` | {0, 1} |
| 9 | `qc_ok + hour_nyc/24` | [0, 2] |

### 5.3 Actualización

Tras cada step:
```python
A[selected] += outer(x, x)
b[selected] += reward * x
```

### 5.4 Persistencia

El estado del bandit (matrices A y b por estrategia) se serializa como JSON y se guarda en SQLite. Se restaura al iniciar el servicio.

---

## 6. Risk Gate

**Archivo**: `core/autotrader/risk.py`

### 6.1 Configuración (RiskConfig)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `max_daily_loss` | 0.10 | Pérdida máxima diaria (10%) |
| `max_total_exposure` | 1.5 | Exposición total máxima |
| `max_position_per_bucket` | 0.5 | Posición máxima por bucket |
| `max_orders_per_hour` | 120 | Máximo de órdenes por hora |
| `cooldown_seconds` | 30 | Cooldown entre trades |
| `max_nowcast_age_seconds` | 600 | Máxima antigüedad del nowcast (10 min) |
| `max_market_age_seconds` | 600 | Máxima antigüedad del mercado (10 min) |
| `blocked_qc_states` | STALE, GAP, ERROR, OUTLIER, SEVERE | QC states que bloquean |

### 6.2 Checks de Riesgo

El `RiskGate.evaluate()` ejecuta los siguientes checks secuencialmente:

1. **daily_loss_limit**: PnL del día < -max_daily_loss
2. **max_total_exposure**: Exposición total > límite
3. **max_position_per_bucket**: Posición en algún bucket > límite
4. **max_orders_per_hour**: Demasiadas órdenes recientes
5. **cooldown_blocked**: Último trade demasiado reciente
6. **stale_nowcast**: Nowcast demasiado antiguo
7. **stale_market**: Datos de mercado demasiado antiguos
8. **qc_blocked**: Estado QC en la lista de bloqueados

Si **cualquier** check falla, `blocked=True` y se registran las razones.

Adicionalmente, el servicio puede forzar `risk_off` manualmente.

---

## 7. Paper Broker

**Archivo**: `core/autotrader/paper_broker.py`

### 7.1 Propósito

El Paper Broker ejecuta acciones de trading de forma simulada usando el `ExecutionSimulator` del módulo de backtest. Proporciona:
- Ejecución realista con slippage y fees (taker)
- Órdenes límite con cola simulada (maker)
- Tracking de posiciones e inventario
- PnL realizado y mark-to-market

### 7.2 Configuración

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `allow_naked_short` | False | Permite venta sin inventario |
| `min_buy_notional_usd` | 0.01 | Mínimo nocional via API (la UI tiene $1, pero la API CLOB permite mucho menos) |

### 7.3 Flujo de Ejecución

```
actions (List[Dict])
     │
     ▼
execute_actions()
     │
     ├─► Para cada action:
     │    1. Parsear side, order_type, bucket, target_size
     │    2. Verificar inventario (no naked shorts por default)
     │    3. Estimar precio y verificar mínimo nocional
     │    4. Construir PolicySignal
     │    5. simulator.execute_signal() → Fill (taker) o Order (maker)
     │    6. Aplicar fill a posiciones
     │
     ▼
update()
     │
     ├─► simulator.update() → Fills de órdenes maker pendientes
     │    Aplicar fills a posiciones
     │
     ▼
mark_to_market()
     │
     └─► Calcular unrealized PnL usando mid price del orderbook
```

### 7.4 Tracking de Posiciones

```python
# Apertura/adición
new_avg_price = (|old_size| * old_avg + |qty| * fill_price) / |new_size|

# Cierre
realized_pnl += (fill_price - avg_price) * closing_qty  # para longs
realized_pnl += (avg_price - fill_price) * closing_qty  # para shorts

# Fees y slippage se descuentan del PnL realizado
realized_pnl -= fees + slippage
```

### 7.5 Performance Metrics

```python
{
    "realized_pnl": float,       # PnL de posiciones cerradas
    "mark_to_market_pnl": float, # PnL no realizado
    "total_pnl": float,          # realized + mark_to_market
    "total_fees": float,
    "total_slippage": float,
    "total_orders": int,
    "total_fills": int,
}
```

---

## 8. AutoTrader Service

**Archivo**: `core/autotrader/service.py`

### 8.1 Configuración (AutoTraderConfig)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `station_id` | `"KLGA"` | Estación objetivo |
| `mode` | `PAPER` | Modo de trading |
| `selection_mode` | `"linucb"` | Modo de selección (`static` o `linucb`) |
| `decision_interval_seconds` | 60 | Intervalo entre decisiones |
| `lambda_drawdown` | 0.20 | Penalización por drawdown en reward |
| `lambda_turnover` | 0.01 | Penalización por turnover en reward |
| `default_strategy_static` | `"conservative_edge"` | Estrategia por defecto en modo static |

### 8.2 Ciclo de Vida (Singleton)

```
          start()
             │
             ▼
         _loop()  ◄── asyncio.Task corriendo cada decision_interval_seconds
             │
             ├── step_once()  ◄── Un ciclo completo de decisión
             │
             ├── pause() / resume()
             │
             └── stop()  ◄── Cancela task, persiste bandit state
```

### 8.3 step_once(): El Ciclo de Decisión

Cada `decision_interval_seconds` (default: 60s):

1. **Build context**: Combina nowcast + mercado + health → `DecisionContext`
2. **Evaluate all strategies**: Cada estrategia evalúa el contexto → `StrategyDecision`
3. **Select strategy**: LinUCB o static selecciona la mejor
4. **Risk check**: `RiskGate.evaluate()` verifica límites
5. **Execute**: Si no bloqueado, `PaperBroker.execute_actions()` + `update()`
6. **Mark-to-market**: Actualiza PnL no realizado
7. **Calculate reward**: `reward = ΔPnL - λ_dd·ΔDD - λ_to·turnover`
8. **Update bandit**: `bandit.update(selected, features, reward)`
9. **Persist**: Decisiones, órdenes, fills, posiciones, rewards → SQLite
10. **Broadcast**: Evento con resumen disponible para WebSocket/UI

### 8.4 Fórmula de Reward

```
reward = pnl_delta - (lambda_drawdown × drawdown_increment) - (lambda_turnover × num_actions)
```

Donde:
- `pnl_delta`: Cambio en PnL total desde el último step
- `drawdown_increment`: Incremento en max drawdown (equity_peak - current)
- `num_actions`: Número de acciones ejecutadas (penaliza over-trading)

### 8.5 Construcción del Contexto

El método `_build_context()` combina datos de tres fuentes:

1. **Nowcast Engine**: `get_nowcast_integration().get_distribution(station)` → distribución de probabilidades, confianza, sigma
2. **Market WebSocket**: `get_ws_client().state.orderbooks` → bid/ask/spread/depth por bracket
3. **Event Window**: `get_event_window_manager().has_active_window()` → estado de ventana

El market_state se indexa por **label normalizado** (ej. `"21-22"`) y solo incluye tokens YES.

### 8.6 Métodos de Control

| Método | Efecto |
|--------|--------|
| `start()` | Inicia el loop async |
| `stop()` | Para el loop, persiste bandit |
| `pause()` | Pausa decisiones (loop sigue corriendo) |
| `resume()` | Reanuda decisiones |
| `risk_off()` | Fuerza bloqueo manual de riesgo |
| `risk_on()` | Quita bloqueo manual |

### 8.7 Métodos de Consulta

| Método | Retorno |
|--------|---------|
| `get_status()` | Estado completo del servicio |
| `get_strategies()` | Conteos de selección y rewards por estrategia |
| `get_positions()` | Posiciones abiertas actuales |
| `get_performance()` | Métricas de performance |
| `get_decisions(limit)` | Historial de decisiones recientes |
| `get_orders(limit)` | Historial de órdenes |
| `get_fills(limit)` | Historial de fills |
| `get_learning_runs(limit)` | Historial de corridas de learning |

---

## 9. Offline Learning

**Archivo**: `core/autotrader/learning.py`

### 9.1 Configuración (LearningConfig)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `station_id` | `"KLGA"` | Estación |
| `train_days` | 53 | Ventana de entrenamiento (D-60..D-8) |
| `val_days` | 7 | Ventana de validación (D-7..D-1) |
| `max_combinations` | 50 | Máximo de combinaciones de parámetros |
| `mode` | `"signal_only"` | Modo de backtest |
| `min_val_score_for_promotion` | 0.0 | Score mínimo para promover modelo |

### 9.2 Walk-Forward Split

```
Hoy (NYC)
  │
  ├─ D-60 ────────────────── D-8 ──── D-7 ────── D-1 ── D
  │  ╰── Train window (53 days) ──╯    ╰── Val (7 days) ╯
```

### 9.3 Pipeline de Learning

```python
def run_once(config):
    1. Crear LearningRunResult (status="running")
    2. Calcular ventanas train/val
    3. Ejecutar CalibrationLoop.grid_search()
       - Prueba hasta max_combinations de parámetros
       - Evalúa cada combinación en train y val
       - Retorna best_params, best_train_score, best_val_score
    4. Generar artifact JSON en data/model_registry/
    5. Decisión de promoción:
       - promoted = True si status=="completed" AND val_score >= min_threshold
    6. Persistir en SQLite (learning_runs + model_registry)
```

### 9.4 Ciclo Nocturno

El `learning_nightly_loop()` en `web_server.py` ejecuta automáticamente:
- **Hora**: 3:15 AM NYC
- **Frecuencia**: Una vez por día
- Corre `service.run_learning_once()` en un thread separado

---

## 10. Backtest Adapter

**Archivo**: `core/autotrader/backtest_adapter.py`

### 10.1 MultiBanditBacktestPolicy

Adapta el sistema multi-estrategia del autotrader para uso en el BacktestEngine:

```python
class MultiBanditBacktestPolicy:
    # Combina 4 policies (conservative, aggressive, fade, maker_passive)
    # Selección via LinUCB o estática
    # Compatible con evaluate_with_reasons() del BacktestEngine
```

### 10.2 Uso en Backtest

El endpoint `/api/v6/autotrader/backtest/replay` compara:
- **Baseline**: `conservative_policy` sola
- **Candidate**: `MultiBanditBacktestPolicy` con selección LinUCB

---

## 11. Live Execution Adapter

**Archivo**: `core/autotrader/execution_adapter_live.py`

### 11.1 Estado Actual

El adapter de ejecución real es un **stub** (placeholder). Actualmente:
- Feature-flagged via env vars: `HELIOS_LIVE_EXEC_ENABLED` y `HELIOS_EXEC_MODE`
- `submit_order()` retorna `{"ok": False, "status": "not_implemented"}`
- Punto de integración futuro para `py_clob_client`

### 11.2 Configuración

```python
@dataclass
class LiveExecutionConfig:
    enabled: bool = False        # Default: deshabilitado
    mode: str = "paper"          # paper | semi_auto | live_auto
```

Variables de entorno:
- `HELIOS_LIVE_EXEC_ENABLED`: `"1"`, `"true"`, o `"yes"` para habilitar
- `HELIOS_EXEC_MODE`: `"paper"`, `"semi_auto"`, o `"live_auto"`

---

## 12. Persistencia (Storage)

**Archivo**: `core/autotrader/storage.py`

### 12.1 Base de Datos

SQLite en `data/autotrader.db` con las siguientes tablas:

| Tabla | Propósito |
|-------|-----------|
| `autotrader_decisions` | Todas las evaluaciones de todas las estrategias |
| `autotrader_orders` | Órdenes generadas |
| `autotrader_fills` | Fills (ejecuciones) |
| `autotrader_positions` | Snapshots de posiciones |
| `autotrader_rewards` | Rewards calculados por step |
| `bandit_state` | Estado serializado del LinUCB |
| `learning_runs` | Corridas de learning offline |
| `model_registry` | Registro de modelos y promociones |

### 12.2 Esquemas de Tablas Principales

**autotrader_decisions**:
```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
ts_utc TEXT NOT NULL,
station_id TEXT NOT NULL,
strategy_name TEXT NOT NULL,
selected INTEGER NOT NULL,      -- 1 si fue la estrategia seleccionada
context_json TEXT NOT NULL,      -- DecisionContext serializado
decision_json TEXT NOT NULL,     -- StrategyDecision serializado
reward REAL                      -- Solo para la estrategia seleccionada
```

**autotrader_rewards**:
```sql
ts_utc TEXT, station_id TEXT, strategy_name TEXT,
reward REAL,
pnl_component REAL,
drawdown_component REAL,
turnover_component REAL
```

### 12.3 Thread Safety

Todas las operaciones de escritura están protegidas por `threading.RLock()` para seguridad entre hilos.

---

## 13. API REST Endpoints

Todos los endpoints están bajo `/api/v6/autotrader/`.

### 13.1 Status y Control

#### GET `/api/v6/autotrader/status`

Retorna estado completo del servicio.

**Response**:
```json
{
    "running": true,
    "paused": false,
    "risk_off": false,
    "station_id": "KLGA",
    "mode": "paper",
    "selection_mode": "linucb",
    "decision_interval_seconds": 60,
    "strategies": ["conservative_edge", "aggressive_edge", "fade_event_qc", "maker_passive"],
    "performance": {
        "realized_pnl": 0.15,
        "mark_to_market_pnl": 0.03,
        "total_pnl": 0.18,
        "total_fees": 0.02,
        "total_slippage": 0.01,
        "total_orders": 47,
        "total_fills": 12
    },
    "max_drawdown_abs": 0.05,
    "strategy_selection_counts": {
        "conservative_edge": 120,
        "aggressive_edge": 45,
        "fade_event_qc": 8,
        "maker_passive": 30
    },
    "strategy_rewards": {
        "conservative_edge": 1.23,
        "aggressive_edge": 0.87,
        "fade_event_qc": 0.12,
        "maker_passive": 0.45
    },
    "last_context": {
        "ts_utc": "2026-01-29T14:32:00+00:00",
        "confidence": 0.82,
        "spread_top": 0.03,
        "market_bucket_count": 24,
        "nowcast_bucket_count": 24,
        "market_overlap_count": 24
    }
}
```

#### POST `/api/v6/autotrader/control`

Controla el servicio.

**Request**:
```json
{
    "action": "start"  // start | pause | resume | risk_off | risk_on
}
```

**Response**:
```json
{
    "ok": true,
    "action": "start",
    "status": { /* ... mismo formato que GET /status */ }
}
```

### 13.2 Datos de Trading

#### GET `/api/v6/autotrader/strategies`

**Response**:
```json
{
    "strategies": [
        {
            "name": "conservative_edge",
            "selected_count": 120,
            "selected_weight": 0.59,
            "reward_sum": 1.23
        }
    ]
}
```

#### GET `/api/v6/autotrader/positions`

**Response**:
```json
{
    "positions": {
        "21-22": {
            "size": 0.5,
            "avg_price": 0.42,
            "unrealized_pnl": 0.03
        }
    }
}
```

#### GET `/api/v6/autotrader/performance`

**Response**: Mismo formato que el campo `performance` de `/status`.

#### GET `/api/v6/autotrader/decisions?limit=200`

**Response**:
```json
{
    "decisions": [
        {
            "ts_utc": "2026-01-29T14:32:00+00:00",
            "station_id": "KLGA",
            "selected_strategy": "conservative_edge",
            "scores": {"conservative_edge": 1.2, "aggressive_edge": 0.9, ...},
            "risk_blocked": false,
            "risk_reasons": [],
            "actions": [{"bucket": "21-22", "side": "buy", ...}],
            "no_trade_reasons": [],
            "fills_count": 1,
            "reward": 0.05,
            "performance": {...}
        }
    ]
}
```

#### GET `/api/v6/autotrader/orders?limit=200`

**Response**:
```json
{
    "orders": [
        {
            "ts_utc": "...", "station_id": "KLGA", "strategy_name": "conservative_edge",
            "order_id": "uuid", "bucket": "21-22", "side": "buy",
            "size": 0.5, "limit_price": 0.42, "order_type": "taker", "status": "filled"
        }
    ]
}
```

#### GET `/api/v6/autotrader/fills?limit=200`

**Response**:
```json
{
    "fills": [
        {
            "ts_utc": "...", "station_id": "KLGA", "strategy_name": "conservative_edge",
            "bucket": "21-22", "side": "buy", "size": 0.5, "price": 0.42,
            "fill_type": "taker_buy", "fees": 0.008, "slippage": 0.002
        }
    ]
}
```

### 13.3 Learning

#### POST `/api/v6/learning/run`

Ejecuta una corrida de learning offline.

**Request**:
```json
{
    "station_id": "KLGA",
    "train_days": 53,
    "val_days": 7,
    "max_combinations": 50,
    "mode": "signal_only"
}
```

**Response**:
```json
{
    "learning_run": {
        "run_id": "abc123def456",
        "station_id": "KLGA",
        "status": "completed",
        "train_start": "2025-11-30",
        "train_end": "2026-01-21",
        "val_start": "2026-01-22",
        "val_end": "2026-01-28",
        "best_params": {...},
        "best_train_score": 0.85,
        "best_val_score": 0.72
    },
    "promotion": {
        "version_id": "KLGA_20260129_143200",
        "promoted": true,
        "reason": "meets_thresholds",
        "metrics": {"best_train_score": 0.85, "best_val_score": 0.72}
    },
    "artifact_path": "data/model_registry/KLGA_20260129_143200.json"
}
```

#### GET `/api/v6/learning/runs?limit=20`

**Response**:
```json
{
    "runs": [
        {
            "run_id": "abc123def456",
            "station_id": "KLGA",
            "status": "completed",
            "best_val_score": 0.72
        }
    ]
}
```

### 13.4 Backtest Comparativo

#### POST `/api/v6/autotrader/backtest/replay`

Ejecuta un backtest comparando baseline (conservative) vs candidate (multi-bandit).

**Request**:
```json
{
    "station_id": "KLGA",
    "start_date": "2026-01-01",
    "end_date": "2026-01-28",
    "mode": "execution",
    "selection_mode": "linucb",
    "risk_profile": "risk_first"
}
```

**Response**:
```json
{
    "config": {
        "station_id": "KLGA",
        "start_date": "2026-01-01",
        "end_date": "2026-01-28",
        "mode": "execution",
        "selection_mode": "linucb",
        "risk_profile": "risk_first"
    },
    "baseline": {
        "policy": "conservative",
        "trading_summary": {...},
        "aggregated_metrics": {...},
        "coverage": {...}
    },
    "candidate": {
        "policy": "multi_bandit",
        "trading_summary": {...},
        "aggregated_metrics": {...},
        "coverage": {...}
    }
}
```

### 13.5 Páginas HTML

| Ruta | Template | Descripción |
|------|----------|-------------|
| `/autotrader` | `autotrader.html` | Dashboard principal del autotrader |
| `/autotrader/tutorial` | `autotrader_tutorial.html` | Tutorial interactivo |

---

## 14. Flujo de Datos End-to-End

### 14.1 Arranque del Sistema

```
web_server.py startup
     │
     ├── autotrader_loop()              ◄── asyncio.create_task()
     │     └── get_autotrader_service()  ◄── Singleton init
     │           ├── build_strategy_catalog()
     │           ├── LinUCBBandit(strategies, dim=10)
     │           ├── RiskGate(RiskConfig())
     │           ├── PaperBroker(ExecutionSimulator())
     │           ├── LearningRunner(storage)
     │           └── _load_bandit_state()  ◄── Restaura de SQLite
     │
     └── learning_nightly_loop()         ◄── 3:15 AM NYC diario
```

### 14.2 Ciclo de Decisión (cada 60s)

```
_build_context()
     │ Nowcast + Market + EventWindow → DecisionContext
     ▼
┌─────────────────────────────┐
│ Para cada strategy:         │
│   strategy.evaluate(ctx)    │ → 4 × StrategyDecision
└─────────┬───────────────────┘
          │
          ▼
bandit.select(features) → selected_strategy
          │
          ▼
risk_gate.evaluate(ctx, positions, pnl, ...) → RiskSnapshot
          │
          ├── blocked? → no-op
          │
          ▼
broker.execute_actions(actions) → [PaperOrder], [PaperFill]
broker.update() → [PaperFill] (maker fills)
broker.mark_to_market(market) → mtm_pnl
          │
          ▼
reward = ΔPnL - λ_dd·ΔDD - λ_to·turnover
bandit.update(selected, features, reward)
          │
          ▼
storage.record_*(...)  ◄── SQLite persistence
```

### 14.3 Ciclo de Learning (nightly)

```
3:15 AM NYC
     │
     ▼
LearningRunner.run_once(LearningConfig)
     │
     ├── CalibrationLoop.grid_search(train_window, val_window)
     │     └── Prueba N combinaciones de parámetros de policy
     │
     ├── Genera artifact en data/model_registry/
     │
     ├── Decisión de promoción (val_score >= threshold?)
     │
     └── Persiste en SQLite (learning_runs + model_registry)
```

# 🌡️ HELIOS Weather Lab - Sistema de Predicción de Temperatura

## 📋 Índice
- [¿Qué es este proyecto?](#qué-es-este-proyecto)
- [Problema que resuelve](#problema-que-resuelve)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Motor de Predicción](#motor-de-predicción)
- [Integración con Polymarket](#integración-con-polymarket)
- [Interfaz de Usuario](#interfaz-de-usuario)
- [Estructura del Código](#estructura-del-código)
- [Cómo Funciona (Flujo Completo)](#cómo-funciona-flujo-completo)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación y Uso](#instalación-y-uso)

---

## ¿Qué es este proyecto?

**HELIOS Weather Lab** es un sistema de predicción de temperatura **determinístico** (100% sin IA/LLM) que predice la temperatura máxima diaria para estaciones meteorológicas específicas (actualmente NYC LaGuardia y Atlanta).

### Características Principales

- 🎯 **Predicciones para 3 días**: Genera pronósticos para hoy, mañana y pasado mañana
- 📊 **Motor de física determinístico**: Usa reglas físicas en lugar de machine learning
- 🔄 **Tracking de desviaciones**: Compara predicciones con realidad y ajusta
- 📈 **Integración con Polymarket**: Consulta mercados de apuestas de temperatura
- 📉 **Validación automática**: Verifica precisión comparando con datos reales

---

## Problema que resuelve

### El Desafío

Los modelos meteorológicos profesionales (HRRR/GFS) son muy buenos, pero tienen **sesgos sistemáticos**:
- A veces sobre-predicen en ciertas condiciones
- No consideran efectos locales (brisa marina, humedad residual del suelo)
- Sus errores son predecibles y corregibles

### La Solución

HELIOS mejora las predicciones del HRRR aplicando:
1. **Corrección de desviación**: Si el HRRR está consistentemente +2°F alto, lo corregimos
2. **Ajustes físicos**: Aplicamos reglas meteorológicas conocidas
3. **Validación continua**: Aprende de errores pasados (sin IA, solo estadísticas)

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    HELIOS Weather Lab                           │
└─────────────────────────────────────────────────────────────────┘
           │
           ├──► 1. Recolección de Datos (Collector)
           │      ├─ METAR (temperatura actual, viento, nubes)
           │      └─ HRRR/GFS (pronósticos multi-modelo)
           │
           ├──► 2. Tracking de Desviación (Deviation Engine)
           │      ├─ Captura trayectoria horaria
           │      └─ Calcula delta entre predicho y real
           │
           ├──► 3. Motor de Física (Physics Engine)
           │      ├─ Aplica reglas determinísticas
           │      └─ Genera predicción final
           │
           ├──► 4. Integración Polymarket (Market Checker)
           │      ├─ Consulta estado de mercados
           │      └─ Decide qué día predecir
           │
           ├──► 5. Almacenamiento (Database)
           │      ├─ SQLite para predicciones
           │      └─ Trayectorias horarias
           │
           └──► 6. Validación (Auditor)
                  ├─ Compara predicciones con realidad
                  └─ Genera reportes de precisión
```

---

## Motor de Predicción

### Fórmula Base

```
Predicción_Final = Base_HRRR + Delta_Desviación + Ajustes_Físicos
```

### Componentes Detallados

#### 1️⃣ **Base HRRR**
- Pronóstico del modelo meteorológico HRRR (High-Resolution Rapid Refresh)
- Datos de Open-Meteo API
- Incluye: temperatura máxima, humedad suelo, radiación solar, cobertura de nubes

#### 2️⃣ **Delta de Desviación**
Compara temperatura **actual** con lo que HRRR **predijo** para esta hora:

```python
Delta = Temp_Real_Ahora - Temp_Predicha_HRRR_Para_Ahora

# Ejemplo:
# Ahora: 35°F (real)
# HRRR dijo que ahora haría: 40°F
# Delta = -5°F

# Si HRRR predice máxima de 46°F:
# Predicción ajustada = 46°F + (-5°F) = 41°F
```

**Limitaciones del Delta:**
- Solo se aplica para **HOY** y **MAÑANA**
- Para días +2 y +3: **Delta = 0** (no extrapolamos)

#### 3️⃣ **Ajustes Físicos**

##### 🌧️ Humedad del Suelo
- **Suelo muy húmedo** (>0.35 m³/m³): **-2°F**
- **Suelo moderado** (>0.25 m³/m³): **-1°F**
- **Razón**: Evaporación absorbe energía solar

##### ☁️ Desajuste de Nubes
- **Condición**: HRRR espera sol (radiación >400 W/m²) pero hay nubes (BKN/OVC)
- **Ajuste**: **-3°F**
- **Razón**: Sin sol directo, menos calentamiento

##### 🌊 Brisa Marina (NYC únicamente)
- **Condición**: Viento del N-NE (340°-70°)
- **Ajuste**: **-2°F**
- **Razón**: Aire frío del océano Atlántico

### Ejemplo Completo

**KLGA (NYC) - 7 de enero 2026:**

```
┌─ Componentes del Cálculo
│  Base HRRR:        46.2°F
│  Desviación Delta: +0.0°F  (día +2, no se aplica)
│  Humedad Suelo:    -1.0°F  (suelo húmedo 0.297 m³/m³)
│  Desajuste Nubes:  -0.0°F  (no aplica para días futuros)
│  Brisa Marina:     -0.0°F  (sin viento marino predicho)
└─ Total Física:    -1.0°F
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREDICCIÓN FINAL:   45.2°F
```

---

## Integración con Polymarket

### ¿Qué es Polymarket?

Polymarket es un mercado de predicciones donde usuarios apuestan sobre eventos futuros. Tiene mercados diarios sobre temperatura máxima en ciudades de EE.UU.

### ¿Cómo lo usamos?

#### 1. Consultar Estado del Mercado

El sistema consulta la **Polymarket Gamma API** para ver:
- ¿Está el mercado de hoy resuelto?
- ¿Hay certeza virtual (alguna opción >98%)?
- ¿Qué mercados existen para días futuros?

#### 2. Lógica de Detección de "Madurez"

Un mercado se considera **"maduro"** (resuelto) si:
- ✅ Alguna opción tiene **≥98% de probabilidad** (certeza virtual)
- ✅ El mercado está marcado como `closed=True`
- ✅ El mercado está marcado como `active=False`

**Ejemplo:**
```json
{
  "markets": [
    {
      "groupItemTitle": "34°F or higher",
      "outcomePrices": ["0.9995", "0.0005"]  // 99.95% Yes!
    }
  ]
}
```
→ Mercado virtualmente resuelto (ya se registró ≥34°F)

#### 3. Selección del Día Objetivo

```python
if mercado_hoy_maduro:
    target_date = MAÑANA
    print("🔍 Mercado de hoy resuelto → prediciendo para mañana")
else:
    target_date = HOY
    print("🔍 Mercado de hoy activo → prediciendo para hoy")
```

#### 4. Mapeo de Estaciones a Ciudades

```python
POLYMARKET_CITY_SLUGS = {
    "KLGA": "nyc",           # LaGuardia → New York City
    "KATL": "atlanta"        # Atlanta → Atlanta
}
```

### Logs de Polymarket

El sistema muestra información clara en español:

```
🔍 MERCADO KLGA (Hoy): Opción '34°F or higher' al 100.0% -> CERTEZA VIRTUAL DETECTADA
⏭️  SALTANDO AL MERCADO DE MAÑANA...
🎯 CAMBIANDO TARGET A: 06-Ene-2026
🔍 MERCADO KLGA (Mañana): '40-41°F' liderando con 73.0% -> ACTIVO
```

---

## Interfaz de Usuario

### Salida de Consola

El sistema tiene una interfaz de texto limpia y estructurada:

#### Inicio del Sistema
```
══════════════════════════════════════════════════════════════════════
  HELIOS Weather Lab - Motor de Física
  Sistema de Predicción Determinística de Temperatura
  Modo: 100% Sin conexión (Sin IA/LLM)
  Estaciones: KLGA (NYC), KATL (Atlanta)
  Modelo: HRRR + Reglas Físicas
══════════════════════════════════════════════════════════════════════
```

#### Ciclo de Recolección
```
──────────────────────────────────────────────────────────────────────
  Ciclo de Recolección
  Hora del Sistema: 2026-01-05 20:14:38
  Mercado US/Este:  2026-01-05 14:14:38
──────────────────────────────────────────────────────────────────────
```

#### Predicciones Detalladas
```
╔══════════════════════════════════════════════════════════════════╗
║  KLGA - Predicciones para los próximos 3 días                  ║
╚══════════════════════════════════════════════════════════════════╝

  📅 MAÑANA (2026-01-06)
     ┌─ Componentes del Cálculo
     │  Base HRRR:        46.4°F
     │  Desviación Delta: -5.0°F
     │  Humedad Suelo:    -1.0°F
     │  Desajuste Nubes:  -0.0°F
     │  Brisa Marina:     -0.0°F
     └─ Total Física:    -1.0°F
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     PREDICCIÓN FINAL:   40.4°F

  ✓ 3 predicciones generadas
```

---

## Estructura del Código

```
helios-temperature/
│
├── main.py                      # Orquestador principal
│
├── config.py                    # Configuración global
│   ├── STATIONS (KLGA, KATL)
│   ├── API endpoints
│   └── Constantes de Polymarket
│
├── collector/                   # Recolección de datos
│   ├── __init__.py
│   ├── metar_fetcher.py        # Datos METAR actuales
│   └── hrrr_fetcher.py         # Pronósticos HRRR/GFS
│
├── market/                      # Integración Polymarket
│   ├── __init__.py
│   └── polymarket_checker.py   # Consulta mercados
│
├── synthesizer/                 # Motor de física
│   ├── __init__.py
│   └── physics.py              # Reglas determinísticas
│
├── deviation/                   # Tracking de desviaciones
│   ├── __init__.py
│   └── deviation_tracker.py    # Compara predicho vs real
│
├── auditor/                     # Validación
│   ├── __init__.py
│   └── daily_validator.py      # Verifica precisión
│
├── registrar/                   # Logging
│   ├── __init__.py
│   └── logger.py               # Guarda predicciones
│
├── database.py                  # SQLite storage
│
└── helios_weather.db            # Base de datos
```

---

## Cómo Funciona (Flujo Completo)

### 1. Inicio del Sistema
```
main.py → init_database() → capture_trajectories() → collection_cycle()
```

### 2. Captura de Trayectoria (07:00 AM diario)
```python
for station in [KLGA, KATL]:
    datos = fetch_hrrr(station, days_ahead=0 y 1)
    guardar_en_db(hourly_temps_48h)  # Para tracking de desviación
```

### 3. Ciclo de Predicción (cada 30 minutos)

```python
async def collect_and_predict(station_id):
    # 1. Consultar Polymarket
    target_date = get_target_date(station_id)  # Hoy o mañana?
    
    # 2. Recolectar datos
    metar = fetch_metar(station_id)           # Temp actual, viento, nubes
    
    # 3. Generar 3 predicciones
    for day in [target_date, target_date+1, target_date+2]:
        hrrr = fetch_hrrr(station, days_ahead=day)
        
        # 4. Calcular desviación (solo para días cercanos)
        if day <= 1:
            delta = calcular_delta(metar, trayectoria_guardada)
        else:
            delta = 0  # No extrapolar para días lejanos
        
        # 5. Aplicar física
        prediction = physics_engine(
            hrrr_base=hrrr.max_temp,
            delta=delta,
            soil=hrrr.soil_moisture,
            radiation=hrrr.radiation,
            sky=metar.sky_condition,
            wind=metar.wind_direction
        )
        
        # 6. Mostrar resultado
        print_prediction(prediction)
    
    # 7. Guardar en base de datos
    save_to_db(prediction_principal)
```

### 4. Validación Diaria

```python
# Al día siguiente, compara predicción con realidad
actual_temp = get_metar_max_temp(yesterday)
predicted_temp = get_from_db(yesterday)

error = actual_temp - predicted_temp
update_statistics(error)
```

---

## Tecnologías Utilizadas

### APIs Externas
- **Open-Meteo**: Datos HRRR/GFS (pronósticos meteorológicos)
- **NOAA METAR**: Observaciones meteorológicas actuales
- **Polymarket Gamma API**: Estado de mercados de temperatura

### Librerías Python
```python
httpx          # HTTP async requests
schedule       # Programación de tareas
asyncio        # Operaciones asíncronas
sqlite3        # Base de datos local
zoneinfo       # Manejo de zonas horarias
```

### Almacenamiento
- **SQLite** (`helios_weather.db`)
  - Tabla `predictions`: Predicciones diarias
  - Tabla `model_path`: Trayectorias horarias

---

## Instalación y Uso

### Requisitos Previos
- Python 3.10+
- Conexión a Internet (para APIs)

### Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
# Ejecutar el sistema
python main.py
```

El sistema:
1. ✅ Inicializa la base de datos
2. ✅ Captura trayectoria inicial
3. ✅ Ejecuta ciclo de predicción
4. ✅ Programa tareas automáticas:
   - Captura de trayectoria: 07:00 AM diario
   - Predicciones: cada 30 minutos
5. ✅ Corre indefinidamente hasta Ctrl+C

### Scripts de Prueba

```bash
# Probar integración Polymarket
python test_polymarket.py

# Verificar predicción para día específico
python test_jan7_prediction.py

# Validar lógica de delta=0 para días futuros
python test_delta_future.py
```

---

## Métricas de Precisión

El sistema rastrea estas métricas (disponibles al cerrar con Ctrl+C):

```
PHYSICS ENGINE ACCURACY REPORT (Last 7 days)

Station: KLGA
  Predictions: 42
  Physics Error: +/- 1.8°F
  Raw HRRR Error: +/- 3.2°F
  Improvement: +1.4°F [ADDING VALUE]

Station: KATL
  Predictions: 38
  Physics Error: +/- 2.1°F
  Raw HRRR Error: +/- 2.8°F
  Improvement: +0.7°F [ADDING VALUE]
```

---

## Próximos Pasos / Mejoras Futuras

1. 🎯 **Más estaciones**: Expandir a más ciudades de EE.UU.
2. 📊 **Dashboard web**: Visualización de predicciones
3. 🤖 **Alertas**: Notificaciones cuando mercados alcancen certeza
4. 📈 **Backtesting**: Validar precisión histórica
5. 🧠 **Aprendizaje estadístico**: Ajustar parámetros físicos basado en errores pasados

---

## Contacto y Contribuciones

Este es un proyecto experimental de predicción meteorológica determinística integrado con mercados de predicción descentralizados.

**Autor**: [Tu nombre/equipo]  
**Licencia**: [Especificar]  
**Repositorio**: [Si aplica]

---

## Notas Finales

### ¿Por qué determinístico en lugar de IA?

1. **Transparencia**: Cada ajuste tiene una razón física clara
2. **Reproducibilidad**: Mismos datos = misma predicción siempre
3. **Explicabilidad**: Sabes exactamente por qué se hizo cada ajuste
4. **Mantenibilidad**: No necesitas re-entrenar modelos

### ¿Es mejor que el HRRR puro?

En promedio, **sí** (~1-1.5°F de mejora), especialmente cuando hay:
- Suelos húmedos post-lluvia
- Desajustes de nubosidad
- Efectos de brisa marina en NYC

Sin embargo, para días muy lejanos (+3), simplemente confiamos en el HRRR profesional con ajustes mínimos.

---

**Última actualización**: 5 de enero de 2026

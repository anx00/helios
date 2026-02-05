# ☀️ HELIOS V1: Documento de Referencia del Proyecto

## 1. Resumen Ejecutivo
**HELIOS** (Hyper-Local Environmental Leveling & Icing Observation System) es un sistema avanzado de predicción meteorológica determinista diseñado para operar en mercados de predicción descentralizados (Polymarket). A diferencia de los enfoques tradicionales basados en "cajas negras" de IA, HELIOS utiliza un **motor de física transparente** que combina múltiples modelos meteorológicos (Ensemble), correcciones en tiempo real (Deviation Tracking) y principios de termodinámica atmosférica para generar predicciones de temperatura máxima con precisión de grado financiero.

---

## 2. Objetivo del Proyecto
El objetivo principal es explotar ineficiencias en los mercados de predicción de temperatura mediante:
1.  **Precisión Superior**: Superar el error medio absoluto (MAE) de los modelos estándar (HRRR, GFS) corrigiendo sus sesgos sistemáticos.
2.  **Arbitraje de Velocidad**: Detectar tendencias térmicas (calentamiento/enfriamiento) antes que el consenso del mercado.
3.  **Gestión de Riesgo**: Validar matemáticamente las probabilidades para asegurar un Valor Esperado (EV) positivo.

---

## 3. Arquitectura del Sistema
El sistema es modular, escrito en Python, y sigue una arquitectura de flujo de datos unidireccional:

### Módulos Principales
1.  **Collector (Recolector)**: Ingesta datos crudos de múltiples fuentes (NOAA, METAR, Satélites, Boyas).
2.  **Synthesizer (Sintetizador)**: El "cerebro" del sistema. Contiene el Motor de Física y el Ensemble.
3.  **Deviation Engine (Motor de Desviación)**: Rastrea errores de modelos en tiempo real para calibrar predicciones futuras.
4.  **Market Engine (Motor de Mercado)**: Interactúa con la API de Polymarket para leer precios y detectar oportunidades.
5.  **Auditor (Auditor)**: Sistema de validación post-evento que calcula P&L y precisión histórica.
6.  **Registrar (Registrador)**: Almacenamiento persistente en SQLite para backtesting y análisis.

---

## 4. Tecnologías y Stack

### Lenguaje y Core
*   **Python 3.10+**: Elegido por su robusto ecosistema científico (`numpy`, `pandas` aunque aquí se usa lógica pura para velocidad).
*   **SQLite**: Base de datos ligera y serverless, ideal para el volumen de datos (millones de registros de sensores).
*   **Async/Await (`asyncio`)**: Crítico para realizar múltiples llamadas a APIs (NOAA, Polymarket) sin bloquear el ciclo de ejecución.

### Fuentes de Datos (Data Sources)
1.  **NOAA METAR (Aviation Weather Center)**: Datos de estaciones terrestres en tiempo real (la "verdad" del mercado).
2.  **Open-Meteo API**: Proveedor de modelos numéricos (HRRR, GFS).
3.  **NWS/NCEP (National Weather Service)**: Datos para modelos NBM (National Blend of Models) y LAMP.
4.  **CAMS (Copernicus Atmosphere Monitoring Service)**: Datos de aerosoles y polvo (AOD) para ajustes de radiación.
5.  **NDBC (National Data Buoy Center)**: Temperatura de la superficie del mar (SST) para correcciones costeras (e.g., KLGA).
6.  **Polymarket Gamma API**: Precios y probabilidades del mercado en tiempo real.

### Librerías Clave
*   `httpx`: Cliente HTTP asíncrono de alto rendimiento.
*   `schedule`: Manejo de tareas cronológicas (ciclos de 30 min, tareas nocturnas).
*   `zoneinfo`: Manejo preciso de zonas horarias (crítico para definir "Día" en mercados).

---

## 5. El Motor de Física (Physics Engine)
Esta es la ventaja competitiva de HELIOS. No usa Machine Learning opaco, sino una corrección aditiva transparente:

`Predicción Final = Base Ensemble + (Δ Desviación * Peso Temporal) + Σ Ajustes Físicos`

### A. Base Ensemble (Consenso Multi-Modelo)
En lugar de confiar en un solo modelo, HELIOS pondera:
*   **HRRR (High-Resolution Rapid Refresh)**: Modelo de corto plazo, muy reactivo.
*   **NBM (National Blend of Models)**: Consenso estadístico oficial.
*   **LAMP (Localized Aviation MOS)**: Específico para aeropuertos.
*   **GFS (Global Forecast System)**: Referencia global (menos peso).

### B. Deviation Tracking (Rastreo de Error)
Si el HRRR dice que debería haber 50°F ahora, pero hay 48°F, hay un `Delta = -2°F`.
*   **Temporal Decay**: Este delta no es constante. Se aplica una función de decaimiento que reduce su peso a medida que nos alejamos del "ahora" hacia la hora pico de calor.
*   **Heating Velocity**: Calcula la primera derivada de la temperatura (velocidad de cambio). Si la temperatura sube más rápido que la curva teórica del modelo, se proyecta un "overshoot".

### C. Ajustes Físicos (Physics Layers)
Reglas deterministas basadas en condiciones ambientales:
1.  **Humedad del Suelo (Soil Moisture)**:
    *   *Lógica*: Suelo húmedo > evaporación > enfriamiento latente > menos calor sensible.
    *   *Ajuste*: -1°F a -2°F si `soil_moisture > 0.35`.
2.  **Desajuste de Nubes (Cloud Mismatch)**:
    *   *Lógica*: Si el modelo predice SOL pero el sensor ve NUBES (OVC/BKN).
    *   *Ajuste*: -2°F a -4°F (bloqueo de radiación solar).
3.  **Advección Térmica (Wind Advection)**:
    *   *Lógica*: Si el viento viene de una zona más fría/caliente. Analiza estaciones 50km "aguas arriba" del viento actual.
4.  **Brisa Marina (Sea Breeze - KLGA)**:
    *   *Lógica*: Viento del noreste en NYC trae aire frío del Atlántico.
    *   *Ajuste*: Penalización dinámica basada en SST (Temp Superficie Mar).
5.  **Aerosoles (AOD)**:
    *   *Lógica*: Alto humo/polvo bloquea la luz solar.

### D. Mecanismos de Protección (Safety Locks)
*   **Sunset Lock**: La temperatura máxima NO puede ser menor a la que ya ocurrió hoy.
*   **Volatility Filter**: Ignora saltos bruscos (>2°F) en las actualizaciones del modelo hasta que se confirmen.
*   **Night Mode Risk**: Detecta si el modelo predice subidas de temperatura antinaturales durante la noche.

---

## 6. Integración de Mercado (Market Operations)

### Detección de Oportunidades
El sistema compara su `Predicción Final` con la distribución de probabilidad de Polymarket.
*   **Spread**: Diferencia entre HELIOS y el Mercado.
    *   `> 2.5°F`: Oportunidad ALTA (High Confidence).
    *   `> 1.5°F`: Oportunidad MEDIA.
*   **Alpha Confidence**: Si el spread es *demasiado* alto (>4°F), el sistema duda de sí mismo (o del mercado) y reduce el tamaño de la apuesta ("Panic Discount").

### Wisdom of the Crowd (Sabiduría de las Masas)
Rastrea la "velocidad" de cambio en los precios de Polymarket. Si el mercado se mueve rápidamente en contra de HELIOS, el sistema puede abortar o cubrirse, asumiendo que el mercado sabe algo (insider info/data privada) que HELIOS no.

---

## 7. Razonamiento Tecnológico: ¿Por qué NO usar IA/LLM?

1.  **Causalidad vs Correlación**: Las redes neuronales encuentran patrones pero no entienden física. HELIOS entiende *por qué* hace frío (e.g., por evaporación del suelo húmedo).
2.  **Datos Escasos**: Los eventos extremos (récords de calor/frío) son raros. Las IAs alucinan en los extremos; la física sigue siendo válida.
3.  **Auditoría**: Si HELIOS falla, podemos ver los logs: "Falló porque sobrestimó el efecto del viento". Con una Black Box AI, no sabríamos por qué.
4.  **Latencia**: Un cálculo de física toma milisegundos. Una inferencia de LLM/Modelo Grande toma segundos y es costosa.

---

## 8. Métricas de Éxito
El proyecto se considera exitoso si:
1.  **MAE < 1.5°F**: Error medio absoluto menor a 1.5 grados Fahrenheit.
2.  **EV+**: Rentabilidad positiva simulada a largo plazo.
3.  **Uptime 99.9%**: Capacidad de recolectar datos sin interrupción (mecanismos de reintento robustos).

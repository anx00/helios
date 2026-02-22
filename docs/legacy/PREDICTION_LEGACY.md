# HELIOS Engine: Guía Exhaustiva de Predicción y Cálculos

HELIOS es un motor de predicción meteorológica determinística de alta precisión, diseñado específicamente para predecir la temperatura máxima diaria con el fin de operar en mercados de predicción. A diferencia de los modelos meteorológicos tradicionales, HELIOS utiliza una arquitectura de **"Corrección sobre Referencia"**, donde toma un modelo base de alta resolución y aplica capas de física y observación en tiempo real para refinar el resultado.

---

## 1. Fuentes de Datos (Data Ingestion)

HELIOS monitoriza constantemente múltiples fuentes para alimentar su motor de física:

### A. Modelos Numéricos (Referencia Base)
*   **HRRR (High-Resolution Rapid Refresh):** Modelo principal de 3km de resolución. Se actualiza cada hora y proporciona la trayectoria de temperatura, radiación solar y humedad del suelo.
*   **GFS (Global Forecast System):** Utilizado como ensamble secundario para promediar la temperatura máxima teórica y evitar sesgos de un solo modelo.

### B. Observación en Tiempo Real (Reality Check)
*   **Protocolo NOAA RACER:** Triple validación de METAR (Aeropuertos) mediante:
    *   **JSON API:** Respuesta estructurada rápida.
    *   **TDS XML:** Datos del servidor de texto de NOAA.
    *   **TG-FTP TXT:** Acceso directo a archivos de texto Raw (el más rápido en actualizar).
*   **Wunderground (PWS):** Red de estaciones personales para validación cruzada y obtención del histórico horario (utilizado para el "Reality Floor").

### C. Variables "Alpha" (Física Avanzada)
*   **NDBC (Buoy SST):** Datos de boyas de NOAA para obtener la temperatura de la superficie del mar (SST) y calcular brisas marinas.
*   **CAMS (Copernicus Aerosol):** Datos de profundidad óptica de aerosoles (AOD) para detectar humo o calima que bloquee la radiación solar.
*   **Open-Meteo:** Fuente secundaria para modelos globales y datos de advección térmica.

---

## 2. El Proceso de Predicción

El cálculo se divide en tres fases principales que ocurren cada 5 minutos:

### Fase 1: El Cálculo del "Delta" (Desviación en Tiempo Real)
HELIOS no solo mira el futuro, mira el PRESENTE. 
1.  Compara la temperatura actual observada contra lo que el modelo HRRR *predijo* que debería hacer a esta misma hora.
2.  Si el HRRR dice que debería hacer 20°C y hace 22°C, tenemos un **Delta de +2°F**.
3.  **Temporal Decay:** Este Delta se pondera según la hora del día. A las 9:00 AM, el Delta tiene mucho peso (0.6x). A medida que nos acercamos al pico (15:00), el peso cae exponencialmente, ya que el modelo suele corregir sus propios errores al llegar al máximo.

### Fase 2: Capas de Ajuste Físico
Sobre el máximo del modelo, HELIOS aplica penalizaciones o bonos basados en reglas físicas determinísticas:

1.  **Humedad del Suelo:** Si el suelo está muy húmedo (>30%), parte de la energía solar se gasta en evaporar agua en lugar de calentar el aire. HELIOS resta entre **-1.0°F y -2.4°F**.
2.  **Inercia Térmica (Ensemble):** Promedia el máximo de HRRR y GFS para reducir la volatilidad de un solo modelo.
3.  **Advección Térmica (Dinámica de Fluidos):** Analiza el viento y la temperatura a 50km "río arriba". Si viene una masa de aire mucho más fría/caliente, aplica un ajuste de **+/- 3.0°F**.
4.  **Brisa Marina / Efecto Estuario:** En estaciones costeras (KLGA) o fluviales (EGLC), si el viento viene del agua y el agua está más fría que el aire proyectado, aplica una penalización de hasta **-2.0°F**.
5.  **Bloqueo por Aerosoles:** Si el AOD detecta humo denso (incendios, calima), resta hasta **-2.0°F** por la reducción de radiación solar directa.
6.  **Velocidad de Calentamiento:** Si la temperatura sube más rápido de lo proyectado en la última hora, aplica un bono de **+1.0°F** por "momentum".

### Fase 3: Reglas de Seguridad (Hard Limits)
*   **Reality Floor:** La predicción nunca puede ser inferior a la temperatura máxima ya alcanzada en el día.
*   **Sunset Hard Limit:** Una vez pasado el atardecer, la predicción se bloquea al máximo observado + 0.5°F, evitando que los modelos "alucinen" subidas nocturnas imposibles.

---

## 3. Ejemplo de Cálculo Matemático

Supongamos una tarde en NYC (KLGA):
1.  **Base HRRR Max:** 60.0°F
2.  **Obs. Actual:** 58.0°F | **HRRR Proyectado ahora:** 56.0°F → **Delta:** +2.0°F
3.  **Hora:** 11:00 AM (Confianza Delta: 0.45x) → **Impacto Delta:** +0.9°F
4.  **Ajustes Físicos:**
    *   Suelo húmedo (Post-lluvia): -1.2°F
    *   Viento del Norte (Brisa Marina): -1.0°F
    *   Advección cálida detectada: +0.5°F
    *   **Total Física:** -1.7°F
5.  **Cálculo:** `60.0 (Base) + 0.9 (Delta) - 1.7 (Física) = 59.2°F`

**Predicción HELIOS: 59.2°F** (Mientras que el HRRR original seguía marcando 60.0°F).

---

## 4. Comparación con el Mercado (Opportunity Analysis)

Finalmente, HELIOS compara su `59.2°F` con los brackets de Polymarket. Si el mercado da un 60% de probabilidad al bracket `60-61°F` y HELIOS está convencido de que no pasará de `59.2°F`, el sistema detecta una **Divergencia (Alpha)** y marca una oportunidad de **SHORT (Vender el favorito)**. 

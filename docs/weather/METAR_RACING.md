# Implementación de Datos NOAA METAR (Protocolo RACING)

Este documento detalla la lógica de implementación para la obtención de datos meteorológicos en tiempo real desde los servidores de la NOAA, con especial atención a la fuente de datos crudos en formato TXT (TG-FTP).

## Visión General: El Protocolo RACING

Para garantizar la máxima velocidad y disponibilidad, HELIOS no depende de una única fuente. En su lugar, utiliza un **Protocolo RACING** (Competencia de Datos) implementado en `collector/metar_fetcher.py`.

Cada vez que se solicita un dato, el sistema lanza tres peticiones concurrentes a diferentes APIs de la NOAA:
1.  **NOAA JSON API**: Datos estructurados rápidos.
2.  **AWC TDS XML**: Fuente redundante en formato XML.
3.  **NOAA TG-FTP TXT**: Datos crudos de alta precisión (el foco de este documento).

El sistema espera un máximo de 3 segundos y selecciona al "ganador" basándose en la frescura del dato y la completitud de la información.

---

## Implementación de la Fuente TG-FTP (Cruda TXT)

La fuente TG-FTP es a menudo la más actualizada, ya que es donde se depositan los ficheros directamente desde las estaciones de medición.

### 1. Origen de los Datos
El fetcher (`collector/metar/tgftp_fetcher.py`) accede directamente a la URL:
`https://tgftp.nws.noaa.gov/data/observations/metar/stations/{STATION_ID}.TXT`

### 2. Formato del Fichero TXT
El fichero devuelto por la NOAA tiene una estructura de dos líneas:
```text
2024/01/12 21:00
KATL 122052Z 31008KT 10SM FEW250 09/01 A3012 RMK AO2 SLP198 T00890006
```
*   **Línea 1**: Timestamp de la observación en formato UTC.
*   **Línea 2**: Cadena METAR completa con todos los parámetros técnicos.

### 3. Lógica de Parsing de Alta Precisión
A diferencia de otras aplicaciones que solo leen el bloque `09/01` (que solo da precisión de grados enteros), HELIOS implementa un parser de **T-Group** mediante expresiones regulares.

#### El T-Group (Precisión Decimétrica)
En la sección `RMK` (Remarks), muchas estaciones incluyen el código de temperatura de alta precisión: `T00890006`.
*   **T**: Prefijo del grupo.
*   **0**: Signo de temperatura (0 = Positivo, 1 = Negativo).
*   **089**: Valor de temperatura (8.9°C).
*   **0**: Signo de punto de rocío (0 = Positivo, 1 = Negativo).
*   **006**: Valor de punto de rocío (0.6°C).

#### Algoritmo de Extracción
```python
# Regex para capturar el grupo T de precisión
t_group = re.search(r' T([01])(\d{3})([01])(\d{3})', raw_ob)

if t_group:
    # 1 indica negativo, 0 indica positivo
    t_sign = -1 if t_group.group(1) == '1' else 1
    t_val = int(t_group.group(2)) / 10.0
    temp_c = t_sign * t_val
```

### 4. Mecanismos de Seguridad (Fallback)
Si el T-Group no está presente en el mensaje METAR, el sistema recurre al bloque estándar:
*   Regex: `(M?\d{2})/(M?\d{2}| )`
*   Ejemplo: `09/01` -> 9.0°C / 1.0°C.
*   El prefijo `M` se traduce automáticamente a valor negativo.

---

## Flujo de Datos

1.  **Captura**: `tgftp_fetcher.py` descarga el TXT y extrae la temperatura precisa.
2.  **Conversión**: La temperatura se normaliza a Celsius y luego se calcula su equivalente en Fahrenheit con un decimal de precisión.
3.  **Validación**: Se compara el timestamp del TXT con el de las otras fuentes del "Race".
4.  **Almacenamiento**: El dato ganador se guarda en la tabla `performance_logs`, permitiendo analizar qué fuente fue la más rápida en cada ciclo.

Esta implementación garantiza que HELIOS siempre trabaje con el dato más cercano a la realidad física antes de que sea redondeado o procesado por APIs de terceros.

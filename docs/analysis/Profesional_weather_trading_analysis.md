# Trading profesional en mercados de weather en Polymarket

**El edge real en weather markets no viene del pronóstico meteorológico en sí, sino de entender exactamente cómo se resuelven los mercados y explotar las ineficiencias entre modelos de predicción y precios del mercado.** Los traders más rentables — como gopfan2 con más de $2M en ganancias — combinan datos de ensemble forecasting gratuitos, conocimiento profundo de las estaciones ASOS/METAR de resolución, y automatización con bots que reaccionan en minutos cuando sale un nuevo model run. El ecosistema ya cuenta con herramientas dedicadas como wethr.net y Climate Sight, bots open-source funcionales, y un edge técnico crítico que pocos explotan: el error de redondeo °C↔°F en las observaciones METAR de 5 minutos que puede mover ±1°F el resultado final.

---

## Cómo resuelven exactamente los mercados y por qué esto es lo primero que debes dominar

Cada mercado de weather en Polymarket especifica en sus reglas una **estación ICAO concreta** y un URL exacto de Weather Underground History. La resolución se basa en la **temperatura más alta registrada** en esa estación específica durante el día, según aparece en la página de historial de WU una vez que los datos están finalizados. Las revisiones posteriores a la finalización no cuentan.

El mapeo confirmado de ciudades a estaciones es:

| Ciudad | Estación | ICAO | Unidad de resolución |
|--------|----------|------|---------------------|
| NYC | LaGuardia Airport | **KLGA** | °F |
| Chicago | O'Hare International | **KORD** | °F |
| Miami | Miami International | **KMIA** | °F |
| Dallas | Dallas Love Field | **KDAL** | °F |
| Seattle | Seattle-Tacoma | **KSEA** | °F |
| Atlanta | Hartsfield-Jackson | **KATL** | °F |
| London | London City Airport | **EGLC** | °F |
| Seoul | Incheon International | **RKSI** | **°C** |

**Dato crítico para NYC**: la estación es LaGuardia, **no Central Park** (que es la que usa NWS para registros climáticos oficiales de NYC). LaGuardia tiene microclima costero y puede diferir varios grados de Central Park en cualquier día dado. Si modelas basándote en pronósticos genéricos para "New York City," estás introduciendo error sistemático. Lo mismo aplica para Seoul: Incheon está a ~30 km de Seúl central, en zona costera, con temperaturas significativamente distintas al centro urbano.

Para las páginas de historial de WU con código ICAO, la fuente de datos es exclusivamente **METAR/ASOS** de esa estación de aviación — no se mezclan datos de Personal Weather Stations. Las PWS solo aparecen cuando navegas por ubicación/código postal en condiciones actuales.

El formato de URL para verificar manualmente es:
`https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA/date/2026-3-1`

### El bug de redondeo °C↔°F: el edge técnico más explotable

Este es probablemente el edge técnico más importante y menos conocido. Las estaciones ASOS en EE.UU. almacenan temperatura internamente en °F, pero los METAR se transmiten en °C entero. La cadena de conversión es: **°F medido → °C redondeado → transmitido vía METAR → reconvertido a °F → redondeado de nuevo**. Esto puede crear discrepancias de **±1°F** entre la temperatura real y lo que muestra Weather Underground.

Ejemplo documentado por NWS Phoenix: una estación mide un pico de 116°F. El promedio de 5 minutos da 116°F = 46.7°C, que se redondea a 47°C. Cuando WU reconvierte 47°C a °F, obtiene **117°F** — un grado más alto que la temperatura real. Para los METARs horarios (~:51-:54 de cada hora), existe un "T-group" con precisión de décimas de °C que mitiga esto. Pero **los METARs de 5 minutos carecen del T-group**, así que dependen del °C entero redondeado.

Implicación directa: cuando la temperatura está cerca de un límite de bucket (ej. 41-42°F vs 43-44°F), este error de redondeo puede determinar qué bucket gana. Los traders que entienden esta mecánica pueden anticipar mejor el valor de resolución usando datos OMO (One Minute Observations) disponibles con 1-2 días de retraso, y calibrando contra los METARs horarios con T-group en lugar de los de 5 minutos.

Para los mercados de **Seoul en °C**, este problema es menor porque METAR reporta nativamente en °C entero sin reconversión.

---

## Fuentes de datos y modelos: el stack gratuito que iguala al profesional

La buena noticia es que prácticamente toda la infraestructura de datos necesaria es gratuita. El stack óptimo para un weather trader serio tiene cinco capas.

**Capa 1: Ensemble forecasting para distribuciones de probabilidad.** Open-Meteo (open-meteo.com) proporciona acceso gratuito sin API key a los ensembles de ECMWF (51 miembros), GEFS (31 miembros), e ICON-EPS (40 miembros). El endpoint de ensemble es: `https://ensemble-api.open-meteo.com/v1/ensemble?latitude=40.71&longitude=-74.01&hourly=temperature_2m&models=ecmwf_ifs025`. Esto devuelve las 51 predicciones individuales de temperatura. Contando cuántos miembros caen en cada bucket del mercado se obtiene una distribución de probabilidad cruda. Combinando los tres sistemas (122+ miembros totales) se construye un "Grand Ensemble" con estimaciones más robustas.

**Capa 2: Pronósticos corregidos por bias (MOS y NBM).** El **National Blend of Models (NBM)** es el producto más valioso para weather trading — combina GFS, ECMWF, HRRR, NAM y otros modelos con corrección estadística, actualizado cada hora, a **2.5 km de resolución**. Proporciona directamente percentiles de probabilidad para temperatura máxima/mínima. Es, esencialmente, la distribución de probabilidad pre-calculada por NWS. Accesible vía AWS Open Data o NOMADS. El MOS clásico (MAV, MEX, LAMP) está disponible gratuitamente vía la API JSON de Iowa Environmental Mesonet: `https://mesonet.agron.iastate.edu/api/1/mos.json?station=KLGA&model=GFS`.

**Capa 3: Observaciones METAR en tiempo real.** AviationWeather.gov proporciona METAR decodificado en JSON gratis: `https://aviationweather.gov/api/data/metar?ids=KLGA&format=json`. Esto es esencial para el día del evento, cuando puedes monitorear temperaturas reales vs pronóstico y operar en consecuencia.

**Capa 4: Red de estaciones cercanas para tendencias.** Synoptic Data (synopticdata.com), sucesor de MesoWest, agrega **170,000+ estaciones** de 320+ redes. Permite monitorear temperaturas de estaciones cercanas a la de resolución para detectar tendencias antes de que el METAR oficial se actualice. Python library: `pip install SynopticPy`.

**Capa 5: Modelo específico para el corto plazo.** Para pronósticos de las próximas 0-48 horas en EE.UU., el **HRRR** (High-Resolution Rapid Refresh) a 3 km de resolución, actualizado cada hora, es superior a cualquier modelo global. Para 3-10 días, ECMWF mantiene ~1 día de ventaja sobre GFS (su pronóstico a 6 días ≈ GFS a 5 días en precisión).

### Comparación de modelos por horizonte temporal

| Horizonte | Mejor modelo | Resolución | Actualización |
|-----------|-------------|------------|---------------|
| 0-18 horas | **HRRR** | 3 km | Cada hora |
| 18-48 horas | HRRR (extendido) + NBM | 3 km / 2.5 km | Cada hora |
| 2-5 días | **ECMWF IFS** | ~9 km | Cada 6 horas |
| 5-10 días | ECMWF IFS | ~9 km | Cada 6 horas |
| Probabilístico (cualquier rango) | **NBM** | 2.5 km | Cada hora |

---

## Las estrategias documentadas de los traders más rentables

Cuatro perfiles de traders con ganancias verificables en leaderboards on-chain ilustran el espectro de estrategias que funcionan.

**gopfan2** (~$2M+ en ganancias) opera con reglas de precio simples pero extremadamente disciplinadas: compra YES cuando el precio está por debajo de $0.15 y NO cuando está por encima de $0.45, con un máximo de **$1 por posición** y miles de posiciones simultáneas. Su win rate reportado es ~70-80%. La lógica subyacente es que los mercados meteorológicos frecuentemente subvaloran el bucket correcto y sobrevaloran los adyacentes. Al mantener posiciones minúsculas, la ley de grandes números trabaja a su favor. Esta estrategia ha sido codificada en el skill open-source "simmer-weather" del Simmer SDK.

**neobrother** (~$20K+ en ganancias) usa **temperature laddering**: compra YES simultáneamente en múltiples buckets adyacentes (ej. 29°C, 30°C, 31°C, 32°C, 33°C, 34°C+). Es conceptualmente similar a un grid trading o straddle amplio en opciones. La idea es que mientras uno de los buckets gane, el costo total de las posiciones perdedoras sea inferior al payout del ganador.

**Hans323** ($1.11M en una sola apuesta de $92K) practica **black swan hunting**: apuestas grandes a eventos de baja probabilidad (2-8¢) con ratios asimétricos extremos. En mercados de weather, esto significa apostar a outcomes que los modelos asignan baja probabilidad pero que el mercado subvalora aún más — típicamente durante regímenes meteorológicos transicionales donde la incertidumbre real es mayor de lo que los participantes retail perciben.

**Bots automatizados** como meropi (~$30K con micro-apuestas de $1-3) operan escaneando continuamente los precios del mercado contra pronósticos actualizados. Un bot documentado convirtió $1,000 en $24,000 desde abril 2025 operando exclusivamente mercados de London weather.

### El flujo de trabajo de un trade típico basado en ensembles

El proceso paso a paso que usan los sistemas automatizados es: primero, obtener los 31 miembros del ensemble GFS (o 51 de ECMWF) para la ubicación y fecha del mercado. Segundo, para cada bucket de temperatura del mercado, contar cuántos miembros predicen temperatura dentro de ese rango — esto da la probabilidad del modelo. Tercero, comparar esa probabilidad con el precio implícito del mercado (el precio del share). Cuarto, si la divergencia supera un umbral mínimo (típicamente **8-15%**), generar señal de trading. Quinto, dimensionar la posición usando **quarter-Kelly**: `position = bankroll × (edge × confidence) / (1 - market_probability) × 0.25`.

El quarter-Kelly (25% del Kelly óptimo) reduce la volatilidad significativamente mientras mantiene ~75% de la tasa de crecimiento óptima. La configuración típica de un bot documentado en GitHub usa: bankroll base $10,000, edge mínimo 8%, Kelly fraction 0.25, máximo 10% del bankroll por trade.

---

## Timing, bots y el edge de velocidad

**Cuándo entrar importa tanto como qué comprar.** Los model runs de GFS y ECMWF se publican cada 6 horas (00, 06, 12, 18 UTC), con datos disponibles ~3.5-5.5 horas después de la inicialización. Los mercados de Polymarket frecuentemente tardan **30-60 minutos** en incorporar los nuevos datos después de que un model run se publica. Este lag es el edge de timing más consistente.

El run de **12Z es particularmente importante** porque incorpora datos de radiosondas matutinas (globos meteorológicos), lo que típicamente produce los cambios más significativos en el pronóstico. Un trader automatizado que parsea la salida del modelo inmediatamente obtiene ventaja temporal sobre participantes manuales.

Para el día del evento, la ventaja se traslada a observaciones en tiempo real. Si a las 2 PM hora local las estaciones cercanas (monitoreadas vía Synoptic Data) corren 3°F más calientes de lo que sugería el pronóstico matutino, la máxima probablemente superará las expectativas. Operar antes de que el mercado ajuste es el edge intraday.

### Ecosistema de bots funcionales

El ecosistema de bots ya está maduro. **Simmer SDK + OpenClaw** proporciona una solución no-code: OpenClaw es un agente AI local que se conecta a Polymarket vía el skill "simmer-weather", escaneando pronósticos NOAA contra precios cada 2 minutos y ejecutando trades automáticamente. Se configura con umbrales de entrada (15% divergencia), salida (45%), y tamaño máximo ($2-5 por trade). Integración con Telegram para comandos y alertas.

En GitHub, **suislanchez/polymarket-kalshi-weather-bot** es un framework Python que usa Open-Meteo para ensemble GFS, implementa Kelly criterion, y tiene un frontend React con globo 3D — aunque es solo simulación, no ejecuta trades reales. Para trading real vía API, Polymarket ofrece el **CLOB API** (clob.polymarket.com) con SDKs oficiales en Python (`py-clob-client`) y TypeScript, autenticación EIP-712, y WebSocket para precios en tiempo real.

**NautilusTrader** proporciona integración profesional con Polymarket para algo trading de grado institucional.

---

## Machine learning y calibración avanzada contra la estación de resolución

Los enfoques más sofisticados van más allá de ensemble counting. **Climate Sight** (climatesight.app) está desarrollando modelos ML entrenados específicamente en sensores co-localizados cerca de estaciones ASOS. La hipótesis es que sensores públicos cercanos no exhiben los errores de redondeo de ASOS, y con suficientes datos históricos de la misma ubicación, un modelo ML puede producir pronósticos más precisos que los modelos gridded a 0.25° de resolución.

Para feature engineering de un modelo ML orientado a weather trading, los inputs relevantes incluyen: datos METAR/ASOS históricos y en tiempo real (temperatura, punto de rocío, viento, presión, nubosidad), salidas de modelos determinísticos y ensemble (GFS, ECMWF, HRRR, NAM), **ensemble spread** como proxy de incertidumbre, normales climatológicas para la fecha y estación, correcciones históricas de bias del modelo vs la estación específica, posición de frentes y patrones sinópticos, hora de amanecer/atardecer y ángulo solar, y datos de sensores cercanos vía Synoptic Data. Las arquitecturas que funcionan incluyen gradient-boosted trees (XGBoost/LightGBM) para features tabulares y redes neuronales para relaciones no lineales entre múltiples salidas de modelos.

**Nonhomogeneous Gaussian Regression (NGR)** es la técnica estándar en meteorología profesional para convertir ensembles en distribuciones probabilísticas calibradas. Usa una transformación lineal del ensemble spread para definir el ancho de la distribución predictiva, superando la regresión lineal simple. Para un trader, esto significa que no basta con contar miembros del ensemble — la dispersión del ensemble contiene información real sobre la forma de la distribución, incluyendo skewness y kurtosis.

Un edge adicional documentado: **PolyMaster** ejecuta localmente el algoritmo GISTEMP v4 de NASA para mercados de anomalía de temperatura global, usando datos GHCN v4 y ERSSTv5, computando el resultado antes de la publicación oficial de NASA. El IC del 95% es ±0.05°C, pero en prediction markets diferencias de 0.01-0.02°C determinan ganadores y perdedores.

---

## Dónde encontrar comunidad y seguir aprendiendo

El ecosistema de recursos dedicados ha crecido significativamente. **wethr.net** es la plataforma analítica más completa, diseñada específicamente para weather trading en prediction markets. Ofrece gráficos de temperatura en tiempo real, 16+ capas de datos (METAR, DSM, CLI), comparación de modelos (GFS, HRRR, NAM, NBM, ECMWF, ICON), buscador de días análogos, y tracking de precipitación para 29+ mercados. Su sección educativa (wethr.net/edu) incluye guías de trading y explicaciones de la mecánica de redondeo ASOS. Tiene tier gratuito con datos retrasados 3 minutos y tiers pagados con acceso instantáneo y API.

**Climate Sight** (climatesight.app) cubre 30+ ciudades entre Polymarket, Kalshi y ForecastEx, con datos en tiempo real y una agenda de investigación ML publicada. **Weather Better** (weatherbetter.app) ofrece análisis ML-powered y comparaciones de modelos.

En Twitter/X, las cuentas relevantes son **@r_gopfan** (gopfan2), **@TheSpartanLabs** (desarrolladores de Simmer SDK), **@Wethrnet** y **@climatesight** (creadores de las plataformas analíticas), y **@0xMovez** que publica sobre estrategias tipo gopfan2. En Discord, el **servidor oficial de Polymarket** (~90K miembros) tiene discusiones de weather en canales generales. **PolyOdds** ofrece señales diarias data-driven. No existe un Discord dedicado exclusivamente a weather trading.

Los artículos más detallados sobre estrategias están en Medium por Ezekiel Njuguna (con caveat: contienen affiliate links), en Publish0x (guía no-code de bots), y en el blog de Kalshi (news.kalshi.com/p/trading-the-weather). En el ámbito académico, el **Bulletin of the American Meteorological Society** (Vol. 105, Issue 10, 2024) publicó "Can Expert Prediction Markets Forecast Climate-Related Risks?" documentando que en mercados con miles de trades, muchos participantes usan la API para colocar apuestas automatizadas informadas por sus propios modelos probabilísticos.

---

## Conclusión: los tres edges accionables ahora mismo

El weather trading en Polymarket opera en la intersección de meteorología cuantitativa, microestructura de mercado y automatización. Tres edges concretos emergen de esta investigación.

**Primero, el edge de información**: los datos de ensemble y NBM son gratuitos y proporcionan distribuciones de probabilidad superiores a lo que los participantes retail incorporan en los precios. Un sistema que compara probabilidades del ensemble contra precios del mercado y opera con quarter-Kelly sobre divergencias >8% genera retornos positivos consistentes con win rates del 70-80%.

**Segundo, el edge de resolución**: entender que NYC resuelve en KLGA (no Central Park), que el redondeo °C↔°F puede introducir ±1°F de error en las observaciones de 5 minutos, y que la página de historial de WU usa exclusivamente datos METAR de la estación ICAO especificada, permite calibrar modelos contra la fuente real de resolución en lugar de pronósticos genéricos por ciudad.

**Tercero, el edge de velocidad**: los model runs se publican en horarios fijos y los mercados tardan 30-60 minutos en ajustarse. Un bot que parsea salida de modelos inmediatamente y opera vía CLOB API captura valor antes de que el mercado alcance equilibrio. Este edge se erosiona a medida que más bots entran al mercado — múltiples fuentes notan que la ventana de oportunidad se está cerrando — pero la naturaleza de los mercados de weather (nuevos mercados cada día, múltiples ciudades, múltiples buckets) crea suficiente superficie de ataque para que traders disciplinados continúen encontrando valor.
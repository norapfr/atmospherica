# ATMOSPHERICA

> *Un sistema de Machine Learning que convierte datos atmosféricos en tiempo real en pintura abstracta generativa. Cada cuadro es único, irrepetible, y codifica el estado del clima del día en un lenguaje visual con gramática propia.*

---

## Índice

1. [Qué es ATMOSPHERICA](#qué-es-atmospherica)
2. [La idea central](#la-idea-central)
3. [Gramática visual](#gramática-visual)
4. [Variable dominante — el motor de la composición](#variable-dominante--el-motor-de-la-composición)
5. [Arquitectura del sistema](#arquitectura-del-sistema)
6. [Módulo 1 — Ingesta de datos](#módulo-1--ingesta-de-datos)
7. [Módulo 2 — Mapeador visual](#módulo-2--mapeador-visual)
8. [Módulo 3 — Generador pictórico (v2)](#módulo-3--generador-pictórico-v2)
9. [Estado actual del proyecto](#estado-actual-del-proyecto)
10. [Lo que queda por construir](#lo-que-queda-por-construir)
11. [Módulo 4 — Modelo predictivo ML](#módulo-4--modelo-predictivo-ml)
12. [Módulo 5 — Automatización y archivo](#módulo-5--automatización-y-archivo)
13. [Módulo 6 — Web de portfolio](#módulo-6--web-de-portfolio)
14. [Stack tecnológico](#stack-tecnológico)
15. [Estructura del repositorio](#estructura-del-repositorio)
16. [Cómo ejecutar](#cómo-ejecutar)
17. [Por qué esto es ML engineering, no solo arte](#por-qué-esto-es-ml-engineering-no-solo-arte)

---

## Qué es ATMOSPHERICA

ATMOSPHERICA es un sistema que ingiere datos meteorológicos y de calidad del aire en tiempo real de una ciudad, los procesa con un pipeline de normalización y mapeo paramétrico, y genera automáticamente una obra de arte abstracta única que codifica el estado atmosférico de ese día.

El sistema no genera imágenes decorativas ni visualizaciones de datos convencionales. Genera pinturas con una gramática visual definida y defendible donde cada decisión formal —color, tamaño, dirección, opacidad, fragmentación— proviene directamente de un dato climático real. El cuadro del 25 de abril de 2026 en Sevilla es diferente al del 26 de abril, y diferente al del mismo día en Oslo. La obra es el dato.

El proyecto tiene tres módulos completamente funcionales y tres módulos en desarrollo. El resultado final será un sistema autónomo que genera y publica una nueva obra cada día, construye un archivo histórico, y añade una capa de predicción de eventos climáticos extremos mediante modelos de series temporales.

---

## La idea central

Existe una distinción fundamental entre **visualización de datos** y **traducción de datos a lenguaje artístico**.

Una visualización muestra el dato. Un gráfico de temperatura a lo largo del día muestra líneas que suben y bajan. El dato es legible pero no se experimenta.

ATMOSPHERICA propone otra cosa: el dato **es** la forma. La temperatura no se muestra en un eje Y — determina el rango de color completo del cuadro. El viento no aparece como una flecha — sus cintas de pintura recorren el lienzo en la dirección exacta del viento real, con una longitud proporcional a su velocidad en metros por segundo. La presión atmosférica no es un número — decide si la composición se organiza en bandas horizontales arquitectónicas (anticiclón) o en rectángulos girados e inestables en los bordes (borrasca).

Esto tiene una consecuencia importante para el portfolio de ML: el sistema demuestra que su autor sabe construir pipelines de datos reales, normalizar variables con rangos históricos, diseñar sistemas de mapeo paramétrico complejos, y pensar en el output como un sistema con reglas, no como generación aleatoria.

---

## Gramática visual

La gramática visual es el corazón del proyecto. Es el conjunto de reglas que traduce cada variable climática en una decisión pictórica. Sin esta gramática, el proyecto sería "IA que genera imágenes bonitas". Con ella, es un sistema de codificación visual con semántica propia.

Cada variable produce una **forma característica** en el cuadro:

| Variable | Forma | Color | Comportamiento |
|---|---|---|---|
| Temperatura | Círculos concéntricos | Azul (frío) → naranja-rojo (calor), escala fija por °C | Más calor = anillos más grandes y numerosos |
| Viento | Curvas Bézier | Azul (más oscuro = más rápido) | Orientadas en la dirección geográfica exacta del viento |
| Humedad | Óvalos difusos / triángulos | Verde (más saturado = más húmedo) | <35% HR → triángulos nítidos; >35% → elipses con halos |
| Presión | Rectángulos y bandas | Ocre/siena (más sólido = más alta) | Alta presión → bandas horizontales; baja → rect. girados |
| Nubes | Rombos aplastados | Gris azulado (más oscuro = más cobertura) | Concentrados en la mitad superior del canvas |
| PM2.5 | Puntos y veladura | Violeta (más saturado = más contaminado) | La veladura afecta todas las capas subyacentes |

### Temperatura → Color y círculos concéntricos

El color de los círculos responde a un espectro continuo de seis rangos. Este color **no varía** con el dominante — es el único canal visual que mantiene su significado semántico absoluto independientemente del contexto:

| Rango | Color |
|---|---|
| ≤ 5°C | Azul profundo |
| 6–12°C | Azul-verde |
| 13–18°C | Verde |
| 19–24°C | Amarillo-naranja |
| 25–30°C | Naranja |
| ≥ 31°C | Rojo |

### Viento → Curvas Bézier orientadas

Los trazos de viento apuntan en la dirección geográfica real. El ángulo se descompone en componentes cartesianas (`wind_dx`, `wind_dy`) para orientar cada curva con precisión vectorial. La longitud es proporcional a la energía del viento. Cuando el fondo es claro (presión o nubes dominantes, `pL > 65`), el color del trazo se oscurece automáticamente para mantener el contraste — las líneas siempre se leen aunque el viento sea suave.

### Humedad → Triángulos o elipses según el nivel

Con humedad baja (< 35% y no dominante) la forma es un triángulo equilátero nítido — aire seco, geometría definida. Por encima del 35% o cuando la humedad domina, la forma cambia a óvalos con halos borrosos que se expanden: la pintura "se humedece".

### Presión → Bandas estructurales

Alta presión produce bandas horizontales que barren el canvas como arquitectura — el día es estable, el orden visual también. Baja presión produce rectángulos inclinados y fragmentados que se dispersan sin orden compositivo.

### Nubes → Rombos con peso vertical

Los rombos se concentran en la mitad superior del canvas (la nubosidad "cae desde arriba"). Con 0% de nubes no aparecen; con cobertura total llenan el cielo del cuadro.

### PM2.5 → Contaminación que ensucia todo

La veladura violácea se aplica como última capa sobre todo lo demás. Con aire limpio (< 5 μg/m³) es casi invisible. Con contaminación alta cubre el cuadro con niebla de partículas y venas de smog horizontales.

---

## Variable dominante — el motor de la composición

Esta es la pieza conceptual central de la versión 2 del generador. Las versiones anteriores pintaban todas las variables con el mismo peso visual. La v2 introduce un mecanismo de **dominancia**: la variable con mayor valor normalizado en ese momento **se apodera de la composición entera** y deforma el comportamiento de todas las demás.

### Cómo se calcula

Primero, el mapeador (`mapper.py`) normaliza cada variable a [0, 1] con rangos históricos locales de Sevilla:

```python
temperature_norm = normalize(data["temperature"], 0, 46)   # 0°C → 46°C máximo histórico
wind_energy      = normalize(data["wind_speed"], 0, 20)    # 0 calma → 20 m/s vendaval
humidity_norm    = veil_opacity / 80.0                     # 0 = seco, 80 = niebla densa
pressure_norm    = normalize(data["pressure"], 990, 1030)  # rango típico isobárico
cloud_norm       = data["clouds"] / 100.0                  # 0% despejado → 100% cubierto
pm_norm          = normalize(data["pm2_5"], 0, 75)         # 0 = limpio, 75 = muy malo (OMS)
```

Después, `generator.py` compara los seis valores normalizados y elige el mayor:

```python
def _compute_dominant(params):
    scores = {
        'temperatura': params['temperature_norm'],
        'viento':      params['wind_energy'],
        'humedad':     params['veil_opacity'] / 80.0,
        'presion':     params['density'],
        'nubes':       params['raw'].get('clouds', 15) / 100.0,
        'pm25':        params['fragmentation'],
    }
    dominant = max(scores, key=scores.get)
    return dominant, scores[dominant], scores
```

En el navegador, la misma lógica se replica en JavaScript sobre el objeto `C` (datos del día inyectados como JSON):

```javascript
const VARS = {
  temperatura: C.temp_norm,
  viento:      C.wind_energy,
  humedad:     C.humidity_norm,
  presion:     C.pressure_norm,
  nubes:       C.cloud_norm,
  pm25:        C.pm_norm,
};

let domKey = 'temperatura', domVal = 0;
for (const [k, v] of Object.entries(VARS))
  if (v > domVal) { domVal = v; domKey = k; }

const DOM = domKey;           // nombre de la variable ganadora
const DOM_STRENGTH = domVal;  // su valor normalizado, entre 0 y 1
```

El dominante no es un ranking ni una media ponderada. Es simplemente el **argmax** de los seis valores normalizados: la variable que más se aleja de su cero histórico en este momento. En un día de ola de calor, la temperatura domina. En un día de levante fuerte, domina el viento. En un día de alta presión estable con cielos despejados, la presión misma puede ganar aunque no parezca un dato "dramático".

### Qué cambia cuando una variable domina

El dominante modifica el sistema en cuatro niveles:

**1. Paleta global del canvas**

El color de fondo se mezcla en un 35% con el tinte característico de la variable dominante. Con presión dominante el fondo vira a aguamarina ocre; con temperatura dominante, a naranja cálido; con PM2.5, a una niebla violácea.

```javascript
const mix = DOM_STRENGTH * 0.35;
const pH = bgH * (1 - mix) + dt.h * mix;  // tono del fondo interpolado
```

**2. Modificadores geométricos globales**

```javascript
const MOD = {
  globalAngle: DOM === 'viento'      ? Math.atan2(C.wind_dy, C.wind_dx) * 0.7 : 0,
  globalScale: DOM === 'temperatura' ? 0.85 + DOM_STRENGTH * 0.35 : 1.0,
  blur:        DOM === 'humedad'     ? DOM_STRENGTH * 4 : 0,
  squish:      DOM === 'nubes'       ? 1 - DOM_STRENGTH * 0.3 : 1,
  rigid:       DOM === 'presion'     ? DOM_STRENGTH : 0,
};
```

- `globalAngle` (viento dominante): todo el canvas rota levemente en la dirección del viento real.
- `globalScale` (temperatura dominante): las formas de temperatura se escalan hacia arriba, invadiendo más superficie.
- `blur` (humedad dominante): las formas se pintan con desenfoque proporcional a la humedad, simulando aire saturado.
- `squish` (nubes dominantes): los rombos de nube se aplastan verticalmente, como nubes bajo presión.
- `rigid` (presión dominante): las bandas de presión se vuelven más estrechas y disciplinadas.

**3. Orden de renderizado**

Las capas se dibujan de fondo a frente, pero el dominante siempre se pinta el último, encima de todo:

```javascript
const order = ['presion', 'nubes', 'viento', 'humedad', 'pm25', 'temperatura'];
const idx = order.indexOf(DOM);
if (idx > -1) { order.splice(idx, 1); order.push(DOM); }
```

Esto garantiza que la variable que más energía tiene ese día no quede enterrada por las demás.

**4. Comportamiento dentro de cada función de dibujo**

Cada función (`drawTemperatura`, `drawViento`, etc.) recibe `isDom = DOM === 'clave'` y cambia su comportamiento:

- En modo no dominante: pocas formas, tamaño reducido, alpha bajo.
- En modo dominante: muchas formas, tamaño máximo, alpha alto, formas que invaden zonas que normalmente no ocuparían.

Por ejemplo, el viento en modo no dominante produce 6–20 curvas cortas dispersas. En modo dominante con energía alta (>0.5) produce 50–90 líneas largas que atraviesan el canvas de lado a lado, con el canvas entero rotado en la dirección del viento.

### Ejemplo: día del 27 de abril de 2026, Sevilla, 13h

```
temperatura  : 0.45  (20.8°C de 46°C máximo)
viento       : 0.31  (6.2 m/s de 20 m/s máximo)
humedad      : 0.64  (HR 65% / 80)
presion      : 0.72  ← DOMINANTE (1014 hPa, el 72% del rango 990-1030)
nubes        : 0.00  (0% de cobertura)
pm25         : 0.06  (4.8 μg/m³, aire muy limpio)
```

La presión gana con 0.72 porque ese día la presión absoluta (1014 hPa) ocupa el 72% de su rango histórico, mientras que la temperatura (20.8°C) solo ocupa el 45% del suyo. El cuadro resultante tiene bandas horizontales que estructuran la composición, un fondo con tinte aguamarina-ocre, y el viento —aunque perceptible— aparece como trazos oscuros sobre fondo claro, subordinado al orden de la presión.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      ATMOSPHERICA v2                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MÓDULO 1    │    │  MÓDULO 2    │    │  MÓDULO 3    │  │
│  │  Ingesta     │───▶│  Mapeador    │───▶│  Generador   │  │
│  │  de datos    │    │  visual      │    │  pictórico   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                  │                    │           │
│    APIs externas      Normalización        HTML autónomo    │
│    OpenWeatherMap     + dominancia         Canvas 2D        │
│    Air Pollution API  + ML params          → PNG exportable │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MÓDULO 4    │    │  MÓDULO 5    │    │  MÓDULO 6    │  │
│  │  Modelo ML   │    │  Automati-   │    │  Web de      │  │
│  │  predictivo  │    │  zación      │    │  portfolio   │  │
│  │  (pendiente) │    │  (pendiente) │    │  (pendiente) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│    ERA5 + AEMET        GitHub Actions       GitHub Pages    │
│    Series temporales   Cron diario         Archivo vivo     │
└─────────────────────────────────────────────────────────────┘
```

El flujo de datos completo es:

```
API → fetcher.py → mapper.py → generator.py → painting.html → PNG
```

Cada paso transforma los datos: el fetcher los obtiene crudos, el mapper los normaliza y calcula la variable dominante, el generador construye el HTML con los datos incrustados como JSON, y el navegador renderiza la pintura final usando Canvas 2D nativo.

---

## Módulo 1 — Ingesta de datos

**Archivos:** `data/fetcher.py`, `data/mock.py`

### Fuentes de datos

El módulo 1 consume dos endpoints de la API de OpenWeatherMap:

**Current Weather API** — datos meteorológicos en tiempo real:
- `temp` — temperatura en °C
- `temp_min`, `temp_max` — rango diario
- `humidity` — humedad relativa en %
- `pressure` — presión atmosférica en hPa
- `wind_speed` — velocidad del viento en m/s
- `wind_deg` — dirección del viento en grados (0=Norte, 90=Este, 180=Sur, 270=Oeste)
- `clouds` — cobertura nubosa en %
- `visibility` — visibilidad en metros
- `weather_id` — código de fenómeno meteorológico (lluvia, tormenta, niebla...)

**Air Pollution API** — calidad del aire:
- `pm2_5` — partículas finas en μg/m³ (el indicador principal de contaminación)
- `no2` — dióxido de nitrógeno en μg/m³ (tráfico)
- `o3` — ozono en μg/m³ (reacciones fotoquímicas)

### Flujo de obtención

```python
# 1. Llamada a Weather API con ciudad y país
weather = get_weather()  # → JSON con todos los campos

# 2. Extraer coordenadas para la Air API
lat = weather["coord"]["lat"]
lon = weather["coord"]["lon"]

# 3. Llamada a Air Pollution API con coordenadas
air = get_air_quality(lat, lon)  # → JSON con componentes

# 4. Unificar en un dict limpio
data = {
    "temperature": weather["main"]["temp"],
    "pressure":    weather["main"]["pressure"],
    "wind_speed":  weather["wind"]["speed"],
    "wind_deg":    weather["wind"]["deg"],
    "humidity":    weather["main"]["humidity"],
    "clouds":      weather["clouds"]["all"],
    "pm2_5":       air["list"][0]["components"]["pm2_5"],
    ...
}
```

### Mock para desarrollo

El archivo `data/mock.py` contiene datos estáticos de Sevilla para desarrollar sin necesidad de llamadas a la API. Esto permite iterar en el generador visual sin consumir el límite de 1000 llamadas/día del plan gratuito.

### Gestión de credenciales

La API key se almacena en un archivo `.env` que nunca se sube al repositorio (está en `.gitignore`). Se carga con `python-dotenv`. Este es un patrón fundamental de seguridad en proyectos de ML con datos externos.

---

## Módulo 2 — Mapeador visual

**Archivo:** `visual/mapper.py`

El mapeador es la pieza más conceptual del proyecto. Recibe el diccionario de datos crudos y devuelve un diccionario de parámetros visuales. Aquí es donde la gramática visual se implementa matemáticamente.

### Normalización

Todas las variables se normalizan al rango [0, 1] con rangos históricos de referencia para Sevilla:

```python
def normalize(value, min_val, max_val):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

temperature_norm = normalize(data["temperature"], 0, 46)   # 0°C a 46°C (máximo histórico Sevilla)
pressure_norm    = normalize(data["pressure"], 990, 1030)  # rango típico de presión
pm_norm          = normalize(data["pm2_5"], 0, 75)         # 0 = aire limpio, 75 = muy malo (OMS)
humidity_norm    = normalize(data["humidity"], 10, 95)
wind_energy      = normalize(data["wind_speed"], 0, 20)    # 0 = calma, 20 = vendaval
```

Usar rangos históricos locales es importante: una temperatura de 25°C en Oslo es calor extremo; en Sevilla es primavera normal. El sistema normaliza con contexto local, no con rangos globales.

### Descomposición vectorial del viento

El ángulo del viento se descompone en componentes cartesianas para usarlo como vector de dirección en el generador:

```python
wind_angle_rad = math.radians(data["wind_deg"])
wind_dx = math.sin(wind_angle_rad)   # componente horizontal
wind_dy = -math.cos(wind_angle_rad)  # componente vertical (invertido por convención de pantalla)
```

Esta descomposición permite que los trazos de viento en el cuadro apunten exactamente en la dirección geográfica correcta.

### Parámetros visuales de salida

El mapeador produce los siguientes parámetros:

```python
{
    "temperature_norm":  float,   # 0-1 para usar en el generador
    "density":           float,   # 0-1 desde presión (también es pressure_norm)
    "wind_energy":       float,   # 0-1 desde velocidad del viento
    "wind_dx":           float,   # componente horizontal del viento
    "wind_dy":           float,   # componente vertical del viento
    "fragmentation":     float,   # 0-1 desde PM2.5
    "veil_opacity":      float,   # 0-80 para el velo de humedad
    "risk_score":        float,   # 0-1 desde el modelo ML (0.0 si no disponible)
    "event_type":        str,     # "heat"|"cold"|"rain"|"wind"|"none"
    "ml_ready":          bool,    # True si el modelo está entrenado
    "raw":               dict,    # datos originales sin normalizar
}
```

---

## Módulo 3 — Generador pictórico (v2)

**Archivo:** `visual/generator.py`

El generador toma los parámetros del mapeador y produce un archivo HTML autónomo que renderiza la pintura en el navegador usando Canvas 2D nativo (sin p5.js). Es el módulo más extenso y el que produce el resultado visible.

### Por qué Canvas 2D nativo en v2

La v1 del generador usaba p5.js (el port JavaScript de Processing). La v2 lo reemplaza con Canvas 2D nativo del navegador por tres razones:

- **Sin dependencias externas**: el HTML es completamente autónomo. No necesita CDN, no falla si p5.js cambia de versión, funciona offline.
- **RNG determinista seeded**: p5.js no tiene RNG seeded nativo. El nuevo generador usa un RNG propio (`seededRng`) que garantiza que el mismo día y la misma ciudad producen siempre exactamente el mismo cuadro, sin variación entre renders.
- **Control total del pipeline de render**: el orden de capas, los modificadores globales y el sistema de dominancia requieren intervenir en cada paso del render de forma que p5.js no permitía sin hackearlo.

### RNG determinista por cuadro

El generador no usa `Math.random()`. Usa un generador XORShift seeded con una clave compuesta por ciudad + fecha + hora:

```javascript
function seededRng(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

// Semilla construida desde los metadatos del cuadro
const ks = C.date.replace(/-/g, '') + C.hour + C.city;
for (let i = 0; i < ks.length; i++) sv = (sv * 31 + ks.charCodeAt(i)) | 0;
const SR = seededRng(Math.abs(sv) || 99991);
```

Esto garantiza que `atmospherica_Seville_2026-04-27_13h.html` produce siempre el mismo cuadro, en cualquier navegador, en cualquier máquina. La reproducibilidad es un requisito de cualquier pipeline de ML.

### Inyección de datos sin `.format()`

Los datos climáticos se pasan al JavaScript como JSON mediante sustitución de texto, **no** mediante `.format()` de Python. Esto evita el conflicto entre las llaves `{}` de Python y las de JavaScript:

```python
html = HTML_TEMPLATE
html = html.replace("__CITY__",       city)
html = html.replace("__CITY_UPPER__", city.upper())
html = html.replace("__CLIMATE_JSON__",
                    json.dumps(climate_data, indent=2, ensure_ascii=False))
```

El objeto `C` queda disponible globalmente en el JavaScript con todos los datos del día y sus valores normalizados.

### Los seis pases de pintura

La pintura se construye en seis capas secuenciales. El dominante siempre se pinta en último lugar:

**Capa 1 — Fondo (`drawFondo`)**

Relleno sólido en `P.paper` (color base del día según hora y dominante) más un gradiente radial cuyo foco varía con la hora: amanecer arriba-izquierda, mediodía arriba-centro, puesta de sol a la derecha. Si el dominante es `nubes`, se añade un gradiente lineal que cae desde arriba. Si es `pm25`, una veladura violácea cubre todo el fondo.

**Capa 2 — Presión (`drawPresion`)**

Rectángulos y bandas. Con presión dominante y valor > 0.5, las bandas son casi horizontales y barren el canvas de arriba a abajo con distribución uniforme. Con valor bajo o no dominante, son rectángulos inclinados con ángulo aleatorio. En modo dominante con valor alto, además aparecen bloques en las cuatro esquinas del canvas.

**Capa 3 — Nubes (`drawNubes`)**

Rombos con ejes desiguales (más anchos que altos). Concentrados en la mitad superior cuando son dominantes. El parámetro `MOD.squish` los aplana verticalmente en proporción a la fuerza del dominante.

**Capa 4 — Viento (`drawViento`)**

Curvas Bézier cuadráticas orientadas en la dirección del viento (`wind_dx`, `wind_dy`). Con energía > 0.5 y el viento como dominante, el canvas completo rota levemente en la dirección del viento antes de pintar las líneas. La longitud varía desde trazos cortos de 15px (calma) hasta líneas que cruzan el canvas entero (vendaval). El color del trazo se adapta al fondo: si el fondo es claro (`pL > 65`), la luminosidad del azul baja a 18–42% para mantener contraste.

**Capa 5 — Humedad (`drawHumedad`)**

Con humedad < 35% y no dominante: triángulos equiláteros pequeños y nítidos. Con humedad > 35% o dominante: óvalos con `MOD.blur` aplicado mediante `ctx.filter` para simular los bordes difusos del aire húmedo. Con valor alto, un gradiente radial adicional crea halos de vapor alrededor de cada forma.

**Capa 6 — PM2.5 (`drawPM`)**

Veladura de fondo + puntos dispersos + venas de smog horizontales (curvas Bézier de grosor variable). La veladura se aplica sobre todo lo anterior como una capa semitransparente global, ennegreciendo y ensuciando el conjunto.

**Capa 7 — ML (`drawML`)**

Si el modelo está activo y `risk_score > 0.06`, se añaden señales de advertencia del evento predicho para mañana. Estas señales usan la **forma del dominante actual** pero con el **color del evento futuro** (rojo para calor, azul para frío, verde para viento, azul-gris para lluvia). Las señales se activan por umbrales progresivos de riesgo: triángulos en los bordes desde el 6%, fracturas internas desde el 25%, marco de alerta desde el 75%.

**Capa 8 — Ruido (`addNoise`)**

Último paso: ruido gaussiano leve aplicado pixel a pixel sobre toda la imagen para eliminar el aspecto "digital" y añadir textura de superficie. La intensidad aumenta si el dominante es PM2.5.

### Sistema de color adaptativo

El objeto `P` (paleta) se construye en función del dominante y la hora del día. Cada variable tiene su función de color dedicada que responde a su valor normalizado:

```javascript
windC:  (a=1) => {
  const bgLight = pL > 65;  // fondo claro → trazo oscuro
  const wL = bgLight ? Math.max(18, 42 - C.wind_energy * 28) : Math.min(78, 62 - C.wind_energy * 28);
  const wS = bgLight ? 70 + C.wind_energy * 25 : 45 + C.wind_energy * 40;
  return `hsla(220, ${wS}%, ${wL}%, ${a})`;
},
tempC:  (a=1) => { /* espectro fijo por °C, independiente del dominante */ },
humC:   (a=1) => `hsla(150, ${35 + C.humidity_norm * 50}%, ${55 - C.humidity_norm * 25}%, ${a})`,
presC:  (a=1) => `hsla(38,  ${38 + C.pressure_norm * 38}%, ${58 - C.pressure_norm * 22}%, ${a})`,
cloudC: (a=1) => `hsla(210, ${18 + C.cloud_norm * 22}%,   ${65 - C.cloud_norm * 30}%,   ${a})`,
pmC:    (a=1) => `hsla(285, ${30 + C.pm_norm * 50}%,      ${52 - C.pm_norm * 28}%,      ${a})`,
```

Los parámetros de cada color son funciones continuas de la variable — no tablas de lookup ni condicionales. Más valor normalizado produce más saturación y menos luminosidad en todos los casos.

### Exportación del PNG

El botón GUARDAR PNG crea un canvas HTML5 temporal, dibuja un fondo opaco (el canvas principal tiene fondo transparente), copia la pintura encima, y usa `canvas.toBlob()` para generar el archivo PNG directamente en el navegador sin pasar por un servidor. El nombre incluye ciudad, fecha y hora automáticamente.

---

## Estado actual del proyecto

### ✅ Completado y funcional

- **Ingesta de datos en tiempo real** desde OpenWeatherMap (Weather + Air Pollution APIs)
- **Mock de datos** para desarrollo sin API
- **Mapeador visual completo** con normalización de todas las variables y cálculo del dominante
- **Generador v2** con Canvas 2D nativo, RNG seeded, sistema de dominancia, seis capas climáticas + ML
- **Sistema de dominancia**: la variable con mayor valor normalizado deforma geometría, paleta, orden de capas y comportamiento de todas las demás
- **Paleta adaptativa** que ajusta contraste automáticamente según luminosidad del fondo
- **Integración ML visual**: señales de riesgo que usan la gramática del dominante actual en el color del evento futuro
- **Leyenda interactiva**: tabla de capas ML con estado en tiempo real (activo/inactivo) y umbrales de activación
- **Exportación PNG** funcional con fondo opaco y nombre automático
- **Gestión segura de credenciales** con `.env` y `python-dotenv`

### 🔄 En iteración activa

- Refinamiento de los pesos del dominante para climas ambiguos (días en que dos variables están muy próximas)
- Calibración de alphas en cada capa para garantizar legibilidad en los seis tipos de día posibles
- Tests de edge cases: dominante = nubes al 0%, dominante = PM2.5 en día limpio

---

## Lo que queda por construir

Los módulos 4, 5 y 6 son los que convierten ATMOSPHERICA de un generador de imágenes en un sistema de ML real y en un producto de portfolio completo.

---

## Módulo 4 — Modelo predictivo ML

**Archivos previstos:** `ml/trainer.py`, `ml/predictor.py`, `ml/features.py`

Este es el módulo que justifica que el proyecto sea de **Machine Learning** y no solo arte generativo.

### El problema

Los cuadros actuales reflejan el clima del momento. El módulo 4 añade una capa predictiva: el sistema analiza los patrones de los últimos 7-14 días y predice si mañana habrá un **evento climático extremo** (ola de calor, episodio de contaminación alta, tormenta, calima sahariana).

Cuando el modelo predice un evento extremo, el cuadro de hoy incorpora señales visuales de advertencia usando la gramática del dominante actual pero en el color del evento futuro. El cuadro advierte antes de que el evento ocurra.

### Dataset

El dataset de entrenamiento se construirá con **ERA5 de Copernicus**, el reanálisis climático más completo del mundo. Contiene datos horarios desde 1940 hasta hoy a resolución de ~31km. Se accede vía la librería `cdsapi` de Python.

Para Sevilla se descargarán los últimos 10 años de:
- Temperatura a 2 metros
- Presión al nivel del mar
- Velocidad y dirección del viento a 10 metros
- Humedad relativa
- Precipitación total

Como etiquetas de entrenamiento se usarán los registros históricos de **AEMET OpenData** (Agencia Estatal de Meteorología), que tiene datos de estaciones meteorológicas y registros de eventos extremos verificados.

### Arquitectura del modelo

**Fase 1 — Random Forest con features de series temporales**

Features de entrada (ventana de 7 días):
- Medias móviles de temperatura, presión y PM2.5
- Diferencias entre días consecutivos (gradiente de presión)
- Tendencia de temperatura (pendiente de regresión lineal en la ventana)
- Hora del año codificada con seno/coseno para capturar estacionalidad
- Temperatura máxima y mínima del período

Target: clasificación binaria de evento extremo al día siguiente (o regresión de temperatura máxima esperada).

**Fase 2 — LSTM para dependencias temporales largas**

```
Input: secuencia de 30 días × 8 features
LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(32) → Dense(1, sigmoid)
```

**Integración con el generador**

```python
risk_score = predictor.predict(last_14_days)
# risk_score llega al generador como parámetro visual
# y activa las capas de alerta por umbrales progresivos
```

### Evaluación

- Accuracy y F1-score para clasificación
- RMSE para regresión de temperatura
- Curva ROC para análisis de umbral de clasificación
- Análisis SHAP para interpretabilidad de features

---

## Módulo 5 — Automatización y archivo

**Archivos previstos:** `.github/workflows/daily.yml`, `archive/logger.py`

### Generación automática diaria

```yaml
# .github/workflows/daily.yml
on:
  schedule:
    - cron: '0 22 * * *'  # 23:00 hora española (UTC+1)
jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - name: Generate painting
        env:
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
        run: python main.py --headless --save-png
```

### Exportación headless

```python
# main.py con flag --headless
if args.headless:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(html_path)}")
        page.locator('#save').click()
```

### Archivo histórico

```json
{
  "2026-04-27": {
    "city": "Seville",
    "image": "output/atmospherica_Seville_2026-04-27.png",
    "climate": { "temperature": 20.8, "pressure": 1014, "wind_speed": 6.2, ... },
    "dominant": { "variable": "presion", "strength": 0.72 },
    "ml": { "risk_score": 0.149, "event_type": "none" }
  }
}
```

---

## Módulo 6 — Web de portfolio

**Archivos previstos:** `web/index.html`, `web/gallery.html`, `web/about.html`

La web se publicará en **GitHub Pages** con tres secciones:

- **Galería principal**: grid de cuadros por fecha, filtrable por tipo de día y dominante
- **Vista individual**: imagen en alta resolución, datos climáticos, leyenda visual, puntuación ML, comparación con el día anterior
- **Documentación técnica**: sistema, gramática visual, arquitectura ML

---

## Stack tecnológico

### Completado

| Componente | Tecnología | Propósito |
|---|---|---|
| Lenguaje principal | Python 3.11 | Pipeline de datos y generación |
| Ingesta de datos | `requests` | Llamadas a APIs externas |
| Variables de entorno | `python-dotenv` | Gestión segura de credenciales |
| Arte generativo | Canvas 2D (HTML5 nativo) | Renderizado pictórico sin dependencias |
| RNG determinista | XORShift seeded | Reproducibilidad entre renders |
| Exportación | HTML5 Canvas API | PNG desde el navegador |

### En desarrollo (Módulo 4)

| Componente | Tecnología | Propósito |
|---|---|---|
| Dataset histórico | ERA5 via `cdsapi` | Datos climáticos para entrenamiento |
| Feature engineering | `pandas`, `numpy` | Construcción de features temporales |
| Modelo v1 | `scikit-learn` (Random Forest) | Clasificación de eventos extremos |
| Modelo v2 | `PyTorch` (LSTM) | Dependencias temporales largas |
| Interpretabilidad | `shap` | SHAP values para explicabilidad |
| Evaluación | `scikit-learn` metrics | Accuracy, F1, ROC AUC |

### Pendiente (Módulos 5 y 6)

| Componente | Tecnología | Propósito |
|---|---|---|
| Automatización | GitHub Actions | Generación diaria sin intervención |
| Headless rendering | Playwright | Exportar PNG sin navegador manual |
| Hosting web | GitHub Pages | Portfolio público y archivo |
| Datos AEMET | AEMET OpenData API | Etiquetas de eventos extremos reales |

---

## Estructura del repositorio

```
atmospherica/
├── .env                        ← API key (nunca en git)
├── .gitignore
├── requirements.txt
├── main.py                     ← Punto de entrada principal
│
├── data/
│   ├── __init__.py
│   ├── fetcher.py              ← Ingesta desde OpenWeatherMap
│   └── mock.py                 ← Datos estáticos para desarrollo
│
├── visual/
│   ├── __init__.py
│   ├── mapper.py               ← Normalización y cálculo del dominante
│   └── generator.py            ← Generador HTML + Canvas 2D (v2)
│
├── ml/                         ← PENDIENTE
│   ├── __init__.py
│   ├── features.py             ← Construcción de features temporales
│   ├── trainer.py              ← Entrenamiento del modelo
│   └── predictor.py            ← Inferencia en producción
│
├── archive/                    ← PENDIENTE
│   ├── logger.py               ← Registro de cuadros generados
│   └── index.json              ← Base de datos histórica
│
├── web/                        ← PENDIENTE
│   ├── index.html              ← Galería principal
│   ├── gallery.html            ← Vista de cuadros
│   └── about.html              ← Documentación técnica
│
├── output/                     ← Cuadros generados (no en git excepto los finales)
│   └── atmospherica_Seville_2026-04-27_13h.html
│
└── .github/
    └── workflows/
        └── daily.yml           ← PENDIENTE — automatización
```

---

## Cómo ejecutar

### Requisitos

- Python 3.11+
- pip
- Cuenta gratuita en [openweathermap.org](https://openweathermap.org/api)
- Navegador moderno (Chrome, Firefox, Edge)

### Instalación

```bash
git clone https://github.com/tu-usuario/atmospherica
cd atmospherica
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Configuración

Crear el archivo `.env`:
```
OPENWEATHER_API_KEY=tu_api_key_aqui
```

### Ejecución

```bash
python main.py
```

El script:
1. Obtiene los datos climáticos actuales de Sevilla
2. Los normaliza y calcula la variable dominante
3. Genera un archivo HTML autónomo en `output/`
4. Abre el navegador automáticamente
5. Pinta el cuadro (render instantáneo con Canvas 2D)
6. El botón GUARDAR PNG exporta la imagen final

### Cambiar ciudad

En `config.py`:
```python
CITY = "Madrid"        # o cualquier ciudad
COUNTRY_CODE = "ES"
```

---

## Por qué esto es ML Engineering, no solo arte

Esta es la pregunta que cualquier reclutador técnico hará. La respuesta tiene cuatro partes.

**1. Pipeline de datos real con APIs externas**

El sistema ingiere datos de dos APIs externas, gestiona autenticación con credenciales seguras, parsea JSON anidado, normaliza variables con rangos históricos contextuales, y descompone vectores. Este es el trabajo del 80% de los proyectos de ML en producción.

**2. Sistema de mapeo paramétrico con lógica de negocio**

El mapeador implementa la normalización de seis variables con rangos locales y calcula el dominante mediante argmax. El generador aplica 15+ transformaciones condicionales sobre ese dominante. Esto es feature engineering aplicado a un dominio no convencional: la habilidad de transformar datos crudos en representaciones útiles para un sistema downstream es exactamente lo que se hace en ML.

**3. Modelo predictivo de series temporales (Módulo 4)**

El LSTM entrenado con ERA5 es ML estándar: dataset real, feature engineering, entrenamiento, evaluación con métricas, inferencia en producción. Lo que lo diferencia es que la inferencia del modelo afecta directamente al output artístico, haciendo el sistema completo end-to-end.

**4. Sistema autónomo en producción**

El resultado final es un sistema que corre solo, sin intervención manual, genera outputs cada día, los persiste en un archivo histórico, y los publica en una web. Esto demuestra capacidad de construir sistemas, no solo notebooks de Jupyter.

La capa de arte no es decorativa para el portfolio. Es lo que hace que el proyecto sea memorable en una entrevista, lo que permite explicar decisiones técnicas de forma narrativa, y lo que demuestra que el autor puede pensar en un sistema de ML como algo más que un modelo en aislamiento.

---

*Proyecto desarrollado por Nora Peñaloza Friqui — Sevilla, 2026*

*Stack: Python · Canvas 2D · scikit-learn · PyTorch · ERA5 · OpenWeatherMap API · GitHub Actions · GitHub Pages*
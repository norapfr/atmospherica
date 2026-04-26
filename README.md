# ATMOSPHERICA

> *Un sistema de Machine Learning que convierte datos atmosféricos en tiempo real en pintura abstracta generativa. Cada cuadro es único, irrepetible, y codifica el estado del clima del día en un lenguaje visual con gramática propia.*

---

## Índice

1. [Qué es ATMOSPHERICA](#qué-es-atmospherica)
2. [La idea central](#la-idea-central)
3. [Gramática visual](#gramática-visual)
4. [Arquitectura del sistema](#arquitectura-del-sistema)
5. [Módulo 1 — Ingesta de datos](#módulo-1--ingesta-de-datos)
6. [Módulo 2 — Mapeador visual](#módulo-2--mapeador-visual)
7. [Módulo 3 — Generador pictórico](#módulo-3--generador-pictórico)
8. [Estado actual del proyecto](#estado-actual-del-proyecto)
9. [Lo que queda por construir](#lo-que-queda-por-construir)
10. [Módulo 4 — Modelo predictivo ML](#módulo-4--modelo-predictivo-ml)
11. [Módulo 5 — Automatización y archivo](#módulo-5--automatización-y-archivo)
12. [Módulo 6 — Web de portfolio](#módulo-6--web-de-portfolio)
13. [Stack tecnológico](#stack-tecnológico)
14. [Estructura del repositorio](#estructura-del-repositorio)
15. [Cómo ejecutar](#cómo-ejecutar)
16. [Por qué esto es ML engineering, no solo arte](#por-qué-esto-es-ml-engineering-no-solo-arte)

---

## Qué es ATMOSPHERICA

ATMOSPHERICA es un sistema que ingiere datos meteorológicos y de calidad del aire en tiempo real de una ciudad, los procesa con un pipeline de normalización y mapeo paramétrico, y genera automáticamente una obra de arte abstracta única que codifica el estado atmosférico de ese día.

El sistema no genera imágenes decorativas ni visualizaciones de datos convencionales. Genera pinturas con una gramática visual definida y defendible donde cada decisión formal —color, tamaño, dirección, opacidad, fragmentación— proviene directamente de un dato climático real. El cuadro del 25 de abril de 2026 en Sevilla es diferente al del 26 de abril, y diferente al del mismo día en Oslo. La obra es el dato.

El proyecto tiene tres módulos completamente funcionales y tres módulos en desarrollo. El resultado final será un sistema autónomo que genera y publica una nueva obra cada día, construye un archivo histórico, y añade una capa de predicción de eventos climáticos extremos mediante modelos de series temporales.

---

## La idea central

Existe una distinción fundamental entre **visualización de datos** y **traducción de datos a lenguaje artístico**.

Una visualización muestra el dato. Un gráfico de temperatura a lo largo del día muestra líneas que suben y bajan. El dato es legible pero no se experimenta.

ATMOSPHERICA propone otra cosa: el dato **es** la forma. La temperatura no se muestra en un eje Y — determina el rango de color completo del cuadro. El viento no aparece como una flecha — sus cintas de pintura recorren el lienzo en la dirección exacta del viento real, con una longitud proporcional a su velocidad en metros por segundo. La presión atmosférica no es un número — decide si la composición se organiza en curvas amplias y fluidas (anticiclón) o en gestos cortos y tensos en los bordes (borrasca).

Esto tiene una consecuencia importante para el portfolio de ML: el sistema demuestra que su autor sabe construir pipelines de datos reales, normalizar variables con rangos históricos, diseñar sistemas de mapeo paramétrico complejos, y pensar en el output como un sistema con reglas, no como generación aleatoria.

---

## Gramática visual

La gramática visual es el corazón del proyecto. Es el conjunto de reglas que traduce cada variable climática en una decisión pictórica. Sin esta gramática, el proyecto sería "IA que genera imágenes bonitas". Con ella, es un sistema de codificación visual con semántica propia.

### Temperatura → Color base y tamaño de las formas

La temperatura es la variable dominante. Define la paleta de color completa del cuadro mediante un espectro continuo de seis rangos:

| Rango | Color | HSB aproximado | Significado visual |
|-------|-------|----------------|-------------------|
| < 0°C | Azul profundo | (220, 80, 78) | Frío extremo, quietud |
| 0–10°C | Azul-verde | (195, 72, 80) | Fresco, tensión contenida |
| 10–20°C | Ocre-dorado | (38, 68, 82) | Templado, equilibrio |
| 20–28°C | Ámbar | (28, 80, 88) | Cálido, expansión |
| 28–38°C | Naranja | (18, 85, 90) | Calor, energía |
| > 38°C | Rojo-naranja | (8, 88, 88) | Calor extremo, tensión máxima |

La temperatura también controla el tamaño de las pinceladas: más calor produce formas más grandes y expansivas. Un día de ola de calor en Sevilla a 44°C produce pinceladas que casi cubren el cuadro. Un día frío de enero produce formas pequeñas y contraídas.

### Presión atmosférica → Estructura compositiva

La presión determina la estructura de la composición mediante arcos de contorno:

- **Alta presión (> 1013 hPa, anticiclón):** arcos amplios, curvatura suave, distribuidos en las zonas del cuadro. El día es estable, y el cuadro lo refleja con ritmo y orden.
- **Baja presión (< 1005 hPa, borrasca):** arcos cortos, curvatura tensa, concentrados en los bordes del lienzo. El día es inestable, y la composición se fragmenta hacia los márgenes.
- **La luminosidad del arco** es proporcional al valor de presión: días de alta presión tienen arcos brillantes; días de borrasca los tienen oscuros.

### Viento → Dirección y longitud de los trazos

El viento es quizás la variable más narrativa del cuadro. Produce cintas de pintura gestual (inspiradas en la técnica de ribbons en p5.js) orientadas en la **dirección geográfica exacta del viento real**:

- Un viento del suroeste produce trazos inclinados hacia el noreste.
- La **longitud del trazo es proporcional a la velocidad en m/s**. Un día calmo como el 25 de abril de 2026 en Sevilla (1.3 m/s) produce trazos cortos y casi imperceptibles —un susurro. Un día de levante a 8 m/s produciría ríos de pintura que recorrerían el lienzo.
- La **opacidad** también responde a la velocidad: viento fuerte domina visualmente, viento calmo apenas se nota.
- La saturación del color del viento (azul-gris) aumenta con la velocidad.

### PM2.5 → Fragmentación y grano

La contaminación por partículas en suspensión PM2.5 (microgramos/m³) introduce ruido, erosión y fragmentación en la superficie pictórica:

- **Aire limpio (< 10 μg/m³, como hoy):** casi sin intervención. Alguna mota suelta apenas visible.
- **Aire moderado (10–25 μg/m³):** grano disperso que erosiona levemente las capas inferiores.
- **Aire contaminado (> 25 μg/m³):** niebla de partículas con color verde-gris enfermizo que opacifica el cuadro. Los trazos de las otras variables se interrumpen.
- **Crisis de contaminación (> 50 μg/m³):** el cuadro entero queda cubierto de una capa de grano que hace casi ilegible todo lo demás. El cuadro documenta la emergencia.

### Humedad → Velos y disolución de bordes

La humedad relativa actúa como el agua en una pintura al óleo recién aplicada:

- **Humedad baja (< 40%, como hoy):** el cuadro tiene bordes nítidos, colores saturados y definidos. El día es seco y los colores "saltan".
- **Humedad media (40–65%):** velos translúcidos orgánicos que suavizan los bordes de las formas. La pintura parece húmeda.
- **Humedad alta (> 65%):** los velos se hacen dominantes y difuminan todo lo anterior. La escena "sangra".
- **Lluvia (humedad > 80% + nubes > 70%):** trazos verticales finos que caen desde arriba, como gotas sobre el lienzo.

### Composición → Asimétrica y sin foco central

La composición no tiene un foco central. Las formas se distribuyen en seis zonas que cubren las esquinas y los bordes del lienzo, dejando el centro relativamente vacío. Esta decisión tiene una justificación climática: en un día estable (como hoy, con anticiclón sobre Sevilla), la energía atmosférica se distribuye de forma homogénea, no converge en un punto. El centro vacío crea tensión compositiva y evita que el cuadro parezca una visualización convencional con un foco de atención.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      ATMOSPHERICA                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MÓDULO 1    │    │  MÓDULO 2    │    │  MÓDULO 3    │  │
│  │  Ingesta     │───▶│  Mapeador    │───▶│  Generador   │  │
│  │  de datos    │    │  visual      │    │  pictórico   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                       │           │
│    APIs externas                          HTML + p5.js      │
│    OpenWeatherMap                         → PNG exportable  │
│    Air Pollution API                                        │
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

Cada paso transforma los datos: el fetcher los obtiene crudos, el mapper los normaliza y asigna parámetros visuales, el generador los convierte en instrucciones pictóricas para p5.js, y el navegador renderiza la pintura final que se exporta como PNG de alta calidad.

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
    "base_color":        (r, g, b),    # color RGB base desde temperatura
    "temperature_norm":  float,        # 0-1 para usar en el generador
    "density":           float,        # 0.3-1.0 desde presión
    "num_layers":        int,          # número de capas desde presión
    "fragmentation":     float,        # 0-1 desde PM2.5
    "fragment_count":    int,          # número de fragmentos
    "opacity_base":      int,          # 160-255 desde humedad
    "veil_opacity":      int,          # 0-80 para el velo de humedad
    "wind_dx":           float,        # componente horizontal del viento
    "wind_dy":           float,        # componente vertical del viento
    "wind_energy":       float,        # 0-1 desde velocidad del viento
    "stroke_length":     int,          # longitud de trazos desde viento
    "stroke_width":      int,          # grosor de trazos desde viento
    "bg_darkness":       int,          # oscuridad del fondo desde nubes
    "raw":               dict,         # datos originales sin normalizar
}
```

---

## Módulo 3 — Generador pictórico

**Archivo:** `visual/generator.py`

El generador toma los parámetros del mapeador y produce un archivo HTML con p5.js que renderiza la pintura en el navegador. Es el módulo más extenso y el que produce el resultado visible.

### Por qué p5.js y no Python puro

Las versiones anteriores del proyecto intentaron generar las imágenes con Pillow (librería de imágenes de Python). El resultado fue técnicamente correcto pero visualmente plano: los trazos no tenían fade natural, las formas no tenían la textura orgánica del óleo, y el control de opacidad era limitado.

p5.js (el port JavaScript de Processing) está específicamente diseñado para arte generativo. Tiene:
- Control de opacidad por punto dentro de un trazo
- Modos de color HSB nativos (más intuitivos para arte que RGB)
- Curvas de Bézier para trazos fluidos
- `noise()` (ruido Perlin) integrado para organicidad
- Renderizado en canvas HTML5 con anti-aliasing de calidad

El pipeline final es: Python genera el HTML con los datos incrustados como JSON → el navegador renderiza la pintura → el usuario exporta el PNG con el botón GUARDAR PNG.

### Estructura del generador

El generador produce un HTML autónomo (no necesita servidor) que contiene:

1. **Los datos climáticos del día** incrustados como JSON en la variable `C`
2. **La paleta de colores** calculada en Python y pasada al JavaScript
3. **El sketch de p5.js** con toda la lógica pictórica
4. **La interfaz** con metadatos del clima y leyenda explicativa

### Los cinco pases de pintura

La pintura se construye en cinco pases secuenciales, uno por variable climática. El navegador muestra el progreso en tiempo real:

**Pase 1 — Temperatura (420 partículas)**

Pinceladas ovales de óleo. Son la capa base y el cuerpo principal del cuadro. La función `brushMark()` genera formas de pincelada realista mediante curvas de Bézier con tres capas superpuestas de opacidad decreciente, simulando el grosor y la transparencia del óleo fresco. El tamaño de cada pincelada y su color responden directamente al valor de temperatura normalizado.

**Pase 2 — Presión atmosférica (220 partículas)**

Arcos de contorno generados con `p.arc()`. En días de alta presión los arcos son amplios, luminosos y se distribuyen por todas las zonas del lienzo. En días de borrasca son cortos, tensos y se concentran en los bordes. Los arcos se dibujan en tres capas con grosor y opacidad decreciente, creando profundidad.

**Pase 3 — Viento (260 partículas)**

Cintas de pintura gestual generadas con `windRibbon()`. Tres cintas paralelas con desplazamiento vertical, rotadas exactamente en la dirección del viento real. La longitud es proporcional a la velocidad: 1.3 m/s produce trazos de 15-70px; 10 m/s produciría trazos de 300-380px que recorrerían el lienzo de lado a lado.

**Pase 4 — PM2.5 (160 partículas)**

Granos de polvo generados con `drawDustBlot()`. Formas irregulares con ruido Perlin en sus bordes para simular partículas en suspensión. Con aire limpio (< 10 μg/m³), el 78% de las iteraciones se saltan aleatoriamente, produciendo apenas unas pocas motas dispersas.

**Pase 5 — Humedad (180 partículas)**

Velos orgánicos o trazos de lluvia según el nivel de humedad. Los velos se generan con formas de ruido Perlin en múltiples capas de opacidad muy baja, creando la sensación de niebla o humedad en el aire.

### Distribución espacial asimétrica

La función `sampleZone()` es la responsable de que el cuadro no se concentre en el centro. Divide el lienzo en seis zonas que cubren las esquinas y los bordes, y selecciona aleatoriamente entre ellas con probabilidad uniforme:

```javascript
function sampleZone() {
    const z = Math.floor(p.random(6));
    switch(z) {
        case 0: return [p.random(W*0.02, W*0.33), p.random(H*0.03, H*0.43)]; // sup izq
        case 1: return [p.random(W*0.63, W*0.97), p.random(H*0.03, H*0.41)]; // sup dcha
        case 2: return [p.random(W*0.02, W*0.35), p.random(H*0.57, H*0.97)]; // inf izq
        case 3: return [p.random(W*0.61, W*0.97), p.random(H*0.55, H*0.97)]; // inf dcha
        case 4: return [p.random(W*0.08, W*0.90), p.random(H*0.02, H*0.19)]; // borde sup
        default:return [p.random(W*0.08, W*0.90), p.random(H*0.81, H*0.98)]; // borde inf
    }
}
```

Esto garantiza que ninguna forma aparezca en la zona central (aproximadamente W*0.35–0.60, H*0.35–0.55), creando una composición con tensión entre los bordes y un espacio de respiración en el interior.

### Paleta de colores dinámica

La paleta no tiene colores fijos. Se calcula en Python en `generate_html()` a partir de los datos del día y se pasa al JavaScript como JSON. Los colores en formato HSB (Hue-Saturation-Brightness) son más intuitivos para arte generativo que RGB porque permiten modificar brillo y saturación independientemente del tono.

Ejemplo para el 25 de abril de 2026 en Sevilla (23.4°C, 1014 hPa, 1.3 m/s, PM2.5 9.4, HR 42%):

```python
palette = {
    "temp":     [38, 68, 82],   # ocre-dorado — templado primaveral
    "pressure": [42, 35, 57],   # neutro tierra, luminosidad media-alta
    "wind":     [205, 19, 67],  # azul-gris muy sutil — viento calmo
    "pm":       [88, 21, 45],   # verde-gris oscuro — aire limpio, casi invisible
    "humidity": [212, 26, 78],  # azul suave — humedad moderada
    "bg":       [32, 5, 7],     # fondo muy oscuro con tinte ámbar
}
```

Un día de ola de calor en Sevilla a 43°C produciría:
```python
"temp": [8, 88, 88]  # rojo-naranja intenso
```

Un día de invierno frío a 4°C produciría:
```python
"temp": [220, 80, 78]  # azul profundo
```

### Exportación del PNG

El botón GUARDAR PNG crea un canvas HTML5 temporal, dibuja un fondo opaco (el canvas de p5.js tiene fondo transparente por defecto), copia el canvas de la pintura encima, y usa `canvas.toBlob()` para generar el archivo PNG sin pasar por un servidor. El nombre del archivo incluye la ciudad y la fecha automáticamente.

---

## Estado actual del proyecto

### ✅ Completado y funcional

- **Ingesta de datos en tiempo real** desde OpenWeatherMap (Weather + Air Pollution APIs)
- **Mock de datos** para desarrollo sin API
- **Mapeador visual completo** con normalización de todas las variables
- **Generador pictórico** en p5.js con cinco pases climáticos diferenciados
- **Paleta dinámica** calculada desde temperatura, presión, viento, PM2.5 y humedad
- **Composición asimétrica** con distribución en seis zonas que evitan el centro
- **Exportación PNG** funcional con fondo opaco
- **Interfaz** con metadatos del clima y leyenda explicativa
- **Gestión segura de credenciales** con `.env` y `python-dotenv`

### 🔄 En iteración activa

- Refinamiento de la paleta de colores para mayor impacto visual
- Ajuste de alphas y densidades por pase
- Balance entre variables: que cada clima produzca un cuadro claramente distinto

---

## Lo que queda por construir

Los módulos 4, 5 y 6 son los que convierten ATMOSPHERICA de un generador de imágenes en un sistema de ML real y en un producto de portfolio completo.

---

## Módulo 4 — Modelo predictivo ML

**Archivos previstos:** `ml/trainer.py`, `ml/predictor.py`, `ml/features.py`

Este es el módulo que justifica que el proyecto sea de **Machine Learning** y no solo arte generativo.

### El problema

Los cuadros actuales reflejan el clima del momento. El módulo 4 añade una capa predictiva: el sistema analiza los patrones de los últimos 7-14 días y predice si mañana habrá un **evento climático extremo** (ola de calor, episodio de contaminación alta, tormenta, calima sahariana).

Cuando el modelo predice un evento extremo, el cuadro de hoy incorpora señales visuales de advertencia: zonas de tensión cromática, fragmentación anómala, o colores fuera de la paleta habitual. El cuadro advierte antes de que el evento ocurra.

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

El Random Forest es el modelo inicial porque es interpretable (SHAP values), no requiere GPU, funciona bien con datos tabulares, y su entrenamiento es rápido. La interpretabilidad es importante porque queremos entender qué features predicen cada tipo de evento.

**Fase 2 — LSTM para dependencias temporales largas**

Un Random Forest con features manuales captura patrones de 7-14 días. Un LSTM puede aprender dependencias más largas (patrones estacionales, ciclos de varios meses) directamente de la secuencia temporal sin necesidad de feature engineering manual.

La arquitectura prevista:
```
Input: secuencia de 30 días × 8 features
LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(32) → Dense(1, sigmoid)
```

**Integración con el generador**

El predictor generará una puntuación de riesgo entre 0 y 1 para el día siguiente. Esta puntuación se pasará al mapeador como una variable adicional que modifica la paleta y la composición:

```python
risk_score = predictor.predict(last_14_days)

# Riesgo alto: introduce colores de advertencia en el cuadro
if risk_score > 0.7:
    visual_params["warning_hue"] = 15    # rojo de alerta
    visual_params["warning_alpha"] = int(risk_score * 60)
```

### Evaluación

El modelo se evaluará con:
- Accuracy y F1-score para clasificación
- RMSE para regresión de temperatura
- Curva ROC para análisis de umbral de clasificación
- Análisis SHAP para interpretabilidad de features

---

## Módulo 5 — Automatización y archivo

**Archivos previstos:** `.github/workflows/daily.yml`, `archive/logger.py`

### Generación automática diaria

Se configurará un **GitHub Action** con trigger `schedule` que ejecuta el pipeline completo cada día a las 23:00 hora local:

```yaml
# .github/workflows/daily.yml
name: Daily painting

on:
  schedule:
    - cron: '0 22 * * *'  # 23:00 hora española (UTC+1)

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Generate painting
        env:
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
        run: python main.py --headless --save-png
      - name: Commit painting
        run: |
          git config --local user.email "atmospherica@bot"
          git config --local user.name "Atmospherica Bot"
          git add output/
          git commit -m "painting: $(date +%Y-%m-%d)"
          git push
```

La API key se almacena como **secret cifrado de GitHub**, nunca en el código.

### Exportación headless

Para ejecutar en un servidor sin navegador, se integrará **Puppeteer** (Node.js) o **Playwright** (Python) para renderizar el HTML en un Chrome headless y exportar el PNG automáticamente sin intervención manual:

```python
# main.py con flag --headless
if args.headless:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{os.path.abspath(html_path)}")
        page.wait_for_selector('#progress:has-text("listo")')
        page.locator('#save').click()
        # esperar descarga y mover a output/
```

### Archivo histórico

Cada cuadro generado se registra en un archivo JSON (`archive/index.json`) con todos sus metadatos:

```json
{
  "2026-04-25": {
    "city": "Seville",
    "image": "output/atmospherica_Seville_2026-04-25.png",
    "climate": {
      "temperature": 23.4,
      "pressure": 1014,
      "wind_speed": 1.3,
      "wind_deg": 210,
      "pm2_5": 9.4,
      "humidity": 42,
      "clouds": 15
    },
    "visual": {
      "day_type": "stable",
      "palette_temp": [38, 68, 82],
      "seed": 84729
    },
    "ml": {
      "risk_score": 0.12,
      "prediction": "no_event"
    }
  }
}
```

Este archivo es la base de datos del proyecto. Permite reconstruir cualquier cuadro, analizar la relación entre clima y estética a lo largo del tiempo, y entrenar el modelo predictivo con datos propios generados por el sistema.

---

## Módulo 6 — Web de portfolio

**Archivos previstos:** `web/index.html`, `web/gallery.html`, `web/about.html`

### Estructura

La web se publicará en **GitHub Pages** (gratuito, hosting estático) y tendrá tres secciones:

**Galería principal**

Grid de cuadros organizados por fecha. Al pasar el ratón sobre cada cuadro se muestran los datos climáticos que lo generaron. Se puede filtrar por tipo de día (estable, ventoso, frío, lluvia, contaminado) y por estación del año.

**Vista de cuadro individual**

Cada cuadro tiene su propia página con:
- La imagen en alta resolución
- Los datos climáticos del día
- La leyenda de la gramática visual aplicada
- La puntuación de riesgo del modelo predictivo
- Comparación con el cuadro del día anterior

**Documentación técnica**

Descripción del sistema, la gramática visual, la arquitectura del modelo ML, y las decisiones de diseño. Esta página es la que leerán los reclutadores técnicos.

### Automatización de la web

El GitHub Action que genera el cuadro diario también actualiza la web: regenera el `index.json`, actualiza la galería con el nuevo cuadro, y hace push a la rama `gh-pages`. La web está siempre actualizada sin intervención manual.

---

## Stack tecnológico

### Completado

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| Lenguaje principal | Python 3.11 | Pipeline de datos y generación |
| Ingesta de datos | `requests` | Llamadas a APIs externas |
| Variables de entorno | `python-dotenv` | Gestión segura de credenciales |
| Arte generativo | p5.js 1.9.0 | Renderizado pictórico en canvas |
| Exportación | HTML5 Canvas API | PNG desde el navegador |

### En desarrollo (Módulo 4)

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| Dataset histórico | ERA5 via `cdsapi` | Datos climáticos para entrenamiento |
| Feature engineering | `pandas`, `numpy` | Construcción de features temporales |
| Modelo v1 | `scikit-learn` (Random Forest) | Clasificación de eventos extremos |
| Modelo v2 | `PyTorch` (LSTM) | Dependencias temporales largas |
| Interpretabilidad | `shap` | SHAP values para explicabilidad |
| Evaluación | `scikit-learn` metrics | Accuracy, F1, ROC AUC |

### Pendiente (Módulos 5 y 6)

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
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
│   ├── mapper.py               ← Normalización y mapeo de parámetros
│   └── generator.py            ← Generación del HTML + p5.js
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
│   └── atmospherica_Seville_2026-04-25.html
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
2. Los mapea a parámetros visuales
3. Genera un archivo HTML en `output/`
4. Abre el navegador automáticamente
5. Pinta el cuadro en tiempo real (se puede ver el proceso)
6. El botón GUARDAR PNG exporta la imagen final

### Cambiar ciudad

En `config.py`:
```python
CITY = "Madrid"        # o cualquier ciudad
COUNTRY_CODE = "ES"
```

### Usar datos de prueba

En `main.py`, cambiar la línea de importación:
```python
from data.mock import get_mock_data as get_all_data
```

---

## Por qué esto es ML Engineering, no solo arte

Esta es la pregunta que cualquier reclutador técnico hará. La respuesta tiene cuatro partes.

**1. Pipeline de datos real con APIs externas**

El sistema ingiere datos de dos APIs externas, gestiona autenticación con credenciales seguras, parsea JSON anidado, normaliza variables con rangos históricos contextuales, y descompone vectores. Este es el trabajo del 80% de los proyectos de ML en producción.

**2. Sistema de mapeo paramétrico con lógica de negocio**

El mapeador implementa 15 transformaciones diferentes, cada una con justificación técnica y estética. Esto es feature engineering aplicado a un dominio no convencional. La habilidad de transformar datos crudos en representaciones útiles para un sistema downstream es exactamente lo que se hace en ML.

**3. Modelo predictivo de series temporales (Módulo 4)**

El LSTM entrenado con ERA5 es ML estándar: dataset real, feature engineering, entrenamiento, evaluación con métricas, inferencia en producción. Lo que lo diferencia es que la inferencia del modelo afecta directamente al output artístico, haciendo el sistema completo end-to-end.

**4. Sistema autónomo en producción**

El resultado final es un sistema que corre solo, sin intervención manual, genera outputs cada día, los persiste en un archivo histórico, y los publica en una web. Esto demuestra capacidad de construir sistemas, no solo notebooks de Jupyter.

La capa de arte no es decorativa para el portfolio. Es lo que hace que el proyecto sea memorable en una entrevista, lo que permite explicar decisiones técnicas de forma narrativa, y lo que demuestra que el autor puede pensar en un sistema de ML como algo más que un modelo en aislamiento.

---

*Proyecto desarrollado por [tu nombre] — Sevilla, 2026*

*Stack: Python · p5.js · scikit-learn · PyTorch · ERA5 · OpenWeatherMap API · GitHub Actions · GitHub Pages*

# ATMOSPHERICA — Pipeline de producción completo

> Documentación técnica del sistema de generación automática de pintura generativa a partir de datos atmosféricos en tiempo real. Cubre el pipeline de ML, la integración con la web y la automatización con GitHub Actions.

---

## Índice

1. [Visión general](#1-visión-general)
2. [El problema de producción](#2-el-problema-de-producción)
3. [Solución: historial acumulado desde la API](#3-solución-historial-acumulado-desde-la-api)
4. [Módulo ML — features.py](#4-módulo-ml--featurespy)
5. [Módulo ML — trainer.py](#5-módulo-ml--trainerpy)
6. [Módulo ML — predictor.py (corregido)](#6-módulo-ml--predictorpy-corregido)
7. [Módulo de historia — history.py](#7-módulo-de-historia--historypy)
8. [Módulo de archivo — archive.py](#8-módulo-de-archivo--archivepy)
9. [main.py — punto de entrada unificado](#9-mainpy--punto-de-entrada-unificado)
10. [Automatización — GitHub Actions](#10-automatización--github-actions)
11. [Conexión con la web](#11-conexión-con-la-web)
12. [Estructura final del repositorio](#12-estructura-final-del-repositorio)
13. [Flujo completo de datos](#13-flujo-completo-de-datos)
14. [Preguntas frecuentes](#14-preguntas-frecuentes)

---

## 1. Visión general

ATMOSPHERICA es un sistema que:

1. Obtiene datos meteorológicos en tiempo real desde la API de OpenWeatherMap
2. Los normaliza y mapea a parámetros visuales
3. Predice si mañana habrá un evento climático extremo usando un modelo Random Forest
4. Genera una pintura abstracta en HTML5 Canvas que codifica el clima del momento
5. Exporta la pintura como PNG
6. Publica todo automáticamente en una galería web en GitHub Pages

El sistema corre de forma autónoma 3 veces al día mediante GitHub Actions y no requiere intervención manual una vez configurado.

---

## 2. El problema de producción

### El desajuste entre entrenamiento e inferencia

El modelo Random Forest fue entrenado con datos históricos de ERA5 (reanálisis climático de Copernicus). Las features incluyen:

- Rolling means de 3 y 7 días (`temp_c_max_ma3`, `pressure_hpa_mean_ma7`...)
- Lags de 1, 2 y 3 días (`temp_c_max_lag1`, `precip_mm_sum_lag2`...)
- Gradientes entre días consecutivos (`temp_grad`, `pressure_grad`)

El problema: en producción solo tienes **un punto de datos** — el momento actual de la API. Si pasas el mismo valor para lag1, lag2 y lag3, el modelo recibe datos que nunca vio en entrenamiento. Las predicciones son formalmente incorrectas.

### Las opciones evaluadas

**Opción A — Acumular historia desde la API** ✓ *elegida*
Guardar los datos de cada ejecución en un CSV local. A partir del día 7 los rolling son reales. A partir del día 14 los lags son completos. Simple, funciona en GitHub Actions, el historial vive en el propio repo.

**Opción B — Reentrenar sin features históricas**
Posible pero peor. El modelo perdería las señales más informativas (gradiente de presión, tendencia de temperatura) que son las que más aportan al AUC.

**Opción C — ERA5 como historia**
ERA5 tiene un delay de ~5 días y fue descargado en un momento fijo. En producción con GitHub Actions el "presente" siempre sería el mismo punto del pasado. Inservible sin re-descargar ERA5 continuamente.

### Por qué la Opción A no jode el historial con 3 ejecuciones diarias

`append_today()` usa `df[~df.index.duplicated(keep="last")]`. Si el workflow corre a las 8h, 12h y 20h del mismo día, el CSV mantiene **una sola fila por día** — la última sobreescribe. El modelo siempre ve exactamente una observación diaria, igual que durante el entrenamiento con ERA5.

---

## 3. Solución: historial acumulado desde la API

### Evolución de la calidad de predicción

| Días acumulados | Estado |
|---|---|
| 0 | Sin historia — proxy aproximado para todas las features |
| 1–6 | Rolling parciales (min_periods=1), lags como proxy |
| 7+ | Rolling ma3 y ma7 reales |
| 14+ | Todos los lags reales — predicción equivalente al entrenamiento |

### Arquitectura de archivos nueva

```
data/
├── fetcher.py       # obtiene datos de la API
├── history.py       # acumula datos diarios → data/history.csv
└── mock.py          # datos estáticos para desarrollo sin API

ml/
├── features.py      # genera featuresAll.csv desde ERA5
├── trainer.py       # entrena el modelo RF
└── predictor.py     # inferencia en producción

archive.py           # actualiza web/data/archive.json
main.py              # punto de entrada
```

---

## 4. Módulo ML — features.py

### Qué hace

Carga los archivos NetCDF de ERA5, los procesa y genera `ml/data_todo/featuresAll.csv` con todas las features listas para entrenar.

### Pipeline interno

```
ERA5 NetCDF (instant + accum)
    │
    ├── xr.open_mfdataset() → merge por coordenadas
    ├── mean(dim=["latitude","longitude"]) → serie temporal puntual
    ├── rename columns → nombres semánticos
    │
    ├── Conversiones físicas
    │     K → °C, Pa → hPa, precip diff × 1000 (m→mm)
    │     humedad via fórmula de Magnus desde dewpoint
    │     wind_speed = √(u² + v²)
    │
    ├── Resample diario → max, min, mean, sum según variable
    │
    ├── Corrección de bias ERA5 (+2.5°C en Sevilla)
    │
    ├── Rolling features (ma3, ma7) para temp, presión, viento, humedad
    ├── Lags (1, 2, 3 días) para temp_max, precip, wind_max
    ├── Gradientes (diff de un día)
    ├── Estacionalidad (sin/cos del día del año)
    ├── Features adicionales:
    │     temp_range, heat_intensity, pressure_deficit,
    │     humidity_range, wind_spike, dry_index, pressure_norm
    │
    ├── Targets:
    │     event_heat  (temp_max ≥ 38°C)
    │     event_cold  (temp_max ≤ 10°C)
    │     event_wind  (wind_max ≥ 8 m/s)
    │     event_rain  (precip ≥ 1 mm)
    │     event_extreme (OR de los anteriores)
    │     target = event_extreme.shift(-1)  ← predecir mañana
    │
    └── featuresAll.csv
```

### Nombres de columnas reales (los que usa el modelo)

```
temp_c_max, temp_c_min, temp_c_mean
pressure_hpa_mean, pressure_hpa_min
wind_speed_max, wind_speed_mean
humidity_max, humidity_mean
precip_mm_sum
cloud_cover_mean
temp_c_max_ma3, pressure_hpa_mean_ma3, wind_speed_max_ma3, humidity_mean_ma3
temp_c_max_ma7, pressure_hpa_mean_ma7, wind_speed_max_ma7, humidity_mean_ma7
temp_c_max_lag1..3, precip_mm_sum_lag1..3, wind_speed_max_lag1..3
temp_grad, pressure_grad
sin_doy, cos_doy
temp_range, heat_intensity, pressure_deficit
humidity_range, wind_spike, dry_index, pressure_norm
```

---

## 5. Módulo ML — trainer.py

### Configuración del modelo

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced_subsample",  # compensa el desbalance (12.2% positivos)
    random_state=42,
    n_jobs=-1
)
```

### Evaluación con TimeSeriesSplit

5 folds estrictamente secuenciales — ningún dato futuro se filtra al entrenamiento. Métricas por fold: F1 y ROC-AUC.

### Lo que guarda trainer.py

```
ml/models_15years/
├── rf_model.pkl     # modelo final entrenado sobre todo el dataset
├── features.pkl     # lista ordenada de nombres de columnas
└── metrics.json     # F1, AUC, top features, metadatos
```

**Importante:** trainer.py **no guarda ningún scaler**. Random Forest no necesita escalado. El predictor original tenía un bug intentando cargar `rf_scaler.pkl` que no existe.

### Ruta del CSV de features

`trainer.py` carga desde `ml/data_Sevilla/features_rf.csv` en su versión original. La ruta correcta donde `features.py` guarda es `ml/data_todo/featuresAll.csv`. Hay que pasarla explícitamente al llamar a `load_features()`.

---

## 6. Módulo ML — predictor.py (corregido)

### Los 3 bugs del predictor original

**Bug 1 — Nombres de columnas inventados**

El predictor original usaba nombres que no existen en el CSV:

```python
# INCORRECTO (original)
"temp_max", "wind_max", "precip_total", "cloud_mean"

# CORRECTO (nombres reales de features.py)
"temp_c_max", "wind_speed_max", "precip_mm_sum", "cloud_cover_mean"
```

**Bug 2 — Cargaba un scaler inexistente**

```python
# INCORRECTO — rf_scaler.pkl nunca fue guardado por trainer.py
self.scaler = joblib.load(self.model_dir / "rf_scaler.pkl")

# CORRECTO — RF no necesita scaler, se elimina completamente
x = np.array([[...]], dtype=np.float32)
risk_score = float(self.model.predict_proba(x)[0][1])
```

**Bug 3 — Nombre del archivo de features**

```python
# INCORRECTO
joblib.load(self.model_dir / "feature_cols.pkl")

# CORRECTO (nombre real que usa trainer.py)
joblib.load(self.model_dir / "features.pkl")
```

### Dos modos de inferencia

**`predict(current_data)`** — datos de la API directamente

Usa el dato puntual de la API con estimaciones para las features históricas. Válido para el primer día o cuando no hay historial. Las estimaciones son conservadoras:

```python
temp_max  = temp + 3.0    # estimación del máximo diario
temp_min  = temp - 5.0    # estimación del mínimo diario
wind_max  = wind * 1.5    # estimación de la ráfaga máxima
```

**`predict_from_history_df(df_row)`** — desde el historial acumulado

Recibe el DataFrame de una fila devuelto por `history.build_features_from_history()`. Las features son reales porque se calculan desde el historial de días anteriores. Esta es la versión de producción.

---

## 7. Módulo de historia — history.py

### Propósito

Acumular los datos diarios de la API en `data/history.csv` para poder construir features reales (rolling, lags, gradientes) en producción.

### `append_today(data)`

Guarda una fila por día. Si se llama varias veces el mismo día (3 ejecuciones del workflow), sobreescribe la fila existente con los datos más recientes:

```python
df = df[~df.index.duplicated(keep="last")]
```

Esto garantiza que el historial siempre tiene exactamente una observación por día, igual que ERA5.

### `build_features_from_history()`

Lee `data/history.csv` y calcula todas las features con la historia real disponible:

- `rolling(3, min_periods=1)` y `rolling(7, min_periods=1)` — funcionan desde el día 1, mejoran progresivamente
- `.shift(lag)` — produce NaN los primeros días, se rellenan con el valor actual como proxy
- Todas las features adicionales derivadas

Devuelve un DataFrame de una sola fila (el último día disponible) listo para pasarlo a `predict_from_history_df()`.

### `history_status()`

Devuelve un string descriptivo del estado actual del historial:

```
"sin historia — predicción desactivada"           # día 0
"3 días — predicción aproximada (faltan 4 para rolling real)"
"8 días — predicción buena (faltan 6 para lags completos)"
"15 días — predicción completa ✓"
```

---

## 8. Módulo de archivo — archive.py

### Propósito

Mantener `web/data/archive.json` actualizado con los metadatos de cada cuadro generado. La web (`index.html`) lee este archivo para construir la galería completa.

### Estructura de cada entrada

```json
{
  "date": "2026-04-28",
  "hour": 12,
  "city": "Sevilla",
  "image_path": "../output/atmospherica_Sevilla_2026-04-28_12h.png",
  "temp_c": 22.4,
  "pressure": 1014,
  "wind_speed": 5.2,
  "wind_dir": "ONO",
  "humidity": 48,
  "clouds": 20,
  "pm25": 8.3,
  "temp_norm": 0.424,
  "pressure_norm": 0.600,
  "wind_energy": 0.260,
  "humidity_norm": 0.600,
  "cloud_norm": 0.200,
  "pm_norm": 0.111,
  "dominant": "presion",
  "dominant_strength": 0.600,
  "dominant2": "temperatura",
  "ml_ready": true,
  "risk_score": 0.142,
  "event_type": "none"
}
```

### Comportamiento

- Si ya existe una entrada para esa fecha+hora la sobreescribe (idempotente)
- Mantiene el array ordenado por fecha descendente (más reciente primero)
- La ruta de la imagen es relativa a `web/index.html` → `../output/nombre.png`

---

## 9. main.py — punto de entrada unificado

### Flujo completo

```
1. get_all_data()                    → datos de la API
2. append_today(data)                → data/history.csv
3. map_to_visual(data)               → parámetros visuales normalizados
4. AtmosphericPredictor()
   ├── build_features_from_history() → features reales si hay historial
   └── predict_from_history_df()     → risk_score, event_type
5. generate_html(visual)             → output/nombre.html
6. Playwright (headless)             → output/nombre.png
7. update_archive(visual, png_path)  → web/data/archive.json
```

### Modo local vs headless

```bash
# Modo local: abre el navegador, el PNG lo guarda el usuario manualmente
python main.py

# Modo headless: Playwright exporta el PNG automáticamente (GitHub Actions)
python main.py --headless
```

### Lógica de predicción

```python
if predictor.is_ready():
    features_today = build_features_from_history()
    if features_today is not None:
        # Historia disponible → features reales
        prediction = predictor.predict_from_history_df(features_today)
    else:
        # Primer día → proxy desde datos de la API
        prediction = predictor.predict(data)
else:
    prediction = {"risk_score": 0.0, "event_type": "unknown", "ready": False}
```

---

## 10. Automatización — GitHub Actions

### Archivo: `.github/workflows/daily.yml`

```yaml
name: Generar cuadro diario

on:
  schedule:
    - cron: '0 7 * * *'    # 8h hora española
    - cron: '0 11 * * *'   # 12h hora española
    - cron: '0 19 * * *'   # 20h hora española
  workflow_dispatch:        # lanzar manualmente desde GitHub

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Instalar dependencias
        run: |
          pip install -r requirements.txt
          playwright install chromium
      - name: Generar cuadro
        env:
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
        run: python main.py --headless
      - name: Commit
        run: |
          git config user.name  "atmospherica-bot"
          git config user.email "bot@atmospherica"
          git add data/history.csv output/ web/
          git diff --staged --quiet || git commit -m "cuadro $(date +%Y-%m-%d_%Hh)"
          git push
```

### Por qué 3 crons y no uno

El cuadro visual cambia según la hora del día porque `C.hour` afecta directamente la paleta de fondo en `buildPalette()` y la posición del gradiente de luz en `drawFondo()`:

| Hora | Paleta |
|---|---|
| < 5h | Azul muy oscuro (noche) |
| 5–8h | Naranja amanecer |
| 8–12h | Amarillo claro (mañana) |
| 12–16h | Azul-blanco (mediodía) |
| 16–20h | Naranja atardecer |
| > 20h | Azul oscuro (noche) |

Con 3 ejecuciones diarias la galería tiene variedad de luz. Los PNGs se nombran con la hora exacta y no se sobreescriben entre sí.

### El historial no se corrompe con 3 ejecuciones

`append_today()` desduplicar por fecha antes de guardar. El CSV siempre tiene exactamente una fila por día calendario independientemente de cuántas veces corra el workflow ese día.

### Configuración en GitHub

Solo hay que añadir un secret en el repositorio:

```
Settings → Secrets and variables → Actions → New repository secret
Nombre: OPENWEATHER_API_KEY
Valor:  tu_api_key_de_openweathermap
```

El `GITHUB_TOKEN` para hacer commit ya viene incluido automáticamente en todos los repos de GitHub.

---

## 11. Conexión con la web

### Cómo lee la web los cuadros

`web/index.html` hace un fetch al inicializar:

```javascript
async function init() {
    const r = await fetch('data/archive.json');
    const archive = await r.json();
    // construye la galería desde el array
}
```

El cuadro más reciente (`archive[0]`) se muestra como la obra principal en el centro de la sala. El resto se muestra en el grid del archivo en la parte inferior.

### Rutas de imagen

Los PNGs están en `output/` en la raíz del repo. La web está en `web/`. La ruta relativa desde `web/index.html` hasta un PNG es `../output/nombre.png`.

Por eso `archive.py` guarda:

```python
"image_path": "../output/atmospherica_Sevilla_2026-04-28_12h.png"
```

### Configuración de GitHub Pages

En el repositorio: `Settings → Pages → Source: Deploy from a branch → Branch: main → Folder: /web`

GitHub Pages servirá `web/index.html` como raíz. Las rutas `../output/` funcionan porque GitHub Pages sirve el repo completo, no solo la carpeta `/web`.

### Flujo de publicación

```
GitHub Actions corre
    └── main.py --headless
          ├── genera output/cuadro.png
          ├── actualiza web/data/archive.json
          └── git commit + push
                └── GitHub Pages detecta el push
                      └── galería actualizada en ~30 segundos
```

---

## 12. Estructura final del repositorio

```
atmospherica/
│
├── .github/
│   └── workflows/
│       └── daily.yml              ← automatización (3 crons diarios)
│
├── data/
│   ├── __init__.py
│   ├── fetcher.py                 ← OpenWeatherMap API
│   ├── history.py                 ← acumula datos diarios → history.csv
│   ├── history.csv                ← generado automáticamente, commiteado
│   └── mock.py                    ← datos estáticos para desarrollo
│
├── visual/
│   ├── __init__.py
│   ├── mapper.py                  ← normalización + cálculo del dominante
│   └── generator.py               ← genera HTML + Canvas 2D (v2)
│
├── ml/
│   ├── __init__.py
│   ├── data_todo/
│   │   └── featuresAll.csv        ← generado por features.py
│   ├── models_15years/
│   │   ├── rf_model.pkl           ← modelo entrenado
│   │   ├── features.pkl           ← lista de columnas (orden exacto)
│   │   └── metrics.json           ← F1, AUC, top features
│   ├── features.py                ← procesa ERA5 → featuresAll.csv
│   ├── trainer.py                 ← entrena el RF
│   └── predictor.py               ← inferencia en producción (corregido)
│
├── web/
│   ├── index.html                 ← galería principal
│   ├── about.html                 ← descripción del proyecto
│   ├── technical.html             ← documentación ML
│   └── data/
│       └── archive.json           ← generado automáticamente, commiteado
│
├── output/                        ← PNGs generados (commiteados)
│   └── atmospherica_Sevilla_2026-04-28_12h.png
│
├── archive.py                     ← actualiza archive.json (raíz)
├── main.py                        ← punto de entrada
├── requirements.txt
├── .env                           ← API key (nunca en git)
└── .gitignore
```

---

## 13. Flujo completo de datos

```
OpenWeatherMap API
    │
    ▼
get_all_data()          → dict con temp, presión, viento, humedad, nubes, PM2.5
    │
    ├──▶ append_today()         → data/history.csv  (una fila por día)
    │
    ▼
map_to_visual()         → parámetros normalizados + dominante calculado
    │
    ├──▶ build_features_from_history()
    │       └── rolling reales, lags reales, gradientes reales
    │           ▼
    │       predict_from_history_df()
    │           └── rf_model.predict_proba()  → risk_score (0-1)
    │
    ▼
generate_html(visual)   → output/nombre.html
    │
    ▼ (headless)
Playwright              → output/nombre.png
    │
    ▼
update_archive()        → web/data/archive.json
    │
    ▼
git commit + push
    │
    ▼
GitHub Pages            → galería pública actualizada
```

---

## 14. Preguntas frecuentes

**¿Por qué el modelo no usa directamente los datos de la API sin historial?**

Porque fue entrenado con features que dependen de historia (rolling 7 días, lags 3 días). Pasarle el mismo valor repetido para lag1, lag2 y lag3 es pasarle datos que nunca vio en entrenamiento. Las predicciones serían formalmente incorrectas aunque pudieran acertar por casualidad.

**¿Cuántos días tarda en funcionar bien?**

A partir del día 7 los rolling son reales y la predicción es decente. A partir del día 14 todos los lags son completos y la predicción es equivalente al entrenamiento. Durante los primeros días el sistema funciona pero las predicciones son aproximadas.

**¿Por qué no se usa ERA5 en tiempo real?**

ERA5 tiene un delay de ~5 días entre la realidad y la disponibilidad de datos. Además el dataset fue descargado en un momento fijo — en producción el "presente" siempre sería el mismo punto del pasado salvo que se re-descargue continuamente, lo que requiere credenciales de Copernicus CDS y tiempo de descarga en cada ejecución.

**¿El workflow de GitHub Actions puede fallar si no hay internet o la API devuelve error?**

Sí. Es conveniente añadir manejo de errores en `fetcher.py` y que el workflow no haga commit si la generación falla. Una mejora futura sería añadir un step de validación antes del commit.

**¿Se pueden añadir más ciudades?**

Sí. Cambiando `CITY` y `COUNTRY_CODE` en `config.py` y ejecutando `main.py`. Cada ciudad tendría su propio historial y sus propios cuadros. La web los mostraría mezclados en la galería (ordenados por fecha).

**¿Por qué GitHub Pages y no un servidor propio?**

Para un portfolio de ML la prioridad es que el proyecto sea accesible, mantenible y que demuestre capacidad de construir sistemas end-to-end. GitHub Pages + Actions cubre todo eso sin coste, sin mantenimiento de servidor y con CI/CD incluido.

---

*Documentación generada para ATMOSPHERICA v2 — Sevilla, 2026*
*Stack: Python 3.11 · scikit-learn · ERA5 · OpenWeatherMap API · HTML5 Canvas · GitHub Actions · GitHub Pages*

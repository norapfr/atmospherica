# ATMOSPHERICA: Real-Time Atmospheric Data as Generative Painting with Integrated Extreme Event Prediction

> *A machine learning system that translates real-time atmospheric observations into deterministic generative painting, with an embedded predictive layer for extreme weather events using Random Forest and LSTM classifiers trained on ERA5 reanalysis data.*

**[Live exhibition](https://norapfr.github.io/atmospherica)** &nbsp;·&nbsp; [Project](https://norapfr.github.io/atmospherica/about.html) &nbsp;·&nbsp; [Technical](https://norapfr.github.io/atmospherica/technical.html)

---

## Abstract

ATMOSPHERICA is an autonomous system that, three times daily, ingests meteorological and air quality data from the OpenWeatherMap API, maps each variable to a formal visual parameter through a normalization pipeline calibrated on local historical ranges, and generates a unique abstract painting encoding the atmospheric state of that moment. A deterministic seeded RNG guarantees that the same atmospheric conditions always produce the same image. The dominant variable — defined as the argmax of normalized values across six climate variables — deforms the global composition, palette, and rendering behavior of every other layer.

A predictive component trained on ERA5 reanalysis data (1940–1990 and 2010–2024, ~24,800 daily observations) estimates the probability of an extreme weather event the following day. When risk exceeds defined thresholds, visual alarm signals appear in the painting using the morphology of today's dominant variable rendered in the event's canonical color. The full pipeline — data ingestion, feature engineering, prediction, rendering, PNG export, and web publication — runs autonomously via GitHub Actions and self-updates a GitHub Pages gallery with each execution.

---

## 1. Motivation and Design Rationale

### 1.1 The problem with data visualization

Conventional data visualization translates numerical values into chart primitives: axes, bars, lines. The datum is represented but not experienced. A temperature chart shows that it was 42°C on a given day; it does not communicate what 42°C feels like in the Guadalquivir basin in August — the violence of it, the persistence of it, the way it overrides everything else.

ATMOSPHERICA proposes a different relationship between datum and representation. Rather than mapping climate data *onto* a visual form, it uses climate data to *generate* form through a grammar of rules where each variable controls a family of visual elements with fixed color semantics, parametric geometry, and dynamic behavior that changes qualitatively, not just quantitatively, as the variable's intensity grows.

### 1.2 Why this constitutes an ML engineering problem

The project is not a data art exercise with some Python attached. It is an end-to-end ML system with the following engineering requirements:

- Real API ingestion with credential management and error handling
- Historical normalization with local calibration (25°C is unremarkable in Seville; it is a heat event in Edinburgh — global ranges are meaningless)
- Feature engineering for time series prediction (rolling means, lags, gradients, seasonal encoding, domain-specific composites)
- Model training with appropriate evaluation methodology for imbalanced temporal data (TimeSeriesSplit, AUC as primary metric, threshold optimization)
- Production inference that degrades gracefully when temporal context is unavailable
- Automated deployment with state persistence (history CSVs committed to the repository)
- A public, live artifact that accumulates daily

The painting is the output format. The system underneath is the contribution.

---

## 2. Data

### 2.1 ERA5 Reanalysis

The primary dataset is ERA5 reanalysis from the Copernicus Climate Data Store, downloaded via `cdsapi` for a 1.5° × 1.5° grid cell centred on Seville (38°N–36.5°N, 6.5°W–5°W). Two temporal periods were selected:

- **1940–1990**: 51 years of pre-industrial and early-industrial climate. Provides the long-run statistical baseline for extreme event frequency and seasonal distribution in the Guadalquivir basin.
- **2010–2024**: 15 years of contemporary climate with documented warming trend and increased frequency of extreme heat events in southern Iberia.

The split is deliberate. Training exclusively on recent data would underestimate the historical frequency of cold and precipitation extremes. Training exclusively on historical data would miss the distributional shift in summer maxima driven by climate change. The combined dataset exposes the model to both the climatological baseline and the contemporary signal.

**Variables downloaded** at 06:00, 12:00, 18:00 UTC: 2m temperature (`t2m`), surface pressure (`sp`), 10m wind components (`u10`, `v10`), 2m dewpoint temperature (`d2m`), total precipitation (`tp`, accumulated), total cloud cover (`tcc`).

**Processing pipeline:**

| Step | Operation |
|---|---|
| Unit conversion | K → °C; Pa → hPa; m → mm (precipitation via diff); wind components → speed |
| Humidity derivation | Magnus formula from dewpoint: `RH = 100 · exp(17.625·Td/(243.04+Td) − 17.625·T/(243.04+T))` |
| Spatial aggregation | Mean over the 1.5° bounding box |
| Temporal aggregation | Resample to daily: T_max, T_min, T_mean; P_mean, P_min; wind_max, wind_mean; precip_sum; humidity_max, humidity_mean; cloud_mean |
| Bias correction | +2.5°C applied to all temperature fields; ERA5 systematically underestimates T_max in Seville at this grid resolution (confirmed against AEMET station records) |

**Final dataset:** 24,838 daily observations across both periods.

### 2.2 Label definition

Binary classification target: will there be an extreme weather event tomorrow? The target is the event indicator shifted one day forward (`shift(-1)`).

Four event triggers (OR logic):

| Type | Criterion | Source |
|---|---|---|
| Heat | T_max ≥ 38°C | AEMET heat wave definition for Seville |
| Cold | T_max ≤ 10°C | Local climatological threshold |
| Wind | Speed ≥ 8 m/s | Gale-force onset |
| Rain | Precip ≥ 1 mm/day | Measurable precipitation |

**Event rate: 5.0%** of days across the full dataset. This strong class imbalance is the primary modeling challenge and drove every decision about loss function, class weighting, evaluation metric, and threshold selection.

### 2.3 Real-time data

Production ingestion uses two OpenWeatherMap endpoints:

- **Current Weather API**: temperature, pressure, wind speed and direction, humidity, cloud cover, precipitation in last hour, weather condition code
- **Air Pollution API** (called with coordinates from weather response): PM2.5, NO₂, O₃, CO, SO₂, NH₃, AQI index

Credentials are stored as a GitHub Actions secret and locally in `.env` (gitignored). Three executions daily (08h, 12h, 20h Spain local time) aggregate into one canonical daily row through deduplication logic in `data/history.py`, matching the daily frequency of the ERA5 training data.

---

## 3. Feature Engineering

### 3.1 Random Forest features (39)

| Group | Features | Count |
|---|---|---|
| Raw daily aggregates | T_max, T_min, T_mean, P_mean, P_min, wind_max, wind_mean, humidity_max, humidity_mean, precip_sum, cloud_mean | 11 |
| 3-day rolling means | temp_c_max_ma3, pressure_hpa_mean_ma3, wind_speed_max_ma3, humidity_mean_ma3 | 4 |
| 7-day rolling means | temp_c_max_ma7, pressure_hpa_mean_ma7, wind_speed_max_ma7, humidity_mean_ma7 | 4 |
| Temporal lags (1–3 days) | temp_c_max_lag{1,2,3}, precip_mm_sum_lag{1,2,3}, wind_speed_max_lag{1,2,3} | 9 |
| First-order gradients | temp_grad, pressure_grad | 2 |
| Seasonal encoding | sin_doy, cos_doy | 2 |
| Composite features | temp_range, heat_intensity, pressure_deficit, humidity_range, wind_spike, dry_index, pressure_norm | 7 |

**Dry index** (`temp_range × (100 − humidity_mean)`) is the most domain-specific feature. It captures the combination of wide diurnal temperature swings and very low relative humidity that characterizes Seville's pre-heatwave atmospheric signature. Ranked 10th by SHAP, it contributes independently of the individual temperature and humidity features already in the set.

**Pressure deficit** (`pressure_hpa_mean − pressure_hpa_mean_ma7`) captures the short-term deviation of pressure from its recent mean. A negative deficit signals a developing low-pressure system even when absolute pressure values appear normal.

### 3.2 LSTM features (17)

The LSTM receives a reduced set of physical and derived features, without the full lag structure, to avoid redundancy with the sequence's own temporal context:

**Raw physical variables (9):** temp_c, dewpoint_c, pressure_hpa, wind_u, wind_v, precip_mm, cloud_cover, season_sin, season_cos

**Derived temporal features (8):** temp_ma3, temp_ma7, pressure_ma3, pressure_grad, wind_speed, wind_ma3, precip_ma3, dry_index

**Critical implementation note:** All features are computed over the complete daily DataFrame before sequence construction. The StandardScaler is fitted exclusively on training days (`feats[:train_end]`) and applied to the full array. Constructing sequences first and scaling afterward introduces overlap bias: the same day appears in multiple windows with different weights, distorting the scaler's statistics.

---

## 4. Models

### 4.1 Random Forest

**Configuration:**

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced_subsample",  # per-tree resampling
    random_state=42,
    n_jobs=-1
)
```

`balanced_subsample` recomputes class weights independently for each bootstrap sample. This is more appropriate than global `balanced` weighting under strong imbalance because each tree sees a different effective class distribution, reducing co-variance between trees on the minority class.

No feature scaling is applied. Random Forest is scale-invariant and any preprocessing step introduces unnecessary complexity with no benefit.

**Evaluation methodology:** TimeSeriesSplit with 5 folds. Each fold's validation set is strictly after its training set in chronological order. No shuffling at any point. This reflects the realistic deployment scenario: the model predicts future events based on past observations.

**Results:**

| Fold | F1 | ROC-AUC |
|---|---|---|
| 1 | 0.043 | 0.806 |
| 2 | 0.127 | 0.773 |
| 3 | 0.190 | 0.888 |
| 4 | 0.156 | 0.825 |
| 5 | 0.229 | 0.886 |
| **Mean** | **0.149** | **0.836** |

**On the F1 score:** F1 = 0.149 is not a failure metric. With a 5.0% event rate, a classifier that predicts all negatives achieves F1 = 0.0 and accuracy = 95%. The RF's F1 reflects genuine positive detections at a useful precision-recall trade-off, achieved against a strongly imbalanced baseline. **ROC-AUC 0.836 is the operative metric**: it measures discriminative power across all thresholds and is not affected by class imbalance. A model that reliably ranks high-risk days above low-risk days is exactly what the visual alarm system requires.

**SHAP feature importance (top 10, mean |SHAP|):**

| Feature | SHAP |
|---|---|
| pressure_hpa_min | 0.0324 |
| temp_c_max | 0.0285 |
| cos_doy | 0.0226 |
| pressure_norm | 0.0201 |
| temp_c_mean | 0.0197 |
| cloud_cover_mean | 0.0185 |
| pressure_hpa_mean | 0.0181 |
| humidity_mean | 0.0176 |
| wind_speed_mean | 0.0155 |
| dry_index | 0.0153 |

Pressure features (min, mean, norm) collectively contribute more than temperature, confirming that synoptic-scale dynamics — not just local temperature thresholds — are the primary signal for next-day extreme events.

### 4.2 AtmosphericLSTM

**Architecture:**

```
Input  → (batch, 14, 17)        # 14-day window × 17 features
LSTM₁  → hidden=128             # short-range synoptic patterns
LSTM₂  → hidden=64              # regime-level compression
LayerNorm → (batch, 64)         # last timestep; LayerNorm > BatchNorm for short seqs
Linear(64→32) + GELU + Dropout(0.25)
Linear(32→1) + Sigmoid          # P(extreme event tomorrow)
```

**Trainable parameters:** 127,169

**Training protocol:**
- Optimizer: AdamW, lr=1×10⁻⁴, weight_decay=1×10⁻⁴
- Scheduler: Cosine annealing over 150 epochs
- Loss: Focal Loss (α=0.80, γ=1.5)
- Early stopping: patience=25 on validation AUC-PR
- Batch size: 64 (larger batches → more stable gradients on CPU)
- Split: 70/10/20 train/val/test with 14-day gap between partitions

**Focal Loss rationale:** `FL(p_t) = −α_t · (1−p_t)^γ · log(p_t)`. The modulating factor `(1−p_t)^γ` down-weights confidently correct predictions (the majority of negatives) and concentrates gradient signal on the rare, hard positives. γ=1.5 was chosen over the canonical γ=2.0 because with a 5% event rate the model struggles to find positive examples early in training; a softer focus parameter allows more stable initial convergence.

**Results:**

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.678 |
| Test F1 (th=0.38) | 0.135 |
| Test AUC-PR | 0.091 |
| Best val AUC-PR | 0.121 |
| Epochs to early stopping | 40 / 150 |

**Training dynamics:** AUC-PR peaked at epoch 15 (0.121) and did not improve over the following 25 epochs. ROC-AUC continued rising slowly (0.658 → 0.712) while AUC-PR stagnated — a characteristic signature of Focal Loss on imbalanced data where the model has exhausted its ability to discriminate the positive class under the precision-recall constraint but continues improving ranking quality.

### 4.3 Comparative analysis

| Dimension | Random Forest | AtmosphericLSTM |
|---|---|---|
| Test ROC-AUC | **0.836** | 0.678 |
| Test F1 | 0.149 | 0.135 |
| Features | 39 (with explicit lags) | 17 (raw + derived) |
| Sequence context | Encoded as lag features | 14-day window |
| Training data | 5,472 (RF subset) | 24,838 |
| Scaling required | No | Yes (StandardScaler) |
| Inference complexity | O(n_trees · depth) | O(seq_len · hidden) |

The RF outperforms the LSTM by 0.158 AUC points despite training on a smaller dataset (5,472 vs 24,838 days). The primary reason is feature engineering: the RF receives pre-computed 7-day rolling means, 3-day lags, and domain-specific composites that encode temporal context explicitly. The LSTM must derive equivalent representations from raw 14-day sequences, which requires more capacity and more data than available. With ~24k daily observations and 127k parameters, the LSTM is not capacity-constrained — it is information-constrained: the temporal patterns needed for accurate prediction are better captured by explicit engineered features than by learning them from sequences of this length and this dataset size.

This finding has practical implications: for tabular climate prediction at daily resolution with datasets under 50k observations, explicit temporal feature engineering paired with gradient-boosted trees or random forests will generally outperform sequence models. The LSTM becomes competitive when raw high-frequency data (hourly or sub-hourly) is available and explicit feature engineering becomes intractable.

---

## 5. Visual System

### 5.1 Normalization and dominance

Six variables are normalized to [0,1] using historical ranges calibrated for Seville:

| Variable | Min | Max | Formula |
|---|---|---|---|
| Temperature | 0°C | 46°C | `(T - 0) / 46` |
| Wind energy | 0 m/s | 20 m/s | `W / 20` |
| Humidity | 0 | `veil_opacity / 80` | mapper output |
| Pressure | 990 hPa | 1030 hPa | `(P - 990) / 40` |
| Clouds | 0% | 100% | `C / 100` |
| PM2.5 | 0 μg/m³ | 75 μg/m³ | `PM / 75` |

The **dominant variable** is the argmax of these six values. It controls global composition through four mechanisms: (1) background palette tinting (35% mix toward the dominant's canonical hue), (2) geometric modifiers (canvas rotation for wind, blur for humidity, squish for clouds, scale for temperature), (3) render order (dominant always rendered last, on top of all other layers), (4) within-function behavior (dominant mode produces 2–4× more elements at larger scale with higher alpha).

### 5.2 Visual grammar

Each variable maps to a fixed family of forms with fixed color semantics:

| Variable | Form | Color |
|---|---|---|
| Temperature | Concentric circles | Blue (≤5°C) → green (13–18°C) → orange (25–30°C) → red (≥31°C) |
| Wind | Bézier curves | HSL(220) with luminosity inversely proportional to background brightness |
| Humidity | Ovals (>35% RH) / triangles (<35% RH) | HSL(150), saturation ∝ humidity |
| Pressure | Horizontal bands (high P) / tilted rectangles (low P) | HSL(38), opacity ∝ pressure |
| Clouds | Flattened diamonds, upper-canvas weighted | HSL(210), darkness ∝ cloud cover |
| PM2.5 | Dots + violet global haze | HSL(285), saturation ∝ PM2.5 |

Temperature color is the only channel that maintains absolute semantic invariance: it never changes regardless of which variable dominates, making temperature always readable even when it is not dominant.

### 5.3 Predictive alarm system

When `risk_score > 0.06`, the painting encodes the prediction through progressive visual signals. Each alarm layer uses the **morphology of today's dominant variable** rendered in the **color of tomorrow's predicted event type**:

| Threshold | Signal | Description |
|---|---|---|
| > 6% | Edge triangles | Triangles penetrating inward from all four margins; depth ∝ risk |
| > 20% | Background tint | Semi-transparent global wash in event color |
| > 25% | Internal fractures | Dominant-morphology shapes in event color scattered across canvas |
| > 75% | Alert border | Double rectangle frame (diffuse + sharp) around full canvas |
| > 85% | Radial focal points | Diffuse light sources in event color contaminating background gradient |

Event colors are fixed and do not appear in the base grammar: heat = HSL(8, 85%, 50%); cold = HSL(215, 78%, 42%); wind = HSL(155, 62%, 38%); rain = HSL(200, 72%, 40%). Any tint in these hues in a painting is a prediction signal, not climate description.

### 5.4 Deterministic reproducibility

The renderer uses an XORShift PRNG seeded from the concatenation of `city + date + hour`. This guarantees that the same atmospheric observation always produces the same painting in any browser on any machine. Reproducibility is treated as a first-class requirement: the painting is a data artifact, not a stochastic sample.

```javascript
function seededRng(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}
```

---

## 6. System Architecture

### 6.1 End-to-end pipeline

```
OpenWeatherMap + Air Pollution APIs
        │
        ▼
   data/fetcher.py         real-time weather + air quality
        │
        ├──▶ data/history.py       append to history_raw.csv
        │                          aggregate to history_daily.csv (1 row/day)
        │                          deduplicate: 3 runs → 1 canonical daily row
        ▼
   visual/mapper.py        normalize 6 variables, compute dominant (argmax)
        │
        ├──▶ ml/predictor.py       build feature vector from daily history
        │                          rf_model.predict_proba() → risk_score ∈ [0,1]
        │                          fallback chain: history → ERA5 CSV → API estimates
        ▼
   visual/generator.py     HTML5 Canvas, seeded RNG, 8 render passes
        │
        ▼
   Playwright              headless PNG export (900×1080px)
        │
        ▼
   archive.py              update docs/data/archive.json
        │
        ▼
   git commit + push       via GitHub Actions
        │
        ▼
   GitHub Pages            gallery updated in ~30s
```

### 6.2 Production inference fallback chain

The RF requires temporal features unavailable from a single API snapshot. Three fallback modes:

1. **Full history (≥14 days):** All lag, rolling, and gradient features computed from real observed data. Equivalent to training conditions.
2. **Partial history (1–13 days):** Rolling means use `min_periods=1` (functional from day 1, converging). Missing lags filled with current value. Prediction is valid but conservative.
3. **No history (day 0):** `predict_from_history()` called on `ml/data_todo/featuresAll.csv`. Uses the last ERA5 observation as the feature vector. Prediction reflects 2024 climate conditions, not live data.

### 6.3 Automation

GitHub Actions workflow (`.github/workflows/daily.yml`):
- **Triggers:** `schedule` at 07:00, 11:00, 19:00 UTC (08h, 12h, 20h Spain time) + `workflow_dispatch`
- **Permissions:** read/write (required for commit)
- **Steps:** checkout → setup Python 3.11 → install dependencies → install Playwright Chromium → run `main.py --headless` → `git add data/ docs/ output/` → conditional commit (skipped if no changes) → push

Three daily paintings per day serve different atmospheric moments (morning, midday, evening) while also exercising the deduplication logic that maintains exactly one daily row in the history CSV.

---

## 7. Repository Structure

```
atmospherica/
├── .github/workflows/daily.yml    ← 3 daily crons + manual dispatch
├── data/
│   ├── fetcher.py                 ← OpenWeatherMap + Air Pollution APIs
│   ├── history.py                 ← raw readings → daily aggregates
│   ├── history_raw.csv            ← up to 3 readings/day, committed
│   └── history_daily.csv          ← one canonical row/day, committed
├── visual/
│   ├── mapper.py                  ← normalization + dominant computation
│   └── generator.py               ← HTML5 Canvas painter (v2)
├── ml/
│   ├── features.py                ← ERA5 NetCDF → featuresAll.csv (RF)
│   ├── lstm_trainer.py            ← LSTM training pipeline
│   ├── trainer.py                 ← RF training + SHAP
│   ├── predictor.py               ← RF production inference (3 fallback modes)
│   ├── data_todo/
│   │   ├── featuresAll.csv        ← RF features, ERA5 1940–1990 + 2010–2024
│   │   └── features_lstm.csv      ← LSTM features (17 columns)
│   └── final_model/
│       ├── rf_model.pkl           ← trained RF
│       ├── features.pkl           ← ordered RF feature list
│       ├── metrics.json           ← RF: AUC 0.836, F1 0.149, SHAP values
│       ├── lstm_model.pt          ← trained LSTM weights
│       ├── lstm_scaler.pkl        ← StandardScaler fitted on train days
│       ├── lstm_feature_cols.pkl  ← ordered LSTM feature list
│       └── lstm_metrics.json      ← LSTM: AUC 0.678, F1 0.135, history
├── docs/
│   ├── index.html                 ← live exhibition gallery
│   ├── about.html                 ← project + visual grammar description
│   ├── technical.html             ← ML documentation (this document in HTML)
│   ├── data/archive.json          ← painting metadata, auto-updated
│   └── output/                    ← generated PNGs, committed
├── archive.py                     ← updates archive.json
├── main.py                        ← entry point
└── requirements.txt
```

---

## 8. Setup and Execution

### Requirements

- Python 3.11+
- Free [OpenWeatherMap API key](https://openweathermap.org/api) (1,000 calls/day on free tier)
- Copernicus CDS account (for ERA5 download and model retraining only)

### Local setup

```bash
git clone https://github.com/norapfr/atmospherica
cd atmospherica
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

```
# .env
OPENWEATHER_API_KEY=your_key_here
```

### Run

```bash
python main.py             # generate painting, open browser
python main.py --headless  # generate PNG without browser (CI mode)
```

### Train RF

```bash
python ml/features.py      # ERA5 NetCDF → ml/data_todo/featuresAll.csv
python ml/trainer.py       # train RF, save model + SHAP metrics
```

### Train LSTM

```bash
# If features_lstm.csv doesn't exist yet:
python ml/lstm_trainer.py --mode train --build_features

# If features_lstm.csv already exists:
python ml/lstm_trainer.py --mode train
```

### Deploy to GitHub Actions

1. `Settings → Actions → General → Workflow permissions → Read and write permissions`
2. `Settings → Secrets → Actions → New secret: OPENWEATHER_API_KEY`
3. `Settings → Pages → Branch: main → Folder: /docs`

---

## 9. Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | Random Forest, StandardScaler, TimeSeriesSplit, metrics |
| `torch` | LSTM architecture, Focal Loss, AdamW |
| `shap` | TreeExplainer SHAP values for RF interpretability |
| `xarray`, `netCDF4` | ERA5 NetCDF4 loading and merging |
| `cdsapi` | Copernicus CDS API client for ERA5 download |
| `pandas`, `numpy` | Feature engineering and data manipulation |
| `joblib` | Model serialization |
| `playwright` | Headless Chromium for PNG export |
| `requests`, `python-dotenv` | API calls and credential management |

---
---

# ATMOSPHERICA (Español)

> *Un sistema de machine learning que traduce observaciones atmosféricas en tiempo real a pintura generativa determinista, con una capa predictiva integrada para eventos climáticos extremos mediante Random Forest y LSTM entrenados sobre datos de reanálisis ERA5.*

---

## Resumen

ATMOSPHERICA es un sistema autónomo que, tres veces al día, ingiere datos meteorológicos y de calidad del aire de la API de OpenWeatherMap, mapea cada variable a un parámetro visual formal a través de un pipeline de normalización calibrado con rangos históricos locales, y genera una pintura abstracta única que codifica el estado atmosférico de ese momento. Un generador de números pseudoaleatorios determinista con semilla garantiza que las mismas condiciones atmosféricas produzcan siempre la misma imagen. La variable dominante — definida como el argmax de los valores normalizados de seis variables climáticas — deforma la composición global, la paleta y el comportamiento de renderizado de todas las demás capas.

Un componente predictivo entrenado sobre datos de reanálisis ERA5 (1940–1990 y 2010–2024, ~24.800 observaciones diarias) estima la probabilidad de que ocurra un evento climático extremo al día siguiente. Cuando el riesgo supera umbrales definidos, aparecen señales de alarma visuales en la pintura usando la morfología de la variable dominante del día renderizada en el color canónico del evento predicho. El pipeline completo — ingesta, feature engineering, predicción, renderizado, exportación PNG y publicación web — corre de forma autónoma mediante GitHub Actions y actualiza una galería en GitHub Pages con cada ejecución.

---

## 1. Motivación y justificación del diseño

### 1.1 El problema de la visualización convencional

La visualización de datos convencional traduce valores numéricos a primitivas de gráfico: ejes, barras, líneas. El dato se representa pero no se experimenta. Un gráfico de temperatura muestra que hubo 42°C un día determinado; no comunica qué significa 42°C en la cuenca del Guadalquivir en agosto — su violencia, su persistencia, cómo anula todo lo demás.

ATMOSPHERICA propone una relación diferente entre dato y representación. En lugar de mapear datos climáticos *sobre* una forma visual, usa datos climáticos para *generar* forma a través de una gramática de reglas donde cada variable controla una familia de elementos visuales con semántica de color fija, geometría paramétrica y comportamiento dinámico que cambia cualitativamente, no solo cuantitativamente, a medida que crece la intensidad de la variable.

### 1.2 Por qué esto es un problema de ML engineering

El proyecto no es un ejercicio de data art con algo de Python. Es un sistema de ML end-to-end con los siguientes requisitos de ingeniería:

- Ingesta de APIs reales con gestión de credenciales y manejo de errores
- Normalización histórica con calibración local (25°C es irrelevante en Sevilla; es un evento de calor en Edimburgo — los rangos globales no tienen sentido)
- Feature engineering para predicción de series temporales (medias móviles, lags, gradientes, codificación estacional, composites específicos del dominio)
- Entrenamiento de modelos con metodología de evaluación apropiada para datos temporales desbalanceados (TimeSeriesSplit, AUC como métrica principal, optimización de umbral)
- Inferencia en producción que degrada de forma controlada cuando el contexto temporal no está disponible
- Despliegue automatizado con persistencia de estado (CSVs de historial commiteados al repositorio)
- Un artefacto público y vivo que se acumula diariamente

La pintura es el formato de salida. El sistema que hay debajo es la contribución.

---

## 2. Datos

### 2.1 Reanálisis ERA5

El dataset primario es el reanálisis ERA5 del Copernicus Climate Data Store, descargado vía `cdsapi` para una celda de cuadrícula de 1,5° × 1,5° centrada en Sevilla (38°N–36,5°N, 6,5°W–5°W). Se seleccionaron dos períodos temporales:

- **1940–1990**: 51 años de clima pre-industrial y de industrialización temprana. Proporciona la línea base estadística a largo plazo para la frecuencia de eventos extremos y la distribución estacional en la cuenca del Guadalquivir.
- **2010–2024**: 15 años de clima contemporáneo con tendencia de calentamiento documentada y mayor frecuencia de eventos de calor extremo en el sur de Iberia.

La división es deliberada. Entrenar exclusivamente con datos recientes subestimaría la frecuencia histórica de episodios de frío y precipitación. Entrenar exclusivamente con datos históricos perdería el cambio distribucional en los máximos estivales impulsado por el cambio climático. El dataset combinado expone al modelo tanto a la línea base climatológica como a la señal contemporánea.

**Variables descargadas** a las 06:00, 12:00, 18:00 UTC: temperatura 2m (`t2m`), presión superficial (`sp`), componentes del viento 10m (`u10`, `v10`), temperatura de punto de rocío 2m (`d2m`), precipitación total acumulada (`tp`), cobertura nubosa total (`tcc`).

**Dataset final:** 24.838 observaciones diarias en ambos períodos.

### 2.2 Definición de etiquetas

Target de clasificación binaria: ¿habrá un evento climático extremo mañana? El target es el indicador de evento desplazado un día hacia adelante (`shift(-1)`).

Cuatro disparadores de evento (lógica OR):

| Tipo | Criterio | Fuente |
|---|---|---|
| Calor | T_max ≥ 38°C | Definición de ola de calor AEMET para Sevilla |
| Frío | T_max ≤ 10°C | Umbral climatológico local |
| Viento | Velocidad ≥ 8 m/s | Inicio de viento de gale-force |
| Lluvia | Precip ≥ 1 mm/día | Precipitación medible |

**Tasa de eventos: 5,0%** de los días en el dataset completo.

---

## 3. Modelos

### 3.1 Random Forest

**Configuración:**

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

**Evaluación:** TimeSeriesSplit con 5 folds estrictamente secuenciales.

**Resultados:**

| Fold | F1 | AUC |
|---|---|---|
| 1 | 0,043 | 0,806 |
| 2 | 0,127 | 0,773 |
| 3 | 0,190 | 0,888 |
| 4 | 0,156 | 0,825 |
| 5 | 0,229 | 0,886 |
| **Media** | **0,149** | **0,836** |

**Sobre el F1:** Con una tasa de eventos del 5%, un clasificador que predice siempre negativo obtiene F1 = 0,0 y accuracy = 95%. El F1 = 0,149 del RF refleja detecciones positivas reales en un trade-off precision-recall útil. AUC 0,836 es la métrica operativa.

### 3.2 AtmosphericLSTM

**Arquitectura:**
```
Input  → (batch, 14, 17)    # ventana de 14 días × 17 features
LSTM₁  → hidden=128
LSTM₂  → hidden=64
LayerNorm → (batch, 64)     # último timestep
Linear(64→32) + GELU + Dropout(0,25)
Linear(32→1) + Sigmoid
```

**Parámetros entrenables:** 127.169

**Resultados:**

| Métrica | Valor |
|---|---|
| Test ROC-AUC | 0,678 |
| Test F1 (th=0,38) | 0,135 |
| Test AUC-PR | 0,091 |
| Mejor val AUC-PR | 0,121 |
| Epochs hasta early stopping | 40 / 150 |

### 3.3 Análisis comparativo

El RF supera al LSTM en 0,158 puntos de AUC pese a entrenarse con menos días (5.472 vs 24.838). La razón principal es el feature engineering: el RF recibe medias móviles de 7 días, lags de 3 días e índices compuestos específicos del dominio que codifican contexto temporal explícito. El LSTM debe derivar representaciones equivalentes a partir de secuencias de 14 días en bruto. Con ~24k observaciones diarias y 127k parámetros, el LSTM no está limitado por capacidad sino por información: los patrones temporales necesarios para una predicción precisa se capturan mejor con features engineered que aprendiéndolos de secuencias de esta longitud y este tamaño de dataset.

Esta observación tiene implicaciones prácticas: para predicción climática tabular a resolución diaria con datasets menores de 50k observaciones, el feature engineering explícito combinado con modelos de árboles generalmente superará a los modelos de secuencias. El LSTM se vuelve competitivo cuando hay datos de alta frecuencia disponibles (horarios o sub-horarios) donde el feature engineering explícito se vuelve intratable.

---

## 4. Configuración y ejecución

### Instalación local

```bash
git clone https://github.com/norapfr/atmospherica
cd atmospherica
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Crear `.env`:
```
OPENWEATHER_API_KEY=tu_api_key_aquí
```

### Ejecutar

```bash
python main.py             # genera pintura, abre navegador
python main.py --headless  # genera PNG sin navegador
```

### Entrenar modelos

```bash
# Random Forest
python ml/features.py && python ml/trainer.py

# LSTM
python ml/lstm_trainer.py --mode train --build_features
```

### Despliegue automático en tu fork

1. `Settings → Actions → General → Workflow permissions → Read and write permissions`
2. `Settings → Secrets → Actions → OPENWEATHER_API_KEY`
3. `Settings → Pages → Branch: main → Folder: /docs`

---

## 5. Stack

`Python 3.11` · `scikit-learn` · `PyTorch` · `shap` · `xarray` · `cdsapi` · `netCDF4` · `pandas` · `numpy` · `Playwright` · `GitHub Actions` · `GitHub Pages`

---

*Sevilla, 2026*
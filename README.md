# ATMOSPHERICA

Real-time atmospheric data translated into deterministic generative painting, with a machine-learning layer that predicts tomorrow's extreme-weather risk.

[Live exhibition](https://norapfr.github.io/atmospherica) | [Project notes](https://norapfr.github.io/atmospherica/about.html) | [Technical page](https://norapfr.github.io/atmospherica/technical.html)

## Why This Project Exists

ATMOSPHERICA is a portfolio project built to show an end-to-end ML engineering workflow outside the usual notebook format. It ingests live weather and air-quality data for Seville, converts the measurements into a visual grammar, generates a reproducible abstract painting, predicts the probability of an extreme event tomorrow, exports the result as a PNG, and publishes it automatically through GitHub Pages.

The painting is not decorative output. Every shape, colour, density, blur, direction, and alarm mark is traceable to a real atmospheric variable.

## What It Does

- Fetches real-time weather from OpenWeatherMap and air-quality data from the Air Pollution API.
- Aggregates daily production history into CSV files committed to the repository.
- Normalizes local atmospheric variables using Seville-calibrated ranges.
- Generates a deterministic HTML5 Canvas painting from the current climate state.
- Computes a Random Forest probability for tomorrow's extreme-weather risk.
- Encodes risk visually through progressive warning layers in the artwork.
- Exports PNGs with Playwright.
- Updates a public GitHub Pages gallery automatically.

## Visual Grammar

The system maps climate variables to formal visual rules:

| Variable | Visual Encoding |
|---|---|
| Temperature | Concentric circles. Colour is fixed by temperature range. |
| Wind | Bezier curves oriented by the real wind direction. |
| Humidity | Triangles when dry, diffuse ovals when humid. |
| Pressure | Rectangles and horizontal bands. High pressure becomes rigid and architectural. |
| Cloud cover | Flattened diamonds concentrated toward the upper canvas. |
| PM2.5 | Violet dots and haze applied as a final contamination layer. |
| Rain | Blue-grey diagonal strokes tilted by the wind. |

The dominant variable is the strongest normalized signal at generation time. It takes over the composition and changes how the other variables behave. When pressure dominates, the second strongest variable becomes co-dominant, so the painting carries both pressure structure and a secondary climate force.

## Tomorrow's Risk In The Painting

The Random Forest outputs a probability for an extreme event on the following day. If `risk_score >= 0.06`, the painting begins to show future-event signals:

| Risk Threshold | Visual Signal |
|---|---|
| > 6% | Edge triangles enter from the margins. |
| > 20% | A transparent background tint appears. |
| > 25% | Internal fractures use today's dominant morphology. |
| > 75% | A double alert frame encloses the canvas. |
| > 85% | Radial focal points contaminate the background light. |

Event colours are reserved for prediction signals:

| Event Type | Colour |
|---|---|
| Heat | Red-orange |
| Cold | Steel blue |
| Rain | Slate blue |
| Wind | Green-teal |

The model's primary output is probability. The event type is a secondary heuristic label. When risk is active but no specific event type crosses its physical threshold, the interface labels it as a generic extreme-event signal instead of over-claiming certainty.

## Machine Learning

### Dataset

Training data comes from ERA5 reanalysis via Copernicus CDS, using a grid centred on Seville:

- 1940-1990: long-run climatological baseline.
- 2010-2024: contemporary climate regime.
- Final RF dataset: 5,472 samples after feature construction and filtering.
- Event rate: 5.0%, making this a strongly imbalanced binary classification problem.

Target definition: will there be an extreme event tomorrow?

Extreme-event triggers:

| Type | Criterion |
|---|---|
| Heat | Daily max temperature >= 38 C |
| Cold | Daily max temperature <= 10 C |
| Wind | Max wind speed >= 8 m/s |
| Rain | Precipitation >= 1 mm/day |

### Random Forest Production Model

The production model is a `RandomForestClassifier` trained with time-series-aware validation.

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
```

Results:

| Metric | Value |
|---|---:|
| Mean ROC-AUC | 0.836 |
| Mean F1 | 0.149 |
| Features | 39 |
| Event rate | 5.0% |

The low F1 is expected under 5% event prevalence. ROC-AUC is the operational metric because the system needs to rank tomorrow-risk days above normal days, then translate probability into graduated visual thresholds.

Top RF features by SHAP:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | `pressure_hpa_min` | 0.0324 |
| 2 | `temp_c_max` | 0.0285 |
| 3 | `cos_doy` | 0.0226 |
| 4 | `pressure_norm` | 0.0201 |
| 5 | `temp_c_mean` | 0.0197 |
| 6 | `cloud_cover_mean` | 0.0185 |
| 7 | `pressure_hpa_mean` | 0.0181 |
| 8 | `humidity_mean` | 0.0176 |
| 9 | `wind_speed_mean` | 0.0155 |
| 10 | `dry_index` | 0.0153 |

### LSTM Baseline

An LSTM baseline was trained on 14-day sequences of 17 physical and derived features. It is kept as a documented experiment, not the production model.

| Metric | Value |
|---|---:|
| Test ROC-AUC | 0.678 |
| Test F1 | 0.135 |
| Test AUC-PR | 0.091 |
| Best validation AUC-PR | 0.121 |
| Epochs run | 40 / 150 |

The Random Forest wins because its explicit rolling means, lags, pressure gradients, seasonality, and dry-index features are stronger for this tabular, imbalanced forecasting problem than the sequence model's reduced feature set.

## Architecture

```text
OpenWeatherMap + Air Pollution API
        |
        v
data/fetcher.py
        |
        v
ml/history.py             ml/final_model/rf_model.pkl
daily CSV state    --->   ml/predictor.py
        |                         |
        v                         v
visual/mapper.py          risk_score + event_type
        |                         |
        v                         |
visual/generator.py <-----+
        |
        v
HTML Canvas artwork
        |
        v
Playwright PNG export
        |
        v
docs/output + docs/data/archive.json
        |
        v
GitHub Pages gallery
```

## Repository Structure

```text
atmospherica/
├── data/
│   ├── fetcher.py
│   ├── history_daily.csv
│   └── history_raw.csv
├── ml/
│   ├── features.py
│   ├── trainer.py
│   ├── predictor.py
│   ├── history.py
│   └── final_model/
│       ├── rf_model.pkl
│       ├── features.pkl
│       ├── metrics.json
│       └── lstm_metrics.json
├── visual/
│   ├── mapper.py
│   └── generator.py
├── docs/
│   ├── index.html
│   ├── about.html
│   ├── technical.html
│   ├── data/archive.json
│   └── output/
├── archive.py
├── main.py
└── requirements.txt
```

## Automation

The GitHub Actions workflow runs twice daily at 08:00 and 20:00 Madrid time, plus manual `workflow_dispatch`.

Each run:

1. Installs Python dependencies and Playwright Chromium.
2. Executes `python main.py --headless`.
3. Generates the HTML painting.
4. Exports the canvas as PNG.
5. Updates `docs/data/archive.json`.
6. Commits the new image, history CSVs, and gallery data.
7. Publishes through GitHub Pages.

## Run Locally

Create a `.env` file with:

```text
OPENWEATHER_API_KEY=your_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
playwright install chromium
```

Generate a painting:

```bash
python main.py
```

Run without opening a browser:

```bash
python main.py --headless
```

Serve the gallery locally:

```bash
python -m http.server 8000 --directory docs
```

Then open:

```text
http://localhost:8000
```

## Engineering Highlights

- Time-series validation instead of shuffled splits.
- Explicit feature persistence to guarantee model-column order in production.
- Graceful inference fallback while local daily history is still short.
- Deterministic seeded rendering for reproducible generative output.
- Browser-based PNG export through Playwright.
- GitHub Actions as a lightweight production scheduler.
- Public artifact, not just offline experimentation.

## Stack

Python 3.11, pandas, NumPy, scikit-learn, PyTorch, SHAP, xarray, netCDF4, cdsapi, joblib, HTML5 Canvas, JavaScript, Playwright, GitHub Actions, GitHub Pages.

## Author

Built by Nora P. as a portfolio project focused on ML engineering, data pipelines, generative systems, and production-minded automation.

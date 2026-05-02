import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
class AtmosphericPredictor:
    """
    Carga el modelo RF entrenado con trainer.py y predice el riesgo de evento
    extremo para el día siguiente a partir de los datos actuales de la API.

    Compatibilidad garantizada con features.py (columnas reales del dataset).
    """
    

    def __init__(self, model_dir: str = None):
        self.model_dir = BASE_DIR / "final_model"
        self.model        = None
        self.feature_cols = None   # lista guardada por trainer.py
        self._load()

    # ─────────────────────────────────────────────
    def _load(self):
        rf_path   = self.model_dir / "rf_model.pkl"
        feat_path = self.model_dir / "features.pkl"   # trainer.py guarda "features.pkl"

        if not rf_path.exists():
            print("  Modelo no entrenado todavía — predictor desactivado")
            return

        if not feat_path.exists():
            print("  features.pkl no encontrado — predictor desactivado")
            return

        self.model        = joblib.load(rf_path)
        self.feature_cols = joblib.load(feat_path)
        print(f"  Modelo cargado: {len(self.feature_cols)} features")

    # ─────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self.model is not None and self.feature_cols is not None

    # ─────────────────────────────────────────────
    def predict(self, current_data: dict) -> dict:
        """
        Recibe el dict de datos actuales de la API (fetcher.py) y devuelve
        una puntuación de riesgo 0-1 y el tipo de evento probable.

        Parámetros esperados en current_data:
            temperature  float  °C
            pressure     float  hPa
            wind_speed   float  m/s
            humidity     float  %
            clouds       float  %
            pm2_5        float  μg/m³  (opcional)
        """
        if not self.is_ready():
            return {"risk_score": 0.0, "event_type": "unknown", "ready": False}

        # ── Datos de entrada ──────────────────────
        temp   = current_data.get("temperature", 20.0)
        pres   = current_data.get("pressure",    1013.0)
        wind   = current_data.get("wind_speed",  2.0)
        hum    = current_data.get("humidity",    50.0)
        clouds = current_data.get("clouds",      30.0)

        # Estimaciones razonables de max/min/mean a partir del dato puntual
        temp_max  = temp + 3.0
        temp_min  = temp - 5.0
        temp_mean = temp

        wind_max  = wind * 1.5
        wind_mean = wind

        hum_max  = min(100.0, hum + 10.0)
        hum_mean = hum

        pres_mean = pres
        pres_min  = pres - 2.0

        precip = 5.0 if clouds > 70 else 0.0
        cloud_frac = clouds / 100.0    # features.py guarda tcc sin convertir (0-1 ERA5)

        # ── Estacionalidad ────────────────────────
        doy       = datetime.now().timetuple().tm_yday
        sin_doy   = np.sin(2 * np.pi * doy / 365)
        cos_doy   = np.cos(2 * np.pi * doy / 365)

        # ── Rolling features (aproximadas con datos actuales) ─
        # En producción real se usarían los últimos N días del archivo histórico.
        # Aquí se usa el valor puntual como proxy neutral.
        ma3_temp     = temp_max
        ma7_temp     = temp_max
        ma3_pres     = pres_mean
        ma7_pres     = pres_mean
        ma3_wind     = wind_max
        ma7_wind     = wind_max
        ma3_hum      = hum_mean
        ma7_hum      = hum_mean

        # ── Lags (sin historia → 0 como proxy neutro) ─────────
        lag1_temp  = temp_max
        lag2_temp  = temp_max
        lag3_temp  = temp_max
        lag1_prec  = 0.0
        lag2_prec  = 0.0
        lag3_prec  = 0.0
        lag1_wind  = wind_max
        lag2_wind  = wind_max
        lag3_wind  = wind_max

        # ── Gradientes (sin historia → 0) ────────────────────
        temp_grad     = 0.0
        pressure_grad = 0.0

        # ── Features adicionales (mirrors exactos de features.py) ─
        temp_range      = temp_max - temp_min
        heat_intensity  = temp_max - temp_mean

        # pressure_hpa_mean_ma7 se calcula igual que ma7_pres
        pressure_deficit = pres_mean - ma7_pres   # = 0 sin historia real

        humidity_range   = hum_max - hum_mean
        wind_spike       = wind_max - wind_mean
        dry_index        = temp_range * (100 - hum_mean)

        # pressure_norm necesita media y std del dataset histórico.
        # Sin ellas usamos el rango típico de Sevilla: ~1013 hPa, std ~5 hPa
        PRES_HIST_MEAN = 1013.0
        PRES_HIST_STD  = 5.0
        pressure_norm  = (pres_mean - PRES_HIST_MEAN) / PRES_HIST_STD

        # ── Mapa nombre_columna → valor ───────────────────────
        # Los nombres deben coincidir EXACTAMENTE con los de features.py
        feature_map = {
            # básicas diarias
            "temp_c_max":          temp_max,
            "temp_c_min":          temp_min,
            "temp_c_mean":         temp_mean,
            "pressure_hpa_mean":   pres_mean,
            "pressure_hpa_min":    pres_min,
            "wind_speed_max":      wind_max,
            "wind_speed_mean":     wind_mean,
            "humidity_max":        hum_max,
            "humidity_mean":       hum_mean,
            "precip_mm_sum":       precip,
            "cloud_cover_mean":    cloud_frac,

            # rolling ma3
            "temp_c_max_ma3":          ma3_temp,
            "pressure_hpa_mean_ma3":   ma3_pres,
            "wind_speed_max_ma3":      ma3_wind,
            "humidity_mean_ma3":       ma3_hum,

            # rolling ma7
            "temp_c_max_ma7":          ma7_temp,
            "pressure_hpa_mean_ma7":   ma7_pres,
            "wind_speed_max_ma7":      ma7_wind,
            "humidity_mean_ma7":       ma7_hum,

            # lags
            "temp_c_max_lag1":         lag1_temp,
            "temp_c_max_lag2":         lag2_temp,
            "temp_c_max_lag3":         lag3_temp,
            "precip_mm_sum_lag1":      lag1_prec,
            "precip_mm_sum_lag2":      lag2_prec,
            "precip_mm_sum_lag3":      lag3_prec,
            "wind_speed_max_lag1":     lag1_wind,
            "wind_speed_max_lag2":     lag2_wind,
            "wind_speed_max_lag3":     lag3_wind,

            # gradientes
            "temp_grad":               temp_grad,
            "pressure_grad":           pressure_grad,

            # estacionalidad
            "sin_doy":                 sin_doy,
            "cos_doy":                 cos_doy,

            # features adicionales
            "temp_range":              temp_range,
            "heat_intensity":          heat_intensity,
            "pressure_deficit":        pressure_deficit,
            "humidity_range":          humidity_range,
            "wind_spike":              wind_spike,
            "dry_index":               dry_index,
            "pressure_norm":           pressure_norm,
        }

        # ── Construir vector respetando el orden exacto del modelo ─
        x = np.array(
            [[feature_map.get(col, 0.0) for col in self.feature_cols]],
            dtype=np.float32
        )

        # RF no necesita escalado — se usa directamente
        risk_score = float(self.model.predict_proba(x)[0][1])

        # ── Tipo de evento probable ───────────────────────────
        if temp_max >= 38:
            event_type = "heat"
        elif temp_min <= 8:
            event_type = "cold"
        elif wind_max >= 10:
            event_type = "wind"
        elif clouds > 70 and hum > 80:
            event_type = "rain"
        else:
            event_type = "none"

        return {
            "risk_score": round(risk_score, 3),
            "event_type": event_type,
            "ready":      True,
        }

    # ─────────────────────────────────────────────
    def predict_from_history_df(self, df_row: "pd.DataFrame") -> dict:
        """
        Versión que acepta directamente el DataFrame de una fila
        devuelto por history.build_features_from_history().
        """
        if not self.is_ready():
            return {"risk_score": 0.0, "event_type": "unknown", "ready": False}

        x = df_row.reindex(columns=self.feature_cols, fill_value=0.0).values.astype(np.float32)
        risk_score = float(self.model.predict_proba(x)[0][1])

        temp_max = float(df_row["temp_c_max"].iloc[-1])
        temp_min = float(df_row["temp_c_min"].iloc[-1])
        wind_max = float(df_row["wind_speed_max"].iloc[-1])
        hum_mean = float(df_row["humidity_mean"].iloc[-1])
        cloud    = float(df_row["cloud_cover_mean"].iloc[-1]) * 100

        if temp_max >= 38:
            event_type = "heat"
        elif temp_min <= 8:
            event_type = "cold"
        elif wind_max >= 8:
            event_type = "wind"
        elif cloud > 70 and hum_mean > 80:
            event_type = "rain"
        else:
            event_type = "none"

        return {"risk_score": round(risk_score, 3), "event_type": event_type, "ready": True}

    # ─────────────────────────────────────────────
    def predict_from_history(self, history_csv: str) -> dict:
        """
        Versión de producción: usa los últimos días reales del CSV de features
        para construir el vector con rolling/lags correctos.

        history_csv: ruta al featuresAll.csv generado por features.py
        """
        if not self.is_ready():
            return {"risk_score": 0.0, "event_type": "unknown", "ready": False}

        df = pd.read_csv(history_csv, index_col=0, parse_dates=True).sort_index()

        # El último día disponible es el "hoy" desde el que predecimos mañana
        exclude = {"target", "event_heat", "event_cold",
                   "event_rain", "event_wind", "event_extreme"}
        feat_cols = [c for c in df.columns if c not in exclude]

        last_row = df[feat_cols].iloc[[-1]]

        # Alinear columnas con las del modelo (rellena con 0 si falta alguna)
        x = last_row.reindex(columns=self.feature_cols, fill_value=0.0).values.astype(np.float32)

        risk_score = float(self.model.predict_proba(x)[0][1])

        temp_max = float(df["temp_c_max"].iloc[-1])
        temp_min = float(df["temp_c_min"].iloc[-1])
        wind_max = float(df["wind_speed_max"].iloc[-1])
        hum_mean = float(df["humidity_mean"].iloc[-1])
        cloud    = float(df["cloud_cover_mean"].iloc[-1]) * 100

        if temp_max >= 38:
            event_type = "heat"
        elif temp_min <= 8:
            event_type = "cold"
        elif wind_max >= 8:
            event_type = "wind"
        elif cloud > 70 and hum_mean > 80:
            event_type = "rain"
        else:
            event_type = "none"

        return {
            "risk_score": round(risk_score, 3),
            "event_type": event_type,
            "ready":      True,
        }


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = AtmosphericPredictor()

    if predictor.is_ready():
        # Test rápido con datos de la API
        test_api = {
            "temperature": 23.4,
            "pressure":    1014.0,
            "wind_speed":  1.3,
            "humidity":    42.0,
            "clouds":      15.0,
        }
        result = predictor.predict(test_api)
        print(f"\nPredicción (datos API):")
        print(f"  Riesgo evento mañana : {result['risk_score']:.1%}")
        print(f"  Tipo probable        : {result['event_type']}")

        # Test con el CSV histórico real (más preciso)
        csv_path = "ml/data_todo/featuresAll.csv"
        if Path(csv_path).exists():
            result2 = predictor.predict_from_history(csv_path)
            print(f"\nPredicción (historia real):")
            print(f"  Riesgo evento mañana : {result2['risk_score']:.1%}")
            print(f"  Tipo probable        : {result2['event_type']}")
    else:
        print("Modelo no disponible. Ejecuta trainer.py primero.")
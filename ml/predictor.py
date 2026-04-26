import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class AtmosphericPredictor:
    """
    Carga el modelo entrenado y predice el riesgo de evento extremo
    para el dia siguiente a partir de los datos actuales.
    """

    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = Path(model_dir)
        self.model        = None
        self.scaler       = None
        self.feature_cols = None
        self._load()

    def _load(self):
        rf_path = self.model_dir / "rf_model.pkl"
        if not rf_path.exists():
            print("  Modelo no entrenado todavia — predictor desactivado")
            return
        self.model        = joblib.load(rf_path)
        self.scaler       = joblib.load(self.model_dir / "rf_scaler.pkl")
        self.feature_cols = joblib.load(self.model_dir / "feature_cols.pkl")
        print(f"  Modelo cargado: {len(self.feature_cols)} features")

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, current_data: dict) -> dict:
        """
        Recibe el dict de datos actuales de la API y devuelve
        una puntuacion de riesgo 0-1 y el tipo de evento probable.

        Si el modelo no esta entrenado devuelve riesgo neutro.
        """
        if not self.is_ready():
            return {"risk_score": 0.0, "event_type": "unknown", "ready": False}

        # Construir un vector de features desde los datos actuales
        # Usamos los datos del dia actual como proxy de la ventana historica
        # (en produccion esto usaria los ultimos 14 dias del archivo)
        temp   = current_data.get("temperature", 20)
        pres   = current_data.get("pressure", 1013)
        wind   = current_data.get("wind_speed", 2)
        hum    = current_data.get("humidity", 50)
        clouds = current_data.get("clouds", 30)

        # Dia del año para estacionalidad
        from datetime import datetime
        day_of_year = datetime.now().timetuple().tm_yday
        season_sin  = np.sin(2 * np.pi * day_of_year / 365)
        season_cos  = np.cos(2 * np.pi * day_of_year / 365)

        # Vector de features — valores aproximados desde datos actuales
        feature_values = {
            "temp_max":          temp + 3,        # estimar max desde actual
            "temp_min":          temp - 5,
            "temp_mean":         temp,
            "pressure_mean":     pres,
            "pressure_min":      pres - 2,
            "wind_max":          wind * 1.5,
            "wind_mean":         wind,
            "humidity_max":      min(100, hum + 10),
            "humidity_mean":     hum,
            "precip_total":      5 if clouds > 70 else 0,
            "cloud_mean":        clouds / 100,
            "temp_max_ma3":      temp + 2,
            "temp_mean_ma3":     temp,
            "pressure_mean_ma3": pres,
            "wind_max_ma3":      wind,
            "humidity_mean_ma3": hum,
            "temp_max_ma7":      temp + 1,
            "temp_mean_ma7":     temp,
            "pressure_mean_ma7": pres,
            "wind_max_ma7":      wind,
            "humidity_mean_ma7": hum,
            "temp_max_grad1":    0.5,
            "pressure_mean_grad1": -0.3,
            "humidity_mean_grad1": 1.0,
            "temp_max_grad3":    1.2,
            "pressure_mean_grad3": -0.8,
            "humidity_mean_grad3": 2.5,
            "temp_trend7":       0.1,
            "season_sin":        season_sin,
            "season_cos":        season_cos,
        }

        # Construir vector respetando el orden de features del modelo
        x = np.array([[feature_values.get(col, 0) for col in self.feature_cols]],
                     dtype=np.float32)
        x_scaled     = self.scaler.transform(x)
        risk_score   = float(self.model.predict_proba(x_scaled)[0][1])

        # Determinar tipo de evento probable
        if temp + 3 >= 38:
            event_type = "heat"
        elif wind * 1.5 >= 10:
            event_type = "wind"
        elif clouds > 70 and hum > 80:
            event_type = "rain"
        elif temp - 5 <= 8:
            event_type = "cold"
        else:
            event_type = "none"

        return {
            "risk_score": round(risk_score, 3),
            "event_type": event_type,
            "ready":      True,
        }


if __name__ == "__main__":
    # Test rapido
    predictor = AtmosphericPredictor()
    if predictor.is_ready():
        test = {"temperature": 23.4, "pressure": 1014,
                "wind_speed": 1.3, "humidity": 42, "clouds": 15}
        result = predictor.predict(test)
        print(f"\nPrediccion para hoy en Sevilla:")
        print(f"  Riesgo evento manana: {result['risk_score']:.1%}")
        print(f"  Tipo probable: {result['event_type']}")
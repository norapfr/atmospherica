import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import webbrowser
from data.fetcher import get_all_data
from visual.mapper import map_to_visual
from visual.generator import generate_html
from ml.predictor import AtmosphericPredictor

if __name__ == "__main__":
    print("ATMOSPHERICA\n")

    # Datos en tiempo real
    print("Obteniendo datos atmosfericos...")
    data = get_all_data()
    print(f"  {data['city']} — {data['temperature']:.1f}C | "
          f"{data['pressure']} hPa | {data['wind_speed']:.1f} m/s | "
          f"PM2.5 {data['pm2_5']:.1f}")

    # Mapeo visual
    print("\nMapeando parametros visuales...")
    visual = map_to_visual(data)

    # Prediccion ML
    print("\nPrediccion ML...")
    predictor = AtmosphericPredictor()
    prediction = predictor.predict(data)
    print(f"  Riesgo evento manana: {prediction['risk_score']:.1%}")
    print(f"  Tipo probable: {prediction['event_type']}")

    # Añadir prediccion a los parametros visuales
    visual["risk_score"]  = prediction["risk_score"]
    visual["event_type"]  = prediction["event_type"]
    visual["ml_ready"]    = prediction["ready"]

    # Generar cuadro
    print("\nGenerando cuadro...")
    path = generate_html(visual)

    print("\nAbriendo navegador...")
    webbrowser.open(f"file://{os.path.abspath(path)}")
    print("Espera a que termine de pintar y pulsa GUARDAR PNG.")
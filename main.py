import sys
import os
import argparse
sys.stdout.reconfigure(encoding='utf-8')

import webbrowser
from datetime import datetime
from data.fetcher import get_all_data
from ml.history import append_today, build_features_from_history, history_status
from visual.mapper import map_to_visual
from visual.generator import generate_html, _compute_dominant
from ml.predictor import AtmosphericPredictor
from archive import update_archive

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true",
                    help="No abrir navegador (GitHub Actions)")
args = parser.parse_args()

if __name__ == "__main__":
    print("ATMOSPHERICA\n")

    # ── Datos en tiempo real ──────────────────────────────────
    print("Obteniendo datos atmosfericos...")
    data = get_all_data()
    print(f"  {data['city']} — {data['temperature']:.1f}C | "
          f"{data['pressure']} hPa | {data['wind_speed']:.1f} m/s | "
          f"PM2.5 {data['pm2_5']:.1f}")

    # ── Acumular historia ─────────────────────────────────────
    print("\nActualizando historia...")
    append_today(data)
    print(f"  {history_status()}")

    # ── Mapeo visual ──────────────────────────────────────────
    print("\nMapeando parametros visuales...")
    visual = map_to_visual(data)

    # ── Dominancia (calculada aquí para tenerla en visual_params) ─
    html_path, dominant, dom_strength, dominant2 = generate_html(visual)
    visual["dominant"]         = dominant
    visual["dominant_strength"] = dom_strength
    visual["dominant2"]        = dominant2

    # ── Prediccion ML ─────────────────────────────────────────
    print("\nPrediccion ML...")
    predictor = AtmosphericPredictor()

    if predictor.is_ready():
        features_today = build_features_from_history()
        if features_today is not None:
            prediction = predictor.predict_from_history_df(features_today)
        else:
            prediction = predictor.predict(data)
        print(f"  Riesgo evento manana: {prediction['risk_score']:.1%}")
        print(f"  Tipo probable: {prediction['event_type']}")
    else:
        prediction = {"risk_score": 0.0, "event_type": "unknown", "ready": False}
        print("  Modelo no entrenado — prediccion desactivada")

    visual["risk_score"] = prediction["risk_score"]
    visual["event_type"] = prediction["event_type"]
    visual["ml_ready"]   = prediction["ready"]

    # ── Generar HTML ──────────────────────────────────────────
    print("\nGenerando cuadro...")
    html_path = generate_html(visual)[0]

    # ── Exportar PNG con Playwright (siempre) ─────────────────
    now      = datetime.now()
    city     = data["city"].replace(" ", "_")
    png_name = f"atmospherica_{city}_{now.strftime('%Y-%m-%d')}_{now.hour:02d}h.png"
    png_path = os.path.join("output", png_name)
    png_web  = f"../output/{png_name}"

    os.makedirs("output", exist_ok=True)

    print("\nExportando PNG...")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page    = browser.new_page(viewport={"width": 900, "height": 1080})
        page.goto(f"file://{os.path.abspath(html_path)}")
        page.wait_for_timeout(1500)   # esperar render del canvas
        page.locator("canvas#c").screenshot(path=png_path)
        browser.close()
    print(f"  PNG guardado: {png_path}")

    # ── Abrir navegador en modo local ─────────────────────────
    if not args.headless:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
        print("  Navegador abierto (el PNG ya está guardado en output/)")

    # ── Actualizar archive.json ───────────────────────────────
    update_archive(visual, image_path=png_web)
    print("\nListo.")
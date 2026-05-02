import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from datetime import datetime


ARCHIVE_PATH = Path("web/data/archive.json")


def update_archive(visual_params: dict, image_path: str):
    """
    Añade la entrada del cuadro de hoy a web/data/archive.json.
    Si ya existe una entrada para esta fecha+hora la sobreescribe.

    image_path: ruta relativa al PNG desde la raíz del repo,
                p.ej. "output/atmospherica_Sevilla_2026-04-28_12h.png"
    """
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Cargar archivo existente
    if ARCHIVE_PATH.exists():
        with open(ARCHIVE_PATH, encoding="utf-8") as f:
            archive = json.load(f)
    else:
        archive = []

    data = visual_params["raw"]
    now  = datetime.now()

    entry = {
        # Metadatos
        "date":       now.strftime("%Y-%m-%d"),
        "hour":       now.hour,
        "city":       data.get("city", "Seville"),

        # Ruta de imagen relativa a la raíz web
        # GitHub Pages sirve desde /web/, así que la ruta
        # que ve el navegador es relativa a index.html
        "image_path": image_path,

        # Datos climáticos crudos (los que muestra la leyenda)
        "temp_c":      round(data["temperature"], 1),
        "pressure":    data["pressure"],
        "wind_speed":  round(data["wind_speed"], 1),
        "wind_dir":    _wind_label(data["wind_deg"]),
        "humidity":    data["humidity"],
        "clouds":      data.get("clouds", 0),
        "pm25":        round(data.get("pm2_5", 0), 1),

        # Valores normalizados (para la leyenda interactiva de index.html)
        "temp_norm":     round(visual_params["temperature_norm"], 3),
        "pressure_norm": round(visual_params["density"], 3),
        "wind_energy":   round(visual_params["wind_energy"], 3),
        "humidity_norm": round(visual_params["veil_opacity"] / 80.0, 3),
        "cloud_norm":    round(data.get("clouds", 0) / 100.0, 3),
        "pm_norm":       round(visual_params["fragmentation"], 3),

        # Dominancia
        "dominant":           visual_params.get("dominant", "temperatura"),
        "dominant_strength":  round(visual_params.get("dominant_strength", 0), 3),
        "dominant2":          visual_params.get("dominant2"),

        # ML
        "ml_ready":    visual_params.get("ml_ready", False),
        "risk_score":  round(visual_params.get("risk_score", 0.0), 3),
        "event_type":  visual_params.get("event_type", "none"),
    }

    # Eliminar entrada duplicada (misma fecha+hora) si existe
    key = entry["date"] + f"_{entry['hour']:02d}"
    archive = [e for e in archive if
               not (e.get("date") == entry["date"] and e.get("hour") == entry["hour"])]

    archive.append(entry)

    # Ordenar: más reciente primero
    archive.sort(key=lambda e: e["date"] + f"_{e.get('hour', 0):02d}", reverse=True)

    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    print(f"  Archive actualizado: {len(archive)} entradas → {ARCHIVE_PATH}")


def _wind_label(deg):
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSO','SO','OSO','O','ONO','NO','NNO']
    return dirs[int((deg + 11.25) / 22.5) % 16]

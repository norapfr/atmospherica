"""
ATMOSPHERICA - Test pipeline REAL
Los casos pasan por el flujo completo:
DATA → MAP → ML → GENERATE_HTML

Uso:
    python test_cases.py [indice]
    python test_cases.py all
    python test_cases.py
"""

import sys, os, webbrowser

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── PATH ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(BASE_DIR))

# ── IMPORTS REALES (mismo flujo que main) ───────────────────────────────
from mapper import map_to_visual
from generator import generate_html
from ml.predictor import AtmosphericPredictor

OUTPUT_DIR = "output_tests"

# ========================================================================
# CASOS (SOLO SE USA "raw")
# ========================================================================
CASES = [

    # ── 0. TEMPERATURA domina - calor extremo verano ──────────────────
    {
        "_label": "TEMPERATURA · Calor extremo (45 degC)",
        "raw": {
            "city": "Sevilla", "temperature": 45.0, "pressure": 1005,
            "wind_speed": 1.2, "wind_deg": 180, "pm2_5": 18.0,
            "humidity": 12, "clouds": 0,
        },
        "temperature_norm": 1.0, "density": 0.30, "wind_energy": 0.06,
        "wind_dx": 0.0, "wind_dy": 1.0,
        "fragmentation": 0.22, "veil_opacity": 9.6,
        "risk_score": 0.91, "event_type": "heat", "ml_ready": True,
    },

    # ── 1. TEMPERATURA domina - frio polar ───────────────────────────
    {
        "_label": "TEMPERATURA · Frio polar (-8 degC)",
        "raw": {
            "city": "Burgos", "temperature": -8.0, "pressure": 1028,
            "wind_speed": 3.5, "wind_deg": 0, "pm2_5": 6.0,
            "humidity": 80, "clouds": 10,
        },
        "temperature_norm": 0.0, "density": 0.78, "wind_energy": 0.18,
        "wind_dx": 0.0, "wind_dy": -1.0,
        "fragmentation": 0.08, "veil_opacity": 64.0,
        "risk_score": 0.72, "event_type": "cold", "ml_ready": True,
    },

    # ── 2. VIENTO domina - borrasca fuerte ───────────────────────────
    {
        "_label": "VIENTO · Borrasca (28 m/s, ONO)",
        "raw": {
            "city": "A Coruna", "temperature": 14.0, "pressure": 988,
            "wind_speed": 28.0, "wind_deg": 292, "pm2_5": 4.0,
            "humidity": 88, "clouds": 85,
        },
        "temperature_norm": 0.35, "density": 0.18, "wind_energy": 0.98,
        "wind_dx": -0.92, "wind_dy": -0.38,
        "fragmentation": 0.05, "veil_opacity": 70.4,
        "risk_score": 0.88, "event_type": "wind", "ml_ready": True,
    },

    # ── 3. VIENTO domina - brisa suave (viento dominante por ausencia) ─
    {
        "_label": "VIENTO · Calma chicha (1.5 m/s)",
        "raw": {
            "city": "Valencia", "temperature": 21.0, "pressure": 1016,
            "wind_speed": 1.5, "wind_deg": 90, "pm2_5": 9.0,
            "humidity": 55, "clouds": 20,
        },
        "temperature_norm": 0.50, "density": 0.53, "wind_energy": 0.08,
        "wind_dx": 1.0, "wind_dy": 0.0,
        "fragmentation": 0.12, "veil_opacity": 44.0,
        "risk_score": 0.0, "event_type": "none", "ml_ready": True,
    },

    # ── 4. HUMEDAD domina - saturacion tropical ──────────────────────
    {
        "_label": "HUMEDAD · Saturacion tropical (98 %)",
        "raw": {
            "city": "Vigo", "temperature": 23.0, "pressure": 1010,
            "wind_speed": 4.0, "wind_deg": 225, "pm2_5": 7.0,
            "humidity": 98, "clouds": 70,
        },
        "temperature_norm": 0.52, "density": 0.40, "wind_energy": 0.20,
        "wind_dx": -0.71, "wind_dy": 0.71,
        "fragmentation": 0.09, "veil_opacity": 78.4,   # 98 * 0.8 = 78.4
        "risk_score": 0.55, "event_type": "rain", "ml_ready": True,
    },

    # ── 5. HUMEDAD domina - desierto (baja humedad, todos los demas aun menores) ─
    {
        "_label": "HUMEDAD · Desierto seco (8 %)",
        "raw": {
            "city": "Almeria", "temperature": 32.0, "pressure": 1008,
            "wind_speed": 2.0, "wind_deg": 45, "pm2_5": 28.0,
            "humidity": 8, "clouds": 0,
        },
        "temperature_norm": 0.72, "density": 0.37,  "wind_energy": 0.10,
        "wind_dx": 0.71, "wind_dy": -0.71,
        "fragmentation": 0.35, "veil_opacity": 6.4,   # 8 * 0.8 = 6.4  <- HUMEDAD domina
        "risk_score": 0.12, "event_type": "heat", "ml_ready": True,
    },

    # ── 6. PRESIÓN domina - anticiclon potente ───────────────────────
    {
        "_label": "PRESIÓN · Anticiclon (1035 hPa)",
        "raw": {
            "city": "Madrid", "temperature": 19.0, "pressure": 1035,
            "wind_speed": 2.0, "wind_deg": 315, "pm2_5": 11.0,
            "humidity": 30, "clouds": 5,
        },
        "temperature_norm": 0.46, "density": 0.95, "wind_energy": 0.10,
        "wind_dx": -0.71, "wind_dy": -0.71,
        "fragmentation": 0.14, "veil_opacity": 24.0,
        "risk_score": 0.05, "event_type": "none", "ml_ready": True,
    },

    # ── 7. PRESIÓN domina - borrasca profunda ────────────────────────
    {
        "_label": "PRESIÓN · Borrasca profunda (960 hPa)",
        "raw": {
            "city": "Bilbao", "temperature": 11.0, "pressure": 960,
            "wind_speed": 18.0, "wind_deg": 250, "pm2_5": 3.0,
            "humidity": 92, "clouds": 95,
        },
        "temperature_norm": 0.27, "density": 0.02, "wind_energy": 0.80,
        "wind_dx": -0.94, "wind_dy": 0.34,
        "fragmentation": 0.04, "veil_opacity": 73.6,
        "risk_score": 0.80, "event_type": "wind", "ml_ready": True,
    },

    # ── 8. NUBES dominan - cielo completamente cubierto ──────────────
    {
        "_label": "NUBES · Cielo 100 % cubierto",
        "raw": {
            "city": "Santander", "temperature": 16.0, "pressure": 1008,
            "wind_speed": 6.0, "wind_deg": 200, "pm2_5": 5.0,
            "humidity": 85, "clouds": 100,
        },
        "temperature_norm": 0.39, "density": 0.37, "wind_energy": 0.30,
        "wind_dx": -0.34, "wind_dy": 0.94,
        "fragmentation": 0.06, "veil_opacity": 68.0,
        "risk_score": 0.35, "event_type": "rain", "ml_ready": True,
    },

    # ── 9. PM2.5 domina - episodio de contaminacion grave ────────────
    {
        "_label": "PM2.5 · Contaminacion critica (180 ug/m3)",
        "raw": {
            "city": "Barcelona", "temperature": 22.0, "pressure": 1014,
            "wind_speed": 1.0, "wind_deg": 90, "pm2_5": 180.0,
            "humidity": 60, "clouds": 30,
        },
        "temperature_norm": 0.51, "density": 0.45, "wind_energy": 0.05,
        "wind_dx": 1.0, "wind_dy": 0.0,
        "fragmentation": 1.0,  "veil_opacity": 48.0,
        "risk_score": 0.65, "event_type": "none", "ml_ready": True,
    },

    # ── 10. ML inactivo - dia perfecto sin riesgo ─────────────────────
    {
        "_label": "ML inactivo · Dia perfecto de primavera",
        "raw": {
            "city": "Granada", "temperature": 22.0, "pressure": 1018,
            "wind_speed": 8.0, "wind_deg": 135, "pm2_5": 4.0,
            "humidity": 40, "clouds": 15,
        },
        "temperature_norm": 0.51, "density": 0.58, "wind_energy": 0.40,
        "wind_dx": 0.71, "wind_dy": 0.71,
        "fragmentation": 0.05, "veil_opacity": 32.0,
        "risk_score": 0.0, "event_type": "none", "ml_ready": False,
    },

    # ── 11. Todos en equilibrio - ninguno domina claramente ──────────
    {
        "_label": "EQUILIBRIO · Todas las variables al 50 %",
        "raw": {
            "city": "Toledo", "temperature": 20.0, "pressure": 1013,
            "wind_speed": 9.5, "wind_deg": 180, "pm2_5": 40.0,
            "humidity": 55, "clouds": 50,
        },
        "temperature_norm": 0.47, "density": 0.42, "wind_energy": 0.48,
        "wind_dx": 0.0, "wind_dy": 1.0,
        "fragmentation": 0.50, "veil_opacity": 44.0,
        "risk_score": 0.50, "event_type": "rain", "ml_ready": True,
    },

    # ── 12. Noche - hora 3, todas las variables bajas ─────────────────
    {
        "_label": "NOCHE · 3 h, ciudad dormida",
        "raw": {
            "city": "Sevilla", "temperature": 18.0, "pressure": 1017,
            "wind_speed": 3.0, "wind_deg": 270, "pm2_5": 6.0,
            "humidity": 65, "clouds": 20,
        },
        "temperature_norm": 0.43, "density": 0.57, "wind_energy": 0.15,
        "wind_dx": -1.0, "wind_dy": 0.0,
        "fragmentation": 0.08, "veil_opacity": 52.0,
        "risk_score": 0.08, "event_type": "none", "ml_ready": True,
    },
]

# ========================================================================
# PIPELINE REAL (igual que main)
# ========================================================================
def run_pipeline(data):
    print("\nMapeando parametros visuales...")
    visual = map_to_visual(data)

    print("\nPrediccion ML...")
    predictor = AtmosphericPredictor()
    prediction = predictor.predict(data)

    print(f"  Riesgo evento manana: {prediction['risk_score']:.1%}")
    print(f"  Tipo probable: {prediction['event_type']}")

    visual["risk_score"] = prediction["risk_score"]
    visual["event_type"] = prediction["event_type"]
    visual["ml_ready"] = prediction["ready"]

    print("\nGenerando cuadro...")
    path = generate_html(visual, output_dir=OUTPUT_DIR)

    return path

# ========================================================================
# RUN TEST
# ========================================================================
def run(idx: int):
    case = CASES[idx]
    data = case["raw"]

    print(f"\n{'='*60}")
    print(f"[{idx}] {case.get('_label','')}")
    print(f"{'='*60}")

    print(f"  {data['city']} — {data['temperature']:.1f}C | "
          f"{data['pressure']} hPa | {data['wind_speed']:.1f} m/s | "
          f"PM2.5 {data['pm2_5']:.1f}")

    path = run_pipeline(data)

    url = f"file://{os.path.abspath(path)}"
    webbrowser.open(url)

# ========================================================================
# MENU
# ========================================================================
def menu():
    print("\nATMOSPHERICA - Test pipeline REAL\n")

    for i, c in enumerate(CASES):
        print(f"[{i}] {c.get('_label','?')}")

    print("\n[all] Ejecutar todos")
    print("[q] Salir\n")

    sel = input("> ").strip().lower()

    if sel == "q":
        return
    elif sel == "all":
        for i in range(len(CASES)):
            run(i)
    elif sel.isdigit() and int(sel) < len(CASES):
        run(int(sel))
    else:
        print("Opcion no valida.")

# ========================================================================
# ENTRYPOINT
# ========================================================================
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        menu()
    elif args[0] == "all":
        for i in range(len(CASES)):
            run(i)
    elif args[0].isdigit() and int(args[0]) < len(CASES):
        run(int(args[0]))
    else:
        print(f"Uso: python test_cases.py [0-{len(CASES)-1} | all]")
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import zoneinfo
_TZ = zoneinfo.ZoneInfo("Europe/Madrid")

# ── Rutas ─────────────────────────────────────────────────────────────────
RAW_PATH   = Path("data/history_raw.csv")    # hasta 3 lecturas por día
DAILY_PATH = Path("data/history_daily.csv")  # un agregado por día

MIN_DAYS_ROLLING = 7
MIN_DAYS_FULL    = 14


# ── 1. Guardar lectura bruta ───────────────────────────────────────────────
def append_reading(data: dict):
    """
    Guarda la lectura actual de la API en history_raw.csv.
    Acumula hasta 3 lecturas por día (8h, 12h, 20h).
    Si ya existe la misma fecha+hora la sobreescribe.
    """
    now = datetime.now(_TZ)
    row = {
        "datetime":    now.strftime("%Y-%m-%d %H:%M"),
        "date":        str(now.date()),
        "hour":        now.hour,
        "temperature": data["temperature"],
        "pressure":    data["pressure"],
        "wind_speed":  data["wind_speed"],
        "humidity":    data["humidity"],
        "clouds":      data["clouds"],
        "rain_1h":     data.get("rain_1h", 0.0),
        "pm2_5":       data.get("pm2_5", 0.0),
    }

    df_new = pd.DataFrame([row])

    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        df = df[~((df["date"] == row["date"]) & (df["hour"] == row["hour"]))]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = df_new

    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"  Lectura guardada: {row['date']} {row['hour']}h "
          f"({len(df)} lecturas totales en raw)")
    return df


# ── 2. Agregar un día ─────────────────────────────────────────────────────
def aggregate_day(date_str: str) -> dict | None:
    """
    Calcula los agregados reales de un día desde sus lecturas brutas.

    Con 1 lectura : max = min = mean = ese valor
    Con 2 lecturas: mejor aproximación
    Con 3 lecturas: max/min/mean reales del día
    """
    if not RAW_PATH.exists():
        return None

    df  = pd.read_csv(RAW_PATH)
    day = df[df["date"] == date_str]

    if day.empty:
        return None

    return {
        "date":              date_str,
        "n_readings":        len(day),
        "temp_c_max":        day["temperature"].max(),
        "temp_c_min":        day["temperature"].min(),
        "temp_c_mean":       day["temperature"].mean(),
        "pressure_hpa_mean": day["pressure"].mean(),
        "pressure_hpa_min":  day["pressure"].min(),
        "wind_speed_max":    day["wind_speed"].max(),
        "wind_speed_mean":   day["wind_speed"].mean(),
        "humidity_max":      day["humidity"].max(),
        "humidity_mean":     day["humidity"].mean(),
        # Suma real de precipitación del día desde las lecturas de la API
        "precip_mm_sum":     day["rain_1h"].sum(),
        "cloud_cover_mean":  day["clouds"].mean() / 100.0,
        "pm2_5_mean":        day["pm2_5"].mean(),
    }


# ── 3. Recalcular history_daily.csv ───────────────────────────────────────
def update_daily():
    """
    Recorre todas las fechas en history_raw.csv y agrega cada una.
    El día de hoy siempre se recalcula con las lecturas que tenga hasta ahora.
    """
    if not RAW_PATH.exists():
        return None

    df_raw  = pd.read_csv(RAW_PATH)
    dates   = sorted(df_raw["date"].unique())
    records = [r for r in (aggregate_day(d) for d in dates) if r]

    if not records:
        return None

    df_daily = pd.DataFrame(records).set_index("date")
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily = df_daily.sort_index()

    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(DAILY_PATH)

    n_today = int(df_daily["n_readings"].iloc[-1])
    print(f"  Historial diario: {len(df_daily)} días "
          f"(hoy {n_today}/3 lecturas — "
          f"temp_max={df_daily['temp_c_max'].iloc[-1]:.1f}°C "
          f"temp_min={df_daily['temp_c_min'].iloc[-1]:.1f}°C)")
    return df_daily


# ── 4. Punto de entrada principal ─────────────────────────────────────────
def append_today(data: dict):
    """Llamado desde main.py en cada ejecución."""
    append_reading(data)
    return update_daily()


# ── 5. Build features para el predictor ───────────────────────────────────
def build_features_from_history() -> "pd.DataFrame | None":
    """
    Construye las features del último día con rolling y lags reales
    desde los agregados diarios (no estimaciones).
    """
    if not DAILY_PATH.exists():
        return None

    df = pd.read_csv(DAILY_PATH, index_col=0, parse_dates=True).sort_index()

    if len(df) < 1:
        return None

    # ── Rolling ──────────────────────────────────────────────────────────
    for col in ["temp_c_max", "pressure_hpa_mean", "wind_speed_max", "humidity_mean"]:
        df[f"{col}_ma3"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()

    # ── Lags ─────────────────────────────────────────────────────────────
    for lag in [1, 2, 3]:
        for col in ["temp_c_max", "precip_mm_sum", "wind_speed_max"]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ── Gradientes ───────────────────────────────────────────────────────
    df["temp_grad"]     = df["temp_c_max"].diff()
    df["pressure_grad"] = df["pressure_hpa_mean"].diff()

    # ── Estacionalidad ───────────────────────────────────────────────────
    doy = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    # ── Features adicionales ─────────────────────────────────────────────
    df["temp_range"]     = df["temp_c_max"] - df["temp_c_min"]
    df["heat_intensity"] = df["temp_c_max"] - df["temp_c_mean"]

    df["pressure_hpa_mean_ma7"] = df["pressure_hpa_mean"].rolling(7, min_periods=1).mean()
    df["pressure_deficit"]      = df["pressure_hpa_mean"] - df["pressure_hpa_mean_ma7"]
    df["humidity_range"]        = df["humidity_max"] - df["humidity_mean"]
    df["wind_spike"]            = df["wind_speed_max"] - df["wind_speed_mean"]
    df["dry_index"]             = df["temp_range"] * (100 - df["humidity_mean"])

    PRES_HIST_MEAN = 1013.0
    PRES_HIST_STD  = 5.0
    df["pressure_norm"] = (df["pressure_hpa_mean"] - PRES_HIST_MEAN) / PRES_HIST_STD

    return df.iloc[[-1]]


# ── 6. Helpers ────────────────────────────────────────────────────────────
def days_available() -> int:
    if not DAILY_PATH.exists():
        return 0
    return len(pd.read_csv(DAILY_PATH, index_col=0, parse_dates=True))


def history_status() -> str:
    n       = days_available()
    today   = str(datetime.now(_TZ).date())
    n_today = 0

    if RAW_PATH.exists():
        df_raw  = pd.read_csv(RAW_PATH)
        n_today = len(df_raw[df_raw["date"] == today])

    readings = f"{n_today}/3 lecturas hoy"

    if n == 0:
        return f"sin días aún ({readings})"
    elif n < MIN_DAYS_ROLLING:
        return f"{n} días ({readings}) — faltan {MIN_DAYS_ROLLING - n} para rolling real"
    elif n < MIN_DAYS_FULL:
        return f"{n} días ({readings}) — faltan {MIN_DAYS_FULL - n} para lags completos"
    else:
        return f"{n} días ({readings}) — predicción completa ✓"
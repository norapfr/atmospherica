import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")


def load_era5(path="ml/data_todo") -> pd.DataFrame:
    print("Cargando ERA5...")

    instant_files = sorted(Path(path).glob("era5_20??_instant.nc"))
    accum_files   = sorted(Path(path).glob("era5_20??_accum.nc"))

    print(f"Instant files: {[f.name for f in instant_files]}")
    print(f"Accum files:   {[f.name for f in accum_files]}")

    ds_instant = xr.open_mfdataset(instant_files, combine="by_coords", engine="netcdf4")
    ds_accum   = xr.open_mfdataset(accum_files,   combine="by_coords", engine="netcdf4")

    print(f"Variables instant: {list(ds_instant.data_vars)}")
    print(f"Variables accum:   {list(ds_accum.data_vars)}")

    ds = xr.merge([ds_instant, ds_accum], join="inner")

    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()

    df = df.rename(columns={
        "valid_time": "datetime",
        "time": "datetime",
        "t2m": "temp_k",
        "sp": "pressure_pa",
        "u10": "wind_u",
        "v10": "wind_v",
        "d2m": "dewpoint_k",
        "tp": "precip_accum",  # IMPORTANTE
        "tcc": "cloud_cover",
    })

    df = df.dropna(subset=["datetime"])
    print(f"Registros: {len(df)}")
    return df


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    print("\nLimpiando y resampleando...")

    df = df.set_index("datetime").sort_index()

    # ── Conversiones físicas ──
    df["temp_c"] = df["temp_k"] - 273.15
    df["pressure_hpa"] = df["pressure_pa"] / 100
    df["dewpoint_c"] = df["dewpoint_k"] - 273.15

    df["wind_speed"] = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)

    # ── HUMEDAD REALISTA ──
    df["humidity"] = 100 * np.exp(
        (17.625 * df["dewpoint_c"]) / (243.04 + df["dewpoint_c"]) -
        (17.625 * df["temp_c"])     / (243.04 + df["temp_c"])
    )
    df["humidity"] = df["humidity"].clip(0, 100)

    # ── PRECIPITACIÓN CORRECTA (clave) ──
    # ERA5 accum = acumulado → hay que derivar
    df["precip_mm"] = df["precip_accum"].diff().clip(lower=0) * 1000

    # ── Limpieza básica ──
    df = df.replace([np.inf, -np.inf], np.nan)

    # ── Resample diario ──
    daily = df.resample("D").agg({
        "temp_c": ["max", "min", "mean"],
        "pressure_hpa": ["mean", "min"],
        "wind_speed": ["max", "mean"],
        "humidity": ["max", "mean"],
        "precip_mm": "sum",
        "cloud_cover": "mean"
    })

    daily.columns = ["_".join(col) for col in daily.columns]
    daily = daily.dropna()

    print(f"Días: {len(daily)}")
    return daily


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerando features RF...")

    # ─────────────────────────────────────────────
    # CORRECCIÓN LOCAL (Sevilla bias ERA5)
    # ─────────────────────────────────────────────
    df["temp_c_max"] += 2.5
    df["temp_c_mean"] += 2.5
    df["temp_c_min"] += 2.5

    # ─────────────────────────────────────────────
    # ROLLING FEATURES
    # ─────────────────────────────────────────────
    for col in ["temp_c_max", "pressure_hpa_mean", "wind_speed_max", "humidity_mean"]:
        df[f"{col}_ma3"] = df[col].rolling(3).mean()
        df[f"{col}_ma7"] = df[col].rolling(7).mean()

    # ─────────────────────────────────────────────
    # LAGS (CLAVE RF)
    # ─────────────────────────────────────────────
    for lag in [1, 2, 3]:
        for col in ["temp_c_max", "precip_mm_sum", "wind_speed_max"]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ─────────────────────────────────────────────
    # GRADIENTES
    # ─────────────────────────────────────────────
    df["temp_grad"] = df["temp_c_max"].diff()
    df["pressure_grad"] = df["pressure_hpa_mean"].diff()

    # ─────────────────────────────────────────────
    # ESTACIONALIDAD
    # ─────────────────────────────────────────────
    doy = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    # ─────────────────────────────────────────────
    # FEATURES ADICIONALES (LAS IMPORTANTES QUE PEDÍAS)
    # ─────────────────────────────────────────────

    # Rango térmico diario
    df["temp_range"] = df["temp_c_max"] - df["temp_c_min"]

    # Intensidad térmica
    df["heat_intensity"] = df["temp_c_max"] - df["temp_c_mean"]

    # Déficit de presión (con seguridad)
    df["pressure_hpa_mean_ma7"] = df["pressure_hpa_mean"].rolling(7).mean()
    df["pressure_deficit"] = df["pressure_hpa_mean"] - df["pressure_hpa_mean_ma7"]

    # rango humedad
    df["humidity_range"] = df["humidity_max"] - df["humidity_mean"]

    # spikes viento
    df["wind_spike"] = df["wind_speed_max"] - df["wind_speed_mean"]

    # índice seco Sevilla (MUY IMPORTANTE)
    df["dry_index"] = df["temp_range"] * (100 - df["humidity_mean"])

    # presión normalizada
    df["pressure_norm"] = (
        (df["pressure_hpa_mean"] - df["pressure_hpa_mean"].mean()) /
        df["pressure_hpa_mean"].std()
    )

    # ─────────────────────────────────────────────
    # TARGETS (EVENTOS)
    # ─────────────────────────────────────────────
    df["event_heat"] = (df["temp_c_max"] >= 38).astype(int)
    df["event_cold"] = (df["temp_c_max"] <= 10).astype(int)
    df["event_wind"] = (df["wind_speed_max"] >= 8).astype(int)
    df["event_rain"] = (df["precip_mm_sum"] >= 1).astype(int)

    df["event_extreme"] = (
        df["event_heat"] |
        df["event_cold"] |
        df["event_wind"] |
        df["event_rain"]
    ).astype(int)

    # TARGET futuro (ML correcto)
    df["target"] = df["event_extreme"].shift(-1)

    df = df.dropna()

    print("\nDistribución target:")
    print(df["target"].value_counts(normalize=True))

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"target", "event_heat", "event_cold", "event_rain",
               "event_wind", "event_extreme"}
    return [c for c in df.columns if c not in exclude]


def build_sequences(df: pd.DataFrame, feature_cols: list,
                    seq_len: int = 14) -> tuple:
    X, y = [], []
    feats   = df[feature_cols].values
    targets = df["target"].values

    for i in range(seq_len, len(df)):
        X.append(feats[i-seq_len:i])
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    df_raw   = load_era5()
    df_daily = clean_and_resample(df_raw)
    df_feat  = add_features(df_daily)

    out = Path("ml/data_todo")
    out.mkdir(exist_ok=True)
    df_feat.to_csv(out / "featuresAll.csv")
    print(f"\nFeatures guardadas en ml/data_todo/featuresAll.csv")
    print(df_feat.tail())
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")


def load_era5(path="ml/data"):
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

    if "valid_time" in df.columns:
        df = df.rename(columns={"valid_time": "datetime"})
    else:
        df = df.rename(columns={"time": "datetime"})

    print(f"Registros cargados: {len(df)}")
    return df


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    print("\nAplicando limpieza y resample diario...")
    df = df.set_index("datetime").sort_index()

    rename = {
        "t2m": "temp_k",
        "sp":  "pressure_pa",
        "u10": "wind_u",
        "v10": "wind_v",
        "d2m": "dewpoint_k",
        "tp":  "precip",
        "tcc": "cloud_cover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "temp_k" in df.columns:
        df["temp_c"]       = df["temp_k"] - 273.15
    if "pressure_pa" in df.columns:
        df["pressure_hpa"] = df["pressure_pa"] / 100.0
    if "dewpoint_k" in df.columns:
        df["dewpoint_c"]   = df["dewpoint_k"] - 273.15
    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_speed"]   = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
        df["wind_deg"]     = (np.degrees(np.arctan2(df["wind_u"], df["wind_v"])) + 360) % 360

    if "temp_c" in df.columns and "dewpoint_c" in df.columns:
        df["humidity"] = 100 * np.exp(
            (17.625 * df["dewpoint_c"]) / (243.04 + df["dewpoint_c"]) -
            (17.625 * df["temp_c"])     / (243.04 + df["temp_c"])
        )
        df["humidity"] = df["humidity"].clip(0, 100)

    if "precip" in df.columns:
        df["precip_mm"] = df["precip"] * 1000

    agg = {}
    if "temp_c" in df.columns:
        agg["temp_max"]      = ("temp_c", "max")
        agg["temp_min"]      = ("temp_c", "min")
        agg["temp_mean"]     = ("temp_c", "mean")
    if "pressure_hpa" in df.columns:
        agg["pressure_mean"] = ("pressure_hpa", "mean")
        agg["pressure_min"]  = ("pressure_hpa", "min")
    if "wind_speed" in df.columns:
        agg["wind_max"]      = ("wind_speed", "max")
        agg["wind_mean"]     = ("wind_speed", "mean")
    if "humidity" in df.columns:
        agg["humidity_max"]  = ("humidity", "max")
        agg["humidity_mean"] = ("humidity", "mean")
    if "precip_mm" in df.columns:
        agg["precip_total"]  = ("precip_mm", "sum")
    if "cloud_cover" in df.columns:
        agg["cloud_mean"]    = ("cloud_cover", "mean")

    daily = df.resample("D").agg(**agg).dropna()
    print(f"  Dias disponibles: {len(daily)}")
    return daily


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerando features...")
    df = daily.copy()

    # Correccion de bias ERA5 — el modelo de reanálisis subestima
    # las temperaturas maximas locales de Sevilla ~2-3 grados
    # Fuente: comparacion con datos de estacion AEMET Sevilla
    TEMP_BIAS_CORRECTION = 2.5
    if "temp_max" in df.columns:
        df["temp_max"]  = df["temp_max"]  + TEMP_BIAS_CORRECTION
        df["temp_mean"] = df["temp_mean"] + TEMP_BIAS_CORRECTION
        df["temp_min"]  = df["temp_min"]  + TEMP_BIAS_CORRECTION

    # Medias moviles
    for col in ["temp_max", "temp_mean", "pressure_mean", "wind_max", "humidity_mean"]:
        if col in df.columns:
            df[f"{col}_ma3"] = df[col].rolling(3, min_periods=1).mean()
            df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()

    # Gradientes
    for col in ["temp_max", "pressure_mean", "humidity_mean"]:
        if col in df.columns:
            df[f"{col}_grad1"] = df[col].diff(1)
            df[f"{col}_grad3"] = df[col].diff(3)

    # Tendencia lineal 7 dias
    def rolling_slope(series, window=7):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)

    if "temp_max" in df.columns:
        df["temp_trend7"] = rolling_slope(df["temp_max"])

    # Estacionalidad
    day_of_year = df.index.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    # Features adicionales utiles
    # Rango diario — alta amplitud termica = dia seco y despejado
    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]

    # Deficit de presion respecto a la media movil
    if "pressure_mean" in df.columns:
        df["pressure_deficit"] = df["pressure_mean"] - df["pressure_mean_ma7"]

    # Umbrales absolutos para Sevilla (con correccion de bias)
    # Basados en definicion oficial AEMET de ola de calor
    df["event_heat"] = (df["temp_max"] >= 38.0).astype(int)  # ola de calor oficial
    df["event_cold"] = (df["temp_max"] <= 10.0).astype(int)  # dia muy frio Sevilla
    df["event_wind"] = (df["wind_max"] >= 8.0).astype(int)   # viento fuerte

    if "precip_total" in df.columns:
        # ERA5 da precipitacion en mm — umbral conservador
        df["event_rain"] = (df["precip_total"] >= 0.5).astype(int)
    else:
        df["event_rain"] = 0

    df["event_extreme"] = (
        df["event_heat"] | df["event_cold"] |
        df["event_rain"] | df["event_wind"]
    ).astype(int)

    df["target"] = df["event_extreme"].shift(-1)

    print(f"  Features generadas: {len(df.columns)}")
    print(f"  Dias con evento extremo: {df['event_extreme'].sum()} / {len(df)}")
    print(f"  event_heat: {df['event_heat'].sum()}")
    print(f"  event_cold: {df['event_cold'].sum()}")
    print(f"  event_rain: {df['event_rain'].sum()}")
    print(f"  event_wind: {df['event_wind'].sum()}")
    print(f"  Distribucion target: {df['target'].value_counts().to_dict()}")

    return df.dropna()


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

    out = Path("ml/data")
    out.mkdir(exist_ok=True)
    df_feat.to_csv(out / "features.csv")
    print(f"\nFeatures guardadas en ml/data/features.csv")
    print(df_feat.tail())
"""
LSTM para Predicción de Eventos Climáticos Extremos
====================================================
Datos:  ERA5 — Sevilla (1940-1990 + 2010-2024)
Target: evento extremo en t+1 (binario)

Correcciones respecto a la versión anterior:
  1. Escalado aplicado sobre datos diarios ANTES de construir secuencias
     (evita el solapamiento de ventanas en el StandardScaler)
  2. Features temporales explícitas añadidas (rolling ma3/ma7, gradientes)
     para ayudar al LSTM con el dataset pequeño (~5k días)
  3. Early stopping guarda el modelo en disco directamente (evita bug
     silencioso con CPU/GPU tensors)
  4. predict_tomorrow acepta DataFrame ya preprocesado o raw (con flag)
  5. find_best_threshold busca también sobre F2 (más peso al recall)
  6. DataLoader con shuffle=False en train para respetar orden temporal

Uso:
    python lstm_extreme_events.py --mode both --build_features
    python lstm_extreme_events.py --mode train
    python lstm_extreme_events.py --mode predict
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

CFG = {
    # Rutas
    "data_path":    "ml/data_todo",
    "features_csv": "ml/data_todo/features_lstm.csv",
    "models_dir":   "ml/final_model",

    # Secuencia
    # 14 días = dos semanas: captura ciclos sinópticos sin ventanas demasiado
    # largas que diluyan la señal en un dataset de 24k días
    "seq_len":      14,

    # Arquitectura
    "hidden1":      128,
    "hidden2":      64,
    "dropout":      0.25,

    # Entrenamiento
    "epochs":       150,
    "batch_size":   64,        # más grande → gradientes más estables en CPU
    "lr":           1e-4,      # más bajo: 3e-4 era demasiado agresivo
    "lr_warmup":    10,        # epochs de warmup lineal antes del cosine
    "weight_decay": 1e-4,
    "patience":     25,        # más paciencia: 24k días necesita más epochs
    "gap":          14,

    # Focal Loss — con 4.5% de event rate, gamma=1.5 es más suave que 2.0
    # y produce gradientes más estables al inicio del entrenamiento
    "focal_alpha":  0.80,
    "focal_gamma":  1.5,

    # Split
    "train_frac":   0.70,
    "val_frac":     0.10,

    # Reproducibilidad
    "seed":         42,
}

# Features físicas crudas
RAW_COLS = [
    "temp_c",        # temperatura máxima diaria 2m
    "dewpoint_c",    # punto de rocío 2m
    "pressure_hpa",  # presión superficial
    "wind_u",        # componente zonal viento 10m
    "wind_v",        # componente meridional viento 10m
    "precip_mm",     # precipitación total diaria
    "cloud_cover",   # fracción nubosidad
    "season_sin",    # ciclo anual — sin
    "season_cos",    # ciclo anual — cos
]

# Features temporales explícitas añadidas sobre los datos diarios
# El LSTM las recibe ya calculadas: esto compensa el dataset pequeño
# (~5k días) donde el LSTM no tiene capacidad suficiente para derivarlas solo
DERIVED_COLS = [
    "temp_ma3",        # media móvil 3 días de temperatura
    "temp_ma7",        # media móvil 7 días de temperatura
    "pressure_ma3",    # media móvil 3 días de presión
    "pressure_grad",   # gradiente de presión (diff de un día)
    "wind_speed",      # módulo del viento: sqrt(u²+v²)
    "wind_ma3",        # media móvil 3 días de velocidad de viento
    "precip_ma3",      # media móvil 3 días de precipitación
    "dry_index",       # rango térmico × (100 - humedad_est)
]

FEATURE_COLS = RAW_COLS + DERIVED_COLS


# ══════════════════════════════════════════════════════════════════════
#  CARGA Y PREPROCESADO ERA5
# ══════════════════════════════════════════════════════════════════════

def load_era5(path: str) -> pd.DataFrame:
    """Carga archivos ERA5 netCDF4 y devuelve DataFrame horario."""
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray no instalado: pip install xarray netCDF4")

    path = Path(path)
    instant_files = sorted(path.glob("era5_*_instant.nc"))
    accum_files   = sorted(path.glob("era5_*_accum.nc"))

    if not instant_files:
        raise FileNotFoundError(f"No se encontraron archivos ERA5 en {path}")

    print(f"Instant files: {[f.name for f in instant_files]}")
    print(f"Accum files:   {[f.name for f in accum_files]}")

    ds_instant = xr.open_mfdataset(instant_files, combine="by_coords", engine="netcdf4")
    ds_accum   = xr.open_mfdataset(accum_files,   combine="by_coords", engine="netcdf4")
    ds = xr.merge([ds_instant, ds_accum], join="inner")

    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.rename(columns={time_col: "datetime"})

    print(f"Registros cargados: {len(df)}")
    return df


def preprocess_era5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte variables ERA5 a unidades físicas, resamplea a diario
    y añade las features temporales derivadas.

    FIX: las features derivadas (rolling, gradientes) se calculan AQUÍ,
    sobre los datos diarios completos, ANTES de construir secuencias.
    Así el StandardScaler las ve correctamente sin solapamiento.
    """
    print("\nPreprocesando ERA5...")
    df = df.set_index("datetime").sort_index()

    rename = {
        "t2m":  "temp_k",      "sp":  "pressure_pa",
        "u10":  "wind_u",      "v10": "wind_v",
        "d2m":  "dewpoint_k",  "tp":  "precip",
        "tcc":  "cloud_cover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Conversiones físicas
    if "temp_k"      in df.columns: df["temp_c"]       = df["temp_k"]      - 273.15
    if "pressure_pa" in df.columns: df["pressure_hpa"]  = df["pressure_pa"] / 100.0
    if "dewpoint_k"  in df.columns: df["dewpoint_c"]    = df["dewpoint_k"]  - 273.15
    if "precip"      in df.columns:
        # ERA5 acumulado: derivar diff y convertir m → mm
        df["precip_mm"] = df["precip"].diff().clip(lower=0) * 1000.0

    # Resample diario
    agg_map = {
        "temp_c":       ("temp_c",       "max"),
        "dewpoint_c":   ("dewpoint_c",   "mean"),
        "pressure_hpa": ("pressure_hpa", "mean"),
        "wind_u":       ("wind_u",       "mean"),
        "wind_v":       ("wind_v",       "mean"),
        "precip_mm":    ("precip_mm",    "sum"),
        "cloud_cover":  ("cloud_cover",  "mean"),
    }
    agg = {k: v for k, v in agg_map.items() if v[0] in df.columns}
    daily = df.resample("D").agg(**agg).dropna()

    # Corrección de bias ERA5 para Sevilla
    if "temp_c" in daily.columns:
        daily["temp_c"] = daily["temp_c"] + 2.5

    # Estacionalidad
    doy = daily.index.dayofyear
    daily["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # ── Features temporales derivadas ─────────────────────────────────
    # FIX: se calculan aquí sobre el DataFrame diario completo,
    # NO dentro del loop de secuencias.

    if "temp_c" in daily.columns:
        daily["temp_ma3"] = daily["temp_c"].rolling(3, min_periods=1).mean()
        daily["temp_ma7"] = daily["temp_c"].rolling(7, min_periods=1).mean()

    if "pressure_hpa" in daily.columns:
        daily["pressure_ma3"]  = daily["pressure_hpa"].rolling(3, min_periods=1).mean()
        daily["pressure_grad"] = daily["pressure_hpa"].diff().fillna(0)

    if "wind_u" in daily.columns and "wind_v" in daily.columns:
        daily["wind_speed"] = np.sqrt(daily["wind_u"]**2 + daily["wind_v"]**2)
        daily["wind_ma3"]   = daily["wind_speed"].rolling(3, min_periods=1).mean()

    if "precip_mm" in daily.columns:
        daily["precip_ma3"] = daily["precip_mm"].rolling(3, min_periods=1).mean()

    # Dry index: rango térmico × (100 - humedad estimada)
    # Humedad estimada desde dewpoint/temp via Magnus approximation
    if "dewpoint_c" in daily.columns and "temp_c" in daily.columns:
        hum_est = 100 * np.exp(
            17.625 * daily["dewpoint_c"] / (243.04 + daily["dewpoint_c"]) -
            17.625 * daily["temp_c"]     / (243.04 + daily["temp_c"])
        ).clip(0, 100)
        # Necesitamos temp_min para el rango — aproximamos con dewpoint proxy
        temp_min_proxy = daily["dewpoint_c"] + 2.0
        temp_range = daily["temp_c"] - temp_min_proxy
        daily["dry_index"] = (temp_range * (100 - hum_est)).clip(lower=0)
    else:
        daily["dry_index"] = 0.0

    daily = daily.dropna()
    print(f"  Días disponibles tras features: {len(daily)}")
    return daily


def label_extremes(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Etiqueta eventos extremos y define el target t+1.
    """
    df = daily.copy()

    df["event_heat"] = (df["temp_c"] >= 38.0).astype(int)
    df["event_cold"] = (df["temp_c"] <= 10.0).astype(int)

    if "wind_speed" in df.columns:
        df["event_wind"] = (df["wind_speed"] >= 8.0).astype(int)
    elif "wind_u" in df.columns:
        spd = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
        df["event_wind"] = (spd >= 8.0).astype(int)
    else:
        df["event_wind"] = 0

    df["event_rain"] = (df["precip_mm"] >= 1.0).astype(int) \
                       if "precip_mm" in df.columns else 0

    df["event_extreme"] = (
        df["event_heat"] | df["event_cold"] |
        df["event_rain"] | df["event_wind"]
    ).astype(int)

    df["target"] = df["event_extreme"].shift(-1)

    total = len(df)
    n_ext = int(df["event_extreme"].sum())
    print(f"\n  Distribución eventos:")
    print(f"    event_heat:    {int(df['event_heat'].sum())}")
    print(f"    event_cold:    {int(df['event_cold'].sum())}")
    print(f"    event_rain:    {int(df['event_rain'].sum())}")
    print(f"    event_wind:    {int(df['event_wind'].sum())}")
    print(f"    event_extreme: {n_ext} / {total} ({n_ext/total*100:.1f}%)")

    return df.dropna()


def build_features_csv(data_path: str, out_path: str) -> pd.DataFrame:
    """Pipeline completo ERA5 → CSV de features."""
    df_raw   = load_era5(data_path)
    df_daily = preprocess_era5(df_raw)
    df_feat  = label_extremes(df_daily)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path)
    print(f"\nFeatures guardadas en {out_path}")
    return df_feat


# ══════════════════════════════════════════════════════════════════════
#  DATASET Y SECUENCIAS
# ══════════════════════════════════════════════════════════════════════

def make_sequences(scaled_features: np.ndarray,
                   targets: np.ndarray,
                   seq_len: int) -> tuple:
    """
    Construye ventanas deslizantes desde datos YA ESCALADOS.

    FIX: el escalado se aplica antes de esta función sobre el array
    diario completo. Así cada día aparece con el mismo valor escalado
    independientemente de en qué ventanas participe.
    """
    X, y = [], []
    for i in range(seq_len, len(scaled_features)):
        X.append(scaled_features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class ClimateDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def temporal_split_indices(n: int, train_frac: float,
                           val_frac: float, gap: int) -> dict:
    """
    Devuelve índices de corte para split temporal con gap.

    Esquema sobre el array de días diarios (antes de secuencias):
        [────── TRAIN ──────][gap][── VAL ──][──── TEST ────]

    Los índices de secuencia se calculan DESPUÉS del escalado,
    pero el corte se define sobre los días originales para que
    el gap sea exactamente 'gap' días calendario.
    """
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))
    return {
        "train_end": t_end,
        "val_start": t_end + gap,
        "val_end":   v_end,
        "test_start": v_end,
    }


# ══════════════════════════════════════════════════════════════════════
#  MODELO
# ══════════════════════════════════════════════════════════════════════

class AtmosphericLSTM(nn.Module):
    """
    LSTM de doble capa para predicción de eventos climáticos extremos.

    Input  → (batch, seq_len, n_features)
    LSTM1  → hidden=128  — patrones de corto plazo (frentes, cambios bruscos)
    LSTM2  → hidden=64   — compresión a régimen climático
    LayerNorm → estabiliza antes del clasificador (mejor que BatchNorm
                con secuencias cortas y batch sizes variables)
    MLP    → Linear(64→32, GELU) → Dropout → Linear(32→1, Sigmoid)

    GELU sobre ReLU: gradientes más suaves cerca de cero, relevante
    cuando los eventos positivos son raros y las activaciones tienden
    a valores pequeños.
    """
    def __init__(self, input_size: int,
                 hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.30):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1,
                             num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2,
                             num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden2)
        self.head = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)         # (B, T, hidden1)
        out, _ = self.lstm2(out)       # (B, T, hidden2)
        out    = self.norm(out[:, -1]) # último timestep: (B, hidden2)
        return self.head(out).squeeze(1)


# ══════════════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    Reduce el peso de negativos fáciles y focaliza el gradiente
    en los positivos difíciles — clave con event rate del 5%.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        w   = torch.where(target == 1,
                          torch.full_like(pred, self.alpha),
                          torch.full_like(pred, 1 - self.alpha))
        return (w * (1 - p_t) ** self.gamma * bce).mean()


# ══════════════════════════════════════════════════════════════════════
#  UTILS DE EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════

def find_best_threshold(probs: np.ndarray, targets: np.ndarray,
                        beta: float = 1.0) -> float:
    """
    Optimiza el umbral de decisión sobre val maximizando F-beta.
    beta=1 → F1 (equilibrio precision/recall)
    beta=2 → F2 (penaliza más los falsos negativos; útil si perder
                  un evento es más costoso que una falsa alarma)

    Con clases desbalanceadas al 5%, el umbral óptimo raramente es 0.5.
    """
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.05, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        if beta == 1.0:
            score = f1_score(targets, preds, zero_division=0)
        else:
            score = fbeta_score(targets, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple:
    model.eval()
    all_probs, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        all_probs.extend(model(xb).cpu().numpy())
        all_targets.extend(yb.numpy())
    return np.array(all_probs), np.array(all_targets)


# ══════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════

def train_model(df: pd.DataFrame, cfg: dict = CFG) -> dict:
    """
    Entrena el LSTM con:
      - Escalado sobre datos diarios antes de construir secuencias (FIX)
      - Split temporal con gap
      - Focal Loss + AdamW + cosine annealing
      - Early stopping sobre AUC-PR con guardado directo en disco (FIX)
      - Threshold óptimo buscado sobre validación
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    models_dir = Path(cfg["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = models_dir / "lstm_best.pt"

    seq_len    = cfg["seq_len"]
    epochs     = cfg["epochs"]
    batch_size = cfg["batch_size"]
    patience   = cfg["patience"]

    # ── Selección de features disponibles
    available    = [c for c in FEATURE_COLS if c in df.columns]
    missing      = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n⚠  Features no disponibles (se omiten): {missing}")
    feature_cols = available
    print(f"\nFeatures usadas ({len(feature_cols)}): {feature_cols}")

    feats   = df[feature_cols].values.astype(np.float32)
    targets = df["target"].values.astype(np.float32)
    n       = len(feats)

    # ── Split sobre índices de días (antes del escalado y las secuencias)
    idx = temporal_split_indices(n, cfg["train_frac"],
                                 cfg["val_frac"], cfg["gap"])

    # FIX: el scaler se ajusta solo sobre los días de train
    scaler = StandardScaler()
    scaler.fit(feats[:idx["train_end"]])

    # Escalar el array diario completo de una vez
    feats_scaled = scaler.transform(feats)

    # Construir secuencias desde los datos YA ESCALADOS
    X_all, y_all = make_sequences(feats_scaled, targets, seq_len)

    # Los índices de secuencia están desplazados en seq_len días
    # respecto a los índices de días originales
    def seq_idx(day_idx):
        return max(0, day_idx - seq_len)

    tr_end = seq_idx(idx["train_end"])
    va_st  = seq_idx(idx["val_start"])
    va_end = seq_idx(idx["val_end"])
    te_st  = seq_idx(idx["test_start"])

    X_train, y_train = X_all[:tr_end],       y_all[:tr_end]
    X_val,   y_val   = X_all[va_st:va_end],  y_all[va_st:va_end]
    X_test,  y_test  = X_all[te_st:],        y_all[te_st:]

    print(f"\nSplit temporal:")
    print(f"  Train: {len(X_train)} seq | Val: {len(X_val)} seq | Test: {len(X_test)} seq")
    print(f"  Eventos extremos — train: {int(y_train.sum())} "
          f"({y_train.mean()*100:.1f}%) | "
          f"val: {int(y_val.sum())} | test: {int(y_test.sum())}")

    train_dl = DataLoader(ClimateDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=False)
    val_dl   = DataLoader(ClimateDataset(X_val,   y_val),
                          batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(ClimateDataset(X_test,  y_test),
                          batch_size=batch_size, shuffle=False)

    # ── Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = AtmosphericLSTM(
        input_size=len(feature_cols),
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = FocalLoss(
        alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"]
    )

    # ── Loop de entrenamiento
    best_auc_pr    = 0.0
    patience_count = 0
    history        = []

    print(f"\nEntrenando {epochs} epochs (patience={patience} sobre AUC-PR)...")
    print(f"{'Epoch':>6} {'Loss':>8} {'F1':>7} {'ROC-AUC':>9} {'AUC-PR':>8}")
    print("─" * 46)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / max(len(train_dl), 1)
        scheduler.step()

        # Eval en validación
        probs_val, tgt_val = evaluate(model, val_dl, device)
        preds_val = (probs_val >= 0.5).astype(int)

        f1     = f1_score(tgt_val, preds_val, zero_division=0)
        roc    = roc_auc_score(tgt_val, probs_val) \
                 if len(np.unique(tgt_val)) > 1 else 0.5
        auc_pr = average_precision_score(tgt_val, probs_val) \
                 if len(np.unique(tgt_val)) > 1 else 0.0

        history.append({
            "epoch": epoch, "loss": round(avg_loss, 5),
            "f1": round(f1, 4), "roc_auc": round(roc, 4),
            "auc_pr": round(auc_pr, 4),
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {avg_loss:>8.4f} {f1:>7.3f} "
                  f"{roc:>9.3f} {auc_pr:>8.3f}")

        # FIX: early stopping guarda directamente en disco
        # Evita el bug de copiar state_dict entre CPU/GPU
        if auc_pr > best_auc_pr:
            best_auc_pr    = auc_pr
            torch.save(model.state_dict(), best_ckpt)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping en epoch {epoch} "
                      f"(mejor AUC-PR val: {best_auc_pr:.4f})")
                break

    # ── Restaurar mejor checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"\nMejor modelo restaurado desde {best_ckpt}")

    # ── Threshold óptimo sobre validación (F1 y F2)
    probs_val, tgt_val = evaluate(model, val_dl, device)
    th_f1 = find_best_threshold(probs_val, tgt_val, beta=1.0)
    th_f2 = find_best_threshold(probs_val, tgt_val, beta=2.0)
    print(f"Threshold óptimo — F1: {th_f1:.2f} | F2 (recall): {th_f2:.2f}")

    # Usar F1 como threshold de producción
    best_threshold = th_f1

    # ── Evaluación final sobre TEST
    probs_test, tgt_test = evaluate(model, test_dl, device)
    preds_test = (probs_test >= best_threshold).astype(int)

    final_f1     = f1_score(tgt_test, preds_test, zero_division=0)
    final_roc    = roc_auc_score(tgt_test, probs_test) \
                   if len(np.unique(tgt_test)) > 1 else 0.5
    final_auc_pr = average_precision_score(tgt_test, probs_test) \
                   if len(np.unique(tgt_test)) > 1 else 0.0

    print(f"\n{'═'*46}")
    print("RESULTADO FINAL — TEST SET")
    print(f"{'═'*46}")
    print(f"  F1  (th={best_threshold:.2f}):  {final_f1:.4f}")
    print(f"  ROC-AUC:              {final_roc:.4f}")
    print(f"  AUC-PR:               {final_auc_pr:.4f}")
    print(f"\nClassification report (test):")
    print(classification_report(tgt_test, preds_test,
                                target_names=["normal", "extremo"],
                                zero_division=0))
    print(f"Confusion matrix:\n{confusion_matrix(tgt_test, preds_test)}")

    # ── Guardar artefactos
    final_model_path = models_dir / "lstm_model.pt"
    torch.save(model.state_dict(), final_model_path)
    joblib.dump(scaler,       models_dir / "lstm_scaler.pkl")
    joblib.dump(feature_cols, models_dir / "lstm_feature_cols.pkl")

    metrics = {
        "model":           "AtmosphericLSTM",
        "seq_len":         seq_len,
        "n_features":      len(feature_cols),
        "feature_cols":    feature_cols,
        "epochs_run":      len(history),
        "threshold_f1":    float(th_f1),
        "threshold_f2":    float(th_f2),
        "threshold_used":  float(best_threshold),
        "test_f1":         float(final_f1),
        "test_roc_auc":    float(final_roc),
        "test_auc_pr":     float(final_auc_pr),
        "best_val_auc_pr": float(best_auc_pr),
        "n_train":         int(len(X_train)),
        "n_val":           int(len(X_val)),
        "n_test":          int(len(X_test)),
        "cfg":             {k: v for k, v in cfg.items()},
        "history":         history,
    }
    with open(models_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nArtefactos guardados en {models_dir}/")
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  INFERENCIA
# ══════════════════════════════════════════════════════════════════════

def load_lstm(models_dir: str = None) -> tuple:
    """Carga modelo, scaler, feature_cols y métricas para inferencia."""
    models_dir = Path(models_dir or CFG["models_dir"])
    feature_cols = joblib.load(models_dir / "lstm_feature_cols.pkl")
    scaler       = joblib.load(models_dir / "lstm_scaler.pkl")

    model = AtmosphericLSTM(
        input_size=len(feature_cols),
        hidden1=CFG["hidden1"],
        hidden2=CFG["hidden2"],
        dropout=0.0,   # sin dropout en inferencia
    )
    model.load_state_dict(
        torch.load(models_dir / "lstm_model.pt", map_location="cpu")
    )
    model.eval()

    with open(models_dir / "lstm_metrics.json", encoding="utf-8") as f:
        meta = json.load(f)

    return model, scaler, feature_cols, meta


def predict_tomorrow(df_recent: pd.DataFrame,
                     models_dir: str = None,
                     already_preprocessed: bool = False) -> dict:
    """
    Predice si ocurrirá un evento extremo mañana.

    Args:
        df_recent: DataFrame con al menos seq_len días.
            Si already_preprocessed=False (default), se esperan columnas
            raw ERA5 (temp_k, pressure_pa, etc.) y se llama a preprocess_era5.
            Si already_preprocessed=True, se esperan columnas ya procesadas
            (temp_c, pressure_hpa, wind_speed, etc.).
        models_dir: directorio con artefactos del modelo.
        already_preprocessed: si True, salta el preprocesado.

    Returns:
        dict con 'prob', 'prediction' (0/1) y 'threshold'.
    """
    model, scaler, feature_cols, meta = load_lstm(models_dir)
    seq_len   = meta["seq_len"]
    threshold = meta["threshold_used"]

    # FIX: preprocesar si los datos son crudos
    if not already_preprocessed:
        df_recent = preprocess_era5(df_recent)

    missing = [c for c in feature_cols if c not in df_recent.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en df_recent: {missing}\n"
            f"Pasa already_preprocessed=True si ya tienes las features calculadas."
        )

    if len(df_recent) < seq_len:
        raise ValueError(
            f"Se necesitan al menos {seq_len} días, "
            f"se recibieron {len(df_recent)}."
        )

    # Escalar y construir secuencia de los últimos seq_len días
    X_raw    = df_recent[feature_cols].values[-seq_len:].astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.from_numpy(X_scaled).unsqueeze(0)  # (1, T, F)

    with torch.no_grad():
        prob = model(X_tensor).item()

    return {
        "prob":       round(prob, 4),
        "prediction": int(prob >= threshold),
        "threshold":  threshold,
        "label":      "EXTREMO ⚠" if prob >= threshold else "Normal ✓",
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LSTM Eventos Climáticos Extremos — Sevilla/ERA5"
    )
    parser.add_argument(
        "--mode", choices=["train", "predict", "both"],
        default="both",
        help="train: entrenar | predict: inferencia demo | both: ambos"
    )
    parser.add_argument(
        "--features_csv", default=CFG["features_csv"],
        help="Ruta al CSV de features preprocesadas"
    )
    parser.add_argument(
        "--build_features", action="store_true",
        help="Regenerar features desde ERA5 netCDF4 antes de entrenar"
    )
    parser.add_argument(
        "--models_dir", default=CFG["models_dir"],
        help="Directorio donde guardar / cargar artefactos del modelo"
    )
    args = parser.parse_args()

    # Actualizar rutas en CFG si se pasan por argumento
    CFG["features_csv"] = args.features_csv
    CFG["models_dir"]   = args.models_dir

    # ── 1. Construir features
    if args.build_features:
        df = build_features_csv(CFG["data_path"], CFG["features_csv"])
    else:
        feat_path = Path(args.features_csv)
        if not feat_path.exists():
            print(f"\nNo se encontró {feat_path}.")
            print("Usa --build_features para generarlo desde ERA5.")
            print("O proporciona la ruta con --features_csv.")
            sys.exit(1)
        print(f"Cargando features desde {feat_path}...")
        df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    print(f"Dataset: {df.shape[0]} días, {df.shape[1]} columnas")

    # ── 2. Entrenar
    if args.mode in ("train", "both"):
        metrics = train_model(df, CFG)
        print(f"\nResumen:")
        print(f"  ROC-AUC test: {metrics['test_roc_auc']:.4f}")
        print(f"  F1 test:      {metrics['test_f1']:.4f}")
        print(f"  AUC-PR test:  {metrics['test_auc_pr']:.4f}")

    # ── 3. Demo inferencia sobre los últimos días del dataset
    if args.mode in ("predict", "both"):
        print("\n── Demo inferencia (últimos días del dataset) ──")
        try:
            result = predict_tomorrow(
                df, models_dir=args.models_dir,
                already_preprocessed=True
            )
            print(f"  Probabilidad evento mañana: {result['prob']:.4f}")
            print(f"  Predicción (th={result['threshold']:.2f}): {result['label']}")
        except Exception as e:
            print(f"  Error en inferencia: {e}")
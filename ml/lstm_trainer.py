"""
LSTM para Predicción de Eventos Climáticos Extremos — Paper-Level
=================================================================
Autor:  (adaptar)
Datos:  ERA5 — Sevilla / cualquier punto ibérico
Target: evento extremo en t+1 (binario)

Uso rápido:
    python lstm_extreme_events.py

Requiere:
    pip install torch scikit-learn pandas numpy xarray netCDF4 joblib
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
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════

CFG = {
    # Datos
    "data_path":   "ml/data_Sevilla",
    "features_csv": "ml/data_Sevilla/features_lstm.csv",
    "models_dir":  "ml/models_15years",

    # Secuencia
    "seq_len":     21,          # días de contexto (21 = dos ciclos sinópticos)

    # Arquitectura
    "hidden1":     128,
    "hidden2":     64,
    "dropout":     0.25,

    # Entrenamiento
    "epochs":      80,
    "batch_size":  32,
    "lr":          3e-4,
    "weight_decay": 1e-4,
    "patience":    15,          # early stopping sobre AUC-PR
    "gap":         21,          # días de separación train/val (= seq_len)

    # Focal Loss
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,

    # Split
    "train_frac":  0.70,
    "val_frac":    0.10,
    # test = resto (0.20)

    # Reproducibilidad
    "seed":        42,
}

# Features físicas crudas — sin rolling, sin eventos binarios
FEATURE_COLS = [
    "temp_c",        # temperatura 2m
    "dewpoint_c",    # punto de rocío 2m
    "pressure_hpa",  # presión superficial
    "wind_u",        # componente zonal viento 10m
    "wind_v",        # componente meridional viento 10m
    "precip_mm",     # precipitación acumulada
    "cloud_cover",   # fracción nubosidad
    "season_sin",    # ciclo anual — sin
    "season_cos",    # ciclo anual — cos
]


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
    instant_files = sorted(path.glob("era5_20??_instant.nc"))
    accum_files   = sorted(path.glob("era5_20??_accum.nc"))

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
    Convierte variables ERA5 a unidades físicas y resamplea a diario.
    Solo variables físicas crudas — sin rolling stats ni gradients.
    """
    print("\nPreprocesando ERA5...")
    df = df.set_index("datetime").sort_index()

    # Renombrar variables ERA5
    rename = {
        "t2m": "temp_k",   "sp":  "pressure_pa",
        "u10": "wind_u",   "v10": "wind_v",
        "d2m": "dewpoint_k", "tp": "precip",
        "tcc": "cloud_cover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Conversiones físicas
    if "temp_k"      in df.columns: df["temp_c"]      = df["temp_k"]      - 273.15
    if "pressure_pa" in df.columns: df["pressure_hpa"] = df["pressure_pa"] / 100.0
    if "dewpoint_k"  in df.columns: df["dewpoint_c"]   = df["dewpoint_k"]  - 273.15
    if "precip"      in df.columns: df["precip_mm"]    = df["precip"]      * 1000.0

    # Resample diario — agregaciones físicamente correctas
    agg = {}
    for (col, func, alias) in [
        ("temp_c",       "max",  "temp_c"),        # máxima diaria
        ("dewpoint_c",   "mean", "dewpoint_c"),
        ("pressure_hpa", "mean", "pressure_hpa"),
        ("wind_u",       "mean", "wind_u"),
        ("wind_v",       "mean", "wind_v"),
        ("precip_mm",    "sum",  "precip_mm"),
        ("cloud_cover",  "mean", "cloud_cover"),
    ]:
        if col in df.columns:
            agg[alias] = (col, func)

    daily = df.resample("D").agg(**agg).dropna()

    # Estacionalidad continua — transformación mínima justificada físicamente
    doy = daily.index.dayofyear
    daily["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Corrección de bias ERA5 para Sevilla (Tmax subestimada ~2.5°C vs AEMET)
    # Comentar si se usa otro punto geográfico
    if "temp_c" in daily.columns:
        daily["temp_c"] = daily["temp_c"] + 2.5

    print(f"  Días disponibles: {len(daily)}")
    return daily


def label_extremes(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Define eventos extremos con umbrales físicos para Sevilla.
    Target = evento en t+1 (shift -1).
    Los labels NO se incluyen como features de entrada.
    """
    df = daily.copy()

    # Umbrales AEMET / climatología Sevilla
    df["event_heat"] = (df["temp_c"]    >= 38.0).astype(int)
    df["event_cold"] = (df["temp_c"]    <= 10.0).astype(int)
    df["event_wind"] = (
        np.sqrt(df["wind_u"]**2 + df["wind_v"]**2) >= 8.0
    ).astype(int)
    df["event_rain"] = (df["precip_mm"] >= 0.5).astype(int) \
                       if "precip_mm" in df.columns else 0

    df["event_extreme"] = (
        df["event_heat"] | df["event_cold"] |
        df["event_rain"] | df["event_wind"]
    ).astype(int)

    # Target: evento MAÑANA
    df["target"] = df["event_extreme"].shift(-1)

    print(f"\n  Distribución eventos:")
    print(f"    event_heat:    {df['event_heat'].sum()}")
    print(f"    event_cold:    {df['event_cold'].sum()}")
    print(f"    event_rain:    {df['event_rain'].sum()}")
    print(f"    event_wind:    {df['event_wind'].sum()}")
    print(f"    event_extreme: {df['event_extreme'].sum()} / {len(df)} "
          f"({df['event_extreme'].mean()*100:.1f}%)")

    return df.dropna()


def build_features_csv(data_path: str, out_path: str) -> pd.DataFrame:
    """Pipeline completo ERA5 → CSV de features listo para entrenar."""
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

def make_sequences(df: pd.DataFrame, feature_cols: list,
                   seq_len: int) -> tuple:
    """
    Construye ventanas deslizantes preservando orden temporal.
    X[i] = días [i-seq_len .. i-1], y[i] = target del día i.
    """
    X, y = [], []
    feats   = df[feature_cols].values.astype(np.float32)
    targets = df["target"].values.astype(np.float32)

    for i in range(seq_len, len(df)):
        X.append(feats[i - seq_len:i])
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class ClimateDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def temporal_split(X: np.ndarray, y: np.ndarray,
                   train_frac: float, val_frac: float,
                   gap: int) -> dict:
    """
    Split estrictamente temporal con gap entre particiones.

    Esquema:
        [───── TRAIN ─────][gap][─ VAL ─][─── TEST ───]

    El gap evita que las últimas secuencias de train solapen
    con val cuando seq_len > 1.
    """
    n      = len(X)
    t_end  = int(n * train_frac)
    v_end  = int(n * (train_frac + val_frac))

    return {
        "X_train": X[:t_end],          "y_train": y[:t_end],
        "X_val":   X[t_end+gap:v_end], "y_val":   y[t_end+gap:v_end],
        "X_test":  X[v_end:],          "y_test":  y[v_end:],
    }


# ══════════════════════════════════════════════════════════════════════
#  MODELO
# ══════════════════════════════════════════════════════════════════════

class AtmosphericLSTM(nn.Module):
    """
    LSTM de doble capa para predicción de eventos climáticos extremos.

    Arquitectura:
        Input  → (batch, seq_len, n_features)
        LSTM1  → hidden=128, extrae patrones de corto plazo
        LSTM2  → hidden=64,  comprime a representaciones de régimen
        LayerNorm → estabiliza distribución antes del clasificador
        MLP    → Linear(64→32, GELU) → Dropout → Linear(32→1, Sigmoid)

    Notas de diseño:
        - LayerNorm en lugar de BatchNorm: no depende del tamaño de batch
          y es más estable en secuencias temporales cortas.
        - GELU sobre ReLU: gradientes más suaves, mejor en activaciones
          próximas a cero (relevante con eventos raros).
        - Se usa el último timestep de LSTM2 como representación,
          que resume toda la secuencia causal hasta t.
    """
    def __init__(self, input_size: int,
                 hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.25):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size, hidden_size=hidden1,
            num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden1, hidden_size=hidden2,
            num_layers=1, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden2)
        self.head = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)          # (B, T, hidden1)
        out, _ = self.lstm2(out)        # (B, T, hidden2)
        out    = self.norm(out[:, -1])  # (B, hidden2) — último timestep
        return self.head(out).squeeze(1)


# ══════════════════════════════════════════════════════════════════════
#  FUNCIÓN DE PÉRDIDA
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) para clasificación binaria desbalanceada.

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    El factor (1−p_t)^γ reduce el peso de ejemplos fáciles (negativos
    bien clasificados) y focaliza el gradiente en los positivos difíciles.

    Args:
        alpha: peso de la clase positiva. 0.75 da énfasis a eventos extremos.
        gamma: factor de focalización. 2.0 es el valor canónico (Lin et al.).
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy(pred, target, reduction="none")
        p_t  = pred * target + (1 - pred) * (1 - target)
        w    = torch.where(target == 1,
                           torch.full_like(pred, self.alpha),
                           torch.full_like(pred, 1 - self.alpha))
        loss = w * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════

def find_best_threshold(probs: np.ndarray, targets: np.ndarray,
                        metric: str = "f1") -> float:
    """
    Optimiza el umbral de decisión sobre val.
    Con clases desbalanceadas, 0.5 rara vez es óptimo.
    metric: 'f1' | 'f2'  (f2 penaliza más el recall)
    """
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        if metric == "f2":
            score = f1_score(targets, preds, beta=2, zero_division=0)
        else:
            score = f1_score(targets, preds, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return best_t


def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple:
    """Devuelve (probs, targets) para el loader dado."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            all_probs.extend(model(xb).cpu().numpy())
            all_targets.extend(yb.numpy())
    return np.array(all_probs), np.array(all_targets)


def train_model(df: pd.DataFrame, cfg: dict = CFG) -> dict:
    """
    Entrena el LSTM con validación temporal, Focal Loss y early stopping
    sobre AUC-PR (más informativo que ROC-AUC en clases desbalanceadas).

    Devuelve dict con métricas finales sobre test.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    seq_len     = cfg["seq_len"]
    epochs      = cfg["epochs"]
    batch_size  = cfg["batch_size"]
    patience    = cfg["patience"]
    models_dir  = Path(cfg["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Verificar features disponibles
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n⚠ Features faltantes (se omiten): {missing}")
    feature_cols = available
    print(f"\nFeatures usadas ({len(feature_cols)}): {feature_cols}")

    # ── Secuencias
    X, y = make_sequences(df, feature_cols, seq_len)
    print(f"Secuencias: {len(X)} × {seq_len} días × {len(feature_cols)} features")
    print(f"Eventos extremos: {int(y.sum())} ({y.mean()*100:.1f}%)")

    # ── Split temporal con gap
    splits = temporal_split(X, y,
                            cfg["train_frac"], cfg["val_frac"],
                            cfg["gap"])
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    print(f"\nSplit temporal:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Escalado (ajustado solo sobre train)
    scaler  = StandardScaler()
    n_feat  = len(feature_cols)
    # Reshape para escalar: (N*T, F) → escalar → restaurar
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    def scale(arr):
        return scaler.transform(arr.reshape(-1, n_feat)).reshape(arr.shape)

    X_train = scale(X_train)
    X_val   = scale(X_val)
    X_test  = scale(X_test)

    # ── DataLoaders (shuffle=False — respeta orden temporal)
    train_dl = DataLoader(ClimateDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=False)
    val_dl   = DataLoader(ClimateDataset(X_val,   y_val),
                          batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(ClimateDataset(X_test,  y_test),
                          batch_size=batch_size, shuffle=False)

    # ── Modelo, optimizador, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = AtmosphericLSTM(
        input_size=len(feature_cols),
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        dropout=cfg["dropout"]
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])

    # ── Loop de entrenamiento
    best_auc_pr    = 0.0
    best_state     = None
    patience_count = 0
    history        = []

    print(f"\nEntrenando {epochs} epochs (early stopping patience={patience})...")
    print(f"{'Epoch':>6} {'Loss':>8} {'F1':>7} {'ROC-AUC':>9} {'AUC-PR':>8}")
    print("─" * 45)

    for epoch in range(1, epochs + 1):
        # ── Train
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

        avg_loss = train_loss / len(train_dl)
        scheduler.step()

        # ── Eval en validación
        probs_val, targets_val = evaluate(model, val_dl, device)
        preds_val = (probs_val >= 0.5).astype(int)

        f1     = f1_score(targets_val, preds_val, zero_division=0)
        roc    = roc_auc_score(targets_val, probs_val) \
                 if len(np.unique(targets_val)) > 1 else 0.5
        auc_pr = average_precision_score(targets_val, probs_val) \
                 if len(np.unique(targets_val)) > 1 else 0.0

        history.append({
            "epoch": epoch, "loss": avg_loss,
            "f1": f1, "roc_auc": roc, "auc_pr": auc_pr
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {avg_loss:>8.4f} {f1:>7.3f} {roc:>9.3f} {auc_pr:>8.3f}")

        # ── Early stopping sobre AUC-PR
        if auc_pr > best_auc_pr:
            best_auc_pr    = auc_pr
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping en epoch {epoch} "
                      f"(mejor AUC-PR val: {best_auc_pr:.3f})")
                break

    # ── Restaurar mejor modelo
    model.load_state_dict(best_state)

    # ── Threshold óptimo sobre validación
    probs_val, targets_val = evaluate(model, val_dl, device)
    best_threshold = find_best_threshold(probs_val, targets_val, metric="f1")
    print(f"\nThreshold óptimo (val F1): {best_threshold:.2f}")

    # ── Evaluación final sobre TEST
    probs_test, targets_test = evaluate(model, test_dl, device)
    preds_test = (probs_test >= best_threshold).astype(int)

    final_f1     = f1_score(targets_test, preds_test, zero_division=0)
    final_roc    = roc_auc_score(targets_test, probs_test) \
                   if len(np.unique(targets_test)) > 1 else 0.5
    final_auc_pr = average_precision_score(targets_test, probs_test) \
                   if len(np.unique(targets_test)) > 1 else 0.0

    print(f"\n{'═'*45}")
    print(f"RESULTADO FINAL — TEST SET")
    print(f"{'═'*45}")
    print(f"  F1 (th={best_threshold:.2f}):  {final_f1:.3f}")
    print(f"  ROC-AUC:             {final_roc:.3f}")
    print(f"  AUC-PR:              {final_auc_pr:.3f}")
    print(f"\nClassification Report (test):")
    print(classification_report(targets_test, preds_test,
                                target_names=["normal", "extremo"],
                                zero_division=0))
    print(f"Confusion Matrix:\n{confusion_matrix(targets_test, preds_test)}")

    # ── Guardar artefactos
    torch.save(best_state, models_dir / "lstm_model.pt")
    joblib.dump(scaler,       models_dir / "lstm_scaler.pkl")
    joblib.dump(feature_cols, models_dir / "lstm_feature_cols.pkl")

    metrics = {
        "model":          "AtmosphericLSTM",
        "seq_len":        seq_len,
        "epochs_run":     len(history),
        "threshold":      float(best_threshold),
        "test_f1":        float(final_f1),
        "test_roc_auc":   float(final_roc),
        "test_auc_pr":    float(final_auc_pr),
        "best_val_auc_pr": float(best_auc_pr),
        "n_train":        int(len(X_train)),
        "n_val":          int(len(X_val)),
        "n_test":         int(len(X_test)),
        "n_features":     int(len(feature_cols)),
        "feature_cols":   feature_cols,
        "cfg":            {k: v for k, v in cfg.items()},
        "history":        history,
    }
    with open(models_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nModelo y métricas guardados en {models_dir}/")
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  INFERENCIA
# ══════════════════════════════════════════════════════════════════════

def load_model(models_dir: str = "ml/models_15years") -> tuple:
    """Carga modelo, scaler y feature_cols para inferencia."""
    models_dir = Path(models_dir)
    feature_cols = joblib.load(models_dir / "lstm_feature_cols.pkl")
    scaler       = joblib.load(models_dir / "lstm_scaler.pkl")

    model = AtmosphericLSTM(
        input_size=len(feature_cols),
        hidden1=CFG["hidden1"],
        hidden2=CFG["hidden2"],
        dropout=0.0   # sin dropout en inferencia
    )
    state = torch.load(models_dir / "lstm_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with open(models_dir / "lstm_metrics.json") as f:
        meta = json.load(f)

    return model, scaler, feature_cols, meta


def predict_tomorrow(df_recent: pd.DataFrame,
                     models_dir: str = "ml/models_15years") -> dict:
    """
    Predice si ocurrirá un evento extremo mañana.

    Args:
        df_recent: DataFrame con al menos seq_len días recientes.
                   Debe contener las columnas físicas crudas.
        models_dir: directorio con artefactos del modelo.

    Returns:
        dict con 'prob' y 'prediction' (0/1).
    """
    model, scaler, feature_cols, meta = load_model(models_dir)
    seq_len   = meta["seq_len"]
    threshold = meta["threshold"]

    missing = [c for c in feature_cols if c not in df_recent.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en df_recent: {missing}")

    X = df_recent[feature_cols].values[-seq_len:].astype(np.float32)
    if len(X) < seq_len:
        raise ValueError(f"Se necesitan al menos {seq_len} días, "
                         f"se recibieron {len(X)}")

    X_scaled = scaler.transform(X)
    X_tensor = torch.from_numpy(X_scaled).unsqueeze(0)  # (1, T, F)

    with torch.no_grad():
        prob = model(X_tensor).item()

    return {
        "prob":       round(prob, 4),
        "prediction": int(prob >= threshold),
        "threshold":  threshold,
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LSTM Eventos Climáticos Extremos — Sevilla/ERA5"
    )
    parser.add_argument("--mode", choices=["train", "predict", "both"],
                        default="both",
                        help="train: entrenar | predict: inferencia | both: ambos")
    parser.add_argument("--features_csv", default=CFG["features_csv"],
                        help="Ruta al CSV de features (generado por preprocess)")
    parser.add_argument("--build_features", action="store_true",
                        help="Regenerar features desde ERA5 netCDF4 antes de entrenar")
    args = parser.parse_args()

    # ── 1. Construir features desde ERA5 si se pide
    if args.build_features:
        df = build_features_csv(CFG["data_path"], CFG["features_csv"])
    else:
        if not Path(args.features_csv).exists():
            print(f"No se encontró {args.features_csv}.")
            print("Ejecuta con --build_features para generar desde ERA5,")
            print("o proporciona el CSV con --features_csv.")
            sys.exit(1)
        print(f"Cargando features desde {args.features_csv}...")
        df = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)

    print(f"DataFrame cargado: {df.shape[0]} días, {df.shape[1]} columnas")

    # ── 2. Entrenar
    if args.mode in ("train", "both"):
        metrics = train_model(df, CFG)

    # ── 3. Demo inferencia sobre los últimos días del dataset
    if args.mode in ("predict", "both"):
        print("\n── Demo inferencia (últimos días del dataset) ──")
        try:
            result = predict_tomorrow(df, CFG["models_dir"])
            print(f"  Probabilidad evento mañana: {result['prob']:.4f}")
            print(f"  Predicción (th={result['threshold']:.2f}): "
                  f"{'EXTREMO ⚠' if result['prediction'] else 'Normal ✓'}")
        except Exception as e:
            print(f"  Error en inferencia: {e}")
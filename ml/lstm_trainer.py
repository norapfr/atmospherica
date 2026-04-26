import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
import joblib
import json


# ── DATASET ──────────────────────────────────────────────────────────
class ClimateSequenceDataset(Dataset):
    """
    Cada muestra es una ventana de SEQ_LEN dias consecutivos.
    El target es el evento extremo del dia siguiente.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── ARQUITECTURA LSTM ─────────────────────────────────────────────────
class AtmosphericLSTM(nn.Module):
    """
    LSTM para prediccion de eventos climaticos extremos.

    Arquitectura:
        Input:  (batch, seq_len, n_features)
        LSTM1:  128 unidades
        Dropout: 0.3
        LSTM2:  64 unidades
        Dropout: 0.3
        Dense:  32 → 1 (sigmoid)

    La doble capa LSTM permite capturar patrones a dos escalas:
    - LSTM1: patrones de corto plazo (dias)
    - LSTM2: patrones de medio plazo (semanas)
    """
    def __init__(self, input_size: int, hidden1: int = 128,
                 hidden2: int = 64, dropout: float = 0.3):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm1(x)           # (batch, seq_len, hidden1)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)          # (batch, seq_len, hidden2)
        out = self.dropout2(out)
        out = out[:, -1, :]              # ultimo timestep: (batch, hidden2)
        out = self.fc(out)               # (batch, 1)
        return out.squeeze(1)


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────
def train_lstm(df: pd.DataFrame, seq_len: int = 14,
               epochs: int = 50, batch_size: int = 32) -> dict:
    """
    Entrena el LSTM con validacion temporal.
    Usa los primeros 80% de dias para train y el 20% final para test.
    No mezcla datos futuros con pasados — fundamental en series temporales.
    """
    print("\n── LSTM ────────────────────────────────────────")

    # Features y target
    exclude = {"target", "event_heat", "event_cold",
               "event_rain", "event_wind", "event_extreme"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X_raw = df[feature_cols].values
    y_raw = df["target"].values

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Construir secuencias
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_len:i])
        y_seq.append(y_raw[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    print(f"Secuencias: {len(X_seq)} x {seq_len} dias x {len(feature_cols)} features")
    print(f"Eventos extremos: {int(y_seq.sum())} ({y_seq.mean()*100:.1f}%)")

    # Split temporal 80/20
    split = int(len(X_seq) * 0.80)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Peso para clase positiva (compensar desbalance)
    pos_weight = torch.tensor([(1 - y_train.mean()) / (y_train.mean() + 1e-6)])

    # Datasets y loaders
    train_ds = ClimateSequenceDataset(X_train, y_train)
    test_ds  = ClimateSequenceDataset(X_test,  y_test)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AtmosphericLSTM(
    input_size=len(feature_cols),
    hidden1=64,    # era 128 — la mitad
    hidden2=32,    # era 64  — la mitad
    dropout=0.2
).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Loop de entrenamiento
    best_auc  = 0.0
    best_state = None
    history   = []

    print(f"\nEntrenando {epochs} epochs...")
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Eval
        model.eval()
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                probs = model(xb).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(yb.numpy())

        all_targets = np.array(all_targets)
        all_probs   = np.array(all_probs)
        all_preds   = np.array(all_preds)

        f1  = f1_score(all_targets, all_preds, zero_division=0)
        auc = roc_auc_score(all_targets, all_probs) \
              if len(np.unique(all_targets)) > 1 else 0.5
        avg_loss = train_loss / len(train_dl)

        scheduler.step(avg_loss)
        history.append({"epoch": epoch, "loss": avg_loss, "f1": f1, "auc": auc})

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} — loss: {avg_loss:.4f}  "
                  f"F1: {f1:.3f}  AUC: {auc:.3f}")

        # Guardar mejor modelo
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Cargar mejor estado
    model.load_state_dict(best_state)

    # Evaluacion final
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            probs = model(xb).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_targets.extend(yb.numpy())

    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)
    all_preds   = np.array(all_preds)

    final_f1  = f1_score(all_targets, all_preds, zero_division=0)
    final_auc = roc_auc_score(all_targets, all_probs) \
                if len(np.unique(all_targets)) > 1 else 0.5

    print(f"\nResultado final LSTM:")
    print(f"  F1:  {final_f1:.3f}")
    print(f"  AUC: {final_auc:.3f}")

    # Guardar
    Path("ml/models").mkdir(exist_ok=True)
    torch.save(best_state, "ml/models/lstm_model.pt")
    joblib.dump(scaler,       "ml/models/lstm_scaler.pkl")
    joblib.dump(feature_cols, "ml/models/lstm_feature_cols.pkl")

    metrics = {
        "model":       "lstm",
        "seq_len":     seq_len,
        "epochs":      epochs,
        "final_f1":    float(final_f1),
        "final_auc":   float(final_auc),
        "best_auc":    float(best_auc),
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)),
        "n_features":  int(len(feature_cols)),
        "history":     history[-10:],  # ultimas 10 epochs
    }
    with open("ml/models/lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Modelo LSTM guardado en ml/models/")
    return metrics


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("ml/data/features.csv", index_col=0, parse_dates=True)
    metrics = train_lstm(
        df,
        seq_len=14,
        epochs=30,
        batch_size=64
    )
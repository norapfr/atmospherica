import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
import shap
import joblib
import json


def load_features(path: str = "ml/data_Sevilla/features15Y.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Features cargadas: {df.shape[0]} dias x {df.shape[1]} columnas")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"target", "event_heat", "event_cold",
               "event_rain", "event_wind", "event_extreme"}
    return [c for c in df.columns if c not in exclude]


def train_random_forest(df: pd.DataFrame) -> dict:
    print("\n── RANDOM FOREST ──────────────────────────────")

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["target"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Muestras: {len(X)}")
    print(f"Eventos extremos en target: {int(y.sum())} ({y.mean()*100:.1f}%)")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Proteccion: a veces un fold solo tiene una clase
        if y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob[:, 0]

        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob_pos) if len(np.unique(y_test)) > 1 else 0.5
        fold_metrics.append({"fold": fold+1, "f1": f1, "auc": auc})
        print(f"  Fold {fold+1}: F1={f1:.3f}  AUC={auc:.3f}")

    # Modelo final con todos los datos
    print("\nEntrenando modelo final con todos los datos...")
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_scaled, y)

    # SHAP values
    print("Calculando SHAP values...")
    explainer   = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_scaled[:100])

    # shap_values puede ser array 2D, 3D o lista segun version de shap
    if isinstance(shap_values, list):
        # lista de arrays por clase → usar clase 1
        arr = np.array(shap_values[1])
    elif shap_values.ndim == 3:
        # (n_classes, n_samples, n_features) o (n_samples, n_features, n_classes)
        if shap_values.shape[0] == 2:
            arr = shap_values[1]
        else:
            arr = shap_values[:, :, 1]
    else:
        arr = shap_values  # ya es 2D (n_samples, n_features)

    shap_importance = np.abs(arr).mean(axis=0)  # shape: (n_features,)

    feature_importance = sorted(
        zip(feature_cols, shap_importance.tolist()),
        key=lambda x: x[1], reverse=True
    )

    print("\nTop 10 features mas importantes (SHAP):")
    for feat, imp in feature_importance[:10]:
        print(f"  {feat:30s}: {imp:.4f}")

    Path("ml/models_15years").mkdir(exist_ok=True)
    joblib.dump(final_model,  "ml/models_15years/rf_model.pkl")
    joblib.dump(scaler,       "ml/models_15years/rf_scaler.pkl")
    joblib.dump(feature_cols, "ml/models_15years/feature_cols.pkl")

    metrics = {
        "model":        "random_forest",
        "fold_metrics": fold_metrics,
        "mean_f1":      float(np.mean([m["f1"] for m in fold_metrics])),
        "mean_auc":     float(np.mean([m["auc"] for m in fold_metrics])),
        "top_features": [(f, float(i)) for f, i in feature_importance[:10]],
        "n_samples":    int(len(X)),
        "n_features":   int(len(feature_cols)),
        "event_rate":   float(y.mean()),
    }

    with open("ml/models_15years/rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModelo guardado en ml/models_15years/")
    print(f"F1 medio:  {metrics['mean_f1']:.3f}")
    print(f"AUC medio: {metrics['mean_auc']:.3f}")

    return metrics


if __name__ == "__main__":
    df = load_features()
    metrics = train_random_forest(df)
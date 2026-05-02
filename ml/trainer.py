import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import shap
import joblib
import json


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_features(path="ml/data_todo/featuresAll.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Dataset: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────
def get_features(df):
    exclude = {
        "target",
        "event_heat", "event_cold",
        "event_rain", "event_wind",
        "event_extreme"
    }
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
def train_rf(df):

    print("\n── RANDOM FOREST (PRO) ─────────────────────")

    feature_cols = get_features(df)
    X = df[feature_cols].values
    y = df["target"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Muestras: {len(X)}")
    print(f"Event rate: {y.mean():.3f}")

    model_params = dict(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=5)

    folds = []
    oof_preds = np.zeros(len(X))

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        prob  = model.predict_proba(X_test)[:, 1]

        oof_preds[test_idx] = prob

        f1 = f1_score(y_test, preds, zero_division=0)

        auc = (
            roc_auc_score(y_test, prob)
            if len(np.unique(y_test)) > 1
            else 0.5
        )

        folds.append({"fold": i+1, "f1": f1, "auc": auc})

        print(f"Fold {i+1}: F1={f1:.3f} AUC={auc:.3f}")

    # ─────────────────────────────────────────────
    # FINAL MODEL
    # ─────────────────────────────────────────────
    print("\nEntrenando modelo final...")

    final_model = RandomForestClassifier(**model_params)
    final_model.fit(X, y)

    # ─────────────────────────────────────────────
    # SHAP (robusto)
    # ─────────────────────────────────────────────
    print("SHAP analysis...")

    explainer = shap.TreeExplainer(final_model)
    sample = X[:200]
    shap_values = explainer.shap_values(sample)

    # Handle all possible shapes:
    # - list of 2 arrays (old shap): take index 1 (positive class)
    # - 3D array (n_samples, n_features, n_classes): take [:, :, 1]
    # - 2D array (n_samples, n_features): use directly
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    importance = np.abs(sv).mean(axis=0)

    feature_importance = sorted(
        zip(feature_cols, importance.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop features:")
    for f, v in feature_importance[:10]:
        print(f"{f:30s} {v:.4f}")

    # ─────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────
    Path("ml/final_model").mkdir(exist_ok=True)

    joblib.dump(final_model, "ml/final_model/rf_model.pkl")
    joblib.dump(feature_cols, "ml/final_model/features.pkl")

    metrics = {
        "model": "rf_pro",
        "mean_f1": float(np.mean([f["f1"] for f in folds])),
        "mean_auc": float(np.mean([f["auc"] for f in folds])),
        "folds": folds,
        "top_features": [(f, float(v)) for f, v in feature_importance[:10]],
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "event_rate": float(y.mean())
    }

    with open("ml/final_model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✔ MODELO LISTO")
    print(f"F1 medio: {metrics['mean_f1']:.3f}")
    print(f"AUC medio: {metrics['mean_auc']:.3f}")

    return final_model, metrics


# ─────────────────────────────────────────────
# 4. RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_features()
    model, metrics = train_rf(df)
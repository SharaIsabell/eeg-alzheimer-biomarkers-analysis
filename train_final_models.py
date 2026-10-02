"""
Treina os modelos finais com TODOS os sujeitos do ds004504 e salva em disco.

O models_training.ipynb só faz validação LOSO e não persiste nenhum modelo, então
não há o que aplicar ao BrainLat. Este script reproduz exatamente as mesmas
configurações (RF, SVM linear, XGBoost x 7 conjuntos de features, média das
épocas por sujeito) e ajusta cada uma no conjunto completo.

Rode na pasta do projeto de treino, onde está eeg_features_baseline_data.csv:

    python train_final_models.py \
        --features eeg_features_baseline_data.csv \
        --out models/ds004504_models.joblib

ou, de um notebook:  from train_final_models import export_models
"""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Mesmos conjuntos do models_training.ipynb
FEATURE_SETS = {
    "Baseline (RBP)": ["Delta_RBP", "Theta_RBP", "Alpha_RBP", "Beta_RBP", "Gamma_RBP"],
    "Mean Frequency": ["Delta_Mean_Freq", "Theta_Mean_Freq", "Alpha_Mean_Freq",
                       "Beta_Mean_Freq", "Gamma_Mean_Freq"],
    "Lempel-Ziv Complexity": ["Delta_LZC", "Theta_LZC", "Alpha_LZC", "Beta_LZC", "Gamma_LZC"],
    "Phase-Amplitude Coupling": ["PAC_Delta_Beta", "PAC_Delta_Gamma", "PAC_Theta_Beta",
                                 "PAC_Theta_Gamma", "PAC_Alpha_Beta", "PAC_Alpha_Gamma"],
    "Kuramoto Order": ["Kuramoto_Delta", "Kuramoto_Theta", "Kuramoto_Alpha",
                       "Kuramoto_Beta", "Kuramoto_Gamma"],
    "Mutual Information": ["MI_Delta", "MI_Theta", "MI_Alpha", "MI_Beta", "MI_Gamma"],
}
FEATURE_SETS["Assinatura Completa (Multivariado)"] = [
    c for cols in list(FEATURE_SETS.values()) for c in cols
]
ALL_FEATURES = FEATURE_SETS["Assinatura Completa (Multivariado)"]

POSITIVE_GROUP = "A"   # Alzheimer
NEGATIVE_GROUP = "C"   # Controle


def make_pipeline(model_name: str, y: np.ndarray) -> Pipeline:
    """Mesmos hiperparâmetros do notebook de treino."""
    if model_name == "Random Forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=42, n_jobs=-1)),
        ])
    if model_name == "SVM (Linear)":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced",
                        random_state=42)),
        ])
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        neg, pos = np.sum(y == 0), np.sum(y == 1)
        return Pipeline([
            ("clf", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  scale_pos_weight=neg / pos if pos > 0 else 1.0,
                                  random_state=42, n_jobs=-1, eval_metric="logloss")),
        ])
    raise ValueError(model_name)


MODEL_NAMES = ("Random Forest", "SVM (Linear)", "XGBoost")


def subject_level(df: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame:
    """Média das épocas por sujeito, como em evaluate_feature."""
    return df.groupby(["Subject_ID", "Group"], as_index=False)[list(feature_cols)].mean()


def export_models(features_csv: str | Path, out_path: str | Path,
                  models=MODEL_NAMES, verbose: bool = True) -> dict:
    df = pd.read_csv(features_csv)
    df = df[df["Group"].isin([POSITIVE_GROUP, NEGATIVE_GROUP])]
    subj = subject_level(df)
    y = np.where(subj["Group"] == POSITIVE_GROUP, 1, 0)

    n_nan = int(subj[ALL_FEATURES].isna().sum().sum())
    if verbose:
        print(f"{len(subj)} sujeitos ({int(y.sum())} AD, {int((y == 0).sum())} CN), "
              f"{df.shape[0]} épocas, NaN no nível do sujeito: {n_nan}")

    fitted = {}
    for model_name in models:
        for cat, cols in FEATURE_SETS.items():
            pipe = make_pipeline(model_name, y)
            pipe.fit(subj[cols].values, y)
            fitted[f"{model_name}|{cat}"] = pipe
            if verbose:
                print(f"  ajustado: {model_name:14s} | {cat}")

    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__}
    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass

    bundle = {
        "models": fitted,
        "feature_sets": FEATURE_SETS,
        "train_subject_features": subj,      # usado no diagnóstico de domínio
        "metadata": {
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(Path(features_csv).resolve()),
            "dataset": "OpenNeuro ds004504 (eyes closed), grupos A e C",
            "n_subjects": int(len(subj)), "n_ad": int(y.sum()), "n_cn": int((y == 0).sum()),
            "label_encoding": {"1": "AD (A)", "0": "CN (C)"},
            "signal_space": {
                "channels": 19, "sfreq": 500.0, "reference": "A1+A2 (orelhas ligadas)",
                "filter": "IIR Butterworth ordem 4, 0.5-45 Hz", "asr_cutoff": 17,
                "ica": "infomax estendido, 19 comp., ICLabel exclui eye blink/muscle",
                "epochs": "4 s, overlap 2 s", "aggregation": "média das épocas por sujeito",
            },
            "versions": versions,
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    if verbose:
        print(f"\nsalvo em {out_path} ({len(fitted)} modelos)")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="eeg_features_baseline_data.csv")
    ap.add_argument("--out", default="models/ds004504_models.joblib")
    args = ap.parse_args()
    export_models(args.features, args.out)
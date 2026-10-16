"""
Treina os modelos finais com TODOS os sujeitos do ds004504 e salva em disco.

O models_training.ipynb só faz validação LOSO e não persiste nenhum modelo, então
não há o que aplicar ao BrainLat. Este script reproduz exatamente as mesmas
configurações (RF, SVM linear, XGBoost x 7 conjuntos de features, média das
épocas por sujeito) e ajusta cada uma no conjunto completo.

Rode na pasta do projeto de treino, onde está eeg_features_baseline_data.csv:

    python train_final_models.py --features eeg_features_ds004504.csv --out models/ds004504_models.joblib

ou, de um notebook:  from train_final_models import export_models
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

import modeling
from feature_sets import (FEATURE_VERSION, FEATURE_SETS, MODEL_NAMES, BANDS, ALL_FEATURES,
                          TARGET_REFERENCE, feature_tag)

DEFAULT_FEATURES = "eeg_features_ds004504.csv"
DEFAULT_OUT = "models/ds004504_models.joblib"


def export_models(features_csv: str | Path = DEFAULT_FEATURES, out_path: str | Path = DEFAULT_OUT,
                  models=MODEL_NAMES, verbose: bool = True) -> dict:
    meta_path = Path(features_csv).with_suffix(".meta.json")
    csv_tag = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_tag") if meta_path.exists() else None
    if csv_tag is not None and csv_tag != feature_tag():
        raise ValueError(f"{features_csv} foi gerado com {csv_tag}, mas a configuração atual é "
                         f"{feature_tag()}; regenere as features antes de treinar")
    subj = modeling.subject_table(pd.read_csv(features_csv))
    y = subj["y"].to_numpy()
    n_nan = int(subj[ALL_FEATURES].isna().sum().sum())
    if verbose:
        print(f"{len(subj)} sujeitos ({int(y.sum())} AD, {int((y == 0).sum())} CN); "
              f"valores ausentes no nível do sujeito: {n_nan}")

    fitted = {}
    for model in models:
        for cat, cols in FEATURE_SETS.items():
            fitted[f"{model}|{cat}"] = modeling.make_pipeline(model, y).fit(subj[cols].to_numpy(float), y)

    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__}
    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass

    bundle = {
        "models": fitted,
        "feature_sets": FEATURE_SETS,
        "train_subject_features": subj,
        "metadata": {
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "feature_version": FEATURE_VERSION,
            "feature_tag": feature_tag(),
            "reference": TARGET_REFERENCE,
            "bands": BANDS,
            "source": str(Path(features_csv).resolve()),
            "n_subjects": int(len(subj)), "n_ad": int(y.sum()), "n_cn": int((y == 0).sum()),
            "label_encoding": {"1": "AD (A)", "0": "CN (C)"},
            "score": "SVM: decision_function; RF e XGBoost: probabilidade de AD",
            "versions": versions,
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    if verbose:
        print(f"salvo em {out_path} ({len(fitted)} modelos, features {feature_tag()})")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    export_models(a.features, a.out)
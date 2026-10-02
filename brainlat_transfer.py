"""
Aplica os modelos treinados no ds004504 aos sujeitos do BrainLat.

Estrutura esperada (dentro da pasta do projeto):
    brainlat_eeg/1_AD/AR/sub-30001/...      brainlat_eeg/5_HC/AR/sub-10002/...
    brainlat_eeg/1_AD/CL/sub-30003/...      brainlat_eeg/5_HC/CL/sub-10001/...
Rótulos: Cognition_AD_EEG_data.csv e Cognition_HC_EEG_data.csv
(colunas 'path' = '<grupo>/<sítio>', 'id EEG', 'diagnosis' = AD | CN).

Por sujeito (com cache, porque o PAC por época é lento):
    arquivo -> channel_adaptation.adapt_to_training_space
            -> ds004504_features.clean_like_training
            -> ds004504_features.extract_epoch_features
            -> <cache>/<sid>_epochs.csv + <sid>_provenance.json
"""
from __future__ import annotations

import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

import channel_adaptation
import ds004504_features as feats

EEG_EXTS = (".set", ".bdf", ".edf", ".fif")
# Se houver mais de um arquivo EEG na pasta do sujeito, prefere o que tiver um
# destes termos como PALAVRA do nome (separada por _ - .), p.ex. "s6_sub-30001_rs-HEP_eeg.set"
PREFERRED_TERMS = ("rs", "rest", "resting", "eyesclosed", "closed")


# --------------------------------------------------------------------------
# Coorte: rótulos + localização dos arquivos
# --------------------------------------------------------------------------
def read_csv_any_encoding(path: str | Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"não consegui ler {path}")


def _pick_eeg_file(folder: Path) -> tuple[Path | None, list[Path]]:
    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in EEG_EXTS)
    if len(files) <= 1:
        return (files[0] if files else None), files
    for term in PREFERRED_TERMS:
        hits = [f for f in files if term in re.split(r"[_\-.]", f.stem.lower())]
        if len(hits) == 1:
            return hits[0], files
    return files[0], files


def build_cohort(brainlat_root: str | Path, label_csvs: list[str | Path],
                 positive=("AD",), negative=("CN",)) -> pd.DataFrame:
    """
    Uma linha por sujeito rotulado: sid, diagnosis, y, site, group_dir, folder,
    eeg_file, n_eeg_files, status ('ok' ou o motivo de exclusão).

    A pasta é encontrada por brainlat_root/*/<sítio>/<sid>, não pelo texto da
    coluna 'path' (o CSV dos controles diz 'HC/AR' e a pasta é '5_HC/AR').
    """
    root = Path(brainlat_root)
    labels = pd.concat([read_csv_any_encoding(p) for p in label_csvs], ignore_index=True)
    labels = labels.rename(columns={"id EEG": "sid"})
    labels = labels[labels["diagnosis"].isin(set(positive) | set(negative))].copy()

    rows = []
    for r in labels.itertuples(index=False):
        sid = str(r.sid).strip()
        site = str(r.path).strip().split("/")[-1]
        row = {"sid": sid, "diagnosis": r.diagnosis, "y": int(r.diagnosis in positive),
               "site": site, "group_dir": None, "folder": None, "eeg_file": None,
               "n_eeg_files": 0, "status": "ok"}
        folders = [f for f in root.glob(f"*/{site}/{sid}") if f.is_dir()]
        if len(folders) != 1:
            row["status"] = f"{len(folders)} pastas encontradas"
            rows.append(row)
            continue
        folder = folders[0]
        row["group_dir"], row["folder"] = folder.parent.parent.name, str(folder)
        expected = "AD" if row["y"] == 1 else "HC"
        if expected not in row["group_dir"].upper():
            row["status"] = f"pasta {row['group_dir']} não bate com diagnóstico {r.diagnosis}"
        eeg, files = _pick_eeg_file(folder)
        row["n_eeg_files"] = len(files)
        if eeg is None:
            row["status"] = "nenhum arquivo EEG"
        else:
            row["eeg_file"] = str(eeg)
        rows.append(row)
    cohort = pd.DataFrame(rows)

    labeled = set(cohort["sid"])
    on_disk = [p for g in ("*AD*", "*HC*") for p in root.glob(f"{g}/*/sub-*") if p.is_dir()]
    cohort.attrs["unlabeled_on_disk"] = sorted(p.name for p in on_disk if p.name not in labeled)
    return cohort


# --------------------------------------------------------------------------
# Leitura e processamento
# --------------------------------------------------------------------------
class EpochedFileError(RuntimeError):
    """O .set está epocado (dados já cortados em segmentos), não contínuo."""


def read_raw_any(path: str | Path) -> mne.io.BaseRaw:
    """
    Lê o registro contínuo. Os .set do BrainLat vêm sem .fdt (dados embutidos);
    o MNE precisa do pacote pymatreader para isso.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".set":
        try:
            return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        except (TypeError, ValueError) as exc:
            # O MNE recusa .set epocado em read_raw_eeglab
            if "epoch" in str(exc).lower() or "trials" in str(exc).lower():
                ep = mne.read_epochs_eeglab(path, verbose="ERROR")
                raise EpochedFileError(
                    f"{path.name} está epocado: {len(ep)} épocas de "
                    f"{ep.times[-1] - ep.times[0] + 1 / ep.info['sfreq']:.2f} s. "
                    "O treino usou registro contínuo; veja o passo 2 do notebook.") from exc
            raise
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if ext == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raise ValueError(f"formato não suportado: {path}")


NON_SCALP_PATTERN = channel_adaptation.NON_SCALP_PATTERN


def describe_recording(raw: mne.io.BaseRaw, verbose: bool = True) -> dict:
    """
    Diagnóstico do estado do arquivo ANTES de qualquer processamento.

    Responde: quantos canais e de que tipo; se há canais fora do escalpo
    (ECG/EXG); taxa e duração; filtros já aplicados segundo o cabeçalho;
    se o sinal já está em referência média; se há trechos removidos
    (eventos 'boundary' do EEGLAB); amplitude típica.
    """
    types = pd.Series(raw.get_channel_types()).value_counts().to_dict()
    eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    non_scalp = [c for c in raw.ch_names if NON_SCALP_PATTERN.match(c)]
    scalp = [c for c in eeg if c not in non_scalp]

    X = raw.get_data(picks=scalp)
    ch_std = X.std(axis=1)
    # Em referência média, a soma dos canais é ~0 a cada instante
    common = np.abs(X.mean(axis=0)).mean()
    typical = np.median(np.abs(X - X.mean(axis=1, keepdims=True)))
    avg_ref_ratio = float(common / typical) if typical > 0 else np.nan

    ann = pd.Series(raw.annotations.description).value_counts().to_dict() if len(raw.annotations) else {}
    n_boundary = sum(v for k, v in ann.items() if "boundary" in str(k).lower())
    info = {
        "n_channels": len(raw.ch_names), "types": types,
        "n_scalp_eeg": len(scalp), "non_scalp": non_scalp,
        "sfreq": float(raw.info["sfreq"]), "duration_s": float(raw.times[-1]),
        "highpass_hz": float(raw.info["highpass"]), "lowpass_hz": float(raw.info["lowpass"]),
        "avg_ref_ratio": round(avg_ref_ratio, 4),
        "looks_average_referenced": bool(avg_ref_ratio < 0.01),
        "median_std_uV": round(float(np.median(ch_std)) * 1e6, 2),
        "n_flat": int((ch_std < 1e-8).sum()),
        "bads_in_header": list(raw.info["bads"]),
        "annotations": ann, "n_boundary": int(n_boundary),
        "n_positions": len(channel_adaptation.positions_from_raw(raw)),
    }
    if verbose:
        print(f"  canais: {info['n_channels']} {types} | escalpo: {len(scalp)} | "
              f"fora do escalpo: {non_scalp or 'nenhum'}")
        print(f"  {info['sfreq']:.0f} Hz, {info['duration_s']:.0f} s "
              f"({info['duration_s'] / 60:.1f} min), posições válidas: {info['n_positions']}")
        print(f"  filtros no cabeçalho: passa-alta {info['highpass_hz']} Hz, "
              f"passa-baixa {info['lowpass_hz']} Hz")
        print(f"  referência média? {'sim' if info['looks_average_referenced'] else 'não'} "
              f"(razão {info['avg_ref_ratio']}) | amplitude mediana {info['median_std_uV']} µV | "
              f"planos: {info['n_flat']} | ruins no cabeçalho: {info['bads_in_header'] or 'nenhum'}")
        print(f"  anotações: {ann or 'nenhuma'}")
        for w in recording_warnings(info):
            print("  ATENÇÃO:", w)
    return info


def recording_warnings(info: dict) -> list[str]:
    w = []
    if info["lowpass_hz"] < 45:
        w.append(f"passa-baixa de {info['lowpass_hz']} Hz abaixo dos 45 Hz do treino: "
                 "as features da banda gama (25-45 Hz) ficam sistematicamente diferentes")
    if info["highpass_hz"] > 0.5:
        w.append(f"passa-alta de {info['highpass_hz']} Hz acima dos 0,5 Hz do treino: "
                 "a banda delta (0,5-4 Hz) fica atenuada")
    if info["n_boundary"]:
        w.append(f"{info['n_boundary']} eventos 'boundary': trechos já foram removidos e o sinal "
                 "tem emendas; épocas que cruzam emendas entram no cálculo")
    if info["median_std_uV"] < 1 or info["median_std_uV"] > 200:
        w.append(f"amplitude mediana {info['median_std_uV']} µV fora do usual: confira unidades")
    if info["duration_s"] < 120:
        w.append(f"registro curto ({info['duration_s']:.0f} s): poucas épocas por sujeito")
    return w


def process_subject(subject_id: str, path: str | Path, cache_dir: str | Path,
                    adapter_kwargs: dict | None = None, overwrite: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cache_dir / f"{subject_id}_epochs.csv"
    out_json = cache_dir / f"{subject_id}_provenance.json"
    if out_csv.exists() and not overwrite:
        return pd.read_csv(out_csv)

    t0 = time.time()
    raw = read_raw_any(path)
    recording = describe_recording(raw, verbose=False)
    raw19, prov = channel_adaptation.adapt_to_training_space(raw, verbose=verbose,
                                                        **(adapter_kwargs or {}))
    del raw
    raw_clean, clean_report = feats.clean_like_training(raw19)
    df = pd.DataFrame(feats.extract_epoch_features(raw_clean, subject_id))
    df.to_csv(out_csv, index=False)

    record = {"subject": subject_id, "source": str(path),
              "processed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "recording": recording, "adapter": prov, "cleaning": clean_report,
              "n_epochs": len(df),
              "seconds": round(time.time() - t0, 1)}
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"  {len(df)} épocas, ICs excluídos {clean_report['ic_excluded']}, "
              f"{record['seconds']:.0f} s")
    return df


def run(subject_paths: dict[str, str | Path], cache_dir: str | Path,
        adapter_kwargs: dict | None = None, overwrite: bool = False, verbose: bool = True):
    """Processa vários sujeitos; falhas são registradas e não interrompem o lote."""
    frames, failures = [], {}
    for i, (sid, path) in enumerate(subject_paths.items(), 1):
        if verbose:
            print(f"[{i}/{len(subject_paths)}] {sid}")
        try:
            frames.append(process_subject(sid, path, cache_dir, adapter_kwargs, overwrite, verbose))
        except Exception as exc:
            failures[sid] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  FALHOU: {failures[sid]}")
                traceback.print_exc(limit=1)
    epochs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return epochs, failures


def mapping_summary(cache_dir: str | Path) -> pd.DataFrame:
    """
    Lê os JSONs de proveniência e mostra, por sujeito, a fonte da geometria, o
    deslocamento e quais eletrodos foram usados. Serve para ver se todos os
    sujeitos receberam a mesma correspondência.
    """
    rows = []
    for pf in sorted(Path(cache_dir).glob("*_provenance.json")):
        with open(pf, encoding="utf-8") as fh:
            rec = json.load(fh)
        a = rec["adapter"]
        rows.append({"sid": rec["subject"], "geometry": a["geometry_source"],
                     "median_deg": round(a["median_angle_deg"], 2),
                     "max_deg": round(a["max_angle_deg"], 2),
                     "ref": "+".join(a["ref_channels"]),
                     "interpolated": ",".join(a["interpolated"]),
                     "orig_sfreq": a["orig_sfreq"],
                     "electrodes": " ".join(m["source"] for m in a["mapping"]),
                     "n_ic_excluded": len(rec["cleaning"]["ic_excluded"]),
                     "lowpass_hz": rec.get("recording", {}).get("lowpass_hz"),
                     "avg_ref_in_file": rec.get("recording", {}).get("looks_average_referenced"),
                     "n_boundary": rec.get("recording", {}).get("n_boundary"),
                     "n_epochs": rec["n_epochs"]})
    return pd.DataFrame(rows)


def subject_level(epochs: pd.DataFrame, feature_cols=feats.ALL_FEATURES) -> pd.DataFrame:
    return epochs.groupby("Subject_ID", as_index=False)[list(feature_cols)].mean()


# --------------------------------------------------------------------------
# Predição e métricas
# --------------------------------------------------------------------------
def predict_all(bundle: dict, subj: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, pipe in bundle["models"].items():
        model, cat = key.split("|", 1)
        X = subj[bundle["feature_sets"][cat]].values
        out.append(pd.DataFrame({"Subject_ID": subj["Subject_ID"].astype(str).values,
                                 "Modelo": model, "Categoria": cat,
                                 "pred": pipe.predict(X).astype(int),
                                 "prob_AD": pipe.predict_proba(X)[:, 1]}))
    return pd.concat(out, ignore_index=True)


def evaluate(preds: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """labels: Series indexada pelo ID do sujeito, 1 = AD, 0 = controle."""
    lab = pd.DataFrame({"Subject_ID": labels.index.astype(str), "y": labels.values.astype(int)})
    df = preds.merge(lab, on="Subject_ID", how="inner")
    rows = []
    for (model, cat), g in df.groupby(["Modelo", "Categoria"], sort=False):
        y, p, s = g["y"].values, g["pred"].values, g["prob_AD"].values
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rows.append({"Modelo": model, "Categoria": cat, "n": len(g),
                     "Accuracy": round(accuracy_score(y, p), 4),
                     "Sensibilidade": round(tp / (tp + fn), 4) if tp + fn else np.nan,
                     "Especificidade": round(tn / (tn + fp), 4) if tn + fp else np.nan,
                     "F1": round(f1_score(y, p, zero_division=0), 4),
                     "AUC": round(roc_auc_score(y, s), 4) if len(set(y)) == 2 else np.nan,
                     "frac_pred_AD": round(p.mean(), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Deslocamento de domínio
# --------------------------------------------------------------------------
def _cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def domain_shift_report(train_subj: pd.DataFrame, target_subj: pd.DataFrame,
                        target_labels: pd.Series, feature_cols=feats.ALL_FEATURES) -> pd.DataFrame:
    t = target_subj.assign(Subject_ID=target_subj["Subject_ID"].astype(str)).set_index("Subject_ID")
    lab = pd.Series(target_labels.values, index=target_labels.index.astype(str)).reindex(t.index)
    tr_ad = train_subj[train_subj["Group"] == "A"]
    tr_cn = train_subj[train_subj["Group"] == "C"]
    rows = []
    for c in feature_cols:
        d_shift = _cohen_d(t.loc[lab == 0, c], tr_cn[c])
        d_tr = _cohen_d(tr_ad[c], tr_cn[c])
        d_tg = _cohen_d(t.loc[lab == 1, c], t.loc[lab == 0, c])
        rows.append({"feature": c, "d_HC_shift": round(d_shift, 2),
                     "d_AD_train": round(d_tr, 2), "d_AD_target": round(d_tg, 2),
                     "same_sign": bool(np.sign(d_tr) == np.sign(d_tg))
                     if np.isfinite(d_tr) and np.isfinite(d_tg) else None})
    return pd.DataFrame(rows)


def align_to_training_controls(target_subj: pd.DataFrame, train_subj: pd.DataFrame,
                               calib_ids, feature_cols=feats.ALL_FEATURES) -> pd.DataFrame:
    """
    x' = (x - mu_HC_brainlat) / sd_HC_brainlat * sd_CN_treino + mu_CN_treino
    calculado só com os controles de calibração, que devem sair da avaliação.
    """
    cols = list(feature_cols)
    calib = target_subj[target_subj["Subject_ID"].astype(str).isin(set(map(str, calib_ids)))]
    if len(calib) < 5:
        raise ValueError(f"só {len(calib)} controles de calibração")
    mu_t, sd_t = calib[cols].mean(), calib[cols].std(ddof=1).replace(0, np.nan)
    cn = train_subj[train_subj["Group"] == "C"]
    out = target_subj.copy()
    out[cols] = (target_subj[cols] - mu_t) / sd_t * cn[cols].std(ddof=1) + cn[cols].mean()
    return out
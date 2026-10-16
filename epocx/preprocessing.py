"""
Ramo EPOC X. Pré-processamento do notebook preprocessingGT.ipynb (projeto mvp), com duas
mudanças pedidas:
  - limpeza de artefatos com wICA "suave 1.5 seletivo" (wica.py) no lugar do ICLabel;
  - referência na região das orelhas/mastoides (references.py, alvo "mastoid")
    no lugar de REST/CAR.

Serve às duas bases do projeto, com o mesmo código:
  ds004504  : 88 sujeitos, 19 canais, 500 Hz, rede de 50 Hz (Grécia)
  epocx     : Emotiv EPOC X, 14 canais, 128 Hz, rede de 60 Hz (Brasil)

Etapas, na ordem do notebook:
   1. canais: só o conjunto comum (feature_sets.CHANNELS), nomes 10-10, taxa comum
   2. duração mínima (gravações mais curtas são descartadas)
   3. canais defeituosos: variância < 1e-12 (morto) ou > 3x a mediana (ruído)
   4. mais de MAX_NOISY canais defeituosos -> sujeito descartado;
      senão, interpolação por splines esféricas
   5. filtro FIR de fase zero, 0,1-45 Hz
   6. notch na frequência da rede da base (só harmônicos abaixo de Nyquist)
   7. referência-alvo
   8. ICA (picard) ajustada numa cópia com passa-alta de 1 Hz;
      componentes = mínimo entre o posto teórico e o medido
   9. wICA suave 1.5 seletivo
  10. canais interpolados no passo 4 são interpolados de novo, agora a partir
      do sinal limpo (ver nota abaixo)
  11. épocas de 4 s a cada 2 s
  12. AutoReject (interpola 1 ou 2 canais por época ou rejeita a época)

Diferenças deliberadas em relação ao notebook:
  - não há canais "estruturais" interpolados (AF3, AF4, FC5, FC6): só os canais
    que existem nas duas bases são usados;
  - passo 10: no notebook, os canais interpolados no passo 4 continuavam com o
    conteúdo de antes da limpeza (a ICA não os inclui), levando artefatos para
    as features; aqui eles são reconstruídos a partir dos canais já limpos;
  - frequência da rede por base (o notebook usava 60 Hz para tudo).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from config_epocx import (CHANNELS, CHANNEL_ALIASES, TARGET_SFREQ, TARGET_REFERENCE,  # noqa: I001
                          FEATURE_VERSION, feature_tag)
import references          # raiz do repositório (config_epocx coloca a raiz no caminho)
import wica

# --------------------------------------------------------------------------
# Configuração por base
# --------------------------------------------------------------------------
# line_freq: usada se o _eeg.json (BIDS) não informar PowerLineFrequency.
# single_task: True se a base só tem uma tarefa (o identificador omite a tarefa).
# mastoid_channels: canais de orelha/mastoide gravados no arquivo, se existirem.
# Com eles, a referência é a média desses canais; sem eles, a referência de
# gravação (source_reference) precisa já estar na região das orelhas/mastoides.
DATASETS = {
    "ds004504": {"line_freq": 50.0, "source_reference": references.SOURCE_REFERENCES["ds004504"],
                 "mastoid_channels": None, "single_task": True},
    "epocx": {"line_freq": 60.0, "source_reference": references.SOURCE_REFERENCES["epocx"],
              "mastoid_channels": None, "single_task": False},
}

# --------------------------------------------------------------------------
# Parâmetros do notebook
# --------------------------------------------------------------------------
MIN_DURATION_S = 120.0       # gravações mais curtas são descartadas
CROP_TO_MIN = False          # True corta todas as gravações em MIN_DURATION_S
VAR_DEAD = 1e-12             # variância (V²) abaixo disso: canal morto
VAR_NOISY_FACTOR = 3.0       # variância acima de 3x a mediana: canal ruidoso
MAX_NOISY = 4                # acima disso o sujeito é descartado
FILTER_BAND = (0.1, 45.0)
ICA_HP = 1.0
ICA_SEED = 42
EPOCH_LEN = 4.0
EPOCH_STEP = 2.0
AR_N_INTERPOLATE = [1, 2]
AR_MAX_CV = 10


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------
def load_raw(path: str | Path) -> mne.io.BaseRaw:
    """Lê .set (ds004504) ou .edf (EPOC X); outros formatos dão erro."""
    path = str(path)
    if path.endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if path.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raise ValueError(f"formato não suportado: {path}")


def _normalized_names(names, wanted) -> dict:
    """
    Renomeia T3/T4/T5/T6 para T7/T8/P7/P8 e remove prefixos e sufixos comuns em
    exportações EDF ("EEG AF3", "AF3-REF"), só quando o resultado é um canal
    procurado e o nome original não é.
    """
    out = {}
    for c in names:
        if c in wanted:
            continue
        n = CHANNEL_ALIASES.get(c, c)
        if n not in wanted:
            n = re.sub(r"^(EEG[ _.-]*)", "", c, flags=re.IGNORECASE)
            n = re.sub(r"[-_ ](REF|Ref|ref|LE|AVG)$", "", n).strip()
            n = CHANNEL_ALIASES.get(n, n)
        if n in wanted and n not in names and n not in out.values():
            out[c] = n
    return out


def select_channels(raw: mne.io.BaseRaw, dataset: str, channels=None, target_sfreq=None,
                    target: str = TARGET_REFERENCE):
    """
    Só os canais do conjunto comum, com nomes 10-10, na ordem do conjunto, e a
    taxa comum. Canais extras do EPOC X (marcadores, qualidade de contato,
    bateria) e os do ds004504 que não existem no EPOC X ficam de fora.

    Se a base declarar canais de mastoide e o alvo for "mastoid", a referência é
    aplicada AQUI, antes de esses canais serem descartados. Referência, filtros
    e reamostragem são lineares, então antecipar a referência não muda o resultado.
    Devolve (raw, registro da referência ou None se ela fica para o passo 7).
    """
    channels = list(CHANNELS if channels is None else channels)
    target_sfreq = TARGET_SFREQ if target_sfreq is None else target_sfreq
    raw.rename_channels(_normalized_names(raw.ch_names, channels))
    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise ValueError(f"canais do conjunto ausentes no arquivo: {missing}")

    pre_ref = None
    mastoids = DATASETS[dataset].get("mastoid_channels")
    if target == "mastoid" and mastoids:
        raw.pick(channels + [c for c in mastoids if c not in channels])
        raw.set_channel_types({c: "eeg" for c in raw.ch_names}, verbose="ERROR")
        raw, pre_ref = references.apply_target_reference(raw, "mastoid", ref_channels=list(mastoids))

    raw.pick(channels)
    raw.reorder_channels(channels)
    raw.set_channel_types({c: "eeg" for c in channels}, verbose="ERROR")
    if target_sfreq and not np.isclose(raw.info["sfreq"], target_sfreq):
        if target_sfreq > raw.info["sfreq"]:
            raise ValueError(f"taxa de {raw.info['sfreq']} Hz abaixo da taxa comum de {target_sfreq} Hz")
        raw.resample(target_sfreq, verbose="ERROR")
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose="ERROR")
    return raw, pre_ref


def standardize_duration(raw: mne.io.BaseRaw, min_s: float = MIN_DURATION_S,
                         crop: bool = CROP_TO_MIN) -> mne.io.BaseRaw | None:
    """None se a gravação for mais curta que min_s; corta o excedente se crop=True."""
    n_min = round(min_s * raw.info["sfreq"])
    if raw.n_times < n_min:
        return None
    if crop and raw.n_times > n_min:
        raw.crop(tmin=0, tmax=(n_min - 1) / raw.info["sfreq"])
    return raw


# --------------------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------------------
def detect_noisy_channels(raw: mne.io.BaseRaw) -> list[str]:
    var = np.var(raw.get_data(), axis=1)
    med = np.median(var)
    idx = set(np.where(var < VAR_DEAD)[0]) | set(np.where(var > VAR_NOISY_FACTOR * med)[0])
    return sorted(raw.ch_names[i] for i in idx)


def preprocess(raw: mne.io.BaseRaw, dataset: str, target: str = TARGET_REFERENCE,
               pre_ref: dict | None = None, line_freq: float | None = None,
               prior_bads: list[str] | None = None, verbose: bool = True):
    """
    Etapas 3 a 12 sobre um registro já com os canais e a taxa comuns.
    Devolve (raw limpo contínuo, épocas finais, relatório) ou (None, None, relatório)
    se o sujeito for descartado.
    """
    cfg = DATASETS[dataset]
    rep = {"dataset": dataset, "channels": list(raw.ch_names), "sfreq": float(raw.info["sfreq"]),
           "duration_s": round(float(raw.times[-1]), 1)}

    # 3-4. canais defeituosos: detectados pela variância + marcados como ruins no BIDS
    detected = detect_noisy_channels(raw)
    marked = sorted(c for c in (prior_bads or []) if c in raw.ch_names)
    noisy = sorted(set(detected) | set(marked))
    rep["noisy_channels"] = noisy
    rep["noisy_detected"] = detected
    rep["noisy_marked_in_bids"] = marked
    if len(noisy) > MAX_NOISY:
        rep["status"] = f"descartado: {len(noisy)} canais defeituosos (máximo {MAX_NOISY})"
        if verbose:
            print("  " + rep["status"])
        return None, None, rep
    if noisy:
        raw.info["bads"] = list(noisy)
        raw.interpolate_bads(reset_bads=False, method="spline", verbose="ERROR")

    # 5-6. filtros
    raw.filter(l_freq=FILTER_BAND[0], h_freq=FILTER_BAND[1], method="fir", phase="zero",
               verbose="ERROR")
    nyq = raw.info["sfreq"] / 2
    lf = float(line_freq or cfg["line_freq"])
    line = [f for f in np.arange(lf, 3 * lf + 1, lf) if f < nyq]
    if line:
        raw.notch_filter(freqs=line, method="spectrum_fit", filter_length="10s", verbose="ERROR")
    rep["notch_hz"] = [float(f) for f in line]

    # 7. referência (já aplicada na leitura se a base tiver canais de mastoide)
    if pre_ref is None:
        raw, pre_ref = references.apply_target_reference(raw, target, source=cfg["source_reference"],
                                                         expected_channels=list(raw.ch_names))
    rep["reference"] = pre_ref

    # 8. ICA
    fit = raw.copy().filter(l_freq=ICA_HP, h_freq=None, verbose="ERROR")
    rank_loss = 1 if target in ("average", "rest") else 0
    n_theory = len(raw.ch_names) - len(raw.info["bads"]) - rank_loss
    n_rank = mne.compute_rank(fit, tol="auto", tol_kind="relative", verbose="ERROR").get("eeg", n_theory)
    n_comp = int(max(1, min(n_theory, n_rank)))
    ica = mne.preprocessing.ICA(n_components=n_comp, method="picard", random_state=ICA_SEED)
    ica.fit(fit, verbose="ERROR")
    rep.update({"ica_n_components": n_comp, "ica_n_theory": int(n_theory), "ica_rank": int(n_rank)})

    # 9. wICA
    clean, ic_table, wmetrics = wica.apply_wica(raw, ica)
    rep.update(wmetrics)
    rep["wica_ics"] = ic_table.round(4).to_dict(orient="records")

    # 10. canais interpolados reconstruídos a partir do sinal limpo
    clean.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose="ERROR")
    if noisy:
        clean.info["bads"] = list(noisy)
        clean.interpolate_bads(reset_bads=True, method="spline", verbose="ERROR")
    else:
        clean.info["bads"] = []

    # 11-12. épocas e AutoReject
    from autoreject import AutoReject
    events = mne.make_fixed_length_events(clean, id=1, start=0, duration=EPOCH_STEP)
    epochs = mne.Epochs(clean, events, tmin=0, tmax=EPOCH_LEN, baseline=None, preload=True,
                        verbose="ERROR")
    n_before = len(epochs)
    if n_before == 0:
        rep["status"] = "descartado: nenhuma época"
        return None, None, rep
    cv = min(AR_MAX_CV, n_before)
    if cv >= 2:
        ar = AutoReject(n_interpolate=AR_N_INTERPOLATE, random_state=ICA_SEED, cv=cv, verbose=False)
        epochs = ar.fit_transform(epochs)
    rep.update({"n_epochs_before_ar": int(n_before), "n_epochs": int(len(epochs)),
                "pct_epochs_rejected": round(100 * (1 - len(epochs) / n_before), 1),
                "status": "ok"})
    if verbose:
        print(f"  canais defeituosos {noisy or 'nenhum'} | ICA {n_comp} comp. | wICA tratou "
              f"{wmetrics['wica_n_ics_treated']}/{wmetrics['wica_n_ics']} ICs, "
              f"{wmetrics['wica_var_removed_mean_pct']:.1f}% da variância em média | "
              f"{len(epochs)}/{n_before} épocas mantidas")
    return clean, epochs, rep


# --------------------------------------------------------------------------
# Sujeito completo, com cache
# --------------------------------------------------------------------------
BIDS_ENTITY = re.compile(r"(sub|ses|task|run)-([A-Za-z0-9]+)")


def bids_entities(path) -> dict:
    """{'sub': '012', 'ses': '01', 'task': 'eyesclosed', 'run': '1'} a partir do nome do arquivo."""
    return dict(BIDS_ENTITY.findall(Path(path).name))


def subject_id_from_path(path) -> str:
    """Identificador do participante, p.ex. 'sub-012'."""
    e = bids_entities(path)
    return f"sub-{e['sub']}" if "sub" in e else Path(path).stem


def recording_id(path, dataset: str) -> str:
    """
    Identificador da gravação: participante, sessão, tarefa e run, p.ex.
    'sub-012_ses-01_task-eyesclosed_run-1'. Em bases de tarefa única a tarefa é
    omitida; no ds004504, que tem uma gravação por sujeito, fica só 'sub-001',
    como antes (o cache já gerado continua valendo).
    """
    e = bids_entities(path)
    if "sub" not in e:
        return Path(path).stem
    parts = [f"sub-{e['sub']}"]
    if "ses" in e:
        parts.append(f"ses-{e['ses']}")
    if "task" in e and not DATASETS[dataset].get("single_task", False):
        parts.append(f"task-{e['task']}")
    if "run" in e:
        parts.append(f"run-{e['run']}")
    return "_".join(parts)


SIDECAR_KEYS = ("SamplingFrequency", "PowerLineFrequency", "EEGReference", "EEGGround",
                "SoftwareFilters", "HardwareFilters", "RecordingDuration", "Manufacturer",
                "ManufacturersModelName", "CapManufacturer", "TaskName", "EEGChannelCount")


def read_bids_sidecar(path) -> dict:
    """
    Metadados BIDS da gravação: campos do _eeg.json e canais marcados como
    "bad" no _channels.tsv. Ausência dos arquivos não é erro.
    """
    path = Path(path)
    stem = path.name[: -len("_eeg" + path.suffix)] if path.name.endswith("_eeg" + path.suffix) else path.stem
    info = {"eeg_json": None, "bad_channels": []}
    js = path.with_name(stem + "_eeg.json")
    if js.exists():
        data = json.loads(js.read_text(encoding="utf-8"))
        info["eeg_json"] = {k: data[k] for k in SIDECAR_KEYS if k in data}
    ch = path.with_name(stem + "_channels.tsv")
    if ch.exists():
        t = pd.read_csv(ch, sep="\t")
        if {"name", "status"} <= set(t.columns):
            bad = t.loc[t["status"].astype(str).str.lower() == "bad", "name"].astype(str)
            info["bad_channels"] = sorted(_normalized_names(list(bad), CHANNELS).get(b, b) for b in bad)
    return info


def discover_bids(root, task: str | None = "eyesclosed", extensions=(".edf", ".set", ".bdf")) -> pd.DataFrame:
    """
    Todas as gravações de EEG de uma pasta BIDS (sub-*/[ses-*/]eeg/*_eeg.<ext>),
    uma linha por gravação, filtradas pela tarefa. task=None mantém todas.
    """
    rows = []
    for f in sorted(Path(root).rglob("*_eeg.*")):
        if f.suffix.lower() not in extensions or f.name.endswith(".md.edf"):
            continue
        e = bids_entities(f)
        if task is not None and e.get("task") != task:
            continue
        rows.append({"Subject_ID": subject_id_from_path(f), "session": e.get("ses"), "task": e.get("task"),
                     "run": e.get("run"), "Recording_ID": recording_id(f, "epocx"), "path": f,
                     "has_eeg_json": f.with_name(f.name.replace("_eeg" + f.suffix, "_eeg.json")).exists()})
    return pd.DataFrame(rows, columns=["Subject_ID", "session", "task", "run", "Recording_ID", "path", "has_eeg_json"])


def read_participants(root) -> pd.DataFrame | None:
    """participants.tsv da raiz BIDS, se existir (idade, sexo...)."""
    f = Path(root) / "participants.tsv"
    if not f.exists():
        return None
    t = pd.read_csv(f, sep="\t")
    return t.rename(columns={"participant_id": "Subject_ID"})


def process_subject(path, dataset: str, cache_dir: str | Path, subject_id: str | None = None,
                    overwrite: bool = False, verbose: bool = True) -> pd.DataFrame | None:
    """
    Uma gravação: leitura -> canais e taxa comuns -> duração -> preprocess -> features.
    Grava <gravação>_epochs.csv e <gravação>_qc.json em cache_dir. Devolve as
    features (colunas Subject_ID = participante, Recording_ID = gravação) ou None
    se a gravação foi descartada (o motivo fica no JSON).
    """
    import ds004504_features as feats

    rid = recording_id(path, dataset)
    pid = subject_id or subject_id_from_path(path)
    ents = bids_entities(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv, out_json = cache_dir / f"{rid}_epochs.csv", cache_dir / f"{rid}_qc.json"
    if out_json.exists() and not overwrite:
        return pd.read_csv(out_csv) if out_csv.exists() else None

    t0 = time.time()
    bids = read_bids_sidecar(path)
    sidecar_lf = (bids["eeg_json"] or {}).get("PowerLineFrequency")
    line_freq = float(sidecar_lf) if isinstance(sidecar_lf, (int, float)) and sidecar_lf > 0 else None
    qc = {"subject": pid, "recording": rid, "task": ents.get("task"), "run": ents.get("run"),
          "source": str(path), "feature_tag": feature_tag(), "feature_version": FEATURE_VERSION,
          "bids": bids, "line_freq_used": line_freq or DATASETS[dataset]["line_freq"],
          "processed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    df = None
    raw, pre_ref = select_channels(load_raw(path), dataset)
    raw = standardize_duration(raw)
    if raw is None:
        qc["preprocessing"] = {"status": f"descartado: gravação com menos de {MIN_DURATION_S:.0f} s"}
    else:
        clean, epochs, rep = preprocess(raw, dataset, pre_ref=pre_ref, line_freq=line_freq,
                                        prior_bads=bids["bad_channels"], verbose=verbose)
        qc["preprocessing"] = rep
        if epochs is not None:
            df = pd.DataFrame(feats.extract_features_from_epochs(epochs, clean, rid))
            df["Subject_ID"] = pid
            df.insert(1, "Recording_ID", rid)
            df.insert(2, "Run", ents.get("run"))
            df.to_csv(out_csv, index=False)
            qc["epochs"] = feats.epoch_qc(df, len(epochs.ch_names))
    qc["seconds"] = round(time.time() - t0, 1)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2, ensure_ascii=False, default=float)
    if verbose and df is None:
        print("  " + qc["preprocessing"]["status"])
    return df


def run(paths, dataset: str, cache_dir, verbose: bool = True) -> dict:
    """Processa vários arquivos; erros são registrados e não interrompem o lote."""
    errors = {}
    for i, p in enumerate(paths, 1):
        rid = recording_id(p, dataset)
        if verbose:
            print(f"[{i}/{len(paths)}] {rid}")
        try:
            process_subject(p, dataset, cache_dir, verbose=verbose)
        except Exception as exc:
            errors[rid] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print("  ERRO:", errors[rid])
    return errors


def qc_table(cache_dir) -> pd.DataFrame:
    """Uma linha por sujeito, a partir dos JSONs."""
    rows = []
    for f in sorted(Path(cache_dir).glob("*_qc.json")):
        q = json.load(open(f, encoding="utf-8"))
        p = q.get("preprocessing", {})
        rows.append({"Subject_ID": q["subject"], "Recording_ID": q.get("recording", q["subject"]),
                     "run": q.get("run"), "status": p.get("status"),
                     "duration_s": p.get("duration_s"),
                     "noisy_channels": ",".join(p.get("noisy_channels", [])),
                     "noisy_marked_in_bids": ",".join(p.get("noisy_marked_in_bids", [])),
                     "line_freq_used": q.get("line_freq_used"),
                     "eeg_reference_bids": ((q.get("bids") or {}).get("eeg_json") or {}).get("EEGReference"),
                     "ica_n_components": p.get("ica_n_components"),
                     "wica_n_ics_treated": p.get("wica_n_ics_treated"),
                     "wica_var_removed_mean_pct": p.get("wica_var_removed_mean_pct"),
                     "n_epochs_before_ar": p.get("n_epochs_before_ar"),
                     "n_epochs": p.get("n_epochs"),
                     "pct_epochs_rejected": p.get("pct_epochs_rejected"),
                     "nan_features": ",".join(q.get("epochs", {}).get("nan_features", []))})
    return pd.DataFrame(rows)
"""
Pré-processamento, extração de features e controle de qualidade.

Este módulo é o ÚNICO lugar onde o sinal é limpo e as features são calculadas.
O preprocessingData.ipynb (ds004504) e o brainlat_transfer.py (BrainLat) chamam
as mesmas funções, então as duas bases passam exatamente pelo mesmo código.

Versão das features: feature_sets.FEATURE_VERSION. As mudanças da versão 2.0 em
relação à 1.0 estão listadas em CORRECOES.md; as principais:
  - bandas em intervalos semiabertos (a RBP soma exatamente 1) e delta a partir de 1 Hz;
  - filtros passa-banda em seções de segunda ordem (sos), numericamente estáveis;
  - ICA ajustada numa cópia filtrada em 1-100 Hz, a faixa para a qual o ICLabel
    foi treinado, e aplicada ao sinal de 0,5-45 Hz;
  - PAC local calculado por sujeito sobre o registro contínuo, normalizado contra
    surrogates (escore z) e resumido pela mediana dos canais;
  - discretização da informação mútua com exatamente 10 faixas;
  - relatório de controle de qualidade por sujeito;
  - notch na rede elétrica (50 Hz) na cópia usada para ajustar a ICA.
Versão 2.1: todas as bases são levadas à referência-alvo comum
(feature_sets.TARGET_REFERENCE, padrão "average") por references.py, antes da limpeza.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import hilbert
import antropy as ant
from tensorpac import Pac
from sklearn.metrics import mutual_info_score
import mne
from mne_icalabel import label_components
from meegkit.asr import ASR

from feature_sets import (feature_tag, FEATURE_VERSION, BANDS, BROADBAND, PAC_PHASE_BANDS,
                          PAC_AMP_BANDS, ALL_FEATURES, FEATURE_SETS, PAC_COLS,
                          KURAMOTO_COLS, RBP_COLS, TARGET_REFERENCE,
                          CHANNELS, CHANNEL_ALIASES, TARGET_SFREQ)
import references

logging.getLogger("tensorpac").setLevel(logging.ERROR)


class _DropNPermWarning(logging.Filter):
    """
    O tensorpac avisa que testes de significância pedem >= 200 permutações. Aqui
    os surrogates servem só para normalizar o PAC em escore z, não para testar
    significância; o aviso não se aplica e escondia a saída útil do lote.
    """
    def filter(self, record):
        return "n_perm" not in record.getMessage()


logging.getLogger("tensorpac").addFilter(_DropNPermWarning())

# --------------------------------------------------------------------------
# Parâmetros do pré-processamento
# --------------------------------------------------------------------------
FILTER_BAND = (0.5, 45.0)       # sinal analisado (Miltiadous et al., 2023)
ICA_FIT_BAND = (1.0, 100.0)     # cópia usada só para ajustar a ICA e rodar o ICLabel
LINE_FREQ = 50.0                # rede elétrica (Grécia, Argentina e Chile); 60.0 nas Américas do Norte e Central
ASR_CUTOFF = 17
ICA_SEED = 42
EXCLUDED_IC_LABELS = ("eye blink", "muscle artifact")

EPOCH_DURATION = 4.0
EPOCH_OVERLAP = 2.0

# PAC: por sujeito, sobre até PAC_MAX_SECONDS do registro limpo (trecho central)
PAC_IDPAC = (2, 2, 4)           # KL/Tort; surrogates por troca de blocos de amplitude; escore z
PAC_N_PERM = 100
PAC_MAX_SECONDS = 300.0

FLAT_STD_VOLTS = 1e-7           # desvio-padrão abaixo disso = canal plano (0,1 µV)

DS004504_CHANNELS = 19


# --------------------------------------------------------------------------
# Leitura do ds004504
# --------------------------------------------------------------------------
def load_ds004504_raw(path: str | Path) -> mne.io.BaseRaw:
    """
    Lê um registro do ds004504.

    Os registros já foram adquiridos com referência à média de A1 e A2 (lóbulos
    das orelhas) e não contêm esses canais. A versão 1.0 tinha um bloco que
    adicionava A1/A2 zerados, referenciava a eles e os apagava: uma operação
    nula. Aqui não há re-referência; a referência é a da aquisição.
    """
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if any(c.upper() in ("A1", "A2") for c in raw.ch_names):
        raise ValueError(f"{Path(path).name}: canais A1/A2 presentes; este carregador "
                         "assume a referência de aquisição já aplicada")
    eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    if len(eeg) != DS004504_CHANNELS:
        raise ValueError(f"{Path(path).name}: {len(eeg)} canais EEG (esperava {DS004504_CHANNELS})")
    raw.pick(eeg)
    return raw


def prepare_ds004504_raw(path: str | Path, target: str = TARGET_REFERENCE):
    """
    Lê o registro e o leva à referência-alvo. A referência de origem do ds004504
    é a média de A1 e A2 (aquisição); com o alvo "average" ou "rest" ela não
    precisa ser conhecida, e com "linked_ears" ou "acquisition" o sinal é mantido.
    Devolve (raw, registro da referência).
    """
    raw = load_ds004504_raw(path)
    raw = select_channels(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False,
                    verbose="ERROR")
    raw, ref = references.apply_target_reference(
        raw, target, source=references.SOURCE_REFERENCES["ds004504"],
        expected_channels=list(CHANNELS))
    return raw, ref


def select_channels(raw: mne.io.BaseRaw, channels=None, target_sfreq=None) -> mne.io.BaseRaw:
    """
    Deixa só os canais do conjunto configurado (feature_sets.CHANNELS), com os
    nomes 10-10, na ordem do conjunto, e reamostra para a taxa comum.

    Vale para qualquer base: a seleção acontece ANTES da referência, então a
    referência média é sempre calculada sobre exatamente estes canais, em todas
    as bases. A reamostragem usa o filtro anti-aliasing do MNE.
    """
    channels = list(CHANNELS if channels is None else channels)
    target_sfreq = TARGET_SFREQ if target_sfreq is None else target_sfreq
    raw.rename_channels({k: v for k, v in CHANNEL_ALIASES.items() if k in raw.ch_names})
    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise ValueError(f"canais do conjunto ausentes no arquivo: {missing}")
    raw.pick(channels)
    raw.reorder_channels(channels)
    if target_sfreq and not np.isclose(raw.info["sfreq"], target_sfreq):
        if target_sfreq > raw.info["sfreq"]:
            raise ValueError(f"taxa de {raw.info['sfreq']} Hz abaixo da taxa comum de {target_sfreq} Hz")
        raw.resample(target_sfreq, verbose="ERROR")
    return raw


# --------------------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------------------
def _iir(raw, lo, hi):
    hi = min(hi, 0.45 * raw.info["sfreq"])
    raw.filter(l_freq=lo, h_freq=hi, method="iir",
               iir_params=dict(order=4, ftype="butter"), verbose="ERROR")
    return raw


def _full_rank_basis(x: np.ndarray, tol: float = 1e-10):
    """
    Base ortonormal do subespaço ocupado pelos dados. Em referência média (ou
    REST), os 19 canais têm posto 18: uma direção não tem variância nenhuma.
    Retorna None quando o posto já é completo.
    """
    xc = x - x.mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(xc @ xc.T / xc.shape[1])
    keep = evals > tol * evals.max()
    return None if keep.all() else evecs[:, keep]


ASR_CHANGE_REL = 0.01      # amostra "alterada": algum canal mudou mais de 1% do seu desvio-padrão
ASR_MAX_LAG = 100          # amostras procuradas no teste de atraso (0,2 s a 500 Hz)


def asr_change_stats(x: np.ndarray, y: np.ndarray, sfreq: float) -> dict:
    """
    Quanto o ASR mudou o sinal, em medidas que separam três situações:
      - ruído numérico: variância removida ~0 e nenhuma amostra acima de 1%;
      - atraso no tempo: lag diferente de 0 (o sinal foi devolvido deslocado);
      - reconstrução real: amostras alteradas acima de 1% concentradas em trechos.
    """
    d = y - x
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = np.inf
    changed = np.any(np.abs(d) > ASR_CHANGE_REL * sd, axis=0)

    # atraso: correlação cruzada no canal de maior variância, num trecho de até 60 s
    ch = int(np.argmax(x.var(axis=1)))
    n = int(min(x.shape[1], 60 * sfreq))
    a = x[ch, :n] - x[ch, :n].mean()
    b = y[ch, :n] - y[ch, :n].mean()
    lags = np.arange(-ASR_MAX_LAG, ASR_MAX_LAG + 1)
    cc = [np.dot(a[max(0, -L):n - max(0, L)], b[max(0, L):n - max(0, -L)]) for L in lags]

    runs = np.diff(np.r_[0, changed.astype(int), 0])
    return {
        "asr_frac_samples_changed": round(float(changed.mean()), 4),
        "asr_n_segments_changed": int((runs == 1).sum()),
        "asr_variance_removed": float(np.sum(d ** 2) / np.sum((x - x.mean(axis=1, keepdims=True)) ** 2)),
        "asr_max_abs_change_uV": round(float(np.abs(d).max()) * 1e6, 4),
        "asr_lag_samples": int(lags[int(np.argmax(cc))]),
    }


def _asr(raw):
    """
    ASR (cutoff 17). Se os dados tiverem posto incompleto, o ASR roda no
    subespaço de posto completo e o resultado volta aos canais. A projeção não
    perde informação (os dados estão inteiros nesse subespaço), evita a
    componente de variância zero, que tornaria os limiares do ASR indefinidos,
    e não muda o resultado: o ASR depende só da estrutura de covariância, não
    da base em que os canais são escritos.
    Devolve (raw, estatísticas da alteração; ver asr_change_stats).
    """
    x = raw.get_data()
    basis = _full_rank_basis(x)
    z = x if basis is None else basis.T @ x
    asr = ASR(method="euclid", cutoff=ASR_CUTOFF, sfreq=raw.info["sfreq"])
    asr.fit(z)
    zc = asr.transform(z)
    y = zc if basis is None else basis @ zc
    if not np.all(np.isfinite(y)):
        raise RuntimeError("ASR produziu valores inválidos")
    stats = asr_change_stats(x, y, raw.info["sfreq"])
    raw._data = y
    return raw, stats


def clean_like_training(raw: mne.io.BaseRaw, random_state: int = ICA_SEED):
    """
    Limpa um registro de 19 canais JÁ na referência-alvo (references.py). Os
    filtros e o ASR não alteram a referência, e a ICA é ajustada no mesmo espaço
    de canais em que é aplicada.

    1. Sinal analisado: filtro IIR Butterworth ordem 4, 0,5-45 Hz; montagem
       standard_1020; ASR (cutoff 17).
    2. Cópia para a ICA: mesmo filtro em 1-100 Hz e ASR próprio. A ICA (infomax
       estendido) é ajustada nessa cópia e o ICLabel classifica os componentes
       nela, que é a faixa para a qual o ICLabel foi treinado.
    3. A solução da ICA (mesmo espaço de canais) é aplicada ao sinal analisado,
       removendo os componentes rotulados como piscada ou músculo.

    O ICLabel foi treinado em dados de 1-100 Hz e em referência média. Com o alvo
    padrão ("average") as duas condições são atendidas; com outros alvos, a
    referência fica fora da especificação, e o relatório registra isso.

    Em referência média os 19 canais têm posto 18, e a ICA usa 18 componentes.
    """
    montage = mne.channels.make_standard_montage("standard_1020")

    main = _iir(raw.copy(), *FILTER_BAND)
    main.set_montage(montage, match_case=False, verbose="ERROR")
    main, asr_stats = _asr(main)

    ica_copy = _iir(raw.copy(), *ICA_FIT_BAND)
    # Notch na rede elétrica e harmônicos dentro da faixa: sem isso, a ICA gasta
    # componentes separando variantes do ruído de rede (o sinal analisado vai só
    # até 45 Hz, mas a cópia da ICA vai até 100 Hz).
    line = np.arange(LINE_FREQ, min(ICA_FIT_BAND[1], 0.45 * raw.info["sfreq"]) + 1, LINE_FREQ)
    if len(line):
        ica_copy.notch_filter(line, verbose="ERROR")
    ica_copy.set_montage(montage, match_case=False, verbose="ERROR")
    ica_copy, _ = _asr(ica_copy)

    rank = mne.compute_rank(ica_copy, rank=None, verbose="ERROR").get("eeg", len(raw.ch_names))
    n_components = int(min(len(raw.ch_names), rank))
    ica = mne.preprocessing.ICA(n_components=n_components, method="infomax",
                                fit_params=dict(extended=True), random_state=random_state)
    ica.fit(ica_copy, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = label_components(ica_copy, ica, method="iclabel")

    exclude = [i for i, lab in enumerate(labels["labels"]) if lab in EXCLUDED_IC_LABELS]
    ica.exclude = exclude
    cleaned = ica.apply(main.copy(), verbose="ERROR")

    std = cleaned.get_data().std(axis=1)
    report = {
        "n_ica_components": n_components,
        "ic_labels": list(labels["labels"]),
        "ic_probabilities": [round(float(p), 3) for p in labels["y_pred_proba"]],
        "ic_excluded": exclude,
        "n_ic_excluded": len(exclude),
        "iclabel_in_spec": bool(references.detect_reference(ica_copy)["looks_average"]),
        **asr_stats,
        "flat_channels": [c for c, s in zip(cleaned.ch_names, std) if s < FLAT_STD_VOLTS],
        "median_channel_std_uV": round(float(np.median(std)) * 1e6, 3),
        "duration_s": round(float(cleaned.times[-1]), 1),
    }
    return cleaned, report


def diagnose_asr(path: str | Path, plot: bool = True) -> dict:
    """
    Repete só a leitura, a referência, o filtro e o ASR de um sujeito do
    ds004504 (segundos a poucos minutos) e mostra o que o ASR mudou. Não altera
    o cache nem as features.
    """
    raw, _ = prepare_ds004504_raw(path)
    main = _iir(raw.copy(), *FILTER_BAND)
    x = main.get_data().copy()
    main, stats = _asr(main)
    if plot:
        import matplotlib.pyplot as plt
        y = main.get_data()
        sd = x.std(axis=1, keepdims=True)
        rel = np.max(np.abs(y - x) / sd, axis=0)
        t = main.times
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, rel, lw=0.6)
        ax.axhline(ASR_CHANGE_REL, color="0.5", ls="--", lw=1, label="1% do desvio-padrão")
        ax.set_yscale("log")
        ax.set_xlabel("tempo (s)")
        ax.set_ylabel("maior mudança entre canais\n(em desvios-padrão)")
        ax.set_title(f"{Path(path).name}: {stats['asr_frac_samples_changed']:.1%} das amostras, "
                     f"{stats['asr_variance_removed']:.2%} da variância, atraso {stats['asr_lag_samples']}")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()
    return stats


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------
def _band_mask(freqs, lo, hi):
    """Intervalo semiaberto [lo, hi)."""
    return (freqs >= lo) & (freqs < hi)


def _bandpass(data, sfreq, lo, hi):
    sos = signal.butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


# --------------------------------------------------------------------------
# Features por época
# --------------------------------------------------------------------------
def compute_baseline_rbp(data, sfreq):
    """Potência relativa de cada banda sobre BROADBAND; média dos canais. Soma = 1."""
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(2 * sfreq))
    total = psd[:, _band_mask(freqs, *BROADBAND)].sum(axis=1)
    return {f"{b}_RBP": float(np.mean(psd[:, _band_mask(freqs, lo, hi)].sum(axis=1) / total))
            for b, (lo, hi) in BANDS.items()}


def compute_mean_frequency_per_band(data, sfreq):
    """
    Centroide espectral DENTRO de cada banda (frequência média ponderada pela
    potência), média dos canais. Não é uma medida de lentificação entre bandas.
    """
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(2 * sfreq))
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = _band_mask(freqs, lo, hi)
        p = psd[:, m]
        s = p.sum(axis=1)
        s[s == 0] = 1e-30
        out[f"{b}_Mean_Freq"] = float(np.mean((freqs[m] * p).sum(axis=1) / s))
    return out


def compute_lzc_per_band(data, sfreq):
    """
    Lempel-Ziv normalizado do sinal filtrado em banda estreita e binarizado pela
    mediana; média dos canais. Em banda estreita, reflete principalmente a
    frequência dominante dentro da banda (ver README).
    """
    out = {}
    for b, (lo, hi) in BANDS.items():
        x = _bandpass(data, sfreq, lo, hi)
        vals = [ant.lziv_complexity((ch > np.median(ch)).astype(int), normalize=True) for ch in x]
        out[f"{b}_LZC"] = float(np.mean(vals))
    return out


def compute_kuramoto_per_band(data, sfreq):
    """
    Parâmetro de ordem de Kuramoto entre os canais, médio no tempo. Em espaço de
    sensor e com referência comum, é sensível à condução de volume.
    """
    out = {}
    for b, (lo, hi) in BANDS.items():
        x = _bandpass(data, sfreq, lo, hi)
        ph = np.angle(hilbert(x, axis=1))
        out[f"Kuramoto_{b}"] = float(np.mean(np.abs(np.mean(np.exp(1j * ph), axis=0))))
    return out


def compute_mutual_information_per_band(data, sfreq, n_bins: int = 10):
    """
    Informação mútua média entre os 171 pares de canais, com cada sinal
    discretizado em exatamente n_bins faixas. Sem correção para condução de volume.
    """
    n = data.shape[0]
    out = {}
    for b, (lo, hi) in BANDS.items():
        x = _bandpass(data, sfreq, lo, hi)
        disc = np.array([np.digitize(ch, np.histogram_bin_edges(ch, bins=n_bins)[1:-1]) for ch in x])
        vals = [mutual_info_score(disc[i], disc[j]) for i in range(n) for j in range(i + 1, n)]
        out[f"MI_{b}"] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------
# PAC local por sujeito
# --------------------------------------------------------------------------
def compute_pac_local_subject(raw_cleaned: mne.io.BaseRaw) -> dict:
    """
    Acoplamento fase-amplitude DENTRO de cada canal (não é conectividade entre
    eletrodos), calculado uma vez por sujeito.

    - Até PAC_MAX_SECONDS do trecho central do registro limpo. Épocas de 4 s são
      curtas demais para estimar PAC com fase delta: na versão 1.0 os valores
      ficavam no piso do estimador.
    - tensorpac idpac=(2, 2, 4): índice de modulação de Tort (Kullback-Leibler),
      surrogates por troca de blocos de amplitude, normalização em escore z.
      O valor indica quantos desvios-padrão o acoplamento observado está acima
      do esperado sem acoplamento.
    - Resumo pela mediana dos 19 canais, robusta a um canal com artefato (a
      versão 1.0 usava a média, e um único sujeito ficava 1.500 vezes acima dos
      demais).
    """
    sf = raw_cleaned.info["sfreq"]
    x = raw_cleaned.get_data()
    n = x.shape[1]
    keep = int(min(n, PAC_MAX_SECONDS * sf))
    start = (n - keep) // 2
    x = x[:, start:start + keep]

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for p_name in PAC_PHASE_BANDS:
            for a_name in PAC_AMP_BANDS:
                p = Pac(idpac=PAC_IDPAC, f_pha=list(BANDS[p_name]), f_amp=list(BANDS[a_name]),
                        dcomplex="hilbert", verbose=False)
                try:
                    vals = p.filterfit(sf, x, n_perm=PAC_N_PERM, n_jobs=1,
                                       random_state=ICA_SEED, verbose=False)
                except TypeError:   # versões antigas do tensorpac sem random_state
                    vals = p.filterfit(sf, x, n_perm=PAC_N_PERM, n_jobs=1, verbose=False)
                out[f"PAC_{p_name}_{a_name}"] = float(np.nanmedian(np.ravel(vals)))
    return out


# --------------------------------------------------------------------------
# Extração completa
# --------------------------------------------------------------------------
def _features_from_arrays(data: np.ndarray, sf: float, pac: dict, subject_id: str,
                          epoch_ids=None) -> list[dict]:
    rows = []
    for i, ep in enumerate(data):
        eid = epoch_ids[i] if epoch_ids is not None else i
        rows.append({
            "Subject_ID": subject_id,
            "Epoch_ID": f"{subject_id}_ep{int(eid):04d}",
            **compute_baseline_rbp(ep, sf),
            **compute_mean_frequency_per_band(ep, sf),
            **compute_lzc_per_band(ep, sf),
            **pac,
            **compute_kuramoto_per_band(ep, sf),
            **compute_mutual_information_per_band(ep, sf),
        })
    return rows


def extract_features_from_epochs(epochs: mne.BaseEpochs, raw_continuous: mne.io.BaseRaw,
                                 subject_id: str) -> list[dict]:
    """
    Features de épocas já prontas (as que sobraram do AutoReject, no
    pré-processamento de preprocessing.py). Uma linha por época mantida; o
    Epoch_ID guarda a posição original da época, para mostrar quais foram
    rejeitadas.

    O PAC é estimado sobre o registro contínuo limpo (antes da divisão em
    épocas): juntar épocas sobrepostas duplicaria trechos e criaria emendas que
    distorcem a filtragem.
    """
    data = epochs.get_data(copy=True)
    sf = epochs.info["sfreq"]
    pac = compute_pac_local_subject(raw_continuous)
    return _features_from_arrays(data, sf, pac, subject_id, epoch_ids=epochs.selection)


def extract_epoch_features(raw_cleaned: mne.io.BaseRaw, subject_id: str) -> list[dict]:
    """
    Pipeline anterior (versão 2.x, usada pelo BrainLat): épocas de 4 s com 2 s de
    sobreposição cortadas aqui mesmo, sem AutoReject.
    """
    epochs = mne.make_fixed_length_epochs(raw_cleaned, duration=EPOCH_DURATION,
                                          overlap=EPOCH_OVERLAP, preload=True, verbose="ERROR")
    data = epochs.get_data(copy=True)
    pac = compute_pac_local_subject(raw_cleaned)
    return _features_from_arrays(data, raw_cleaned.info["sfreq"], pac, subject_id)


def epoch_qc(df: pd.DataFrame, n_channels: int | None = None) -> dict:
    """Verificações sobre a tabela de épocas de um sujeito (n_channels: canais usados)."""
    n_channels = n_channels or len(CHANNELS)
    rbp_sum = df[RBP_COLS].sum(axis=1)
    kur = df[KURAMOTO_COLS].to_numpy()
    return {
        "n_epochs": int(len(df)),
        "rbp_sum_max_abs_dev": float(np.max(np.abs(rbp_sum - 1.0))),
        "kuramoto_degenerate_epochs": int(np.any(np.isclose(kur, 1.0 / n_channels, atol=1e-6), axis=1).sum()),
        "nan_features": sorted(c for c in ALL_FEATURES if df[c].isna().any()),
    }


# --------------------------------------------------------------------------
# Lote do ds004504 (com cache)
# --------------------------------------------------------------------------
def process_ds004504_subject(subject_id: str, path: str | Path, cache_dir: str | Path,
                             overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cache_dir / f"{subject_id}_epochs.csv"
    out_json = cache_dir / f"{subject_id}_qc.json"
    if out_csv.exists() and out_json.exists() and not overwrite:
        return pd.read_csv(out_csv)

    t0 = time.time()
    raw, ref = prepare_ds004504_raw(path)
    cleaned, rep = clean_like_training(raw)
    df = pd.DataFrame(extract_epoch_features(cleaned, subject_id))
    df.to_csv(out_csv, index=False)

    qc = {"subject": subject_id, "source": str(path), "feature_version": FEATURE_VERSION, "feature_tag": feature_tag(),
          "processed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "channels": list(cleaned.ch_names), "sfreq": float(cleaned.info["sfreq"]),
          "reference": ref, "cleaning": rep, "epochs": epoch_qc(df, len(cleaned.ch_names)),
          "seconds": round(time.time() - t0, 1)}
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"  {len(df)} épocas, {rep['n_ic_excluded']} ICs removidos, "
              f"ASR: {rep['asr_frac_samples_changed']:.1%} das amostras alteradas (>1%), "
              f"{rep['asr_variance_removed']:.2%} da variância, atraso {rep['asr_lag_samples']}, "
              f"{qc['seconds']:.0f} s")
    return df


def qc_table(cache_dir: str | Path) -> pd.DataFrame:
    """Uma linha por sujeito a partir dos JSONs de QC."""
    rows = []
    for f in sorted(Path(cache_dir).glob("*_qc.json")):
        q = json.load(open(f, encoding="utf-8"))
        c, e = q["cleaning"], q["epochs"]
        rows.append({"Subject_ID": q["subject"], "n_epochs": e["n_epochs"],
                     "reference": q.get("reference", {}).get("target"),
                     "n_ica_components": c["n_ica_components"],
                     "n_ic_excluded": c["n_ic_excluded"],
                     "excluded_labels": ",".join(c["ic_labels"][i] for i in c["ic_excluded"]),
                     # sujeitos processados antes desta versão só têm a medida antiga
                     "asr_frac_changed": c.get("asr_frac_samples_changed"),
                     "asr_variance_removed": c.get("asr_variance_removed"),
                     "asr_lag_samples": c.get("asr_lag_samples"),
                     "asr_frac_modified_old": c.get("asr_fraction_samples_modified"),
                     "flat_channels": ",".join(c["flat_channels"]),
                     "median_channel_std_uV": c["median_channel_std_uV"],
                     "rbp_sum_max_abs_dev": e["rbp_sum_max_abs_dev"],
                     "kuramoto_degenerate_epochs": e["kuramoto_degenerate_epochs"],
                     "nan_features": ",".join(e["nan_features"]),
                     "duration_s": c["duration_s"]})
    return pd.DataFrame(rows)
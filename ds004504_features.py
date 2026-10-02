"""
Pré-processamento (filtro -> montagem -> ASR -> ICA/ICLabel), epocagem e as
36 features do notebook preprocessingData, empacotados para serem aplicados a
qualquer Raw de 19 canais já no espaço do treino (ver channel_adaptation.py).

As funções de feature são cópias literais do notebook. Não altere os corpos:
qualquer diferença aqui vira deslocamento de domínio entre treino e BrainLat.

A única etapa do notebook que NÃO está aqui é a re-referência A1/A2: no BrainLat
os rótulos Biosemi 'A1'/'A2' são eletrodos do escalpo (A1 fica perto de Cz), e
rodar aquele bloco referenciaria o sinal a eles. A referência é feita em
channel_adaptation.adapt_to_training_space. Filtro e re-referência são lineares e
comutam, então a ordem não altera o resultado.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert
import antropy as ant
from tensorpac import Pac
from sklearn.metrics import mutual_info_score
import mne
from mne_icalabel import label_components
from meegkit.asr import ASR

logging.getLogger("tensorpac").setLevel(logging.ERROR)

EPOCH_DURATION = 4.0
EPOCH_OVERLAP = 2.0
EXCLUDED_IC_LABELS = ("eye blink", "muscle artifact")

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
    c for k in ("Baseline (RBP)", "Mean Frequency", "Lempel-Ziv Complexity",
                "Phase-Amplitude Coupling", "Kuramoto Order", "Mutual Information")
    for c in FEATURE_SETS[k]
]
ALL_FEATURES = FEATURE_SETS["Assinatura Completa (Multivariado)"]


# --------------------------------------------------------------------------
# Pré-processamento (idêntico ao notebook, exceto a re-referência)
# --------------------------------------------------------------------------
def clean_like_training(raw: mne.io.BaseRaw, random_state: int = 42):
    """Filtro IIR 0.5-45 Hz -> montagem 10-20 -> ASR (cutoff 17) -> ICA infomax + ICLabel."""
    raw = raw.copy()

    raw.filter(l_freq=0.5, h_freq=45.0, method="iir",
               iir_params=dict(order=4, ftype="butter"), verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False,
                    verbose="ERROR")

    sfreq = raw.info["sfreq"]
    asr = ASR(method="euclid", cutoff=17, sfreq=sfreq)
    raw_data = raw.get_data()
    _, sample_mask = asr.fit(raw_data)
    raw._data = asr.transform(raw_data)

    # Com referência A1+A2 (ou proxy de mastoide) o posto é 19, igual ao treino.
    # Com referência média o posto cai para 18 e o ICA precisa de 18 componentes.
    rank = mne.compute_rank(raw, rank=None, verbose="ERROR").get("eeg", len(raw.ch_names))
    n_components = int(min(19, rank))

    ica = mne.preprocessing.ICA(n_components=n_components, method="infomax",
                                fit_params=dict(extended=True), random_state=random_state)
    ica.fit(raw, verbose="ERROR")
    ic_labels = label_components(raw, ica, method="iclabel")
    exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"])
                   if label in EXCLUDED_IC_LABELS]
    ica.exclude = exclude_idx
    raw_cleaned = ica.apply(raw.copy(), verbose="ERROR")

    report = {
        "n_ica_components": n_components,
        "ic_labels": list(ic_labels["labels"]),
        "ic_excluded": exclude_idx,
    }
    return raw_cleaned, report


# --------------------------------------------------------------------------
# Features (cópia literal do notebook preprocessingData)
# --------------------------------------------------------------------------
def compute_mean_frequency_per_band(data, sfreq):
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(2*sfreq))
    bands = {
        'Delta_Mean_Freq': (0.5, 4.0),
        'Theta_Mean_Freq': (4.0, 8.0),
        'Alpha_Mean_Freq': (8.0, 13.0),
        'Beta_Mean_Freq': (13.0, 25.0),
        'Gamma_Mean_Freq': (25.0, 45.0)
    }
    mean_freq_features = {}
    for band_name, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        freqs_band = freqs[idx_band]
        psd_band = psd[:, idx_band]
        sum_psd = np.sum(psd_band, axis=1)
        sum_psd[sum_psd == 0] = 1e-10
        mean_freqs_channels = np.sum(freqs_band * psd_band, axis=1) / sum_psd
        mean_freq_features[band_name] = np.mean(mean_freqs_channels)
    return mean_freq_features


def compute_lzc_per_band(data, sfreq):
    bands = {
        'Delta_LZC': (0.5, 4.0),
        'Theta_LZC': (4.0, 8.0),
        'Alpha_LZC': (8.0, 13.0),
        'Beta_LZC': (13.0, 25.0),
        'Gamma_LZC': (25.0, 45.0)
    }
    lzc_features = {}
    for band_name, (fmin, fmax) in bands.items():
        nyq = 0.5 * sfreq
        low = fmin / nyq
        high = fmax / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        lzc_values = []
        for ch_data in filtered_data:
            binarized = np.where(ch_data > np.median(ch_data), 1, 0)
            lzc = ant.lziv_complexity(binarized, normalize=True)
            lzc_values.append(lzc)
        lzc_features[band_name] = np.mean(lzc_values)
    return lzc_features


def compute_pac_per_pair(data, sfreq):
    phase_bands = {
        'Delta': [0.5, 4.0],
        'Theta': [4.0, 8.0],
        'Alpha': [8.0, 13.0]
    }
    amp_bands = {
        'Beta': [13.0, 25.0],
        'Gamma': [25.0, 45.0]
    }
    pac_features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for p_name, f_pha in phase_bands.items():
            for a_name, f_amp in amp_bands.items():
                feature_name = f"PAC_{p_name}_{a_name}"
                p = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp)
                pac_matrix = p.filterfit(sfreq, data)
                pac_features[feature_name] = np.nanmean(pac_matrix)
    return pac_features


def compute_kuramoto_per_band(data, sfreq):
    bands = {
        'Kuramoto_Delta': (0.5, 4.0),
        'Kuramoto_Theta': (4.0, 8.0),
        'Kuramoto_Alpha': (8.0, 13.0),
        'Kuramoto_Beta': (13.0, 25.0),
        'Kuramoto_Gamma': (25.0, 45.0)
    }
    kuramoto_features = {}
    for band_name, (fmin, fmax) in bands.items():
        nyq = 0.5 * sfreq
        low = fmin / nyq
        high = fmax / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        analytic_signal = hilbert(filtered_data, axis=1)
        instantaneous_phases = np.angle(analytic_signal)
        complex_phasors = np.exp(1j * instantaneous_phases)
        mean_phasor_time = np.abs(np.mean(complex_phasors, axis=0))
        kuramoto_features[band_name] = np.mean(mean_phasor_time)
    return kuramoto_features


def compute_mutual_information_per_band(data, sfreq):
    bands = {
        'MI_Delta': (0.5, 4.0),
        'MI_Theta': (4.0, 8.0),
        'MI_Alpha': (8.0, 13.0),
        'MI_Beta': (13.0, 25.0),
        'MI_Gamma': (25.0, 45.0)
    }
    n_channels = data.shape[0]

    def discretize(sig, bins=10):
        return np.digitize(sig, np.histogram_bin_edges(sig, bins=bins))

    mi_features = {}
    for band_name, (fmin, fmax) in bands.items():
        nyq = 0.5 * sfreq
        low = fmin / nyq
        high = fmax / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        discrete_data = np.apply_along_axis(discretize, 1, filtered_data)
        mi_values = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                mi = mutual_info_score(discrete_data[i], discrete_data[j])
                mi_values.append(mi)
        mi_features[band_name] = np.mean(mi_values)
    return mi_features


def compute_baseline_rbp(data, sfreq):
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(2*sfreq))
    idx_total = np.logical_and(freqs >= 0.5, freqs <= 45.0)
    total_power = np.sum(psd[:, idx_total], axis=1)
    bands = {
        'Delta_RBP': (0.5, 4.0),
        'Theta_RBP': (4.0, 8.0),
        'Alpha_RBP': (8.0, 13.0),
        'Beta_RBP': (13.0, 25.0),
        'Gamma_RBP': (25.0, 45.0)
    }
    rbp_features = {}
    for band_name, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.sum(psd[:, idx_band], axis=1)
        relative_power = band_power / total_power
        rbp_features[band_name] = np.mean(relative_power)
    return rbp_features


# --------------------------------------------------------------------------
# Epocagem + extração (mesmo esquema do loop de 65 sujeitos)
# --------------------------------------------------------------------------
def extract_epoch_features(raw_cleaned: mne.io.BaseRaw, subject_id: str) -> list[dict]:
    epochs = mne.make_fixed_length_epochs(raw_cleaned, duration=EPOCH_DURATION,
                                          overlap=EPOCH_OVERLAP, preload=True, verbose="ERROR")
    epochs_data = epochs.get_data(copy=True)
    sfreq = raw_cleaned.info["sfreq"]

    rows = []
    for epoch_idx, epoch_data in enumerate(epochs_data):
        rows.append({
            "Subject_ID": subject_id,
            "Epoch_ID": f"{subject_id}_ep{epoch_idx:04d}",
            **compute_mean_frequency_per_band(epoch_data, sfreq),
            **compute_lzc_per_band(epoch_data, sfreq),
            **compute_pac_per_pair(epoch_data, sfreq),
            **compute_kuramoto_per_band(epoch_data, sfreq),
            **compute_mutual_information_per_band(epoch_data, sfreq),
            **compute_baseline_rbp(epoch_data, sfreq),
        })
    return rows
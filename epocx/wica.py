"""
wICA (wavelet-enhanced ICA; Castellanos & Makarov, 2006) para remoção de artefatos.

Portado de comparacao_ica.py (projeto mvp), com a variante escolhida nos testes
como padrão: "suave 1.5 seletivo".

Como funciona, em vez de excluir componentes inteiros (como o ICLabel):
  1. a ICA separa o sinal em componentes independentes (ICs);
  2. só as ICs com curtose (excesso) acima de KURTOSIS_MIN são tratadas: picos
     raros e grandes, típicos de piscadas e artefatos, dão curtose alta; ICs com
     distribuição quase gaussiana, típicas de atividade cerebral, ficam intactas
     ("seletivo");
  3. em cada IC tratada, a transformada wavelet separa o sinal em sub-bandas, e
     em cada sub-banda é removido só o EXCESSO acima de um limiar ("suave"): o
     coeficiente é encurtado até o limiar, não zerado;
  4. o limiar é 1,5 x o limiar universal de Donoho, sqrt(2 ln N) x sigma, com
     sigma estimado pela mediana do desvio absoluto (mais brando que o original);
  5. a parte removida de cada IC volta ao espaço dos canais pela matriz de
     mistura e é subtraída dos dados.

Nos testes com o EPOC X, essa variante removeu bem menos variância que o wICA
original e preservou mais o sinal (correlação média com o sinal de base mais alta).

Dependência: pywavelets (pip install pywavelets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import mne

WAVELET = "coif5"
LEVEL = 8
THRESHOLD_MULT = 1.5      # "1.5"
SOFT = True               # "suave"
KURTOSIS_MIN = 8.0        # "seletivo": só ICs com curtose (excesso) > 8
MAX_PROJECTION_ERROR = 1e-2


def artifact_part(signal: np.ndarray, wavelet: str = WAVELET, level: int = LEVEL,
                  threshold_mult: float = THRESHOLD_MULT, soft: bool = SOFT) -> np.ndarray:
    """
    Parte "artefato" de uma IC: o que excede o limiar em cada sub-banda wavelet.

    Limiar por sub-banda: threshold_mult * sqrt(2 ln N) * sigma, sigma = MAD / 0,6745.
    soft=True : só o excesso acima do limiar é considerado artefato.
    soft=False: coeficientes acima do limiar são considerados artefato inteiros (original).
    """
    import pywt

    n = signal.size
    coefs = pywt.wavedec(signal, wavelet, level=level, mode="symmetric")
    factor = np.sqrt(2 * np.log(n))
    out = []
    for c in coefs:
        sigma = np.median(np.abs(c - np.median(c))) / 0.6745
        thr = threshold_mult * factor * sigma
        if soft:
            out.append(np.sign(c) * np.maximum(np.abs(c) - thr, 0.0))
        else:
            out.append(np.where(np.abs(c) > thr, c, 0.0))
    return pywt.waverec(out, wavelet, mode="symmetric")[:n]


def mixing_by_regression(ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw):
    """
    Fontes das ICs e matriz de mistura (canais do ICA x ICs) por mínimos
    quadrados: Xc ~= V @ Sc. Não depende dos detalhes internos do MNE
    (branqueamento, PCA). O erro relativo de reconstrução deve ser ~0 quando a
    ICA cobre o espaço dos dados (n_components = posto).
    """
    sources = ica.get_sources(raw).get_data()
    picks = [raw.ch_names.index(ch) for ch in ica.ch_names]
    X = raw.get_data()[picks]
    Xc = X - X.mean(axis=1, keepdims=True)
    Sc = sources - sources.mean(axis=1, keepdims=True)
    V = np.linalg.lstsq(Sc.T, Xc.T, rcond=None)[0].T
    error = float(np.linalg.norm(Xc - V @ Sc) / np.linalg.norm(Xc))
    return sources, picks, V, error


def apply_wica(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, wavelet: str = WAVELET,
               level: int = LEVEL, threshold_mult: float = THRESHOLD_MULT, soft: bool = SOFT,
               kurtosis_min: float | None = KURTOSIS_MIN):
    """
    Remove de `raw` a parte artefato das ICs selecionadas. Nenhuma IC é excluída
    inteira. Devolve (raw limpo, tabela por IC, métricas).
    """
    import pywt
    from scipy.stats import kurtosis

    sources, picks, V, err = mixing_by_regression(ica, raw)
    if err > MAX_PROJECTION_ERROR:
        raise RuntimeError(f"a ICA não reconstrói os dados (erro relativo {err:.2e})")

    level_ef = min(level, pywt.dwt_max_level(sources.shape[1], pywt.Wavelet(wavelet).dec_len))
    kurt = kurtosis(sources, axis=1)
    treat = np.ones(len(sources), bool) if kurtosis_min is None else kurt > kurtosis_min
    art = np.vstack([artifact_part(s, wavelet, level_ef, threshold_mult, soft) if t
                     else np.zeros_like(s) for s, t in zip(sources, treat)])

    data = raw.get_data()
    data[picks] -= V @ art
    cleaned = mne.io.RawArray(data, raw.info.copy(), verbose=False)

    var_removed = 100 * art.var(axis=1) / sources.var(axis=1)
    table = pd.DataFrame({"ic": range(len(sources)), "curtose": kurt, "tratada": treat,
                          "var_removida_pct": var_removed})
    metrics = {"wavelet": wavelet, "wica_level": int(level_ef), "wica_threshold_mult": threshold_mult,
               "wica_soft": soft, "wica_kurtosis_min": kurtosis_min,
               "wica_n_ics": int(len(sources)), "wica_n_ics_treated": int(treat.sum()),
               "wica_var_removed_mean_pct": float(var_removed.mean()),
               "wica_var_removed_max_pct": float(var_removed.max()),
               "wica_projection_error": err}
    return cleaned, table, metrics

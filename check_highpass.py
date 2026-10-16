"""
Verifica se os arquivos do BrainLat passaram por um filtro passa-alta antes da
distribuição, e estima a frequência de corte.

Duas evidências independentes:

1. eeglab_history(path)
   Lê o campo EEG.history do .set, onde o EEGLAB registra os comandos aplicados
   (p.ex. "pop_eegfiltnew(EEG, 'locutoff', 1)"). Evidência direta, quando existe.

2. compare_lowfreq_spectra(...)
   Compara o espectro de 0,1 a 10 Hz dos controles do BrainLat com o dos
   controles do ds004504, nas mesmas condições:
     - os mesmos 19 canais (BrainLat passa por adapt_to_training_space, que só
       seleciona canais, reamostra e re-referencia; nada é filtrado);
     - referência média dos 19 nas duas bases, para que a diferença de
       referência não altere o formato do espectro;
     - cada espectro normalizado pela potência média em 2-4 Hz, para que a
       diferença de amplitude entre equipamentos não conte.
   A diferença (BrainLat - ds004504, em dB) por frequência funciona como a
   resposta do filtro desconhecido: perto de 0 dB acima do corte, caindo abaixo.

Uso no notebook (depois do passo 1):
    import check_highpass as chk
    chk.eeglab_history(paths[sid0])
    res = chk.compare_lowfreq_spectra(brainlat_hc_paths, ds004504_hc_paths, ORDER, OUT_DIR)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
import mne

import brainlat_transfer
import channel_adaptation


# --------------------------------------------------------------------------
# 1. Histórico do EEGLAB
# --------------------------------------------------------------------------
def eeglab_history(path: str | Path, verbose: bool = True) -> dict:
    """
    Lê EEG.history (e o que houver em EEG.etc) de um .set sem carregar os dados.
    Requer pymatreader (já necessário para o MNE ler os .set do BrainLat).
    """
    from pymatreader import read_mat

    path = str(path)
    try:
        d = read_mat(path, ignore_fields=["data"])
    except Exception:
        d = read_mat(path)
    eeg = d.get("EEG", d)

    hist = eeg.get("history", "")
    if isinstance(hist, (list, tuple, np.ndarray)):
        hist = "\n".join(str(h) for h in np.ravel(hist))
    hist = str(hist).strip()

    filter_lines = [ln.strip() for ln in hist.splitlines()
                    if any(k in ln.lower() for k in ("filt", "locutoff", "hicutoff", "highpass",
                                                     "detrend", "cleanline", "clean_rawdata"))]
    out = {
        "history": hist,
        "filter_lines": filter_lines,
        "etc_keys": sorted(eeg.get("etc", {}).keys()) if isinstance(eeg.get("etc"), dict) else [],
        "setname": str(eeg.get("setname", "")),
        "comments": str(eeg.get("comments", ""))[:500],
    }
    if verbose:
        print(f"arquivo : {Path(path).name}")
        print(f"setname : {out['setname'] or '(vazio)'}")
        print(f"EEG.etc : {out['etc_keys'] or '(vazio)'}")
        if hist:
            print("\nEEG.history:")
            print(hist)
        else:
            print("\nEEG.history está vazio: o arquivo não registra o processamento aplicado.")
        if filter_lines:
            print("\nlinhas que mencionam filtragem:")
            for ln in filter_lines:
                print("  ", ln)
    return out


# --------------------------------------------------------------------------
# 2. Espectro de baixa frequência
# --------------------------------------------------------------------------
SEGMENT_S = 20.0           # janelas de 20 s -> resolução de 0,05 Hz
NORM_BAND = (2.0, 4.0)     # faixa usada para normalizar cada espectro
FMIN, FMAX = 0.1, 10.0


def _psd_19ch(raw19: mne.io.BaseRaw):
    """PSD mediana dos 19 canais, em referência média, normalizada em 2-4 Hz."""
    r = raw19.copy().set_eeg_reference("average", projection=False, verbose="ERROR")
    x = r.get_data()
    x = x - x.mean(axis=1, keepdims=True)
    fs = r.info["sfreq"]
    nper = int(min(SEGMENT_S * fs, x.shape[1]))
    f, p = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="constant")
    p = np.median(p, axis=0)
    norm = p[(f >= NORM_BAND[0]) & (f <= NORM_BAND[1])].mean()
    keep = (f >= FMIN) & (f <= FMAX)
    return f[keep], p[keep] / norm


def _band_ratio(f, p, a=(0.5, 1.0), b=(1.0, 2.0)) -> float:
    """Potência média em 0,5-1 Hz sobre 1-2 Hz. Um passa-alta em ~1 Hz derruba essa razão."""
    pa = p[(f >= a[0]) & (f < a[1])].mean()
    pb = p[(f >= b[0]) & (f < b[1])].mean()
    return float(pa / pb)


def _crossing(f, db, level):
    """Maior frequência (vindo de cima para baixo) em que a curva cruza `level` dB."""
    above = db > level
    for i in range(len(f) - 1, 0, -1):
        if above[i] and not above[i - 1]:
            f0, f1, d0, d1 = f[i - 1], f[i], db[i - 1], db[i]
            return float(f0 + (level - d0) * (f1 - f0) / (d1 - d0))
    return float("nan")


def compare_lowfreq_spectra(brainlat_paths: dict[str, str | Path],
                            ds004504_paths: dict[str, str | Path],
                            order: list[str],
                            out_dir: str | Path | None = None,
                            adapter_kwargs: dict | None = None,
                            verbose: bool = True) -> dict:
    """
    brainlat_paths : {id: .set} de CONTROLES do BrainLat
    ds004504_paths : {id: .set} de CONTROLES do ds004504 (grupo C)
    Controles nas duas bases, para que o excesso de delta do Alzheimer não
    entre na comparação.
    """
    kw = dict(reference="average", order=order)
    kw.update(adapter_kwargs or {})

    spectra = {"BrainLat": [], "ds004504": []}
    ratios = []
    freqs = None

    skipped = {}
    for sid, p in brainlat_paths.items():
        try:
            raw = brainlat_transfer.read_raw_any(p)
            raw19, _ = channel_adaptation.adapt_to_training_space(raw, **kw)
            f, s = _psd_19ch(raw19)
        except Exception as exc:
            skipped[sid] = f"{type(exc).__name__}: {str(exc)[:120]}"
            if verbose:
                print(f"BrainLat  {sid}: PULADO ({skipped[sid]})")
            continue
        freqs = f if freqs is None else freqs
        spectra["BrainLat"].append(np.interp(freqs, f, s))
        ratios.append({"base": "BrainLat", "sid": sid, "ratio_05_1_over_1_2": _band_ratio(f, s)})
        if verbose:
            print(f"BrainLat  {sid}: razão 0,5-1 / 1-2 Hz = {ratios[-1]['ratio_05_1_over_1_2']:.2f}")

    for sid, p in ds004504_paths.items():
        try:
            raw = mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR")
            raw.pick([c for c in order if c in raw.ch_names])
            raw.reorder_channels(order)
            f, s = _psd_19ch(raw)
        except Exception as exc:
            skipped[sid] = f"{type(exc).__name__}: {str(exc)[:120]}"
            if verbose:
                print(f"ds004504  {sid}: PULADO ({skipped[sid]})")
            continue
        if freqs is None:
            freqs = f
        spectra["ds004504"].append(np.interp(freqs, f, s))
        ratios.append({"base": "ds004504", "sid": sid, "ratio_05_1_over_1_2": _band_ratio(f, s)})
        if verbose:
            print(f"ds004504  {sid}: razão 0,5-1 / 1-2 Hz = {ratios[-1]['ratio_05_1_over_1_2']:.2f}")

    for k, v in spectra.items():
        if len(v) < 3:
            raise RuntimeError(f"só {len(v)} espectros válidos para {k}; pulados: {skipped}")
    S = {k: np.array(v) for k, v in spectra.items()}
    med_db = {k: 10 * np.log10(np.median(v, axis=0)) for k, v in S.items()}
    diff_db = med_db["BrainLat"] - med_db["ds004504"]
    ref = diff_db[(freqs >= NORM_BAND[0]) & (freqs <= NORM_BAND[1])].mean()
    diff_db = diff_db - ref

    summary = {
        "freq_minus3dB": _crossing(freqs, diff_db, -3.0),
        "freq_minus6dB": _crossing(freqs, diff_db, -6.0),
        "diff_at_0p5Hz_dB": float(np.interp(0.5, freqs, diff_db)),
        "diff_at_1Hz_dB": float(np.interp(1.0, freqs, diff_db)),
        "diff_at_2Hz_dB": float(np.interp(2.0, freqs, diff_db)),
    }
    ratio_df = pd.DataFrame(ratios)

    if verbose:
        print("\nDiferença BrainLat - ds004504 (mediana dos controles, normalizada em 2-4 Hz):")
        print(f"  em 2 Hz  : {summary['diff_at_2Hz_dB']:+.1f} dB")
        print(f"  em 1 Hz  : {summary['diff_at_1Hz_dB']:+.1f} dB")
        print(f"  em 0,5 Hz: {summary['diff_at_0p5Hz_dB']:+.1f} dB")
        print(f"  cruza -3 dB em ~{summary['freq_minus3dB']:.2f} Hz e -6 dB em ~{summary['freq_minus6dB']:.2f} Hz")
        print("\nRazão de potência 0,5-1 Hz / 1-2 Hz por base (mediana [mín, máx]):")
        for base, g in ratio_df.groupby("base"):
            r = g["ratio_05_1_over_1_2"]
            print(f"  {base:9s}: {r.median():.2f} [{r.min():.2f}, {r.max():.2f}]  (n={len(r)})")

    if out_dir is not None:
        _plot(freqs, S, diff_db, summary, Path(out_dir))
        ratio_df.to_csv(Path(out_dir) / "passa_alta_razoes.csv", index=False)

    return {"freqs": freqs, "spectra": S, "diff_db": diff_db,
            "summary": summary, "ratios": ratio_df, "skipped": skipped}


def _plot(freqs, S, diff_db, summary, out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"ds004504": "0.45", "BrainLat": "#1f77b4"}
    for k, v in S.items():
        db = 10 * np.log10(v)
        a1.plot(freqs, np.median(db, axis=0), color=colors[k], lw=2, label=f"{k} (n={len(v)})")
        a1.fill_between(freqs, np.percentile(db, 25, axis=0), np.percentile(db, 75, axis=0),
                        color=colors[k], alpha=0.2, lw=0)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xlim(FMIN, FMAX)
        for fx in (0.5, 1.0):
            a.axvline(fx, color="0.6", ls="--", lw=1)
        a.set_xlabel("frequência (Hz)")
        a.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
        a.set_xticklabels(["0,1", "0,2", "0,5", "1", "2", "5", "10"])
    a1.set_ylabel("potência relativa a 2-4 Hz (dB)")
    a1.set_title("Espectro dos controles (mediana e intervalo interquartil)")
    a1.legend(frameon=False)

    a2.plot(freqs, diff_db, color="#1f77b4", lw=2)
    a2.axhline(0, color="0.3", lw=1)
    for lvl in (-3, -6):
        a2.axhline(lvl, color="0.6", ls=":", lw=1)
        a2.text(FMAX * 0.95, lvl + 0.4, f"{lvl} dB", ha="right", fontsize=9, color="0.4")
    a2.set_ylabel("BrainLat - ds004504 (dB)")
    a2.set_title(f"Resposta aparente do filtro: -3 dB em ~{summary['freq_minus3dB']:.2f} Hz, "
                 f"-6 dB em ~{summary['freq_minus6dB']:.2f} Hz", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "passa_alta_espectros.png", dpi=150, bbox_inches="tight")
    plt.show()
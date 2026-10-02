"""
Adaptação de um registro BrainLat (touca Biosemi, 128 canais) para o espaço de
sinal em que os modelos foram treinados (OpenNeuro ds004504):
19 canais 10-20, referência nas orelhas (A1+A2), 500 Hz.

Etapas de adapt_to_training_space, em ordem:
  1. Geometria: obtém a posição 3D de cada eletrodo do arquivo (ou, se o
     arquivo não tiver posições, do modelo padrão 'biosemi128' do MNE).
  2. Correspondência: para cada um dos 19 canais 10-20, escolhe o eletrodo do
     arquivo mais próximo da posição 10-20 ideal (compute_19ch_mapping).
  3. Referência: escolhe os eletrodos mais próximos de TP9/TP10 (mastoides)
     como substitutos das orelhas.
  4. Canais ruins: interpola, a partir de todos os canais, qualquer eletrodo
     escolhido que esteja marcado como ruim ou plano.
  5. Seleção, remoção de offset DC, reamostragem para 500 Hz.
  6. Re-referência, renomeação para os nomes do treino, ordem do treino,
     montagem standard_1020.

Filtro, ASR, ICA e features ficam em ds004504_features.py.
"""
from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import mne

# Ordem dos canais nos arquivos .set do ds004504 (sem A1/A2). Para confirmar,
# use training_channel_order() com um arquivo do treino.
DS004504_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
# Os mesmos 19 em nomenclatura 10-10, que é como aparecem em standard_1005
NOMINAL_19 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
              "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
LEGACY_NAMES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

# Canais fora do escalpo (ECG, eletrodos externos, gatilho). Nunca entram na
# geometria nem na escolha dos 19; EXG só pode virar referência via "explicit".
NON_SCALP_PATTERN = re.compile(r"^(EXG|ECG|EKG|EOG|EMG|HEOG|VEOG|STATUS|STI|TRIG|GSR|RESP|ERG)",
                               re.IGNORECASE)

TRAINING_SFREQ = 500.0
TEMPLATE_MONTAGE = "biosemi128"
NOMINAL_MONTAGE = "standard_1005"
MASTOID_PROXIES = ("TP9", "TP10")

FLAT_STD_VOLTS = 1e-8       # desvio-padrão abaixo disso = canal plano
HEAD_RADIUS_MM = 95.0       # só para converter ângulo em mm aproximados
MIN_POSITIONS = 32          # menos que isso no arquivo: usa o modelo padrão
MAX_MEDIAN_DEG = 12.0       # deslocamento mediano máximo aceitável (check_geometry)
MAX_TEMPLATE_DEG = 20.0     # divergência máxima arquivo x modelo padrão (check_orientation)


# --------------------------------------------------------------------------
# Espaço de treino
# --------------------------------------------------------------------------
def training_channel_order(reference_set: str | Path | None = None) -> list[str]:
    """Ordem dos 19 canais do treino, lida de um .set do ds004504 se fornecido."""
    if reference_set is not None and Path(reference_set).exists():
        raw = mne.io.read_raw_eeglab(reference_set, preload=False, verbose="ERROR")
        names = [c for c in raw.ch_names if c.upper() not in ("A1", "A2")]
        if len(names) != 19:
            raise ValueError(f"esperava 19 canais no arquivo de treino, achei {len(names)}")
        return names
    return list(DS004504_ORDER)


def _to_training_name(nominal: str, order: list[str]) -> str:
    name = LEGACY_NAMES.get(nominal, nominal)
    lookup = {c.lower(): c for c in order}
    return lookup[name.lower()]


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def template_positions(name: str = TEMPLATE_MONTAGE) -> dict[str, np.ndarray]:
    mont = mne.channels.make_standard_montage(name)
    return {k: np.asarray(v, float) for k, v in mont.get_positions()["ch_pos"].items()}


def positions_from_raw(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    """
    Posições 3D dos canais EEG do escalpo (só as válidas: finitas e não nulas).
    Canais cujo nome indica ECG/EXG/etc. ficam de fora mesmo que tenham posição.
    """
    out = {}
    for ch in raw.info["chs"]:
        if ch["kind"] != mne.io.constants.FIFF.FIFFV_EEG_CH or NON_SCALP_PATTERN.match(ch["ch_name"]):
            continue
        xyz = np.asarray(ch["loc"][:3], float)
        if np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 0:
            out[ch["ch_name"]] = xyz
    return out


def directions(pos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Ajusta uma esfera às posições (mínimos quadrados), subtrai o centro e
    devolve a direção radial unitária de cada eletrodo.

    Esfera: |p - c|^2 = r^2  <=>  2 p.c + (r^2 - |c|^2) = |p|^2, que é linear
    em (c, r^2 - |c|^2). A direção não depende do raio nem das unidades.
    """
    names = list(pos)
    P = np.array([pos[n] for n in names])
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    U = P - sol[:3]
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return dict(zip(names, U))


def _angles(src_dirs: dict, targets: list[str], nominal: str = NOMINAL_MONTAGE):
    nom = directions(template_positions(nominal))
    names = list(src_dirs)
    S = np.array([src_dirs[n] for n in names])
    T = np.array([nom[t] for t in targets])
    return names, np.degrees(np.arccos(np.clip(T @ S.T, -1.0, 1.0)))   # (n_targets, n_src)


def compute_19ch_mapping(source_pos: dict[str, np.ndarray],
                         order: list[str] | None = None) -> pd.DataFrame:
    """
    Para cada canal 10-20 do treino, o eletrodo de `source_pos` mais próximo.

    Distância = ângulo entre direções radiais (após a normalização esférica de
    cada montagem). Atribuição gulosa sem repetição: ordena todos os pares
    (canal 10-20, eletrodo) pelo ângulo e fixa do menor para o maior, pulando
    canais já resolvidos e eletrodos já usados.
    """
    order = order or list(DS004504_ORDER)
    names, ang = _angles(directions(source_pos), NOMINAL_19)
    assigned, used = {}, set()
    for flat in np.argsort(ang, axis=None):
        i, j = np.unravel_index(flat, ang.shape)
        if i in assigned or j in used:
            continue
        assigned[i] = j
        used.add(j)
        if len(assigned) == len(NOMINAL_19):
            break
    rows = []
    for i, nm in enumerate(NOMINAL_19):
        j = assigned[i]
        rows.append({"training_name": _to_training_name(nm, order), "nominal": nm,
                     "source": names[j], "angle_deg": round(float(ang[i, j]), 2),
                     "approx_mm": round(float(np.radians(ang[i, j]) * HEAD_RADIUS_MM), 1),
                     "nearest_was_free": bool(j == int(np.argmin(ang[i])))})
    out = pd.DataFrame(rows)
    out["_o"] = out["training_name"].map({c: k for k, c in enumerate(order)})
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def nearest_channels(source_pos: dict[str, np.ndarray], targets=MASTOID_PROXIES,
                     exclude: set[str] | None = None) -> pd.DataFrame:
    """Eletrodo mais próximo de cada posição nominal em `targets`."""
    exclude = exclude or set()
    dirs = {k: v for k, v in directions(source_pos).items() if k not in exclude}
    names, ang = _angles(dirs, list(targets))
    rows = []
    for i, t in enumerate(targets):
        j = int(np.argmin(ang[i]))
        rows.append({"nominal": t, "source": names[j], "angle_deg": round(float(ang[i, j]), 2),
                     "approx_mm": round(float(np.radians(ang[i, j]) * HEAD_RADIUS_MM), 1)})
    return pd.DataFrame(rows)


def check_geometry(mapping: pd.DataFrame) -> tuple[bool, str]:
    """
    Numa touca densa o eletrodo mais próximo fica a poucos graus da posição
    ideal. Mediana acima de MAX_MEDIAN_DEG indica touca esparsa ou posições
    inválidas. ATENÇÃO: isto NÃO detecta eixos trocados (uma touca quase
    uniforme girada continua tendo um eletrodo perto de cada ponto 10-20);
    para isso existe check_orientation.
    """
    med = float(mapping["angle_deg"].median())
    return med <= MAX_MEDIAN_DEG, f"deslocamento mediano {med:.1f} graus (limite {MAX_MEDIAN_DEG})"


def check_orientation(file_pos: dict[str, np.ndarray],
                      template: str = TEMPLATE_MONTAGE) -> tuple[bool | None, str]:
    """
    Compara, eletrodo a eletrodo, a direção no arquivo com a direção do mesmo
    nome no modelo padrão (as duas esferas ajustadas sobre os mesmos nomes).
    Uma convenção de eixos diferente (frente/lado trocados, esquerda/direita
    espelhadas) produz divergências de dezenas de graus.

    Retorna (True/False, msg) ou (None, msg) se os nomes não permitem comparar.
    """
    tpl = template_positions(template)
    common = [n for n in file_pos if n in tpl]
    if len(common) < MIN_POSITIONS:
        return None, f"só {len(common)} nomes em comum com {template}: orientação não verificável"
    fd = directions({n: file_pos[n] for n in common})
    td = directions({n: tpl[n] for n in common})
    ang = np.degrees(np.arccos(np.clip([fd[n] @ td[n] for n in common], -1, 1)))
    med = float(np.median(ang))
    return med <= MAX_TEMPLATE_DEG, (f"divergência mediana arquivo x {template}: {med:.1f} graus "
                                     f"em {len(common)} eletrodos (limite {MAX_TEMPLATE_DEG})")


def resolve_geometry(raw: mne.io.BaseRaw, mapping_source: str = "auto",
                     order: list[str] | None = None):
    """
    Decide de onde vêm as posições e calcula a correspondência.

    mapping_source:
      "file"     : posições gravadas no arquivo (chanlocs do .set)
      "template" : modelo padrão biosemi128 do MNE (exige nomes A1..D32)
      "auto"     : arquivo, se tiver >= MIN_POSITIONS posições e passar nas
                   verificações; senão, modelo padrão.

    Retorna (posições, mapping, fonte, notas). Se a orientação das posições do
    arquivo não puder ser verificada (nomes não-Biosemi), a fonte vem como
    "file_unverified" e a figura do notebook deve ser conferida a olho.
    """
    notes = []
    if mapping_source in ("auto", "file"):
        pos = positions_from_raw(raw)
        if len(pos) >= MIN_POSITIONS:
            orient_ok, orient_msg = check_orientation(pos)
            notes.append(orient_msg)
            if orient_ok is False:
                notes.append("posições do arquivo rejeitadas (orientação)")
            else:
                mp = compute_19ch_mapping(pos, order)
                geo_ok, geo_msg = check_geometry(mp)
                notes.append(geo_msg)
                if geo_ok:
                    return pos, mp, ("file" if orient_ok else "file_unverified"), notes
                notes.append("posições do arquivo rejeitadas (deslocamento)")
        else:
            notes.append(f"arquivo tem só {len(pos)} posições válidas")
        if mapping_source == "file":
            raise RuntimeError("; ".join(notes))

    tpl = template_positions(TEMPLATE_MONTAGE)
    pos = {n: tpl[n] for n in raw.ch_names if n in tpl and not NON_SCALP_PATTERN.match(n)}
    if len(pos) < MIN_POSITIONS:
        raise RuntimeError(
            "sem geometria utilizável: " + "; ".join(notes) +
            f"; só {len(pos)} nomes do arquivo existem em {TEMPLATE_MONTAGE}. "
            f"Exemplo de nomes no arquivo: {raw.ch_names[:8]}")
    mp = compute_19ch_mapping(pos, order)
    geo_ok, geo_msg = check_geometry(mp)
    notes.append(geo_msg)
    if not geo_ok:
        raise RuntimeError(f"modelo padrão reprovado: {geo_msg}")
    return pos, mp, "template", notes


# --------------------------------------------------------------------------
# Adaptação
# --------------------------------------------------------------------------
def adapt_to_training_space(raw: mne.io.BaseRaw,
                            mapping_source: str = "auto",
                            reference: str = "mastoid_approx",
                            ref_channels: list[str] | None = None,
                            order: list[str] | None = None,
                            rename: dict[str, str] | None = None,
                            target_sfreq: float = TRAINING_SFREQ,
                            max_duration_s: float | None = None,
                            verbose: bool = False):
    """
    Returns
    -------
    raw19 : Raw com os 19 canais do treino (nomes e ordem do ds004504), 500 Hz,
            re-referenciado, com montagem standard_1020.
    prov  : dicionário com tudo o que foi decidido (para o JSON de proveniência).
    """
    order = order or list(DS004504_ORDER)
    raw = raw.copy().load_data(verbose="ERROR")
    if rename:
        raw.rename_channels({k: v for k, v in rename.items() if k in raw.ch_names})

    # 1-2. geometria e correspondência
    pos, mapping, geo_source, geo_notes = resolve_geometry(raw, mapping_source, order)
    signal_chs = list(mapping["source"])

    # 3. referência
    if reference == "mastoid_approx":
        if ref_channels is None:
            refs = nearest_channels(pos, MASTOID_PROXIES, exclude=set(signal_chs))
            ref_channels = refs["source"].tolist()
    elif reference == "explicit":
        if not ref_channels:
            raise ValueError("reference='explicit' exige ref_channels")
    elif reference == "average":
        ref_channels = []
    else:
        raise ValueError(f"reference desconhecida: {reference}")
    if set(ref_channels) & set(signal_chs):
        raise ValueError("canais de referência coincidem com canais de sinal")

    needed = signal_chs + list(ref_channels)
    missing = [c for c in needed if c not in raw.ch_names]
    if missing:
        raise KeyError(f"canais necessários ausentes: {missing}")
    to_eeg = {c: "eeg" for c in needed if raw.get_channel_types([c])[0] != "eeg"}
    if to_eeg:
        raw.set_channel_types(to_eeg, verbose="ERROR")

    if max_duration_s is not None and raw.times[-1] > max_duration_s:
        raw.crop(tmax=max_duration_s)

    # 4. ruins/planos entre os escolhidos -> interpolação com todos os canais
    stds = raw.get_data(picks=needed).std(axis=1)
    flat = [c for c, s in zip(needed, stds) if s < FLAT_STD_VOLTS]
    bad_needed = sorted((set(raw.info["bads"]) & set(needed)) | set(flat))
    if bad_needed:
        # Interpola usando só canais do escalpo com posição conhecida (exclui
        # ECG/EXG sem posição, que o MNE não saberia onde colocar)
        if geo_source == "template":  # sem posições no arquivo: usa as do modelo
            raw.set_montage(TEMPLATE_MONTAGE, on_missing="ignore", verbose="ERROR")
        with_pos = set(positions_from_raw(raw))
        raw.pick([c for c in raw.ch_names if c in with_pos])
        raw.info["bads"] = sorted((set(raw.info["bads"]) & with_pos) | set(bad_needed))
        if len(raw.ch_names) - len(raw.info["bads"]) < MIN_POSITIONS:
            raise RuntimeError(f"canais ruins {bad_needed}: poucos canais bons para interpolar")
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")

    # 5. seleção, offset, taxa
    raw.pick(needed)
    raw.info["bads"] = []
    raw.apply_function(lambda x: x - x.mean(), picks="all", verbose="ERROR")
    orig_sfreq = float(raw.info["sfreq"])
    if not np.isclose(orig_sfreq, target_sfreq):
        raw.resample(target_sfreq, verbose="ERROR")

    # 6. referência, nomes, ordem, montagem
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    else:
        raw.set_eeg_reference(ref_channels=list(ref_channels), verbose="ERROR")
        raw.drop_channels(list(ref_channels))
    raw.rename_channels(dict(zip(mapping["source"], mapping["training_name"])))
    raw.reorder_channels(order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        match_case=False, verbose="ERROR")

    assert raw.ch_names == order
    assert np.isclose(raw.info["sfreq"], target_sfreq)

    prov = {
        "geometry_source": geo_source,
        "geometry_notes": geo_notes,
        "n_positions": len(pos),
        "mapping": mapping.to_dict(orient="records"),
        "median_angle_deg": float(mapping["angle_deg"].median()),
        "max_angle_deg": float(mapping["angle_deg"].max()),
        "reference": reference,
        "ref_channels": list(ref_channels),
        "interpolated": bad_needed,
        "orig_sfreq": orig_sfreq,
        "sfreq": float(raw.info["sfreq"]),
        "duration_s": float(raw.times[-1]),
    }
    if verbose:
        print(f"  geometria={geo_source} ({len(pos)} pos.), desloc. mediano "
              f"{prov['median_angle_deg']:.1f} graus, ref={ref_channels}, "
              f"interpolados={bad_needed or 'nenhum'}, {orig_sfreq:.0f}->{prov['sfreq']:.0f} Hz, "
              f"{prov['duration_s']:.0f} s")
    return raw, prov


# --------------------------------------------------------------------------
# Figura de conferência
# --------------------------------------------------------------------------
def plot_mapping(pos: dict[str, np.ndarray], mapping: pd.DataFrame,
                 ref_channels: list[str] | None = None, title: str = "", ax=None):
    """
    Vista de cima (nariz para cima, esquerda do sujeito à esquerda), projeção
    azimutal equidistante das direções radiais. Cinza: todos os eletrodos do
    arquivo. Preto: posição 10-20 ideal. Colorido: eletrodo escolhido, com seta
    a partir da posição ideal. Losangos: referência.

    Conferência a olho: Fp1/Fp2 junto ao nariz, O1/O2 atrás, ímpares (Fp1, F3,
    C3, T3...) à esquerda, pares à direita.
    """
    import matplotlib.pyplot as plt

    def proj(u):
        th = np.arccos(np.clip(u[2], -1, 1))
        n = np.hypot(u[0], u[1])
        return np.array([0.0, 0.0]) if n == 0 else th * np.array([u[0], u[1]]) / n

    src = directions(pos)
    nom = directions(template_positions(NOMINAL_MONTAGE))
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.5))
    allxy = np.array([proj(v) for v in src.values()])
    ax.scatter(allxy[:, 0], allxy[:, 1], s=14, c="0.85", zorder=1)
    for r in mapping.itertuples():
        p0, p1 = proj(nom[r.nominal]), proj(src[r.source])
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", lw=1.2, color="#c0392b"))
        ax.scatter(*p0, s=30, c="black", zorder=3)
        ax.scatter(*p1, s=60, c="#c0392b", zorder=4, edgecolors="white")
        ax.text(p0[0], p0[1] + 0.07, r.training_name, ha="center", fontsize=9, weight="bold")
        ax.text(p1[0], p1[1] - 0.09, r.source, ha="center", fontsize=7, color="#6a1b9a")
    for c in ref_channels or []:
        p = proj(src[c])
        ax.scatter(*p, s=90, marker="D", c="#1f77b4", zorder=4)
        ax.text(p[0], p[1] - 0.1, f"ref {c}", ha="center", fontsize=7, color="#1f77b4")
    lim = max(np.abs(allxy).max(), 1.7) * 1.08
    ax.add_patch(plt.Circle((0, 0), np.pi / 2, fill=False, ls="--", ec="0.6"))
    ax.plot([-0.12, 0, 0.12], [lim * 0.93, lim, lim * 0.93], c="0.4")   # nariz
    ax.text(-lim, 0, "E", fontsize=12, va="center")
    ax.text(lim * 0.95, 0, "D", fontsize=12, va="center")
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_ylim(-lim * 1.05, lim * 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    return ax
"""
Referência do EEG: leva qualquer registro a uma referência-alvo comum.

Cada conjunto de dados chega numa referência diferente (o ds004504 nas orelhas,
o BrainLat em média de 128 canais, outros em Cz, mastoides ou na referência do
amplificador). Medidas como Kuramoto, informação mútua e PAC mudam com a
referência, então comparar ou transferir modelos entre bases exige que todas
cheguem à MESMA referência. Este módulo é o único lugar onde isso acontece.

Referências-alvo disponíveis (feature_sets.TARGET_REFERENCE):

  "mastoid"      Região das orelhas/mastoides. PADRÃO ATUAL.
                 Se a base tiver canais de orelha ou mastoide, declarados em
                 ref_channels, a referência passa a ser a média deles, e eles
                 são removidos. Se não tiver, mas a base foi GRAVADA com
                 referência nessa região (declarado em SOURCE_REFERENCES), o
                 sinal é mantido como está. Caso contrário, erro.
                 Não depende de quais canais foram selecionados, ao contrário
                 da referência média, que com 10 canais é uma aproximação ruim.
                 Diferença residual entre as bases atuais: o ds004504 usa a
                 média das duas orelhas; o EPOC X, um único eletrodo de
                 referência (CMS) atrás de uma orelha.

  "average"      Média dos canais analisados (os 19 do 10-20). PADRÃO.
                 Reproduzível de forma exata a partir de qualquer referência
                 comum de origem: se x_i = v_i - r(t), a média dos 19 canais
                 subtraída de cada um elimina r(t), seja r as orelhas, Cz,
                 mastoides ou a média de 128 canais. Não exige nenhum canal
                 extra nem informação sobre a referência original. Também é a
                 referência para a qual o ICLabel foi treinado.

  "rest"         REST (Reference Electrode Standardization Technique): estima
                 uma referência no infinito a partir de um modelo de cabeça
                 esférico. Também independe da referência de origem, mas
                 depende do modelo de cabeça e das posições dos eletrodos.
                 Experimental neste repositório.

  "linked_ears"  Média das orelhas ou mastoides. Exata só quando a base tem
                 esses canais ou já foi gravada assim (ds004504). Em bases sem
                 eles, usa canais substitutos (p.ex. os eletrodos mais próximos
                 de TP9/TP10 no BrainLat), o que é uma aproximação.

  "acquisition"  Mantém a referência de gravação. Só é válido quando todas as
                 bases comparadas foram gravadas na mesma referência.

Para acrescentar uma base nova com o alvo "average" ou "rest" não é preciso
saber a referência de origem: basta selecionar os 19 canais e chamar
apply_target_reference.
"""
from __future__ import annotations

import warnings

import numpy as np
import mne

TARGETS = ("mastoid", "average", "rest", "linked_ears", "acquisition")

# Referências de gravação que já ficam na região das orelhas/mastoides
MASTOID_SOURCES = {"linked_ears", "linked_mastoids", "mastoid", "cms_mastoid"}
EAR_MASTOID_NAMES = {"A1", "A2", "M1", "M2", "TP9", "TP10", "LM", "RM", "EARL", "EARR"}
MIN_CHANNELS_AVERAGE = 16   # abaixo disso a média é uma aproximação ruim da referência neutra

# Referência de origem declarada de cada base usada no projeto (informativo;
# os alvos "average" e "rest" não dependem disso).
SOURCE_REFERENCES = {
    "ds004504": "linked_ears",   # média de A1 e A2 na aquisição (Miltiadous et al., 2023)
    "epocx": "cms_mastoid",      # eletrodo de referência CMS do headset, atrás da orelha
                                 # (confirmar posição e lado na documentação da Emotiv)
    "brainlat": "common",        # média de 128 canais na maioria dos arquivos distribuídos
}


def detect_reference(raw: mne.io.BaseRaw) -> dict:
    """
    Pistas sobre a referência atual. Em referência média, a soma dos canais é
    zero a cada instante; a razão abaixo compara essa soma com a amplitude típica.
    """
    x = raw.get_data(picks="eeg")
    common = float(np.abs(x.mean(axis=0)).mean())
    typical = float(np.median(np.abs(x - x.mean(axis=1, keepdims=True))))
    ratio = common / typical if typical > 0 else float("nan")
    return {
        "n_eeg": int(x.shape[0]),
        "avg_ratio": round(ratio, 4),
        "looks_average": bool(ratio < 0.01),
        "ear_or_mastoid_channels": sorted(c for c in raw.ch_names if c.upper() in EAR_MASTOID_NAMES),
        "custom_ref_applied": bool(raw.info["custom_ref_applied"]),
    }


_REST_FORWARD_CACHE: dict = {}


def _rest(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    REST com modelo de cabeça esférico. O modelo direto depende só dos nomes e
    posições dos canais; como todas as bases usam as mesmas 19 posições
    standard_1020, ele é calculado uma vez e reutilizado, e a transformação é
    idêntica em todas as bases.
    """
    if raw.get_montage() is None:
        raise ValueError("REST exige posições dos eletrodos (montagem)")
    pos = raw.get_montage().get_positions()["ch_pos"]
    key = tuple((c, tuple(np.round(pos[c], 6))) for c in raw.ch_names if c in pos)
    if key not in _REST_FORWARD_CACHE:
        sphere = mne.make_sphere_model("auto", "auto", raw.info, verbose=False)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0, verbose=False)
        _REST_FORWARD_CACHE[key] = mne.make_forward_solution(raw.info, trans=None, src=src,
                                                             bem=sphere, verbose=False)
    raw.set_eeg_reference("REST", forward=_REST_FORWARD_CACHE[key], verbose=False)
    return raw


def apply_target_reference(raw: mne.io.BaseRaw, target: str, source: str | None = None,
                           ref_channels: list[str] | None = None,
                           expected_channels: list[str] | None = None):
    """
    Aplica a referência-alvo, no próprio objeto (in place), e devolve (raw, registro).

    target            um de TARGETS
    source            referência de origem declarada (p.ex. "linked_ears", "common");
                      obrigatória para "acquisition" e para "linked_ears" sem ref_channels
    ref_channels      canais de orelha/mastoide (reais ou substitutos) para "linked_ears";
                      são subtraídos e depois removidos
    expected_channels se fornecido, confere que os canais EEG presentes são exatamente
                      esses ANTES de referenciar. Para "average" isso é essencial: a
                      média só é a mesma entre bases se for calculada sobre o mesmo
                      conjunto de canais.
    """
    if target not in TARGETS:
        raise ValueError(f"referência-alvo desconhecida: {target}; opções: {TARGETS}")
    before = detect_reference(raw)
    ref_channels = list(ref_channels or [])

    if expected_channels is not None:
        eeg_now = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
        extra = set(eeg_now) - set(expected_channels) - set(ref_channels)
        missing = set(expected_channels) - set(eeg_now)
        if extra or missing:
            raise ValueError(f"conjunto de canais diferente do esperado: sobram {sorted(extra)}, "
                             f"faltam {sorted(missing)}")

    if target == "mastoid":
        # Os canais de orelha/mastoide são sempre declarados (ref_channels), nunca
        # adivinhados pelo nome: no Biosemi, "A1" e "A2" são eletrodos do escalpo.
        if ref_channels:
            chans = ref_channels
            missing = [c for c in chans if c not in raw.ch_names]
            if missing:
                raise KeyError(f"canais de referência ausentes: {missing}")
            raw.set_eeg_reference(ref_channels=chans, verbose="ERROR")
            raw.drop_channels(chans)
            ref_channels = chans
        elif source not in MASTOID_SOURCES:
            raise ValueError(
                f"alvo 'mastoid': o arquivo não tem canais de orelha/mastoide e a referência de "
                f"gravação declarada ({source!r}) não fica nessa região. Declare a referência da "
                f"base em references.SOURCE_REFERENCES ou use outro alvo.")

    elif target == "average":
        if ref_channels:
            raw.drop_channels(ref_channels)
        n = len(mne.pick_types(raw.info, eeg=True))
        if n < MIN_CHANNELS_AVERAGE:
            warnings.warn(f"referência média com só {n} canais", RuntimeWarning)
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    elif target == "rest":
        if ref_channels:
            raw.drop_channels(ref_channels)
        _rest(raw)

    elif target == "linked_ears":
        if ref_channels:
            missing = [c for c in ref_channels if c not in raw.ch_names]
            if missing:
                raise KeyError(f"canais de referência ausentes: {missing}")
            raw.set_eeg_reference(ref_channels=ref_channels, verbose="ERROR")
            raw.drop_channels(ref_channels)
        elif source != "linked_ears":
            raise ValueError("alvo 'linked_ears' exige canais de orelha/mastoide (ref_channels) "
                             "ou dados já gravados nessa referência (source='linked_ears'); "
                             "para bases sem esses canais, use 'average' ou 'rest'")

    elif target == "acquisition":
        if source is None:
            raise ValueError("alvo 'acquisition' exige declarar a referência de origem (source)")

    record = {"target": target, "source": source, "ref_channels": ref_channels,
              "n_channels_after": int(len(mne.pick_types(raw.info, eeg=True))),
              "detected_before": before}
    return raw, record
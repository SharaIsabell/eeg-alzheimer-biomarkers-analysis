"""
Configuração única do estudo: versão das features, bandas e conjuntos de features.

Todos os outros módulos (extração, modelagem, BrainLat) importam daqui, para que
nenhuma definição exista em duas cópias que possam divergir.

Este módulo não depende de MNE nem de bibliotecas de sinal, para poder ser
importado por qualquer etapa.
"""
from __future__ import annotations

# Muda sempre que a definição de alguma feature muda. É usada no nome das
# pastas de cache, para que features de versões diferentes nunca se misturem.
#   1.0  versão original (commit 9374d39)
#   2.0  correções da auditoria de outubro de 2026 (ver CORRECOES.md)
#   2.1  referência-alvo comum a todas as bases (references.py)
#   (o ramo do EPOC X tem versão e configuração próprias: epocx/config_epocx.py)
FEATURE_VERSION = "2.1"

# --------------------------------------------------------------------------
# Referência-alvo (ver references.py)
# --------------------------------------------------------------------------
# Todas as bases são levadas a esta referência antes da limpeza e das features.
# "average" (média dos canais do conjunto CHANNELS) é reproduzível de forma exata a partir de
# qualquer referência de origem e é a referência esperada pelo ICLabel.
# Alternativas: "rest", "linked_ears", "acquisition".
TARGET_REFERENCE = "average"


# --------------------------------------------------------------------------
# Canais e taxa de amostragem
# --------------------------------------------------------------------------
# Nomes na nomenclatura 10-10. No ds004504, T3/T4/T5/T6 são renomeados para
# T7/T8/P7/P8 (mesmas posições), para coincidir com os nomes do EPOC X.
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

DS004504_19 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
               "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
EPOCX_14 = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
            "P8", "T8", "FC6", "F4", "F8", "AF4"]

CHANNEL_SETS = {
    # os 19 canais originais do ds004504
    "ds004504_19": DS004504_19,
    # só os canais que existem com o MESMO nome nos dois aparelhos (10);
    # AF3/AF4 e FC5/FC6 do EPOC X e Fp1/Fp2, Fz, Cz, Pz, C3, C4, P3, P4 do
    # ds004504 ficam de fora
    "epocx_comum": [c for c in EPOCX_14 if c in DS004504_19],
}
CHANNEL_SET = "ds004504_19"
CHANNELS = CHANNEL_SETS[CHANNEL_SET]

# Taxa comum a todas as bases. None mantém a taxa original (500 Hz no ds004504).
# O ramo do EPOC X usa 128 Hz (epocx/config_epocx.py).
TARGET_SFREQ = None


def make_tag(version: str, target: str, channel_set: str, sfreq) -> str:
    """
    Identificador de tudo o que muda as features. Para a configuração original
    (19 canais, taxa original) mantém o formato antigo, "v2.1-average", para que
    as pastas de cache já geradas continuem valendo; outras configurações levam
    canais e taxa, p.ex. "v3.0-mastoid-epocx_comum-128Hz".
    """
    if channel_set == "ds004504_19" and sfreq is None:
        return f"v{version}-{target}"
    sf = "orig" if sfreq is None else f"{sfreq:g}Hz"
    return f"v{version}-{target}-{channel_set}-{sf}"


def feature_tag(target: str | None = None) -> str:
    """Identificador da configuração deste módulo (pipeline original)."""
    return make_tag(FEATURE_VERSION, target or TARGET_REFERENCE, CHANNEL_SET, TARGET_SFREQ)

# --------------------------------------------------------------------------
# Bandas de frequência
# --------------------------------------------------------------------------
# Intervalos semiabertos [início, fim): um bin de frequência pertence a uma
# única banda, e as cinco potências relativas somam exatamente 1.
#
# Delta começa em 1 Hz (antes 0,5 Hz), por dois motivos documentados:
#   1. a faixa 0,5-1 Hz acumula deriva lenta residual e dominava a potência
#      relativa (delta ~78% e alfa ~7% em olhos fechados, na versão 1.0);
#   2. os registros do BrainLat chegam com passa-alta prévio (-3 dB em ~1,26 Hz,
#      -6 dB em ~0,78 Hz; ver check_highpass.py), então 0,5-1 Hz não é
#      comparável entre as bases.
DELTA_LOW_HZ = 1.0
BROADBAND = (DELTA_LOW_HZ, 45.0)

BANDS = {
    "Delta": (DELTA_LOW_HZ, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 25.0),
    "Gamma": (25.0, 45.0),
}
PAC_PHASE_BANDS = ("Delta", "Theta", "Alpha")
PAC_AMP_BANDS = ("Beta", "Gamma")

# --------------------------------------------------------------------------
# Nomes das colunas (mantidos da versão 1.0 para compatibilidade)
# --------------------------------------------------------------------------
RBP_COLS = [f"{b}_RBP" for b in BANDS]
MEANFREQ_COLS = [f"{b}_Mean_Freq" for b in BANDS]
LZC_COLS = [f"{b}_LZC" for b in BANDS]
PAC_COLS = [f"PAC_{p}_{a}" for p in PAC_PHASE_BANDS for a in PAC_AMP_BANDS]
KURAMOTO_COLS = [f"Kuramoto_{b}" for b in BANDS]
MI_COLS = [f"MI_{b}" for b in BANDS]

ALL_FEATURES = RBP_COLS + MEANFREQ_COLS + LZC_COLS + PAC_COLS + KURAMOTO_COLS + MI_COLS
assert len(ALL_FEATURES) == 31

# --------------------------------------------------------------------------
# Conjuntos avaliados. Os nomes descrevem o que cada medida calcula de fato
# (ver README, seção "O que cada família mede").
# --------------------------------------------------------------------------
FEATURE_SETS = {
    "RBP": RBP_COLS,
    "Centroide espectral": MEANFREQ_COLS,
    "LZC (banda estreita)": LZC_COLS,
    "PAC local": PAC_COLS,
    "Kuramoto (sensor)": KURAMOTO_COLS,
    "Informação mútua": MI_COLS,
    "Todas (31)": ALL_FEATURES,
}
BASELINE_SET = "RBP"
COMBINED_SET = "Todas (31)"

MODEL_NAMES = ("Random Forest", "SVM (Linear)", "XGBoost")

# Rótulos do ds004504 (participants.tsv, coluna Group)
POSITIVE_GROUP = "A"   # Alzheimer
NEGATIVE_GROUP = "C"   # controle
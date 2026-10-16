"""
Configuração do ramo EPOC X. Fica separada de feature_sets.py para que o
pipeline original (19 canais, referência média, 500 Hz) não seja afetado.

O que é compartilhado com o pipeline original (na raiz do repositório):
  - feature_sets.py      : bandas, nomes das features, conjuntos de canais
  - ds004504_features.py : cálculo das 31 features
  - references.py        : referências-alvo (inclusive "mastoid")
  - modeling.py          : pipelines e estatística
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # raiz do repositório
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# O ramo depende de versões recentes de três arquivos da raiz. A verificação lê
# o texto dos arquivos (sem importá-los) e diz exatamente o que atualizar.
_REQUIRED = {
    "feature_sets.py": ["CHANNEL_SETS", "def make_tag"],
    "references.py": ['"mastoid"', "MASTOID_SOURCES"],
    "ds004504_features.py": ["def extract_features_from_epochs", "def epoch_qc"],
}
_outdated = []
for _name, _needles in _REQUIRED.items():
    _path = ROOT / _name
    _text = _path.read_text(encoding="utf-8") if _path.exists() else ""
    _missing = [n for n in _needles if n not in _text]
    if _missing:
        _outdated.append(f"  {_path}  (falta: {', '.join(_missing)})")
if _outdated:
    raise ImportError(
        "O ramo EPOC X precisa de versões mais recentes destes arquivos da raiz do projeto:\n"
        + "\n".join(_outdated)
        + "\nSubstitua-os pelas versões atuais e reinicie o kernel.")

from feature_sets import CHANNEL_SETS, CHANNEL_ALIASES, make_tag  # noqa: E402

#   3.0  pré-processamento do preprocessingGT (FIR 0,1-45 Hz, canais defeituosos,
#        notch, ICA picard) com wICA suave 1.5 seletivo no lugar do ICLabel,
#        AutoReject e referência na região das mastoides
FEATURE_VERSION = "3.0"
TARGET_REFERENCE = "mastoid"
CHANNEL_SET = "epocx_comum"
CHANNELS = CHANNEL_SETS[CHANNEL_SET]
TARGET_SFREQ = 128.0          # taxa do EPOC X; o ds004504 é reamostrado para baixo

# Onde ficam as saídas deste ramo (cache, CSVs de features, QC)
OUT_DIR = Path(__file__).resolve().parent


def feature_tag() -> str:
    return make_tag(FEATURE_VERSION, TARGET_REFERENCE, CHANNEL_SET, TARGET_SFREQ)
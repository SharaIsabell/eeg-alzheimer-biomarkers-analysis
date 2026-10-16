"""
Semelhança entre as features do EPOC X (sem rótulos) e as do ds004504 adaptado
(com rótulos AD e CN), no ramo epocx.

Sem rótulos no EPOC X não há como medir acerto de classificação. O que se mede
é se as features das duas bases são comparáveis e, quando não são, se a
diferença entre as bases é maior ou menor que o efeito da doença:

  1. por feature: tamanho de efeito EPOC X x controles (e x AD), comparado ao
     efeito AD x controles dentro do ds004504; posição relativa entre CN e AD;
  2. confiabilidade: primeira metade x segunda metade de cada gravação;
  3. multivariado: classificador de base (distingue as bases?), tipicidade de
     cada gravação em relação aos controles (distância de Mahalanobis) e
     projeção em componentes principais;
  4. estrutura: as features se correlacionam do mesmo jeito nas duas bases?

A unidade de análise é a gravação (média das épocas), nunca a época.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_sets import ALL_FEATURES

SMALL_EFFECT = 0.5        # |g| abaixo disso: diferença pequena
MIN_N = 5                 # abaixo disso, a análise não é feita
SEED = 42


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def load_tables(epocx_root: str | Path, expected_tag: str):
    """Lê as duas tabelas do ramo e confere se foram geradas com a mesma configuração."""
    root = Path(epocx_root)
    out = {}
    for name in ("ds004504", "epocx"):
        csv = root / f"eeg_features_{name}.csv"
        meta = csv.with_suffix(".meta.json")
        if not csv.exists():
            raise FileNotFoundError(f"{csv} não existe: rode o preprocessingData_epocx.ipynb")
        tag = json.loads(meta.read_text(encoding="utf-8")).get("feature_tag") if meta.exists() else None
        if tag != expected_tag:
            raise ValueError(f"{csv.name} foi gerado com {tag}; a configuração atual é {expected_tag}")
        out[name] = pd.read_csv(csv)
    return out["ds004504"], out["epocx"]


def recording_table(epochs: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame:
    """
    Média das épocas por PARTICIPANTE, mantendo o grupo quando existe. Quem tem
    mais de uma gravação (p.ex. run-1 e run-2 no EPOC X) entra uma vez, com a
    média das épocas de todas elas: a unidade de análise é a pessoa.
    """
    keys = ["Subject_ID"] + [c for c in ("Group",) if c in epochs.columns]
    out = epochs.groupby(keys, as_index=False)[list(feature_cols)].mean()
    if "Recording_ID" in epochs.columns:
        n_rec = epochs.groupby("Subject_ID")["Recording_ID"].nunique().rename("n_gravacoes")
        out = out.merge(n_rec, on="Subject_ID", how="left")
    return out


# --------------------------------------------------------------------------
# 1. Por feature
# --------------------------------------------------------------------------
def hedges_g(a, b) -> tuple[float, float, float]:
    """g de Hedges (a - b) com IC95% aproximado. Devolve (g, inferior, superior)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, np.nan)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return (np.nan, np.nan, np.nan)
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = j * (a.mean() - b.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return (float(g), float(g - 1.96 * se), float(g + 1.96 * se))


def _bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for rank in range(len(p) - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * len(p) / (rank + 1))
        adj[i] = prev
    return adj


def classify_shift(g_cn: float, d_disease: float, small: float = SMALL_EFFECT) -> str:
    if np.isnan(g_cn):
        return "indefinido"
    if abs(g_cn) < small:
        return "compatível com os controles"
    if abs(g_cn) <= abs(d_disease):
        return "deslocada, dentro do efeito da doença"
    return "deslocamento maior que o efeito da doença"


def per_feature_table(ds: pd.DataFrame, ep: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame:
    """
    Uma linha por feature:
      g_EPOC_vs_CN (IC95%), g_EPOC_vs_AD, d_AD_vs_CN (efeito da doença no ds004504),
      AUC de Mann-Whitney (probabilidade de uma gravação do EPOC X ter valor maior
      que um controle), estatística KS e p corrigido por Benjamini-Hochberg,
      posição relativa e categoria.

    posicao_CN_AD = (mediana EPOC X - média CN) / (média AD - média CN):
      0 = como os controles, 1 = como os pacientes, fora de [0, 1] = além dos dois.
      Só é calculada quando o efeito da doença é pelo menos pequeno (|d| >= 0,5),
      porque com efeito pequeno o denominador é quase zero.
    """
    cn, ad = ds[ds["Group"] == "C"], ds[ds["Group"] == "A"]
    rows = []
    for c in feature_cols:
        e, n_, a_ = ep[c].dropna(), cn[c].dropna(), ad[c].dropna()
        g_cn = hedges_g(e, n_)
        g_ad = hedges_g(e, a_)
        d_dis = hedges_g(a_, n_)[0]
        auc = stats.mannwhitneyu(e, n_, alternative="two-sided").statistic / (len(e) * len(n_)) if len(e) and len(n_) else np.nan
        ks = stats.ks_2samp(e, n_) if len(e) and len(n_) else None
        pos = ((e.median() - n_.mean()) / (a_.mean() - n_.mean())
               if abs(d_dis) >= SMALL_EFFECT and a_.mean() != n_.mean() else np.nan)
        rows.append({"feature": c, "n_EPOC": len(e), "n_CN": len(n_), "n_AD": len(a_),
                     "mediana_EPOC": e.median(), "media_CN": n_.mean(), "media_AD": a_.mean(),
                     "g_EPOC_vs_CN": g_cn[0], "g_IC_inf": g_cn[1], "g_IC_sup": g_cn[2],
                     "g_EPOC_vs_AD": g_ad[0], "d_AD_vs_CN": d_dis,
                     "AUC_EPOC_vs_CN": auc,
                     "KS": ks.statistic if ks else np.nan, "p_KS": ks.pvalue if ks else np.nan,
                     "posicao_CN_AD": pos})
    out = pd.DataFrame(rows)
    out["p_KS_BH"] = _bh(out["p_KS"].fillna(1.0))
    out["categoria"] = [classify_shift(g, d) for g, d in zip(out["g_EPOC_vs_CN"], out["d_AD_vs_CN"])]
    # Com poucas gravações o IC do g é largo; se ele cruza o limite de |g| = 0,5, a
    # categoria depende do acaso da amostra e não deve sustentar conclusões.
    lo, hi = out["g_IC_inf"], out["g_IC_sup"]
    crosses = ((lo < SMALL_EFFECT) & (hi > SMALL_EFFECT)) | ((lo < -SMALL_EFFECT) & (hi > -SMALL_EFFECT))
    out["classificacao_incerta"] = crosses
    return out


# --------------------------------------------------------------------------
# 2. Confiabilidade (primeira metade x segunda metade)
# --------------------------------------------------------------------------
def _epoch_index(epoch_id: str) -> int:
    m = re.search(r"_ep(\d+)$", str(epoch_id))
    return int(m.group(1)) if m else -1


def split_half_reliability(epochs: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame:
    """
    Para cada gravação, a média das épocas da primeira metade do registro e a da
    segunda. Correlação de Spearman entre as metades (entre gravações), corrigida
    por Spearman-Brown para o registro inteiro. Metades no tempo, e não épocas
    alternadas: épocas vizinhas se sobrepõem em 50% e inflariam a confiabilidade.
    """
    key = "Recording_ID" if "Recording_ID" in epochs.columns else "Subject_ID"
    df = epochs.assign(_i=epochs["Epoch_ID"].map(_epoch_index)).sort_values([key, "_i"])
    df["_half"] = df.groupby(key)["_i"].transform(lambda s: (s.rank(method="first") > len(s) / 2).astype(int))
    halves = df.groupby([key, "_half"])[list(feature_cols)].mean().unstack("_half")
    # Features constantes dentro de cada gravação (o PAC é estimado uma vez por
    # gravação e repetido em todas as épocas) dariam metades idênticas e
    # confiabilidade 1 por construção: ficam sem valor. Para elas, só a
    # confiabilidade entre gravações (run_reliability) tem sentido.
    within_sd = df.groupby(key)[list(feature_cols)].std().max()
    rows = []
    n = halves.shape[0]
    for c in feature_cols:
        if within_sd.get(c, 1.0) < 1e-12:
            rows.append({"feature": c, "n": n, "r_metades": np.nan, "confiabilidade": np.nan,
                         "nota": "constante dentro da gravação"})
            continue
        h = halves[c].dropna()
        if len(h) < MIN_N or h.shape[1] < 2:
            rows.append({"feature": c, "n": len(h), "r_metades": np.nan, "confiabilidade": np.nan})
            continue
        r = stats.spearmanr(h[0], h[1]).statistic
        rows.append({"feature": c, "n": len(h), "r_metades": r,
                     "confiabilidade": 2 * r / (1 + r) if r > -1 else np.nan})
    return pd.DataFrame(rows)


def run_reliability(epochs: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame | None:
    """
    Confiabilidade teste-reteste entre duas gravações do mesmo participante (as
    duas primeiras runs, em ordem): média das épocas de cada run, correlação de
    Spearman entre participantes. É mais forte que a divisão de uma gravação ao
    meio, porque inclui recolocar o aparelho e o estado da pessoa em outro
    momento. 'confiabilidade_media_2_runs' (Spearman-Brown) vale para a média das
    duas runs, que é o que entra nas comparações.
    Devolve None se não houver participantes suficientes com duas runs.
    """
    if not {"Run", "Recording_ID"} <= set(epochs.columns):
        return None
    per_run = epochs.groupby(["Subject_ID", "Run"])[list(feature_cols)].mean().reset_index()
    per_run["_ordem"] = per_run.groupby("Subject_ID")["Run"].rank(method="first")
    two = per_run[per_run["_ordem"] <= 2]
    both = two.groupby("Subject_ID")["_ordem"].nunique()
    ids = both[both == 2].index
    if len(ids) < MIN_N:
        return None
    a = two[(two["_ordem"] == 1) & two["Subject_ID"].isin(ids)].set_index("Subject_ID").loc[ids]
    b = two[(two["_ordem"] == 2) & two["Subject_ID"].isin(ids)].set_index("Subject_ID").loc[ids]
    rows = []
    for c in feature_cols:
        r = stats.spearmanr(a[c], b[c], nan_policy="omit").statistic
        rows.append({"feature": c, "n_participantes": len(ids), "r_entre_runs": r,
                     "confiabilidade_media_2_runs": 2 * r / (1 + r) if r > -1 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Multivariado
# --------------------------------------------------------------------------
def domain_classifier(ds: pd.DataFrame, ep: pd.DataFrame, feature_cols=ALL_FEATURES,
                      reference: str = "C", n_splits: int = 5, n_repeats: int = 20) -> dict:
    """
    Regressão logística regularizada tentando distinguir gravações do EPOC X das
    do ds004504 (só controles, por padrão), com validação cruzada estratificada
    repetida. AUC ~0,5: as bases são indistinguíveis com estas features; AUC
    perto de 1: as bases se separam com facilidade (deslocamento grande).
    """
    ref = ds if reference is None else ds[ds["Group"] == reference]
    X = pd.concat([ref[feature_cols], ep[feature_cols]]).to_numpy(float)
    y = np.r_[np.zeros(len(ref)), np.ones(len(ep))].astype(int)
    keep = ~np.isnan(X).any(axis=1)
    X, y = X[keep], y[keep]
    k = int(min(n_splits, np.bincount(y).min()))
    if k < 2:
        return {"AUC_media": np.nan, "nota": "gravações insuficientes"}
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=SEED)
    aucs = []
    for tr, te in cv.split(X, y):
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.5, max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.5, max_iter=2000))]).fit(X, y)
    coefs = pd.Series(pipe[-1].coef_[0], index=list(feature_cols)).sort_values(key=np.abs, ascending=False)
    return {"AUC_media": float(np.mean(aucs)), "AUC_dp": float(np.std(aucs)),
            "AUC_p2_5": float(np.quantile(aucs, 0.025)), "AUC_p97_5": float(np.quantile(aucs, 0.975)),
            "n_referencia": int((y == 0).sum()), "n_EPOC": int((y == 1).sum()),
            "coeficientes": coefs}


def typicality(ds: pd.DataFrame, ep: pd.DataFrame, feature_cols=ALL_FEATURES,
               reference: str = "C") -> tuple[pd.DataFrame, dict]:
    """
    Distância de Mahalanobis de cada gravação à distribuição dos controles do
    ds004504 (covariância de Ledoit-Wolf, estável com poucos sujeitos).

    Calibração: a distância de cada controle é calculada deixando-o de fora do
    ajuste; o percentil 95 dessas distâncias define o limite "típico de um
    controle". As distâncias dos pacientes com AD mostram quanto essa medida
    separa os grupos dentro do ds004504.
    """
    cols = list(feature_cols)
    ref = ds[ds["Group"] == reference].dropna(subset=cols)

    def fit(df):
        mu, sd = df[cols].mean(), df[cols].std(ddof=1).replace(0, np.nan)
        z = ((df[cols] - mu) / sd).to_numpy(float)
        return mu, sd, LedoitWolf().fit(z)

    def dist(model, df):
        mu, sd, lw = model
        z = ((df[cols] - mu) / sd).to_numpy(float)
        return lw.mahalanobis(z)

    loo = np.array([dist(fit(ref.drop(index=i)), ref.loc[[i]])[0] for i in ref.index])
    limit = float(np.quantile(loo, 0.95))
    full = fit(ref)
    rows = [pd.DataFrame({"Subject_ID": ref["Subject_ID"].values, "base": "ds004504 CN (fora do ajuste)", "d2": loo})]
    ad = ds[ds["Group"] == "A"].dropna(subset=cols)
    if len(ad):
        rows.append(pd.DataFrame({"Subject_ID": ad["Subject_ID"].values, "base": "ds004504 AD", "d2": dist(full, ad)}))
    epc = ep.dropna(subset=cols)
    rows.append(pd.DataFrame({"Subject_ID": epc["Subject_ID"].values, "base": "EPOC X", "d2": dist(full, epc)}))
    table = pd.concat(rows, ignore_index=True)
    table["tipico_de_controle"] = table["d2"] <= limit
    summary = {"limite_p95_controles": limit,
               **{f"frac_tipica_{b}": float(g["tipico_de_controle"].mean()) for b, g in table.groupby("base")}}
    return table, summary


def pca_projection(ds: pd.DataFrame, ep: pd.DataFrame, feature_cols=ALL_FEATURES, n_components: int = 2):
    """Componentes principais ajustados SÓ no ds004504; o EPOC X é projetado nelas."""
    cols = list(feature_cols)
    d = ds.dropna(subset=cols)
    e = ep.dropna(subset=cols)
    scaler = StandardScaler().fit(d[cols])
    pca = PCA(n_components=n_components, random_state=SEED).fit(scaler.transform(d[cols]))
    out = pd.concat([
        pd.DataFrame(pca.transform(scaler.transform(d[cols])), columns=[f"PC{i+1}" for i in range(n_components)])
          .assign(Subject_ID=d["Subject_ID"].values, base=np.where(d["Group"] == "A", "ds004504 AD", "ds004504 CN")),
        pd.DataFrame(pca.transform(scaler.transform(e[cols])), columns=[f"PC{i+1}" for i in range(n_components)])
          .assign(Subject_ID=e["Subject_ID"].values, base="EPOC X"),
    ], ignore_index=True)
    return out, pca.explained_variance_ratio_


# --------------------------------------------------------------------------
# 4. Estrutura de correlação
# --------------------------------------------------------------------------
def correlation_structure(ds: pd.DataFrame, ep: pd.DataFrame, feature_cols=ALL_FEATURES,
                          reference: str = "C") -> dict:
    """
    Correlações de Spearman entre todas as features em cada base, comparadas pela
    correlação entre os triângulos superiores das duas matrizes. Perto de 1: as
    features se relacionam entre si do mesmo jeito nas duas bases.
    """
    cols = [c for c in feature_cols if ds[c].std() > 0 and ep[c].std() > 0]
    ref = ds if reference is None else ds[ds["Group"] == reference]
    if len(ep) < MIN_N:
        return {"r_estrutura": np.nan, "nota": f"menos de {MIN_N} gravações no EPOC X"}
    a = ref[cols].corr(method="spearman").to_numpy()
    b = ep[cols].corr(method="spearman").to_numpy()
    iu = np.triu_indices(len(cols), 1)
    ok = ~(np.isnan(a[iu]) | np.isnan(b[iu]))
    return {"r_estrutura": float(stats.spearmanr(a[iu][ok], b[iu][ok]).statistic),
            "n_pares": int(ok.sum()), "n_features": len(cols)}


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def plot_shift_vs_disease(table: pd.DataFrame, out_path=None):
    """
    Por feature: g do EPOC X contra os controles (ponto, com IC95%) e o efeito da
    doença no ds004504 (faixa cinza de -|d| a +|d|). Pontos dentro da faixa: a
    diferença entre bases é menor que o efeito da doença.
    """
    import matplotlib.pyplot as plt

    t = table.sort_values("g_EPOC_vs_CN", key=np.abs, ascending=True).reset_index(drop=True)
    colors = {"compatível com os controles": "#2b8cbe",
              "deslocada, dentro do efeito da doença": "#fdae61",
              "deslocamento maior que o efeito da doença": "#d7191c", "indefinido": "0.6"}
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(t) + 1.5))
    for i, r in t.iterrows():
        d = abs(r["d_AD_vs_CN"]) if np.isfinite(r["d_AD_vs_CN"]) else 0
        ax.plot([-d, d], [i, i], color="0.85", lw=6, solid_capstyle="butt", zorder=1)
        incerta = bool(r.get("classificacao_incerta", False))
        ax.errorbar(r["g_EPOC_vs_CN"], i, xerr=[[r["g_EPOC_vs_CN"] - r["g_IC_inf"]], [r["g_IC_sup"] - r["g_EPOC_vs_CN"]]],
                    fmt="o", color=colors[r["categoria"]], ms=5, capsize=2, zorder=2,
                    markerfacecolor="white" if incerta else colors[r["categoria"]])
    ax.axvline(0, color="0.4", lw=1)
    for x in (-SMALL_EFFECT, SMALL_EFFECT):
        ax.axvline(x, color="0.6", ls=":", lw=1)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["feature"], fontsize=8)
    ax.set_xlabel("g de Hedges: EPOC X - controles do ds004504 (IC95%)")
    ax.set_title("Diferença entre as bases (pontos) e efeito da doença no ds004504 (faixa cinza)", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items() if k in set(t["categoria"])]
    if "classificacao_incerta" in t and t["classificacao_incerta"].any():
        handles.append(plt.Line2D([], [], marker="o", ls="", color="0.4", markerfacecolor="white",
                                  label="vazado: classificação incerta (IC cruza |g| = 0,5)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pca(proj: pd.DataFrame, var_ratio, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    style = {"ds004504 CN": ("#2b8cbe", "o"), "ds004504 AD": ("#d7191c", "o"), "EPOC X": ("#1a9641", "^")}
    for base, g in proj.groupby("base"):
        c, m = style.get(base, ("0.5", "o"))
        ax.scatter(g["PC1"], g["PC2"], c=c, marker=m, s=40, alpha=0.8, label=f"{base} (n={len(g)})",
                   edgecolors="white", linewidths=0.5)
    ax.set_xlabel(f"PC1 ({100 * var_ratio[0]:.0f}% da variância do ds004504)")
    ax.set_ylabel(f"PC2 ({100 * var_ratio[1]:.0f}%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Componentes ajustados no ds004504; EPOC X projetado", fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_typicality(table: pd.DataFrame, limit: float, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["ds004504 CN (fora do ajuste)", "ds004504 AD", "EPOC X"]
    colors = ["#2b8cbe", "#d7191c", "#1a9641"]
    data = [np.sqrt(table.loc[table["base"] == b, "d2"]) for b in order if b in set(table["base"])]
    labels = [b for b in order if b in set(table["base"])]
    # Compatível com várias versões do matplotlib: na 3.11, 'labels' foi removido
    # do boxplot e 'vert' deu lugar a 'orientation'. Os rótulos são postos à mão.
    try:
        ax.boxplot(data, orientation="horizontal", showfliers=False)
    except TypeError:
        ax.boxplot(data, vert=False, showfliers=False)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    for i, (d, c) in enumerate(zip(data, colors), 1):
        ax.scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), s=12, c=c, alpha=0.7)
    ax.axvline(np.sqrt(limit), color="0.4", ls="--", lw=1, label="percentil 95 dos controles")
    ax.set_xlabel("distância de Mahalanobis aos controles do ds004504")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# 5. Candidatas a features estáveis
# --------------------------------------------------------------------------
MAX_SHIFT = 0.5        # |g| EPOC X x controles abaixo disso: deslocamento pequeno
MIN_DISEASE = 0.8      # |d| AD x CN no ds004504 a partir disso: efeito da doença grande
MIN_RETEST = 0.8       # correlação entre run-1 e run-2 a partir disso: estável entre sessões


def candidate_features(table: pd.DataFrame, reliability: pd.DataFrame,
                       max_shift: float = MAX_SHIFT, min_disease: float = MIN_DISEASE,
                       min_retest: float = MIN_RETEST) -> pd.DataFrame:
    """
    Cruza, para cada feature, os três critérios de "feature estável e útil":

      1. deslocamento pequeno entre as bases: |g EPOC X x controles| < max_shift
      2. efeito da doença grande no ds004504:  |d AD x CN| >= min_disease
      3. estável entre sessões no EPOC X:     r entre run-1 e run-2 >= min_retest

    Candidata = atende aos três. A coluna 'deslocamento_menor_conclusivo' é mais
    exigente: o IC95% inteiro do g fica abaixo do efeito da doença, isto é, a
    diferença entre as bases é comprovadamente menor que a diferença entre AD e
    controles. Com poucos participantes, é raro alguma feature chegar lá.

    table       : saída de per_feature_table
    reliability : tabela com 'feature' e 'r_entre_runs' (seção de confiabilidade)
    """
    cols = ["feature", "r_entre_runs"] + [c for c in ("confiabilidade_ds004504",) if c in reliability]
    t = table.merge(reliability[cols], on="feature", how="left")
    abs_g, abs_d = t["g_EPOC_vs_CN"].abs(), t["d_AD_vs_CN"].abs()
    t["c1_deslocamento_pequeno"] = abs_g < max_shift
    t["c2_efeito_doenca_grande"] = abs_d >= min_disease
    t["c3_estavel_entre_sessoes"] = t["r_entre_runs"] >= min_retest
    crit = ["c1_deslocamento_pequeno", "c2_efeito_doenca_grande", "c3_estavel_entre_sessoes"]
    t["n_criterios"] = t[crit].sum(axis=1)
    t["candidata"] = t["n_criterios"] == 3
    t["deslocamento_menor_conclusivo"] = np.maximum(t["g_IC_inf"].abs(), t["g_IC_sup"].abs()) < abs_d
    keep = ["feature", "g_EPOC_vs_CN", "g_IC_inf", "g_IC_sup", "d_AD_vs_CN", "r_entre_runs"] + \
           [c for c in ("confiabilidade_ds004504",) if c in t] + crit + \
           ["n_criterios", "candidata", "deslocamento_menor_conclusivo"]
    return (t[keep].sort_values(["n_criterios", "d_AD_vs_CN"], ascending=[False, False],
                                key=lambda s: s.abs() if s.name == "d_AD_vs_CN" else s)
            .reset_index(drop=True))


def plot_candidates(cand: pd.DataFrame, max_shift: float = MAX_SHIFT, min_disease: float = MIN_DISEASE,
                    min_retest: float = MIN_RETEST, out_path=None):
    """
    Eixo x: deslocamento entre as bases (|g|). Eixo y: efeito da doença (|d|).
    Cor: estabilidade entre sessões. O canto superior esquerdo (deslocamento
    pequeno, efeito grande) é onde ficam as features úteis; as com borda preta
    também são estáveis entre sessões (as candidatas).
    """
    import matplotlib.pyplot as plt

    x, y = cand["g_EPOC_vs_CN"].abs(), cand["d_AD_vs_CN"].abs()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(0, max_shift, ymin=0, ymax=1, color="#e8f4ea", zorder=0)
    ax.axhline(min_disease, color="0.5", ls="--", lw=1)
    ax.axvline(max_shift, color="0.5", ls="--", lw=1)
    sc = ax.scatter(x, y, c=cand["r_entre_runs"], cmap="viridis", vmin=0, vmax=1, s=60,
                    edgecolors=np.where(cand["candidata"], "black", "none"), linewidths=1.8, zorder=3)
    for xi, yi, name, is_c in zip(x, y, cand["feature"], cand["candidata"]):
        if is_c or (xi < max_shift * 2 and yi >= min_disease * 0.75):
            ax.annotate(name, (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8,
                        fontweight="bold" if is_c else "normal")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("deslocamento entre as bases: |g| EPOC X - controles (escala log acima de 1)")
    ax.set_ylabel("efeito da doença no ds004504: |d| AD - controles")
    fig.colorbar(sc, ax=ax, label="teste-reteste no EPOC X (run-1 x run-2)")
    ax.set_title(f"Candidatas (borda preta): |g| < {max_shift}, |d| ≥ {min_disease}, reteste ≥ {min_retest}",
                 fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
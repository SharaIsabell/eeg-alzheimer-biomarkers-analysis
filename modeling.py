"""
Modelagem: pipelines, validação leave-one-subject-out (LOSO) e estatística.

É usado pelo models_training.ipynb (avaliação), pelo train_final_models.py
(modelo final aplicado ao BrainLat) e pelo brainlat_transfer.py, para que o
modelo avaliado e o modelo salvo sejam exatamente o mesmo objeto.

Correções em relação à versão 1.0 (ver CORRECOES.md):
  - SVM sem calibração de Platt: a pontuação contínua é decision_function, a
    mesma quantidade que define a predição. Na versão 1.0, predict e
    predict_proba discordavam, e a calibração interna de Platt fazia a AUC
    refletir a composição de cada dobra e a ordem das linhas (AUC = 1,0000 com
    acurácia 0,46 no conjunto PAC).
  - Ordem dos sujeitos embaralhada com semente fixa antes do LOSO, para que o
    rótulo não fique correlacionado com a posição da linha.
  - Detecção de modelos degenerados (pontuação constante no treino).
  - Random Forest com o mesmo imputador no notebook e no modelo final.
  - Intervalos de confiança (Wilson para proporções, bootstrap estratificado para
    AUC), McNemar exato, teste de permutação, análise univariada com correção
    para comparações múltiplas e modelo só com covariáveis.
  - Resultados gravados de uma vez (sem modo append).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_sets import (ALL_FEATURES, FEATURE_SETS, MODEL_NAMES, POSITIVE_GROUP,
                          NEGATIVE_GROUP, BASELINE_SET, COMBINED_SET)

SEED = 42
N_BOOT = 2000
DEGENERATE_SHARE = 0.9   # dobra degenerada: >= 90% das pontuações de treino praticamente iguais


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def subject_table(epochs: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame:
    """
    Média das épocas por sujeito (feita só com dados do próprio sujeito, então
    não há vazamento). Mantém covariáveis se existirem. y = 1 para AD.
    """
    df = epochs[epochs["Group"].isin([POSITIVE_GROUP, NEGATIVE_GROUP])]
    cov = [c for c in ("Age", "Gender", "MMSE") if c in df.columns]
    subj = df.groupby(["Subject_ID", "Group"] + cov, as_index=False, dropna=False)[list(feature_cols)].mean()
    subj["y"] = (subj["Group"] == POSITIVE_GROUP).astype(int)
    return subj.sort_values("Subject_ID").reset_index(drop=True)


# --------------------------------------------------------------------------
# Pipelines (fonte única)
# --------------------------------------------------------------------------
def make_pipeline(model_name: str, y: np.ndarray) -> Pipeline:
    if model_name == "Random Forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            # n_jobs=1: com dezenas de sujeitos, paralelizar não compensa, e versões
            # recentes do scikit-learn/joblib emitem avisos em cada dobra do LOSO.
            # O resultado é o mesmo com qualquer n_jobs (a semente é fixa).
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=SEED, n_jobs=1)),
        ])
    if model_name == "SVM (Linear)":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", class_weight="balanced", random_state=SEED)),
        ])
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        neg, pos = int(np.sum(y == 0)), int(np.sum(y == 1))
        return Pipeline([
            ("clf", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  scale_pos_weight=neg / pos if pos else 1.0,
                                  random_state=SEED, n_jobs=-1, eval_metric="logloss")),
        ])
    raise ValueError(model_name)


def continuous_score(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Pontuação contínua coerente com .predict:
      SVM -> decision_function (predição = pontuação > 0)
      RF, XGBoost -> probabilidade de AD (predição = probabilidade > 0,5)
    """
    clf = pipe[-1]
    if isinstance(clf, SVC):
        return pipe.decision_function(X)
    return pipe.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# LOSO
# --------------------------------------------------------------------------
def _is_degenerate(train_scores: np.ndarray) -> bool:
    """
    O modelo não separa os sujeitos de treino: a grande maioria recebe a mesma
    pontuação (foi o que aconteceu no SVM com PAC na versão 1.0, em que 64 de 65
    sujeitos tinham decisão ~ -1). Um único valor extremo não mascara o critério,
    ao contrário do desvio-padrão.
    """
    s = np.asarray(train_scores, float)
    med = np.median(s)
    tol = 1e-3 * (1.0 + abs(med))
    return bool(np.mean(np.abs(s - med) < tol) >= DEGENERATE_SHARE)

def loso_predict(subj: pd.DataFrame, cols, model_name: str, y: np.ndarray | None = None,
                 seed: int = SEED) -> pd.DataFrame:
    """
    Leave-one-subject-out. Uma linha por sujeito: y, pred, score, degenerate.
    Os sujeitos são embaralhados com semente fixa antes da validação.
    """
    order = np.random.default_rng(seed).permutation(len(subj))
    s = subj.iloc[order].reset_index(drop=True)
    X = s[list(cols)].to_numpy(dtype=float)
    yy = (s["y"].to_numpy() if y is None else np.asarray(y)[order]).astype(int)

    preds, scores, degen = np.zeros(len(s), int), np.zeros(len(s)), np.zeros(len(s), bool)
    for i in range(len(s)):
        tr = np.arange(len(s)) != i
        pipe = make_pipeline(model_name, yy[tr])
        pipe.fit(X[tr], yy[tr])
        preds[i] = int(pipe.predict(X[[i]])[0])
        scores[i] = float(continuous_score(pipe, X[[i]])[0])
        degen[i] = _is_degenerate(continuous_score(pipe, X[tr]))
    return pd.DataFrame({"Subject_ID": s["Subject_ID"].values, "y": yy,
                         "pred": preds, "score": scores, "degenerate": degen})


# --------------------------------------------------------------------------
# Métricas e intervalos
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def bootstrap_auc_ci(y, score, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05):
    """Bootstrap estratificado (reamostra AD e controles separadamente)."""
    y, score = np.asarray(y), np.asarray(score)
    if len(set(y)) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs[b] = roc_auc_score(y[idx], score[idx])
    return tuple(np.quantile(aucs, [alpha / 2, 1 - alpha / 2]))


def metrics(y, pred, score) -> dict:
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    acc_lo, acc_hi = wilson_ci(tp + tn, n)
    sens_lo, sens_hi = wilson_ci(tp, tp + fn)
    spec_lo, spec_hi = wilson_ci(tn, tn + fp)
    auc = roc_auc_score(y, score) if len(set(y)) == 2 else np.nan
    auc_lo, auc_hi = bootstrap_auc_ci(y, score)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n": n, "Accuracy": (tp + tn) / n, "Acc_IC_inf": acc_lo, "Acc_IC_sup": acc_hi,
        "Sensibilidade": sens, "Sens_IC_inf": sens_lo, "Sens_IC_sup": sens_hi,
        "Especificidade": spec, "Espec_IC_inf": spec_lo, "Espec_IC_sup": spec_hi,
        "Acuracia_balanceada": np.nanmean([sens, spec]),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": auc, "AUC_IC_inf": auc_lo, "AUC_IC_sup": auc_hi,
        "Classe_majoritaria": max(y.mean(), 1 - y.mean()),
        "Frac_prevista_AD": pred.mean(),
    }


# --------------------------------------------------------------------------
# Bateria completa
# --------------------------------------------------------------------------
def run_all(subj: pd.DataFrame, models=MODEL_NAMES, feature_sets=FEATURE_SETS,
            verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia todas as combinações. Retorna (resultados, predições por sujeito)."""
    res, preds = [], []
    for model in models:
        for cat, cols in feature_sets.items():
            p = loso_predict(subj, cols, model)
            m = metrics(p["y"], p["pred"], p["score"])
            m.update({"Modelo": model, "Categoria": cat,
                      "Dobras_degeneradas": int(p["degenerate"].sum())})
            res.append(m)
            preds.append(p.assign(Modelo=model, Categoria=cat))
            if verbose:
                flag = f"  [ATENÇÃO: {m['Dobras_degeneradas']} dobras degeneradas]" if m["Dobras_degeneradas"] else ""
                print(f"{model:14s} | {cat:22s} | acc {m['Accuracy']:.3f} "
                      f"[{m['Acc_IC_inf']:.3f}; {m['Acc_IC_sup']:.3f}] | AUC {m['AUC']:.3f} "
                      f"[{m['AUC_IC_inf']:.3f}; {m['AUC_IC_sup']:.3f}]{flag}")
    cols = ["Modelo", "Categoria"] + [k for k in res[0] if k not in ("Modelo", "Categoria")]
    return pd.DataFrame(res)[cols], pd.concat(preds, ignore_index=True)


# --------------------------------------------------------------------------
# Comparações
# --------------------------------------------------------------------------
def mcnemar_exact(y, pred_a, pred_b) -> dict:
    """b = só A acerta; c = só B acerta. Teste binomial exato bilateral."""
    y, a, b_ = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = a == y, b_ == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"so_A_acerta": b, "so_B_acerta": c, "p": float(p)}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj


def compare_sets(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Comparação PRIMÁRIA (definida antes de olhar os resultados): todas as
    features contra RBP, em cada classificador, com correção de Holm para os 3
    testes. Comparação SECUNDÁRIA (exploratória): todas as features contra o
    melhor conjunto isolado do mesmo classificador.
    """
    rows = []
    for model, g in preds.groupby("Modelo", sort=False):
        piv = g.pivot(index="Subject_ID", columns="Categoria", values="pred")
        y = g.drop_duplicates("Subject_ID").set_index("Subject_ID").loc[piv.index, "y"]
        acc = {cat: float((piv[cat] == y).mean()) for cat in piv.columns}
        singles = [c for c in piv.columns if c != COMBINED_SET]
        best = max(singles, key=lambda c: acc[c])
        for kind, other in (("primária", BASELINE_SET), ("secundária (exploratória)", best)):
            t = mcnemar_exact(y, piv[COMBINED_SET], piv[other])
            rows.append({"Modelo": model, "Tipo": kind, "A": COMBINED_SET, "B": other,
                         "Acc_A": acc[COMBINED_SET], "Acc_B": acc[other], **t})
    out = pd.DataFrame(rows)
    prim = out["Tipo"] == "primária"
    out.loc[prim, "p_Holm"] = holm(out.loc[prim, "p"])
    return out


def permutation_test(subj: pd.DataFrame, cols, model_name: str, n_perm: int = 200,
                     seed: int = SEED, verbose: bool = True) -> dict:
    """
    AUC observada contra a distribuição de AUC com rótulos embaralhados, usando
    exatamente o mesmo LOSO. p = (1 + nº de AUC permutadas >= observada) / (1 + n_perm).
    """
    obs = loso_predict(subj, cols, model_name)
    auc_obs = roc_auc_score(obs["y"], obs["score"])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = rng.permutation(subj["y"].to_numpy())
        pr = loso_predict(subj, cols, model_name, y=yp)
        null[k] = roc_auc_score(pr["y"], pr["score"])
        if verbose and (k + 1) % 50 == 0:
            print(f"  {k + 1}/{n_perm} permutações")
    return {"AUC_observada": auc_obs, "AUC_nula_media": float(null.mean()),
            "AUC_nula_p95": float(np.quantile(null, 0.95)),
            "p": float((1 + np.sum(null >= auc_obs)) / (1 + n_perm)), "nula": null}


# --------------------------------------------------------------------------
# Estatística univariada, covariáveis e outliers
# --------------------------------------------------------------------------
def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def univariate_table(subj: pd.DataFrame, feature_cols=ALL_FEATURES) -> pd.DataFrame:
    """d de Cohen, AUC univariada (Mann-Whitney) e p com correções de Holm e BH."""
    ad, cn = subj[subj["y"] == 1], subj[subj["y"] == 0]
    rows = []
    for c in feature_cols:
        a, b = ad[c].dropna(), cn[c].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": c, "d_Cohen": cohen_d(a, b), "AUC_univariada": u / (len(a) * len(b)),
                     "p": p, "n_AD": len(a), "n_CN": len(b)})
    out = pd.DataFrame(rows)
    out["p_Holm"] = holm(out["p"])
    out["p_BH"] = benjamini_hochberg(out["p"])
    return out.sort_values("p").reset_index(drop=True)


def covariate_descriptives(subj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, d in subj.groupby("Group"):
        r = {"Grupo": g, "n": len(d)}
        for c in ("Age", "MMSE"):
            if c in d:
                r[f"{c} (média ± DP)"] = f"{d[c].mean():.1f} ± {d[c].std():.1f}"
        if "Gender" in d:
            r["Mulheres"] = int((d["Gender"].astype(str).str.upper() == "F").sum())
        rows.append(r)
    return pd.DataFrame(rows)


def covariate_only_auc(subj: pd.DataFrame) -> dict:
    """
    LOSO com regressão logística usando só idade e sexo. Mostra quanto da
    separação AD/CN as covariáveis demográficas explicariam sozinhas.
    O MMSE não entra: ele faz parte do critério diagnóstico.
    """
    if not {"Age", "Gender"} <= set(subj.columns):
        return {"AUC": np.nan, "nota": "idade/sexo ausentes"}
    X = np.c_[subj["Age"].astype(float), (subj["Gender"].astype(str).str.upper() == "F").astype(float)]
    y = subj["y"].to_numpy()
    scores = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression())]).fit(X[tr], y[tr])
        scores[i] = pipe.predict_proba(X[[i]])[0, 1]
    lo, hi = bootstrap_auc_ci(y, scores)
    return {"AUC": roc_auc_score(y, scores), "AUC_IC_inf": lo, "AUC_IC_sup": hi}


def outlier_report(subj: pd.DataFrame, feature_cols=ALL_FEATURES, z_thr: float = 5.0) -> pd.DataFrame:
    """Escore z robusto (mediana e MAD). Sinaliza para inspeção; não remove ninguém."""
    rows = []
    for c in feature_cols:
        x = subj[c].astype(float)
        med = x.median()
        mad = (x - med).abs().median()
        if not mad or np.isnan(mad):
            continue
        z = 0.6745 * (x - med) / mad
        for i in np.where(z.abs() > z_thr)[0]:
            rows.append({"Subject_ID": subj["Subject_ID"].iloc[i], "feature": c,
                         "valor": x.iloc[i], "mediana": med, "z_robusto": round(float(z.iloc[i]), 1)})
    return pd.DataFrame(rows, columns=["Subject_ID", "feature", "valor", "mediana", "z_robusto"])


# --------------------------------------------------------------------------
# Figuras e resumo
# --------------------------------------------------------------------------
def plot_performance(results: pd.DataFrame, metric: str, out_path: str | Path):
    """Pontos com IC95%, eixo de 0 a 1 e linha de referência (acaso)."""
    import matplotlib.pyplot as plt

    lo, hi = {"Accuracy": ("Acc_IC_inf", "Acc_IC_sup"), "AUC": ("AUC_IC_inf", "AUC_IC_sup")}[metric]
    ref = results["Classe_majoritaria"].iloc[0] if metric == "Accuracy" else 0.5
    ref_label = (f"classe majoritária ({ref:.1%})" if metric == "Accuracy" else "acaso (0,5)")
    cats = list(dict.fromkeys(results["Categoria"]))
    models = list(dict.fromkeys(results["Modelo"]))
    colors = ["#341671", "#AE2D68", "#F54952"]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(cats) * len(models) + 1.5))
    y = 0
    ticks, labels = [], []
    for cat in cats:
        for k, model in enumerate(models):
            r = results[(results["Categoria"] == cat) & (results["Modelo"] == model)].iloc[0]
            ax.errorbar(r[metric], y, xerr=[[r[metric] - r[lo]], [r[hi] - r[metric]]],
                        fmt="o", color=colors[k % 3], capsize=3, label=model if cat == cats[0] else None)
            ticks.append(y)
            labels.append(f"{cat} · {model}")
            y += 1
        y += 0.6
    ax.axvline(ref, color="0.4", ls="--", lw=1.2, label=ref_label)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{'Acurácia' if metric == 'Accuracy' else 'AUC'} (LOSO, IC95%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def results_markdown(results: pd.DataFrame, comparisons: pd.DataFrame,
                     covariates: dict | None = None) -> str:
    """Resumo em Markdown gerado a partir dos resultados (referenciado pelo README)."""
    def ci(v, a, b, pct=False):
        f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
        return f"{f(v)} [{f(a)}; {f(b)}]"

    lines = ["# Resumo dos resultados (gerado por models_training.ipynb)", "",
             f"Sujeitos: {int(results['n'].iloc[0])}. Classe majoritária: "
             f"{100 * results['Classe_majoritaria'].iloc[0]:.1f}%. IC95%: Wilson (acurácia) "
             f"e bootstrap estratificado com {N_BOOT} réplicas (AUC).", "",
             "| Modelo | Conjunto | Acurácia | AUC | Sensib. | Especif. |",
             "|---|---|---|---|---|---|"]
    for r in results.sort_values("AUC", ascending=False).itertuples():
        lines.append(f"| {r.Modelo} | {r.Categoria} | {ci(r.Accuracy, r.Acc_IC_inf, r.Acc_IC_sup, True)} | "
                     f"{ci(r.AUC, r.AUC_IC_inf, r.AUC_IC_sup)} | {100 * r.Sensibilidade:.1f}% | "
                     f"{100 * r.Especificidade:.1f}% |")
    lines += ["", "## Comparações (McNemar exato)", "",
              "| Modelo | Tipo | A | B | Acc A | Acc B | só A acerta | só B acerta | p | p Holm |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comparisons.itertuples():
        ph = "" if pd.isna(getattr(r, "p_Holm", np.nan)) else f"{r.p_Holm:.4f}"
        lines.append(f"| {r.Modelo} | {r.Tipo} | {r.A} | {r.B} | {r.Acc_A:.3f} | {r.Acc_B:.3f} | "
                     f"{r.so_A_acerta} | {r.so_B_acerta} | {r.p:.4f} | {ph} |")
    if covariates and not pd.isna(covariates.get("AUC", np.nan)):
        lines += ["", f"Modelo só com idade e sexo: AUC {ci(covariates['AUC'], covariates['AUC_IC_inf'], covariates['AUC_IC_sup'])}."]
    degen = results[results["Dobras_degeneradas"] > 0]
    if len(degen):
        lines += ["", "Combinações com dobras degeneradas (pontuação constante no treino; "
                  "interpretar com cautela): " + ", ".join(f"{r.Modelo} · {r.Categoria}" for r in degen.itertuples()) + "."]
    return "\n".join(lines) + "\n"


def plot_dashboard(p: pd.DataFrame, model_name: str, category: str, out_dir: str | Path = "images"):
    """Matriz de confusão e curva ROC (LOSO) de uma combinação, a partir de loso_predict."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y, pred, score = p["y"].to_numpy(), p["pred"].to_numpy(), p["score"].to_numpy()
    m = metrics(y, pred, score)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        a1.text(j, i, int(v), ha="center", va="center", fontsize=14)
    a1.set_xticks([0, 1], ["CN", "AD"])
    a1.set_yticks([0, 1], ["CN", "AD"])
    a1.set_xlabel("previsto")
    a1.set_ylabel("verdadeiro")
    a1.set_title(f"acurácia {m['Accuracy']:.3f} [{m['Acc_IC_inf']:.3f}; {m['Acc_IC_sup']:.3f}]")
    fpr, tpr, _ = roc_curve(y, score)
    a2.plot(fpr, tpr, color="#E66101", lw=2.5)
    a2.plot([0, 1], [0, 1], "k--", lw=1)
    a2.set_xlabel("1 - especificidade")
    a2.set_ylabel("sensibilidade")
    a2.set_title(f"AUC {m['AUC']:.3f} [{m['AUC_IC_inf']:.3f}; {m['AUC_IC_sup']:.3f}]")
    fig.suptitle(f"{model_name} · {category} (LOSO, n={m['n']})", fontsize=13)
    fig.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"dashboard_{model_name}_{category}".replace(" ", "_").replace("(", "").replace(")", "").replace("·", "")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out / f"{name}.png"
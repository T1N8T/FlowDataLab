"""islr_escalera.py · La escalera de modelos "a priori" de entrega/justificacion_islr.md: de menos a más flexible, todos con la
misma CV por controlador (datos/folds.csv), las mismas variables y la misma semilla. Los hiperparámetros que se eligen
(λ de ridge/lasso/GAM, tamaño del árbol) se eligen con CV anidada: los 4 folds de entrenamiento hacen de folds internos.

Peldaños (ISLR):
  prior_tipo     §4.4    solo el tipo de tarea: score = log(lift del tipo) aprendido en el train del fold (π_k)
  naive_bayes    §4.4.4  GaussianNB fila a fila
  lda            §4.4.2  LDA fila a fila (con shrinkage de Ledoit-Wolf, por estabilidad)
  logit_fila     §4.3    logística binaria fila a fila (cada fila como si fuera independiente)
  logit_cond     §4.3.5  logística condicional: softmax dentro de cada lista (elegir 1 de m), casi sin penalizar
  ridge_cond     §6.2.1  la misma con penalización L2, λ por CV anidada
  lasso_cond     §6.2.2  la misma con penalización L1, λ por CV anidada (imprime cuántos coeficientes quedan ≠ 0)
  gam_cond       §7.7    GAM: splines cúbicos (5 nudos) en las 40 variables continuas más correladas con y (elegidas en el
                         train del fold) + el resto lineal, logística condicional con L2 y λ por CV anidada
  arbol          §8.1    un árbol de clasificación, nº de hojas por CV anidada
  bagging        §8.2.1  bagging de árboles (random forest con todas las variables en cada corte)
  random_forest  §8.2.2  random forest (√p variables en cada corte)
  boosting       §8.2.3  boosting de árboles (LightGBM binario)
  boosting_rank  —       lo mismo con objetivo de ranking (LambdaRank): fuera de ISLR, es el modelo del proyecto
  red            §10     red con dos capas ocultas (64, 32), MLP de scikit-learn con early stopping

Conjuntos de información (--info), para ver que la escalera se reordena según la señal disponible (ISLR §2.2.2):
  snapshot  las columnas originales del CSV (lo que se tendría "a priori", sin ingeniería de variables)
  realista  features_realista.construir(): snapshot + pasado
  futuro    features_futuro.construir(): + futuro de la sesión (sin oráculo ni segundo paso)

Uso:
  python entrega/codigo/islr_escalera.py --rapido                     # comprobación: fold 0, 20 % del train, no guarda
  python entrega/codigo/islr_escalera.py --info snapshot              # escalera completa (guarda cada OOF al acabar)
  python entrega/codigo/islr_escalera.py --info futuro --modelos ridge_cond boosting
  --forzar recalcula los OOF que ya existan (por defecto se reutilizan: si se corta, al relanzar continúa).
Salidas: salidas/oof/escalera_<info>_<modelo>.csv y salidas/notas/islr_escalera_<info>.csv (resumen + pruebas por pares).
"""
import argparse, time, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from comun import labels, hits, leer, guardar, DATOS, SALIDAS, CV023
from islr_pruebas import perm_signo, boot_ic, p_predictiva, holm, mde

warnings.filterwarnings("ignore")
SEMILLA = 0
META = {'sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index', 'cs', 't',
        'split', 'fold', 'block'}
PELDANOS = ["prior_tipo", "naive_bayes", "lda", "logit_fila", "logit_cond", "ridge_cond", "lasso_cond", "gam_cond",
            "arbol", "bagging", "random_forest", "boosting", "boosting_rank", "red"]
REJILLA = {"ridge_cond": [1e-4, 1e-3, 1e-2, 1e-1], "lasso_cond": [1e-4, 3e-4, 1e-3, 3e-3],
           "gam_cond": [1e-4, 1e-3, 1e-2, 1e-1], "arbol": [16, 64, 256, 1024]}


# ------------------------------------------------------------------------------------------------------------------ datos
def cargar(info):
    """DataFrame de train+validation ordenado por sample_id, row_index, con fold y target_index, y la lista de features."""
    if info == "snapshot":
        d = pd.concat([pd.read_csv(DATOS / f"{f}.csv") for f in ("train", "validation")], ignore_index=True)
        folds = pd.read_csv(DATOS / "folds.csv")
        d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold)))
        cols = [c for c in d.columns if c not in META and pd.api.types.is_numeric_dtype(d[c])]
    else:
        mod = __import__("features_realista" if info == "realista" else "features_futuro")
        d, _, fs = mod.construir()
        cols = [c for c in sum(fs.values(), []) if c not in META]
    d = d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    return d, list(dict.fromkeys(cols))


class Prep:
    """Para los modelos no basados en árboles: recorte a los percentiles 1–99, imputación por la mediana + indicador de
    faltante, estandarización y one-hot de task_index. Todo ajustado solo con el train del fold."""
    def fit(self, D, cols):
        self.num = [c for c in cols if c != "task_index"]
        X = D[self.num].astype("float64")
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        X = X.clip(self.lo, self.hi, axis=1)
        self.med = X.median()
        self.na = [c for c in self.num if D[c].isna().any()]
        X = X.fillna(self.med)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1).fillna(1)
        self.tipos = sorted(D.task_index.unique()) if "task_index" in cols else []
        return self

    def transform(self, D):
        X = ((D[self.num].astype("float64").clip(self.lo, self.hi, axis=1).fillna(self.med) - self.mu) / self.sd).values
        extra = [D[self.na].isna().values.astype("float64")] if self.na else []
        extra += [(D.task_index.values[:, None] == np.array(self.tipos)[None, :]).astype("float64")] if self.tipos else []
        return np.hstack([X] + extra)


# --------------------------------------------------------------------------------------------- logística condicional
def _grupos(sid):
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    return starts, np.diff(np.r_[starts, len(sid)])


def logit_cond_fit(X, y, sid, lam=1e-6, l1=False):
    """Softmax dentro de cada snapshot (ISLR §4.3.5). Pérdida media por snapshot + λ/2·||w||² (ridge) o λ·||w||₁ (lasso,
    con w = w⁺ − w⁻ y L-BFGS-B con cotas). Sin intercepto: dentro de la lista se cancela."""
    starts, cnt = _grupos(sid)
    G, p = len(starts), X.shape[1]
    pos = np.flatnonzero(y == 1)

    def nll(w):
        s = X @ w
        m = np.maximum.reduceat(s, starts)
        e = np.exp(s - np.repeat(m, cnt))
        Z = np.add.reduceat(e, starts)
        P = e / np.repeat(Z, cnt)
        f = -(s[pos] - m - np.log(Z)).sum() / G
        g = X.T @ (P - y) / G
        return f, g

    if not l1:
        r = minimize(lambda w: tuple(a + b for a, b in zip(nll(w), (lam / 2 * w @ w, lam * w))), np.zeros(p),
                     jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        return r.x

    def obj(z):
        f, g = nll(z[:p] - z[p:])
        return f + lam * z.sum(), np.r_[g + lam, -g + lam]
    r = minimize(obj, np.zeros(2 * p), jac=True, method="L-BFGS-B", bounds=[(0, None)] * (2 * p),
                 options=dict(maxiter=1000))
    return r.x[:p] - r.x[p:]


def top3_rapido(score, y, sid):
    """Acierto top-3 de un vector de scores alineado con y/sid (para la CV interna)."""
    df = pd.DataFrame({"sid": sid, "s": score, "y": y})
    df["r"] = df.groupby("sid").s.rank(ascending=False, method="first")
    return (df[df.y == 1].r <= 3).mean()


# ------------------------------------------------------------------------------------------------------------- modelos
def gam_base(Xtr, ytr, ncont, n_max=40):
    """Índices de las columnas continuas a expandir con splines: las n_max más correladas con y en el train."""
    cand = [j for j in range(ncont) if len(np.unique(Xtr[:5000, j])) >= 10]
    c = np.array([abs(np.corrcoef(Xtr[:, j], ytr)[0, 1]) if Xtr[:, j].std() > 0 else 0 for j in cand])
    return [cand[i] for i in np.argsort(-np.nan_to_num(c))[:n_max]]


def gam_expandir(Xtr, Xte, idx):
    from sklearn.preprocessing import SplineTransformer
    st = SplineTransformer(n_knots=5, degree=3, extrapolation="constant").fit(Xtr[:, idx])
    f = lambda X: np.hstack([np.delete(X, idx, axis=1), st.transform(X[:, idx])])
    return f(Xtr), f(Xte)


def ajustar_predecir(nombre, Dtr, Dte, cols, par=None):
    """Entrena `nombre` en Dtr y devuelve el score de cada fila de Dte. `par`: hiperparámetro elegido (o None)."""
    ytr = Dtr.target_index.values.astype("float64")
    sid_tr = Dtr.sample_id.values
    if nombre == "prior_tipo":
        lift = (Dtr[Dtr.target_index == 1].task_index.value_counts(normalize=True)
                / Dtr.task_index.value_counts(normalize=True)).fillna(0)
        return np.log(Dte.task_index.map(lift).fillna(0).values + 1e-9)

    if nombre in ("arbol", "bagging", "random_forest", "boosting", "boosting_rank"):
        Xtr, Xte = Dtr[cols].astype("float32").values, Dte[cols].astype("float32").values
        if nombre.startswith("boosting"):
            import lightgbm as lgb
            prm = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, random_state=SEMILLA, verbose=-1, n_jobs=-1)
            if nombre == "boosting":
                m = lgb.LGBMClassifier(objective="binary", **prm).fit(Xtr, ytr)
                return m.predict_proba(Xte)[:, 1]
            m = lgb.LGBMRanker(objective="lambdarank", **prm).fit(Xtr, ytr, group=_grupos(sid_tr)[1])
            return m.predict(Xte)
        Xtr, Xte = np.nan_to_num(Xtr, nan=-999.0), np.nan_to_num(Xte, nan=-999.0)
        if nombre == "arbol":
            from sklearn.tree import DecisionTreeClassifier
            m = DecisionTreeClassifier(max_leaf_nodes=int(par), min_samples_leaf=20, random_state=SEMILLA)
        else:
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=200 if nombre == "random_forest" else 100, min_samples_leaf=20,
                                       max_features="sqrt" if nombre == "random_forest" else 1.0, max_samples=0.25,
                                       n_jobs=-1, random_state=SEMILLA)
        return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    pp = Prep().fit(Dtr, cols)
    Xtr, Xte = pp.transform(Dtr), pp.transform(Dte)
    if nombre == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        m = GaussianNB(var_smoothing=1e-6).fit(Xtr, ytr)
        lp = m.predict_joint_log_proba(Xte)
        return lp[:, 1] - lp[:, 0]
    if nombre == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).decision_function(Xte)
    if nombre == "logit_fila":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr).decision_function(Xte)
    if nombre == "red":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, max_iter=50, early_stopping=True,
                          n_iter_no_change=5, random_state=SEMILLA).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if nombre == "gam_cond":
        idx = gam_base(Xtr, ytr, len(pp.num))
        Xtr, Xte = gam_expandir(Xtr, Xte, idx)
    lam = par if par is not None else 1e-6
    w = logit_cond_fit(Xtr, ytr, sid_tr, lam=lam, l1=(nombre == "lasso_cond"))
    if nombre == "lasso_cond":
        print(f"      lasso λ={lam:g}: {int((np.abs(w) > 1e-8).sum())}/{len(w)} coeficientes ≠ 0", flush=True)
    return Xte @ w


def elegir_par(nombre, Dtr, cols):
    """CV anidada: los folds de Dtr hacen de folds internos; se elige el valor de REJILLA con mejor top-3 medio."""
    if nombre not in REJILLA:
        return None
    res = {}
    for v in REJILLA[nombre]:
        acc = []
        for fi in sorted(Dtr.fold.unique()):
            a, b = Dtr[Dtr.fold != fi], Dtr[Dtr.fold == fi]
            acc.append(top3_rapido(ajustar_predecir(nombre, a, b, cols, v), b.target_index.values, b.sample_id.values))
        res[v] = np.mean(acc)
    mejor = max(res, key=res.get)
    print(f"      CV interna {nombre}: " + ", ".join(f"{k:g}→{v:.4f}" for k, v in res.items()) + f"  ⇒ {mejor:g}",
          flush=True)
    return mejor


def oof_modelo(nombre, D, cols, folds):
    partes = []
    for f in folds:
        t0 = time.time()
        Dtr, Dte = D[D.fold != f], D[D.fold == f]
        par = elegir_par(nombre, Dtr, cols)
        s = ajustar_predecir(nombre, Dtr, Dte, cols, par)
        partes.append(pd.DataFrame({"sample_id": Dte.sample_id.values, "row_index": Dte.row_index.values, "score": s}))
        print(f"   fold {f}: top-3 {top3_rapido(s, Dte.target_index.values, Dte.sample_id.values):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------- main
def resumen(info, nombres):
    lab = labels()
    H = pd.DataFrame({n: hits(leer(f"escalera_{info}_{n}")) for n in nombres}).reindex(lab.index).dropna()
    atco = lab.atco.loc[H.index]
    Pa = H.groupby(atco).mean()
    filas = []
    for i, n in enumerate(nombres):
        r = dict(modelo=n, cv=H[n].mean(), media_atco=Pa[n].mean(), peor_atco=Pa[n].min(),
                 cv_023=H[n][atco.isin(CV023)].mean())
        if i:
            a = nombres[i - 1]; d = (Pa[n] - Pa[a]).values; ic = boot_ic(d)
            r.update(vs_anterior=a, dif=d.mean(), ic95_inf=ic[0], ic95_sup=ic[1], suben=int((d > 0).sum()),
                     bajan=int((d < 0).sum()), p_perm=perm_signo(d), mde_80=mde(d), P_controlador_nuevo=p_predictiva(d))
        filas.append(r)
    R = pd.DataFrame(filas)
    R["p_perm_holm"] = np.nan
    m = R.p_perm.notna()
    R.loc[m, "p_perm_holm"] = holm(R.loc[m, "p_perm"])
    R.to_csv(SALIDAS / "notas" / f"islr_escalera_{info}.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(R.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default="snapshot", choices=["snapshot", "realista", "futuro"])
    ap.add_argument("--modelos", nargs="*", default=PELDANOS)
    ap.add_argument("--rapido", action="store_true", help="fold 0, 20 %% de los snapshots de train, no guarda")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    D, cols = cargar(a.info)
    print(f"info={a.info}: {D.sample_id.nunique()} snapshots, {len(D)} filas, {len(cols)} variables "
          f"({time.time() - t0:.0f}s)", flush=True)
    folds = sorted(D.fold.unique())
    if a.rapido:
        rng = np.random.default_rng(SEMILLA)
        tr = D[D.fold != folds[0]].sample_id.unique()
        keep = set(rng.choice(tr, size=len(tr) // 5, replace=False)) | set(D[D.fold == folds[0]].sample_id)
        D = D[D.sample_id.isin(keep)].reset_index(drop=True)
        folds = folds[:1]

    hechos = []
    for n in a.modelos:
        nombre_oof = f"escalera_{a.info}_{n}"
        if not a.rapido and not a.forzar and leer(nombre_oof) is not None:
            print(f"== {n}: ya existe {nombre_oof}, se reutiliza", flush=True); hechos.append(n); continue
        print(f"== {n}", flush=True)
        t1 = time.time()
        oof = oof_modelo(n, D, cols, folds)
        if not a.rapido:
            guardar(nombre_oof, oof=oof); hechos.append(n)
        print(f"   {n}: {time.time() - t1:.0f}s", flush=True)
    if not a.rapido and hechos:
        print(f"\n== resumen ({a.info})"); resumen(a.info, hechos)


if __name__ == "__main__":
    main()

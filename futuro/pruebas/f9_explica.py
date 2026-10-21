"""f9_explica.py · ¿cómo predice cada modelo enviado y qué cambia en el público? (hilo f9)
Modelos (1.er pase, 1 semilla, mismos parámetros que m_lgb_future.py), los cuatro enviados a Kaggle:
    M0 lgb_future (público 0.942) · A + clear (0.947) · O + clear + oráculo completo (0.940) · B + clear + oráculo de coord. (0.934)
1. SHAP (pred_contrib) en OOF (modelos por fold) y en test (modelo con todo), agrupado por de dónde viene la información:
   snapshot (columnas originales + logic), pasado (p_*, blk_*, gap_*), futuro (f_*), futuro clear (sx_*), otras sesiones (or_*).
   Se mide sobre los 5 primeros candidatos de cada snapshot (donde se decide el top-3), en CV, test 020 (público) y 023.
2. Desplazamiento: validación adversaria (¿se distingue el test de un controlador de los datos etiquetados?) y variables.
3. Dependencia temporal: SHAP medio de f_next_input (s hasta la siguiente instrucción) y p_task_start (s en la lista) por tramos.
4. Test 020: snapshots en que el top-3 de B difiere del de A y qué grupo de variables lo mueve.
Uso: python futuro/pruebas/f9_explica.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "cv")]
import m_lgb_future as M  # noqa: E402
from metric import hits  # noqa: E402

OUT = ROOT / "futuro" / "pruebas" / "notas"
t0 = time.time()
data = M.build_data()
lab, te, fs = data["lab"], data["te"], data["fs"]
OC, OB = M.oracle_data(data, "coord"), M.oracle_data(data, "base")
BASE = fs["base"] + fs["logic"] + fs["past"] + fs["future"]
MODELOS = {"M0": (BASE, None), "A": (BASE + fs["clear"], None),
           "O": (BASE + fs["clear"], OB), "B": (BASE + fs["clear"], OC)}
a = M.parse(["--threads", "30"])


def grupo(c):
    if c.startswith("or_"): return "otras sesiones"
    if c.startswith("sx_"): return "futuro clear"
    if c.startswith("f_"): return "futuro"
    if c.startswith(("p_", "blk_", "gap_")): return "pasado"
    return "snapshot"


def shap(model, X):
    c = model.booster_.predict(X, pred_contrib=True)
    return pd.DataFrame(c[:, :-1], columns=X.columns, index=X.index)


R = {}
for nom, (F, O) in MODELOS.items():
    F = F + (list(O["lab"].columns) if O is not None else [])
    so, sh_o = np.zeros(len(lab)), []
    for f in range(5):
        L = pd.concat([lab, O[f]], axis=1) if O is not None else lab
        tr, va = (L.fold != f).values, (L.fold == f).values
        m = M.fit(a, L[tr], F, 0)
        so[va] = m.predict(L.loc[va, F]); sh_o.append(shap(m, L.loc[va, F]))
    L = pd.concat([lab, O["lab"]], axis=1) if O is not None else lab
    T = pd.concat([te, O["te"]], axis=1) if O is not None else te
    m = M.fit(a, L, F, 0)
    R[nom] = dict(F=F, so=so, sh_o=pd.concat(sh_o).sort_index(), st=m.predict(T[F]), sh_t=shap(m, T[F]), T=T)
    cv = hits(lab[["sample_id", "row_index"]].assign(score=so)).mean()
    print(f"{nom}: {len(F)} features, CV 1.er pase (1 semilla) {cv:.4f}  ({time.time() - t0:.0f}s)", flush=True)


def top5(d, s):
    return d.assign(_s=s).groupby("sample_id")._s.rank(ascending=False, method="first").values <= 5


# 1 · reparto de la influencia por origen temporal de la información
filas = []
for nom, r in R.items():
    for zona, d, s, sh in (("CV", lab, r["so"], r["sh_o"]), ("test 020", te, r["st"], r["sh_t"]), ("test 023", te, r["st"], r["sh_t"])):
        m = top5(d, s)
        if zona != "CV":
            m &= (d.atco == int(zona[-3:])).values
        g = sh[m].abs().T.groupby(grupo).sum().T.mean()
        filas.append(dict(modelo=nom, zona=zona, **(g / g.sum() * 100).round(1).to_dict()))
T1 = pd.DataFrame(filas).set_index(["modelo", "zona"]).fillna(0)
print("\n1 · % de la influencia (|SHAP|, 5 primeros candidatos) por origen de la información\n" + T1.to_string())

# variables más influyentes de B y A, CV frente a test 020 / 023
for nom in ("A", "B"):
    r = R[nom]
    v = pd.DataFrame({z: (sh[m].abs().mean()) for z, sh, m in
                      (("CV", r["sh_o"], top5(lab, r["so"])),
                       ("020", r["sh_t"], top5(te, r["st"]) & (te.atco == 20).values),
                       ("023", r["sh_t"], top5(te, r["st"]) & (te.atco == 23).values))})
    print(f"\n{nom} · 10 variables más influyentes (|SHAP| medio en los 5 primeros):\n" + v.sort_values("CV", ascending=False).head(10).round(3).to_string())

# 2 · validación adversaria: ¿se distingue cada controlador de test de los datos etiquetados?
Fa = R["B"]["F"]
X = pd.concat([pd.concat([lab, OC["lab"]], axis=1)[Fa].assign(y=0), R["B"]["T"][Fa].assign(y=1, atco=te.atco.values)], ignore_index=True)
for at in (20, 23):
    D = X[(X.y == 0) | (X.atco == at)].drop(columns="atco").sample(frac=0.3, random_state=0)
    idx = np.arange(len(D)) % 3
    p = np.zeros(len(D))
    for k in range(3):
        mm = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, n_jobs=30).fit(D[idx != k][Fa], D[idx != k].y)
        p[idx == k] = mm.predict_proba(D[idx == k][Fa])[:, 1]
    imp = pd.Series(mm.booster_.feature_importance("gain"), Fa).sort_values(ascending=False).head(6)
    print(f"\n2 · adversaria test {at:03d} frente a etiquetados: AUC {roc_auc_score(D.y, p):.3f}; variables que más lo delatan: "
          f"{(imp / imp.sum()).round(2).to_dict()}")

# 3 · dependencia temporal: cómo pesa el tiempo hasta la siguiente instrucción / el tiempo en la lista
for var, cortes in (("f_next_input", [-1, 3, 10, 30, 60, 180, 1e9]), ("p_task_start", [-1, 30, 120, 300, 600, 1e9])):
    filas = []
    for nom in ("A", "B"):
        r = R[nom]
        for zona, d, sh in (("CV", lab, r["sh_o"]), ("020", te[te.atco == 20], r["sh_t"][(te.atco == 20).values]),
                            ("023", te[te.atco == 23], r["sh_t"][(te.atco == 23).values])):
            b = pd.cut(d[var], cortes)
            filas.append(pd.Series(sh[var].groupby(b.values, observed=False).mean().values, index=b.cat.categories.astype(str),
                                   name=f"{nom} {zona}"))
    na = {z: d[var].isna().mean() for z, d in (("CV", lab), ("020", te[te.atco == 20]), ("023", te[te.atco == 23]))}
    print(f"\n3 · SHAP medio de {var} por tramos (vacío: CV {na['CV']:.0%}, 020 {na['020']:.0%}, 023 {na['023']:.0%})\n"
          + pd.DataFrame(filas).round(3).to_string())

# 4 · test 020: dónde cambia B respecto a A y qué lo mueve
k = ["sample_id", "row_index"]
t = te[k + ["atco", "task_index", "session"]].assign(sA=R["A"]["st"], sB=R["B"]["st"])
for s in ("A", "B"):
    t["r" + s] = t.groupby("sample_id")["s" + s].rank(ascending=False, method="first")
t20 = t[t.atco == 20]
entra, sale = (t20.rB <= 3) & (t20.rA > 3), (t20.rA <= 3) & (t20.rB > 3)
gB = R["B"]["sh_t"][(te.atco == 20).values].T.groupby(grupo).sum().T
TY = ['entry_conf', 'exit_conf', 'entry_coord', 'exit_coord', 'assume', 'transfer', 'conflict', 'non_conf', 'qos',
      'entry_cond', 'exit_cond', 'request', 'zone', 'return_route']
print(f"\n4 · test 020: en {t20[entra].sample_id.nunique() / t20.sample_id.nunique():.0%} de los snapshots el top-3 del 1.er pase cambia de A a B")
print("   tipos que entran al top-3 con B:", t20[entra].task_index.map(dict(enumerate(TY))).value_counts().head(5).to_dict())
print("   tipos que salen del top-3 con B:", t20[sale].task_index.map(dict(enumerate(TY))).value_counts().head(5).to_dict())
print("   SHAP medio por origen en B (filas que entran / salen / resto):\n" +
      pd.DataFrame({"entran": gB[entra.values].mean(), "salen": gB[sale.values].mean(),
                    "resto": gB[~(entra | sale).values].mean()}).round(3).to_string())
T1.to_csv(OUT / "f9_influencia.csv")
print(f"\n({time.time() - t0:.0f}s)")

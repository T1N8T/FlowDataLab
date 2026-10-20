"""f8_parejas.py · ¿por qué se confunden las tareas del mismo avión y se puede detectar la diferencia? (hilo f8)
Parejas (respuesta, hermana) del mismo avión (cs) en el mismo snapshot, con los dos tipos dentro de un mismo grupo
(salida: exit_coord, exit_cond, return_route, transfer, zone; entrada: assume, entry_coord).
- modelo  : % de parejas en que el OOF de REF pone la respuesta por encima de la hermana.
- clasif. : LightGBM sobre (features de A, de B, A - B) que predice cuál de las dos es la respuesta; CV por controlador
            (Data/folds.csv). Si supera al modelo, la diferencia es detectable con los datos.
- por qué : las variables más importantes del clasificador y su mediana en la respuesta frente a la hermana.
Uso: python futuro/pruebas/f8_parejas.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import datos, ROOT  # noqa: E402

REF = "suav_lgb_future_clear10_ses"
TY = ['entry_conf', 'exit_conf', 'entry_coord', 'exit_coord', 'assume', 'transfer', 'conflict', 'non_conf', 'qos',
      'entry_cond', 'exit_cond', 'request', 'zone', 'return_route']
GRUPOS = [{3, 10, 13, 5, 12}, {4, 2}]
K = ["sample_id", "row_index"]

lab, _, fs = datos()
F = [c for c in fs["base"] + fs["logic"] + fs["past"] + fs["future"] + fs["clear"] if c != "task_index"]
L = lab.merge(pd.read_csv(ROOT / "futuro" / "pruebas" / "oof" / f"{REF}.csv"), on=K)
a = L[L.target_index == 1]
b = L[L.target_index == 0]
P = a.merge(b, on=["sample_id", "cs"], suffixes=("_a", "_b"))
P = P[[any(x in g and y in g for g in GRUPOS) for x, y in zip(P.task_index_a, P.task_index_b)]].reset_index(drop=True)
P["par"] = [" vs ".join(sorted((TY[x], TY[y]))) for x, y in zip(P.task_index_a, P.task_index_b)]
P["modelo_ok"] = P.score_a > P.score_b
print(f"{len(P)} parejas respuesta/hermana del mismo avión en {P.sample_id.nunique()} snapshots", flush=True)

# dataset simétrico: (A, B) -> 1 y (B, A) -> 0
fa, fb = [c + "_a" for c in F], [c + "_b" for c in F]
def bloque(x, y, lab_):
    X = pd.DataFrame(np.c_[P[x].values, P[y].values, P[x].values.astype(float) - P[y].values.astype(float)],
                     columns=[f"A_{c}" for c in F] + [f"B_{c}" for c in F] + [f"d_{c}" for c in F])
    X["tA"], X["tB"] = P["task_index" + x[0][-2:]].values, P["task_index" + y[0][-2:]].values
    return X.assign(y=lab_, fold=P.fold_a.values, par=P.par.values, idx=np.arange(len(P)))
D = pd.concat([bloque(fa, fb, 1), bloque(fb, fa, 0)], ignore_index=True)
X = [c for c in D.columns if c not in ("y", "fold", "par", "idx")]
p = np.zeros(len(D))
imp = pd.Series(0.0, index=X)
for f in range(5):
    tr, va = (D.fold != f).values, (D.fold == f).values
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, colsample_bytree=0.7,
                           verbose=-1, n_jobs=30).fit(D.loc[tr, X], D.y[tr], categorical_feature=["tA", "tB"])
    p[va] = m.predict_proba(D.loc[va, X])[:, 1]
    imp += pd.Series(m.booster_.feature_importance("gain"), index=X)
# por pareja original: acierta si p(A,B) > p(B,A)
pa = pd.Series(p[:len(P)]); pb = pd.Series(p[len(P):])
P["clasif_ok"] = (pa > pb).values
r = P.groupby("par").agg(n=("modelo_ok", "size"), modelo=("modelo_ok", "mean"), clasif=("clasif_ok", "mean"))
print("\n% de parejas con la respuesta por encima de la hermana:\n" + r[r.n >= 100].sort_values("n", ascending=False).round(3).to_string())
print(f"\ntotal: modelo {P.modelo_ok.mean():.3f}   clasificador de parejas {P.clasif_ok.mean():.3f}")

top = imp.groupby(lambda c: c.split("_", 1)[1] if c[:2] in ("A_", "B_", "d_") else c).sum().sort_values(ascending=False).head(10)
print("\nvariables que distinguen a la respuesta de su hermana (ganancia):", (top / top.sum()).round(3).to_dict())
for par in r[r.n >= 150].sort_values("n", ascending=False).index[:6]:
    g = P[P.par == par]
    t1, t2 = par.split(" vs ")
    rows = []
    for c in top.index[:6]:
        if c in ("tA", "tB") or c + "_a" not in g:
            continue
        def med(tipo, cual):
            m = g[g["task_index_" + cual].map(lambda t: TY[t]) == tipo][c + "_" + cual]
            return m.median()
        rows.append(dict(var=c, **{f"{t1} respuesta": med(t1, "a"), f"{t1} hermana": med(t1, "b"),
                                   f"{t2} respuesta": med(t2, "a"), f"{t2} hermana": med(t2, "b")}))
    print(f"\n{par}: medianas (respuesta frente a hermana)\n" + pd.DataFrame(rows).set_index("var").round(1).to_string())

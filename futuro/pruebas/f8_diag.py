"""f8_diag.py · diagnósticos de cuánto error es reducible (hilo f8). Sobre el mejor OOF sin oráculo y los datos con futuro.
1. acuerdo   : sesiones distintas del mismo escenario con snapshots etiquetados a <= 30 s: si la respuesta de A está en la
               lista de B, ¿B responde la misma tarea? (comportamiento común a los controladores = techo de un modelo general)
2. colision  : en los fallos, ¿alguna fila por encima de la correcta tiene EXACTAMENTE las mismas features? (irreducible)
3. fallos    : LightGBM que predice el fallo (hit=0) desde agregados del snapshot; AUC en CV por controlador
4. curva     : 1.er pase (clear, 1 semilla) entrenado con 25/50/75/100 % de los controladores de entrenamiento
Uso: python futuro/pruebas/f8_diag.py [acuerdo colision fallos curva]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import datos, lgb_cv, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "cv"))
from metric import hits  # noqa: E402

REF = "suav_lgb_future_clear10_ses"
K = ["sample_id", "row_index"]
lab, te, fs = datos()
F = fs["base"] + fs["logic"] + fs["past"] + fs["future"] + fs["clear"]
oof = pd.read_csv(ROOT / "futuro" / "pruebas" / "oof" / f"{REF}.csv")
L = lab.merge(oof, on=K)
L["rk"] = L.groupby("sample_id").score.rank(ascending=False, method="first")
tgt = L[L.target_index == 1].set_index("sample_id")
HIT = tgt.rk <= 3


def acuerdo(ventana=60):
    s = tgt[["session", "scenario", "atco", "t", "task_name"]].reset_index()
    s["t"] = s.t - s.session.map(lab.groupby("session").t.min())   # tiempo desde el inicio de la sesión
    pres = L.groupby("sample_id").task_name.apply(set)
    r = []
    for sc, g in s.groupby("scenario"):
        g = g.sort_values("t")
        for ses, a in g.groupby("session"):
            b = g[(g.session != ses) & (g.atco != a.atco.iloc[0])]
            if b.empty:
                continue
            j = np.searchsorted(b.t.values, a.t.values)
            for i, (ti, tn) in enumerate(zip(a.t.values, a.task_name.values)):
                for k in (j[i] - 1, j[i]):
                    if 0 <= k < len(b) and abs(b.t.values[k] - ti) <= ventana and tn in pres[b.sample_id.values[k]]:
                        r.append(dict(sc=sc, igual=b.task_name.values[k] == tn, hit=HIT[a.sample_id.values[i]]))
    r = pd.DataFrame(r)
    print(f"acuerdo entre controladores (misma tarea presente, <= {ventana} s): {r.igual.mean():.3f} en {len(r)} parejas; "
          f"por escenario {r.groupby('sc').igual.mean().round(3).to_dict()}")
    print(f"  cuando el otro controlador elige lo mismo, acertamos {r[r.igual].hit.mean():.3f}; cuando no, {r[~r.igual].hit.mean():.3f}")


def colision():
    X = L[F].astype(float).fillna(-9e9).round(6)
    h = pd.util.hash_pandas_object(X.drop(columns=["task_index"]).assign(ti=L.task_index), index=False).values
    L["_h"] = h
    fall = tgt[~HIT]
    c_same, c_twin = 0, 0
    for sid, row in fall.iterrows():
        g = L[L.sample_id == sid]
        hc = g.loc[g.target_index == 1, "_h"].iloc[0]
        up = g[g.rk < row.rk]
        c_same += (up._h == hc).any()
        c_twin += (up.cs == row.cs).any()
    print(f"colisiones: en {c_same} de {len(fall)} fallos ({c_same / len(fall):.1%}) hay una fila por encima con features "
          f"idénticas; en {c_twin / len(fall):.1%} la de encima es del mismo avión")


def fallos():
    g = L.groupby("sample_id")
    X = pd.DataFrame({"n": g.size(), "pmax": g.score.max(), "gap12": g.score.apply(lambda s: np.sort(s.values)[-1] - np.sort(s.values)[-2]),
                      "ncoord": g.task_index.apply(lambda s: s.isin([2, 3]).sum()), "blk_len": g.blk_len.first(),
                      "blk_pos": g.blk_pos.first(), "gap_prev": g.gap_prev.first(), "gap_next": g.gap_next.first()})
    for c in fs["future"][:6]:
        X[c + "_min"] = g[c].min()
    top = L[L.rk == 1].set_index("sample_id")
    X["top_type"] = top.task_index
    y = (~HIT).astype(int).reindex(X.index)
    fold = tgt.fold.reindex(X.index)
    p = np.zeros(len(X))
    for f in range(5):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1, n_jobs=16)
        m.fit(X[fold != f], y[fold != f], categorical_feature=["top_type"])
        p[(fold == f).values] = m.predict_proba(X[fold == f])[:, 1]
    imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False).head(6)
    print(f"predecir el fallo: AUC {roc_auc_score(y, p):.3f} (tasa de fallo {y.mean():.3f}); importantes: {imp.to_dict()}")
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    print("  tasa de fallo por decil de riesgo:", pd.Series(y.values).groupby(q).mean().round(3).tolist())


def curva():
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5, 0.75, 1.0):
        t0 = time.time()
        oof = np.zeros(len(lab))
        for f in range(5):
            at = lab.loc[lab.fold != f, "atco"].unique()
            keep = set(rng.choice(at, max(1, round(frac * len(at))), replace=False))
            tr = lab[(lab.fold != f) & lab.atco.isin(keep)]
            va = (lab.fold == f).values
            from base_fut import lgb_fit, PARAMS_FUT
            oof[va] = lgb_fit(tr, F, {**PARAMS_FUT, "n_jobs": 30, "random_state": 0}).predict(lab.loc[va, F])
        cv = hits(lab[K].assign(score=oof)).mean()
        print(f"curva: {frac:.0%} de los controladores de entrenamiento -> CV 1.er pase {cv:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:] or ["acuerdo", "colision", "fallos", "curva"]:
        globals()[m]()

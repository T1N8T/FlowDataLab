"""
auditoria_coord.py · Pablo (A) · RAMA FUTURO · Sección 17 de LGB_futuro.ipynb: estilo de los controladores y ambigüedad
de las coordinaciones. Sin reentrenar (usa los OOF guardados).

    estilo   ¿los controladores de estilo parecido (perfil SIN etiquetas, feat_oracle.estilo) coinciden más en sus
             respuestas? Spearman entre distancia de estilo y acuerdo (misma tarea a ±180 s de tiempo alineado) en
             parejas de sesiones del mismo escenario y distinto controlador. Y lo atípicas que son las sesiones de test.
    temp     el oráculo de coordinaciones ponderado por estilo (T = 2, 1, 0.5) frente al uniforme: ¿gana más en las
             sesiones atípicas?
    ambig    entre las coordinaciones del mismo tipo de los snapshots cuya respuesta es una coordinación: acierto top-1
             y AUC por parejas del modelo (lgb_future_clear_orcoord) frente a un kNN (25 vecinos, por folds) con las
             mismas features, y cuántas veces el vecino más parecido (de otro fold) tiene la etiqueta contraria.

Uso: python futuro/pruebas/auditoria_coord.py [estilo|temp|ambig ...]   (sin argumentos: las tres)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "cv")]
import m_lgb_future as M, feat_oracle as fo, features_future as ff, metricas  # noqa: E402
from metric import hits  # noqa: E402


def estilo(data):
    lab, te = data["lab"], data["te"]
    D = pd.concat([lab, te])
    Z = fo.estilo(D)
    tau = pd.Series(fo.align(D), index=D.index)
    a = lab[lab.target_index == 1].assign(tau=tau.iloc[:len(lab)].values[(lab.target_index == 1).values])
    a = a[["session", "scenario", "atco", "task_name", "tau"]]
    rows = []
    for sc, g in a.groupby("scenario"):
        for x, gx in g.groupby("session"):
            for y, gy in g.groupby("session"):
                if gx.atco.iloc[0] == gy.atco.iloc[0]:
                    continue
                m = gx.merge(gy, on="task_name", suffixes=("", "_y"))
                m = m[(m.tau - m.tau_y).abs() <= 180]
                agree = gx.set_index(["tau", "task_name"]).index.isin(m.set_index(["tau", "task_name"]).index).mean()
                rows.append(dict(x=x, agree=agree, dist=float(np.sqrt(((Z.loc[x] - Z.loc[y]) ** 2).sum()))))
    P = pd.DataFrame(rows)
    rho, p = spearmanr(P.dist, P.agree)
    print(f"parejas: {len(P)} · Spearman(distancia de estilo, acuerdo) = {rho:+.2f} (p = {p:.1g})")
    q = pd.cut(P.groupby("x").dist.rank(pct=True), [0, .33, .67, 1.0], labels=["parecidas", "medias", "lejanas"])
    print("acuerdo por tercil de parecido:", P.groupby(q, observed=True).agree.mean().round(3).to_dict())
    atip = Z.abs().mean(1)
    tr = atip.drop(te.session.unique())
    for s in te.session.unique():
        print(f"  {s}: atipicidad (media |z|) {atip[s]:.2f}, percentil frente a train/val {(tr < atip[s]).mean():.2f}; "
              f"|z| > 2 en {Z.loc[s][Z.loc[s].abs() > 2].round(2).to_dict()}")
    return atip


def temp(data, atip):
    u = hits(pd.read_csv(ROOT / "futuro" / "oof" / "lgb_future_clear_orcoord_ses.csv"))
    ses = metricas.info().session.reindex(u.index)
    for n in ("t2", "t1", "t05"):
        c = hits(pd.read_csv(ROOT / "futuro" / "oof" / f"lgb_future_clear_orcoord_{n}_ses.csv")).reindex(u.index)
        g = (c - u).groupby(ses).mean()
        a = atip.reindex(g.index)
        top = a.sort_values().index[-8:]
        rho, p = spearmanr(a, g)
        print(f"  {n}: Spearman(atipicidad, Δ por sesión) = {rho:+.2f} (p = {p:.2f}); Δ en las 8 sesiones más atípicas "
              f"{g[top].mean() * 100:+.2f}, resto {g.drop(top).mean() * 100:+.2f}")


def ambig(data):
    from sklearn.neighbors import NearestNeighbors
    lab, fs = data["lab"], data["fs"]
    F = [c for b in "base,logic,past,future,clear".split(",") for c in fs[b] if c != "task_index"]
    lab = lab.merge(pd.read_csv(ROOT / "futuro" / "oof" / "lgb_future_clear_orcoord.csv"), on=["sample_id", "row_index"],
                    how="left")
    tgt = lab[lab.target_index == 1].set_index("sample_id").task_index
    res = {}
    for ty in (2, 3):
        C = lab[lab.sample_id.isin(tgt[tgt == ty].index) & (lab.task_index == ty)]
        C = C[C.groupby("sample_id").sample_id.transform("size") >= 2].reset_index(drop=True)
        X = C[F].astype(float)
        X = X.loc[:, X.std() > 0]
        X = ((X - X.mean()) / X.std()).fillna(0).values
        knn, dmin, conf = (np.full(len(C), np.nan) for _ in range(3))
        for f in range(5):
            tr = (C.fold != f).values
            dist, ix = NearestNeighbors(n_neighbors=25).fit(X[tr]).kneighbors(X[~tr])
            yt = C.target_index.values[tr]
            knn[~tr], dmin[~tr] = yt[ix].mean(1), dist[:, 0]
            conf[~tr] = yt[ix[:, 0]] != C.target_index.values[~tr]
        C = C.assign(knn=knn, dmin=dmin, conf=conf)

        def auc(col):
            out = []
            for _, g in C.groupby("sample_id"):
                t, o = g[g.target_index == 1][col].values, g[g.target_index == 0][col].values
                if len(t) and len(o):
                    out.append((t[0] > o).mean() + 0.5 * (t[0] == o).mean())
            return np.mean(out)

        def top1(col):
            return (C.sample(frac=1, random_state=0).sort_values(col, ascending=False, kind="stable")
                    .drop_duplicates("sample_id").target_index.mean())

        t = C[C.target_index == 1]
        near = t.dmin <= C.dmin.quantile(0.1)
        res[ff.TYPES[ty]] = dict(snapshots=C.sample_id.nunique(), azar=(1 / C.groupby("sample_id").size()).mean(),
                                 top1_modelo=top1("score"), top1_knn=top1("knn"), auc_modelo=auc("score"),
                                 auc_knn=auc("knn"), vecino_opuesto=t.conf.mean(),
                                 vecino_opuesto_10pc=t[near].conf.mean(), n_10pc=int(near.sum()))
    print(pd.DataFrame(res).round(3).to_string())


if __name__ == "__main__":
    que = sys.argv[1:] or ["estilo", "temp", "ambig"]
    data = M.build_data()
    atip = estilo(data) if ("estilo" in que or "temp" in que) else None
    if "temp" in que:
        temp(data, atip)
    if "ambig" in que:
        ambig(data)

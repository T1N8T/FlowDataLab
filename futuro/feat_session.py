"""
feat_session.py · Pablo (A) · RAMA FUTURO · Segundo paso de sesión: "cada tarea se trabaja una sola vez".

En train+validation el 98 % de las tareas son la respuesta en UN solo episodio de la sesión, y tras un hueco entre
rachas (block) la respuesta casi nunca se repite (0.2 %). Así que si el primer pase está muy seguro de que la tarea X fue
la respuesta en OTRA racha de la sesión, aquí probablemente no lo es ("explaining away").

A partir del score del primer pase (OOF o test, nunca la etiqueta) se calculan, por tarea:
    lsm   log-softmax del score en el snapshot          lPb   log de su probabilidad media en la racha
    M     máxima probabilidad media en OTRA racha       E     suma de probabilidades medias en otras rachas
    Pprev / Pnext  probabilidad media en la racha anterior / siguiente de la sesión
    Pc / Mc        lo mismo a nivel de avión (suma de sus tareas): en esta racha / máximo en otra
y un logit condicional (softmax dentro del snapshot, 9 pesos) los combina. Los pesos se ajustan con los folds de
entrenamiento y salen casi iguales en los 5 (M ≈ -2.6): poco riesgo de sobreajuste.
Sin fuga: todas las filas de una sesión están en el mismo fold, y su OOF no vio la etiqueta.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

COLS = ["lsm", "lPb", "M", "E", "Pprev", "Pnext", "Pc", "Mc", "top_else"]


def _others_max(g, v):
    """Para cada fila, el máximo de v en las OTRAS filas de su grupo (0 si no hay otras)."""
    x = pd.DataFrame(dict(g=g, v=v))
    s = x.sort_values(["g", "v"], ascending=[True, False])
    first = s.groupby("g").v.transform("first")
    second = s.groupby("g").v.shift(-1).groupby(s.g).transform("first").fillna(0)
    out = np.where(s.v.values >= first.values, second.values, first.values)
    return pd.Series(out, index=s.index).reindex(x.index).values


def session_feats(d, score):
    """d: filas con sample_id, session, block, blk_len, t, task_name, cs. Devuelve DataFrame alineado con d (COLS)."""
    x = d[["sample_id", "session", "block", "blk_len", "t", "task_name", "cs"]].copy()
    x["s"] = np.asarray(score)
    x["s"] -= x.groupby("sample_id").s.transform("max")
    x["p"] = np.exp(x.s); x["p"] /= x.groupby("sample_id").p.transform("sum")
    # probabilidad media de cada tarea en su racha (0 en los snapshots donde no está)
    b = x.groupby(["block", "task_name"]).agg(Pb=("p", "sum"), n=("blk_len", "first"), session=("session", "first"),
                                               cs=("cs", "first")).reset_index()
    b["Pb"] /= b.n
    b["rk"] = b.groupby("block").Pb.rank(ascending=False, method="min")
    st = b.groupby(["session", "task_name"]).ngroup().values
    b["M"] = _others_max(st, b.Pb.values)
    b["E"] = b.groupby(["session", "task_name"]).Pb.transform("sum") - b.Pb
    b["top_else"] = (b.rk == 1).groupby([b.session, b.task_name]).transform("sum") - (b.rk == 1)
    # racha anterior / siguiente de la sesión
    bt = x.groupby("block").agg(session=("session", "first"), tb=("t", "min")).reset_index().sort_values(["session", "tb"])
    bt["prev"], bt["next"] = bt.groupby("session").block.shift(), bt.groupby("session").block.shift(-1)
    bt = bt.set_index("block")
    key = b.set_index(["block", "task_name"]).Pb
    for k, col in (("prev", "Pprev"), ("next", "Pnext")):
        nb = bt[k].reindex(b.block).values
        b[col] = key.reindex(list(zip(nb, b.task_name))).fillna(0).values
    # avión: suma de sus tareas en la racha, máximo en otra racha
    c = b.groupby(["session", "block", "cs"]).Pb.sum().rename("Pc").reset_index()
    c["Mc"] = _others_max(c.groupby(["session", "cs"]).ngroup().values, c.Pc.values)
    y = x[["block", "task_name", "cs"]].merge(b[["block", "task_name", "Pb", "M", "E", "Pprev", "Pnext", "top_else"]],
                                              on=["block", "task_name"], how="left") \
        .merge(c[["block", "cs", "Pc", "Mc"]], on=["block", "cs"], how="left")
    return pd.DataFrame(dict(lsm=np.log(x.p.values), lPb=np.log(y.Pb.values + 1e-4), M=y.M.values, E=y.E.clip(upper=3).values,
                             Pprev=y.Pprev.values, Pnext=y.Pnext.values, Pc=y.Pc.values, Mc=y.Mc.values,
                             top_else=y.top_else.clip(upper=2).values), index=d.index)[COLS]


def _nll(w, X, y, g, ng):
    s = X @ w
    m = np.full(ng, -np.inf); np.maximum.at(m, g, s)
    e = np.exp(s - m[g]); z = np.zeros(ng); np.add.at(z, g, e)
    lp = s - m[g] - np.log(z[g])
    return -(lp * y).sum() / ng, ((np.exp(lp) - y)[:, None] * X).sum(0) / ng


def fit_logit(X, y, sample_id):
    """Logit condicional (softmax dentro de cada snapshot). Devuelve los pesos."""
    g = pd.factorize(np.asarray(sample_id))[0]
    w0 = np.zeros(X.shape[1]); w0[0] = 1.0
    return minimize(_nll, w0, args=(np.asarray(X, float), np.asarray(y, float), g, g.max() + 1), jac=True,
                    method="L-BFGS-B").x


def sesion(lab, oof, te=None, pred=None):
    """Segundo paso. lab con fold y target_index; oof/pred = scores del primer pase alineados con lab/te.
    Devuelve (oof_ses, pred_ses o None, pesos ajustados con todo el OOF)."""
    X = session_feats(lab, oof).values
    out = np.full(len(lab), np.nan)
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        out[va] = X[va] @ fit_logit(X[tr], lab.target_index.values[tr], lab.sample_id.values[tr])
    w = fit_logit(X, lab.target_index.values, lab.sample_id.values)
    p = session_feats(te, pred).values @ w if te is not None else None
    return out, p, dict(zip(COLS, w.round(3)))

"""
feat_session_causal.py · Pablo (A) · RAMA FUTURO · Segundo paso de sesión CAUSAL (modelo "realista", sección 20).

Igual que feat_session.py, pero cada snapshot solo ve su score y los de snapshots ANTERIORES de la sesión:
    lsm      log-softmax del score en el snapshot
    lPa      log de la probabilidad media de la tarea en la racha hasta ahora (actual incluido)
    p1, p2   probabilidad de la tarea 1 y 2 snapshots antes en la racha (0 si no estaba)
    Mp, Ep   máxima / suma de la probabilidad media de la tarea en rachas ANTERIORES de la sesión ("ya se hizo")
    Pprev    probabilidad media en la racha anterior
    Pca, Mcp avión: suma de sus tareas en la racha hasta ahora / máximo en rachas anteriores
    top_prev nº de rachas anteriores en que fue la favorita
Un logit condicional (softmax dentro del snapshot) los combina; pesos ajustados en los folds de entrenamiento.
Hace a la vez de suavizado causal (lPa, p1, p2), así que no hace falta el suavizado de vecinos.
"""
import numpy as np, pandas as pd
from feat_session import fit_logit

COLS = ["lsm", "lPa", "p1", "p2", "Mp", "Ep", "Pprev", "Pca", "Mcp", "top_prev"]


def session_feats(d, score):
    x = d[["sample_id", "session", "block", "blk_pos", "t", "task_name", "cs"]].copy()
    x["s"] = np.asarray(score)
    x["s"] -= x.groupby("sample_id").s.transform("max")
    x["p"] = np.exp(x.s); x["p"] /= x.groupby("sample_id").p.transform("sum")
    x = x.sort_values(["session", "t"], kind="stable")
    # racha hasta ahora: media sobre los snapshots ya vistos de la racha (ausente = 0)
    x["Pa"] = x.groupby(["block", "task_name"]).p.cumsum() / (x.blk_pos + 1)
    pos = x.set_index(["block", "task_name", "blk_pos"]).p
    for k in (1, 2):
        x[f"p{k}"] = pos.reindex(list(zip(x.block, x.task_name, x.blk_pos - k))).fillna(0).values
    x["Pca"] = x.groupby(["sample_id", "cs"]).Pa.transform("sum")
    # rachas anteriores completas (ya pasadas)
    blen = x.groupby("block").blk_pos.max() + 1
    b = x.groupby(["block", "task_name"]).agg(P=("p", "sum"), session=("session", "first"), cs=("cs", "first"),
                                               tb=("t", "min")).reset_index()
    b["P"] /= b.block.map(blen)
    b["top"] = (b.P == b.groupby("block").P.transform("max")).astype(float)
    bt = x.groupby("block").agg(session=("session", "first"), tb=("t", "min")).sort_values(["session", "tb"])
    order = pd.Series(np.arange(len(bt)), index=bt.index)
    b["ord"] = b.block.map(order)
    b = b.sort_values(["session", "task_name", "ord"])
    g = b.groupby(["session", "task_name"])
    b["Mp"] = g.P.transform(lambda s: s.shift().cummax()).fillna(0)
    b["Ep"] = (g.P.cumsum() - b.P).clip(upper=3)
    b["top_prev"] = (g.top.cumsum() - b.top).clip(upper=2)
    prev_blk = bt.reset_index().assign(prev=lambda z: z.groupby("session").block.shift()).set_index("block").prev
    key = b.set_index(["block", "task_name"]).P
    b["Pprev"] = key.reindex(list(zip(b.block.map(prev_blk), b.task_name))).fillna(0).values
    c = b.groupby(["session", "block", "cs"]).agg(Pc=("P", "sum"), ord=("ord", "first")).reset_index()
    c = c.sort_values(["session", "cs", "ord"])
    c["Mcp"] = c.groupby(["session", "cs"]).Pc.transform(lambda s: s.shift().cummax()).fillna(0)
    y = x.merge(b[["block", "task_name", "Mp", "Ep", "Pprev", "top_prev"]], on=["block", "task_name"], how="left") \
        .merge(c[["block", "cs", "Mcp"]], on=["block", "cs"], how="left")
    y.index = x.index
    y = y.reindex(d.index)
    return pd.DataFrame(dict(lsm=np.log(y.p.values),
                             lPa=np.log(y.Pa.values + 1e-4), p1=y.p1.values, p2=y.p2.values, Mp=y.Mp.values,
                             Ep=y.Ep.values, Pprev=y.Pprev.values, Pca=y.Pca.values, Mcp=y.Mcp.values,
                             top_prev=y.top_prev.values), index=d.index)[COLS]


def sesion(lab, oof, te=None, pred=None):
    X = session_feats(lab, oof).values
    out = np.full(len(lab), np.nan)
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        out[va] = X[va] @ fit_logit(X[tr], lab.target_index.values[tr], lab.sample_id.values[tr])
    w = fit_logit(X, lab.target_index.values, lab.sample_id.values)
    p = session_feats(te, pred).values @ w if te is not None else None
    return out, p, dict(zip(COLS, w.round(3)))

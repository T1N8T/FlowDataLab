"""
feat_hist.py · Pablo (A) · RAMA "REALISTA" · Bloque "hist" (h_*): historia de la lista de tareas, solo hacia atrás.

Una tarea que estaba en el snapshot anterior de la racha y ya no está = el controlador la ha resuelto (o AWARE la ha
quitado): es una acción pasada deducida SIN etiquetas. Lo mismo para las tareas que aparecen. Con esos eventos:
    h_prev_in       la tarea estaba en el snapshot anterior de la racha (NaN si es el primero de la racha)
    h_n_seen        nº de snapshots anteriores de la sesión en que ya apareció la tarea
    h_gone_before   la tarea ya desapareció antes en la sesión (reaparece)
    h_cs_gone_*     avión: segundos desde su última tarea resuelta, cuántas en 30/120 s, tipo de la última
    h_ty_gone_since tipo: segundos desde la última tarea resuelta de este tipo en la sesión
    h_ses_gone_*    sesión: segundos desde la última tarea resuelta, ¿era del mismo avión? ¿mismo tipo?
    h_cs_new_*      avión: segundos desde que le apareció una tarea nueva, cuántas en 30 s
    h_snap_gone/new nº de tareas resueltas / nuevas en este snapshot; h_dlen = cambio de longitud de la lista
    + rangos en el snapshot de h_cs_gone_since.
Los eventos de un snapshot tienen la hora de ese snapshot (se ven en él), así que nada mira al futuro.
"""
import numpy as np, pandas as pd


def _events(d):
    st = d.drop_duplicates("sample_id")[["sample_id", "session", "t", "block"]].sort_values(["session", "t"])
    st["prev"] = st.groupby("block").sample_id.shift()
    r = d[["sample_id", "task_name", "cs", "task_index"]]
    cur = r.merge(st[["sample_id", "prev", "session", "t"]], on="sample_id")
    prv = r.rename(columns={"sample_id": "prev"}).merge(st[["sample_id", "prev", "session", "t"]].dropna(), on="prev")
    key_cur = set(zip(cur.sample_id, cur.task_name))
    key_prv = set(zip(prv.prev, prv.task_name))
    cur["prev_in"] = np.where(cur.prev.isna(), np.nan,
                              [(p, n) in key_prv for p, n in zip(cur.prev, cur.task_name)]).astype(float)
    gone = prv[[(s, n) not in key_cur for s, n in zip(prv.sample_id, prv.task_name)]]
    new = cur[cur.prev_in == 0]
    return st, cur, gone[["session", "t", "cs", "task_index", "task_name"]], new[["session", "t", "cs", "task_index", "task_name"]]


def _asof(rows, ev, by, cols=(), dt=0.0, name="t_ev"):
    """Para cada fila: último evento con t_ev <= t - dt del mismo grupo `by` (y su nº acumulado)."""
    e = ev.sort_values(["t", "task_name"], kind="stable").rename(columns={"t": name}).copy()
    e["_n"] = e.groupby(by).cumcount() + 1
    q = rows[by + ["t"]].assign(_q=rows.t - dt, _i=np.arange(len(rows))).sort_values("_q")
    m = pd.merge_asof(q, e[by + [name, "_n", *cols]].rename(columns={name: "_te"}), left_on="_q", right_on="_te",
                      by=by, direction="backward", allow_exact_matches=True)
    return m.sort_values("_i").reset_index(drop=True)


def add_hist(d):
    d = d.copy()
    st, cur, gone, new = _events(d)
    cur = d[["sample_id", "task_name"]].merge(cur[["sample_id", "task_name", "prev_in"]], on=["sample_id", "task_name"],
                                              how="left")
    d["h_prev_in"] = cur.prev_in.values
    order = d.sort_values(["session", "t"]).index
    d.loc[order, "h_n_seen"] = d.loc[order].groupby(["session", "task_name"]).cumcount().values
    rows = d[["session", "t", "cs", "task_index", "task_name"]].reset_index(drop=True)

    a = _asof(rows, gone, ["session", "task_name"])
    d["h_gone_before"] = a._n.notna().astype(float).values
    a = _asof(rows, gone, ["session", "cs"], cols=("task_index",))
    d["h_cs_gone_since"] = (rows.t - a._te).values
    d["h_cs_gone_type"] = a.task_index.values
    n_now = a._n.fillna(0).values
    for w in (30, 120):
        b = _asof(rows, gone, ["session", "cs"], dt=w)
        d[f"h_cs_gone_n{w}"] = n_now - b._n.fillna(0).values
    a = _asof(rows, gone, ["session", "task_index"])
    d["h_ty_gone_since"] = (rows.t - a._te).values
    g = gone.sort_values(["t", "task_name"], kind="stable").drop_duplicates(["session", "t"], keep="last")   # último evento de cada snapshot
    a = _asof(rows, g.rename(columns={"cs": "g_cs", "task_index": "g_ty"}), ["session"], cols=("g_cs", "g_ty"))
    d["h_ses_gone_since"] = (rows.t - a._te).values
    d["h_ses_gone_same_cs"] = np.where(a._te.isna(), np.nan, (a.g_cs.values == rows.cs.values).astype(float))
    d["h_ses_gone_same_ty"] = np.where(a._te.isna(), np.nan, (a.g_ty.values == rows.task_index.values).astype(float))
    a = _asof(rows, new, ["session", "cs"])
    d["h_cs_new_since"] = (rows.t - a._te).values
    b = _asof(rows, new, ["session", "cs"], dt=30)
    d["h_cs_new_n30"] = a._n.fillna(0).values - b._n.fillna(0).values

    sn = st[["sample_id", "session", "t", "prev"]].copy()
    sn["h_snap_gone"] = sn.set_index(["session", "t"]).index.map(gone.groupby(["session", "t"]).size()).fillna(0).values
    sn["h_snap_new"] = sn.sample_id.map(new.merge(st[["sample_id", "session", "t"]], on=["session", "t"])
                                        .groupby("sample_id").size()).fillna(0).values
    ln = d.groupby("sample_id").size()
    sn["h_dlen"] = sn.sample_id.map(ln).values - sn.prev.map(ln).values
    sn.loc[sn.prev.isna(), ["h_snap_gone", "h_snap_new"]] = np.nan
    d = d.merge(sn[["sample_id", "h_snap_gone", "h_snap_new", "h_dlen"]], on="sample_id", how="left")
    g = d.groupby("sample_id")
    d["h_cs_gone_since_rk"] = g.h_cs_gone_since.rank(method="min")
    d["h_cs_new_since_rk"] = g.h_cs_new_since.rank(method="min")
    return d

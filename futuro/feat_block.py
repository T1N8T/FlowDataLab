"""
feat_block.py · Pablo (A) · RAMA FUTURO · Features de futuro medidas por RACHA (episodio), no por snapshot.

La respuesta es la misma en casi toda la racha (block), pero f_*, sx_* y sc_* se miden desde cada snapshot: en el primer
snapshot de una racha larga el evento que delata la tarea puede quedar lejos. Aquí, para cada (racha, tarea), se toma el
momento más temprano del evento visto desde cualquier snapshot de la racha y se expresa respecto al FINAL de la racha
(negativo = ocurre dentro de la racha). Además, cuántas instrucciones recibe el avión durante la racha.
Columnas bk_* (bloque "block" en features_future.feature_sets). No usan etiquetas. NaN = no hay evento.
"""
import numpy as np, pandas as pd

EVENTS = ["f_next_input", "f_next_input2", "f_task_end", "f_cs_end", "sx_task", "sc_task"]


def add_block(d):
    d = d.copy()
    blk = d.groupby("block").t
    t0, t1 = blk.transform("min"), blk.transform("max")
    d["bk_t_from_start"], d["bk_t_to_end"] = d.t - t0, t1 - d.t
    g = ["block", "task_name"]
    for c in EVENTS:
        d["bk_" + c] = (d.t + d[c]).groupby([d.block, d.task_name]).transform("min") - t1
        d["bk_" + c + "_rk"] = d.groupby("sample_id")["bk_" + c].rank(method="min")
    # tipo de la primera instrucción vista desde la racha y si encaja con la tarea
    first = d.t + d.f_next_input
    i = first.fillna(np.inf).groupby([d.block, d.task_name]).transform("idxmin")
    d["bk_next_act"] = d.f_next_act.reindex(i.values).values
    d["bk_next_match"] = d.f_next_match.reindex(i.values).values
    # instrucciones al avión durante la racha (+3 s): número de tiempos de instrucción distintos
    ev = d.t - d.a_since_flight.where(d.a_since_flight < 3600)
    e = pd.DataFrame(dict(session=d.session, cs=d.cs, ev=ev.round())).dropna().drop_duplicates()
    b = d.drop_duplicates(["block", "cs"])[["block", "session", "cs"]].assign(t0=t0, t1=t1)
    m = b.merge(e, on=["session", "cs"])
    m = m[(m.ev > m.t0) & (m.ev <= m.t1 + 3)]
    n = m.groupby(["block", "cs"]).size().rename("bk_n_inputs")
    d = d.merge(n, left_on=["block", "cs"], right_index=True, how="left")
    d["bk_n_inputs"] = d.bk_n_inputs.fillna(0)
    return d

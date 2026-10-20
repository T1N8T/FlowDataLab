"""
feat_rel.py · Pablo (A) · RAMA FUTURO · Features relativas entre candidatos del mismo snapshot (columnas rl_*).

Ya existían rangos en el snapshot, en el tipo y en el avión (f_next_input_rk, f_task_end_rk(_flight), sx_task_rk(_type),
sc_task_rk(_type), l_n_type, l_n_flight...). Aquí solo lo que faltaba:
    rl_<c>_dmin_type / _dmin_all   diferencia con el mejor competidor (mínimo de c) del mismo tipo / de todo el snapshot,
                                   sin contarse a sí mismo (c = f_next_input, f_task_end, sx_task)
    rl_<c>_rk_type                 rango de c dentro del tipo (f_next_input, f_task_end)
    rl_n_match_cs                  tareas del mismo avión cuya siguiente instrucción (≤ 60 s) encaja con su tipo
    rl_n_cs_soon                   aviones del snapshot con una instrucción en ≤ 30 s
No usan etiquetas. Bloque "rel" en features_future.feature_sets.
"""
import numpy as np, pandas as pd


def _min_other(d, c, keys):
    """Mínimo de c entre las OTRAS filas del grupo (NaN si no hay)."""
    g = d.groupby(keys)[c]
    m1 = g.transform("min")
    # segundo mínimo: para la fila que es el mínimo, el mínimo de las demás
    is_min = d[c] == m1
    n_min = is_min.groupby([d[k] for k in keys]).transform("sum")
    second = d[c].where(~is_min).groupby([d[k] for k in keys]).transform("min")
    return np.where(is_min & (n_min == 1), second, m1)


def add_rel(d):
    d = d.copy()
    for c in ("f_next_input", "f_task_end", "sx_task"):
        d[f"rl_{c}_dmin_type"] = d[c] - _min_other(d, c, ["sample_id", "task_index"])
        d[f"rl_{c}_dmin_all"] = d[c] - _min_other(d, c, ["sample_id"])
    for c in ("f_next_input", "f_task_end"):
        d[f"rl_{c}_rk_type"] = d.groupby(["sample_id", "task_index"])[c].rank(method="min")
    d["rl_n_match_cs"] = ((d.f_next_match == 1) & (d.f_next_input <= 60)).groupby([d.sample_id, d.cs]).transform("sum")
    soon = d[d.f_next_input <= 30].groupby("sample_id").cs.nunique()
    d["rl_n_cs_soon"] = d.sample_id.map(soon).fillna(0)
    return d

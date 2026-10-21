"""
extra_ctx2.py · bloque ctx2_*: amplía ctx con lo que más pesa de él, la distancia de la fila a la tarea MÁS URGENTE DE SU
MISMO TIPO en el snapshot (solo filas del mismo sample_id). sentido +1: menor = más urgente (value - min);
sentido -1: mayor = más urgente (max - value).
"""
import pandas as pd

SPEC = [("t5_tts", [5], "time_to_sector", -1),          # transfer: negativo dentro, más cerca de 0 = sale antes
        ("t5_dist", [5], "distance_to_sector", -1),
        ("t4_dist", [4], "distance_to_sector", +1),     # assume: fuera, más cerca del borde
        ("t4_wait", [4], "x_state_age", -1),            # assume: oferta que más lleva esperando
        ("t2_tts", [2], "time_to_sector", +1),          # entry coordination
        ("t3_tte", [3], "k_timeToExitPoint", +1),       # exit coordination
        ("t10_tte", [10], "k_timeToExitPoint", +1),     # exit conditions
        ("t13_dev", [13], "non_conformance_agreed_route_deviation_horizontally", -1),
        ("t12_tz", [12], "logic_t_zone", +1),
        ("t6_md", [6], "time_to_MD", +1),
        ("all_dl", None, "logic_deadline_abs", +1),     # deadline de cada tipo (logic), dentro del tipo
        ("all_age", None, "x_task_age", -1)]            # la tarea más antigua de su tipo


def add(d):
    d = d.copy()
    sid, ty = d.sample_id, d.task_index
    new = {}
    for name, tipos, col, sentido in SPEC:
        v = d[col] if tipos is None else d[col].where(ty.isin(tipos))
        g = v.groupby([sid, ty])
        new[f"ctx2_{name}"] = (v - g.transform("min")) if sentido > 0 else (g.transform("max") - v)
    new = pd.DataFrame(new, index=d.index).astype("float32")
    return pd.concat([d, new], axis=1), list(new.columns)

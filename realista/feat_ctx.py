"""
feat_ctx.py · RAMA "REALISTA" · bloque ctx (ctx_*): contexto de la lista dentro del MISMO snapshot.

LightGBM puntúa cada fila por separado: el gradiente es por lista, pero una fila no ve a las tareas con las que compite.
Estas columnas se lo dicen. Solo usan filas del mismo sample_id (ni otros snapshots ni etiquetas):
  - ctx_n_t*   nº de tareas de cada tipo principal en la lista (2, 3, 4, 5, 6, 10, 12, 13)
  - ctx_n_*    nº de urgentes: assume ofrecido, transfer debido, alerta STCA, instrucción a un avión no asumido
  - ctx_min_* / ctx_dmin_*   mínimo del snapshot de una variable clave y distancia de la fila a ese mínimo
  - ctx_rk_* / ctx_rkt_*     rango (pct) en el snapshot y rango dentro del tipo de variables sin rango en el reto
Prueba: +0.3 sobre b045 con la misma semilla (sección 7 de LGB_realista.ipynb).
"""
import pandas as pd

MAIN_TYPES = [2, 3, 4, 5, 6, 10, 12, 13]


def add_ctx(d):
    d = d.copy()
    sid, ty = d.sample_id, d.task_index
    new = {}
    for t in MAIN_TYPES:
        new[f"ctx_n_t{t}"] = (ty == t).groupby(sid).transform("sum")
    urg = dict(offered=(ty == 4) & (d.x_transfer_in == 1),
               due=(ty == 5) & (d.k_aircraftNeedsToBeTransferred == 1),
               stca=ty.isin([0, 1, 6]) & (d.HMI_conflict_safety_alert == 1),
               nonassumed_instr=ty.isin([1, 3, 6, 7, 8, 10, 12, 13]) & (d.is_assumed == 0))
    for k, m in urg.items():
        new[f"ctx_n_{k}"] = m.groupby(sid).transform("sum")
    for k, col, mask in (("tts4", "time_to_sector", ty == 4), ("tts5", "k_timeToExitPoint", ty == 5),
                         ("zone", "distance_to_zone", ty == 12), ("md", "time_to_MD", ty.isin([0, 1, 6])),
                         ("cz", "cz_task", ty.notna()), ("age", "x_task_age", ty.notna()),
                         ("since", "a_since_flight", ty.notna())):
        v = d[col].where(mask)
        mn = v.groupby(sid).transform("min")
        new[f"ctx_min_{k}"], new[f"ctx_dmin_{k}"] = mn, v - mn
    for col in ("x_task_age", "a_since_flight", "x_state_age", "k_timeToExitPoint", "distance_to_zone", "gxa_tcross",
                "current_FL"):
        new[f"ctx_rk_{col}"] = d[col].groupby(sid).rank(pct=True)
        new[f"ctx_rkt_{col}"] = d[col].groupby([sid, ty]).rank(method="min")
    return pd.concat([d, pd.DataFrame(new, index=d.index).astype("float32")], axis=1)

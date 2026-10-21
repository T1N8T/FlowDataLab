"""
feat_causal.py · Pablo (A) · RAMA FUTURO · Modelo "realista" (sección 20): solo el snapshot actual y snapshots PASADOS.
Columnas cz_* (bloque "causal" en features_future.feature_sets).

Espejo hacia atrás de sc_* (feat_coord.py) y sx_* (feat_clear.py): segundos desde el último cambio, en snapshots
anteriores de la sesión, del estado que resuelve cada tarea (NaN = no ha cambiado aún). Más cz_task = el cambio que
corresponde al tipo de la tarea y su rango en el snapshot / tipo / avión. No usan etiquetas ni snapshots futuros.
"""
import numpy as np, pandas as pd
from feat_coord import add_ranks

KEY = ["session", "cs", "t"]


def since_change(d, cols, kind="any"):
    """Segundos desde el último snapshot (actual incluido) de la sesión en que cambió alguna de cols para el avión.
    kind: 'any', 'up' (sube) o 'down' (baja). Solo mira atrás."""
    a = d.drop_duplicates(KEY)[KEY + cols].sort_values(KEY).reset_index(drop=True)
    g = [a.session, a.cs]
    first = (a.groupby(["session", "cs"], sort=False).cumcount() == 0).values
    ch = np.zeros(len(a), bool)
    for c in cols:
        v = a[c].fillna(-1.0)
        p = v.groupby(g).shift()
        m = (v != p) if kind == "any" else (v > p) if kind == "up" else (v < p)
        ch |= (m & p.notna()).values & ~first
    last = pd.Series(np.where(ch, a.t.values, np.nan)).groupby(g).ffill()
    a["_v"] = a.t.values - last.values
    return d[KEY].merge(a[KEY + ["_v"]], on=KEY, how="left")["_v"].values


def add_causal(d):
    d = d.copy()
    ty = d.task_index.values
    d["cz_isc_any"] = since_change(d, ["x_isc_notified", "x_isc_coordinated", "x_entry_coord_not_done"])
    d["cz_transfer_in"] = since_change(d, ["x_transfer_in"])
    d["cz_assumed"] = since_change(d, ["is_assumed"], "up")
    d["cz_out_any"] = since_change(d, ["k_out_NOTIFIED", "k_out_COORDINATED", "k_outCoordMismatch", "k_outCoordRevision"])
    d["cz_cfl"] = since_change(d, ["k_clearedFL"])
    d["cz_heading"] = since_change(d, ["k_isClearedHeadingAssigned", "k_isLocked"])
    d["cz_exitfl"] = since_change(d, ["exit_FL", "k_isAtExitFL", "k_hasToClimbToExitFL", "k_hasToDescendToExitFL"])
    d["cz_zone"] = since_change(d, ["zone_conflict_ind"])
    d["cz_conflict"] = since_change(d, ["conflict_ind"])
    d["cz_route"] = since_change(d, ["monitoring_rule_activated"])
    d["cz_task"] = np.select(
        [ty == 2, ty == 3, ty == 4, ty == 5, ty == 10, ty == 12, ty == 13, np.isin(ty, [0, 1, 6])],
        [d.cz_isc_any, d.cz_out_any, np.fmin(d.cz_transfer_in, d.cz_assumed), d.cz_assumed,
         np.fmin(d.cz_cfl, d.cz_exitfl), np.fmin(d.cz_zone, d.cz_heading), np.fmin(d.cz_route, d.cz_heading),
         np.fmin(d.cz_conflict, d.cz_cfl)], np.nan)
    return add_ranks(d, "cz_task")

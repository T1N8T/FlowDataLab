"""
feat_coord.py · Pablo (A) · RAMA FUTURO · Cambios de estado FUTUROS del avión ligados a coordinación, assume y transfer.

La señal f_next_input (siguiente instrucción al avión) no ve las coordinaciones con los sectores vecinos, que no son
instrucciones al avión. Aquí se mide, para cada tarea, cuántos segundos faltan hasta el siguiente snapshot de la sesión
en que cambia el estado que esa tarea resuelve:
    entry_coordination -> x_isc_* (notified/coordinated/entry_coord_not_done)
    exit_coordination  -> k_out_* / k_outCoordRevision / k_outCoordMismatch
    assume_aircraft    -> is_assumed pasa a 1          transfer_aircraft -> is_assumed pasa a 0
Columnas sc_* (bloque "coord" en features_future.feature_sets). No usan etiquetas. NaN = no vuelve a cambiar.
"""
import numpy as np, pandas as pd

KEY = ["session", "cs", "t"]


def next_change(d, cols, kind="any"):
    """Segundos hasta el siguiente snapshot (estrictamente posterior) de la sesión en que cambia alguna de `cols` para el
    mismo avión. kind: 'any' = cualquier cambio, 'up' = sube (0->1, NaN->valor), 'down' = baja (1->0).
    Devuelve un array alineado con las filas de d."""
    a = d.drop_duplicates(KEY)[KEY + cols].sort_values(KEY).reset_index(drop=True)
    grp = [a.session, a.cs]
    first = (a.groupby(["session", "cs"], sort=False).cumcount() == 0).values
    ch = np.zeros(len(a), bool)
    for c in cols:
        v = a[c].fillna(-1.0)
        p = v.groupby(grp).shift()
        m = (v != p) if kind == "any" else (v > p) if kind == "up" else (v < p)
        ch |= (m & p.notna()).values & ~first
    ct = pd.Series(np.where(ch, a.t.values, np.nan))
    nxt = ct.groupby(grp).shift(-1).groupby(grp).bfill()
    a["_v"] = nxt.values - a.t.values
    return d[KEY].merge(a[KEY + ["_v"]], on=KEY, how="left")["_v"].values


def add_ranks(d, col):
    d[col + "_rk"] = d.groupby("sample_id")[col].rank(method="min")
    d[col + "_rk_type"] = d.groupby(["sample_id", "task_index"])[col].rank(method="min")
    d[col + "_rk_flight"] = d.groupby(["sample_id", "cs"])[col].rank(method="min")
    return d


def add_coord(d):
    d = d.copy()
    ty = d.task_index.values
    d["sc_isc_coord"] = next_change(d, ["x_isc_coordinated"], "up")
    d["sc_isc_any"] = next_change(d, ["x_isc_notified", "x_isc_coordinated", "x_entry_coord_not_done"])
    d["sc_transfer_in"] = next_change(d, ["x_transfer_in"])
    d["sc_assumed"] = next_change(d, ["is_assumed"], "up")
    d["sc_released"] = next_change(d, ["is_assumed"], "down")
    d["sc_out_rev"] = next_change(d, ["k_outCoordRevision"], "up")
    d["sc_out_any"] = next_change(d, ["k_out_NOTIFIED", "k_out_COORDINATED", "k_outCoordMismatch", "k_outCoordRevision"])
    # el cambio que corresponde al tipo de la tarea
    d["sc_task"] = np.select(
        [ty == 2, ty == 3, ty == 4, ty == 5],
        [np.fmin(d.sc_isc_coord, d.sc_isc_any), np.fmin(d.sc_out_rev, d.sc_out_any), d.sc_assumed, d.sc_released],
        np.nan)
    d = add_ranks(d, "sc_task")
    # gemelo assume / entry_coordination: ¿cambia antes la coordinación de entrada o el assume? (<0 -> coordinación)
    d["sc_coord_minus_assume"] = d.sc_isc_any - d.sc_assumed
    return d

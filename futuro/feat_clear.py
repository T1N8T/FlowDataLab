"""
feat_clear.py · Adri (B) · RAMA FUTURO · Cambios de estado FUTUROS del avión ligados a instrucciones (nivel, rumbo, zona,
conflicto, ruta). Ataca exit_conditions, zone_conflict, return_to_route y conflictos.
Columnas sx_* (bloque "clear" en features_future.feature_sets). No usan etiquetas. NaN = no vuelve a cambiar.
Versión inicial: Adri la amplía o la sustituye.
"""
import numpy as np
from feat_coord import next_change, add_ranks


def add_clear(d):
    d = d.copy()
    ty = d.task_index.values
    d["sx_cfl"] = next_change(d, ["k_clearedFL"])
    d["sx_heading"] = next_change(d, ["k_isClearedHeadingAssigned", "k_isLocked"])
    d["sx_exitfl"] = next_change(d, ["k_isAtExitFL", "k_hasToClimbToExitFL", "k_hasToDescendToExitFL"])
    d["sx_zone_off"] = next_change(d, ["zone_conflict_ind"], "down")
    d["sx_conflict_off"] = next_change(d, ["conflict_ind"], "down")
    d["sx_route_ok"] = next_change(d, ["monitoring_rule_activated"], "down")
    lateral = np.fmin(d.sx_heading, d.sx_route_ok)
    d["sx_task"] = np.select(
        [ty == 10, ty == 12, ty == 13, np.isin(ty, [0, 1, 6]), ty == 7],
        [np.fmin(d.sx_cfl, d.sx_exitfl),
         np.fmin.reduce([d.sx_zone_off, d.sx_heading, d.sx_cfl]),
         lateral,
         np.fmin.reduce([d.sx_conflict_off, d.sx_cfl, d.sx_heading]),
         np.fmin.reduce([d.sx_cfl, d.sx_heading, d.sx_route_ok])],
        np.nan)
    return add_ranks(d, "sx_task")

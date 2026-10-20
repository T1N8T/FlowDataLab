"""
h3_coord.py · hilo h3_coord · features DENTRO del snapshot para coordinaciones y tipos raros.

Todas las features usan solo filas del mismo sample_id (agregados por sample_id, (sample_id, cs) o
(sample_id, task_index)). Ninguna usa target_index.

Uso:
    python pruebas/h3_coord.py ref                    # referencia 1 semilla (99 features de lgb_adri)
    python pruebas/h3_coord.py coord                  # ref + grupo h3 "coord"
    python pruebas/h3_coord.py coord+carga            # varios grupos
    python pruebas/h3_coord.py coord+carga --best     # 2 semillas + preds de test -> h3_coord_best
Grupos: coord, avion2, carga, inter, y los de base.py: exit_coord, deadline, tipo_estado, avion.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/pruebas")
from base import lab_test, build, BEST_GROUPS, FEATURE_GROUPS, lgb_cv, guardar, registrar  # noqa: E402

COORD_TYPES = [2, 3, 10, 13]          # entry_coordination, exit_coordination, exit_conditions, return_to_route
RARE_TYPES = [2, 3, 6, 8, 10, 13]


def _ac(df, s, how):
    """Agregado por avión dentro del snapshot."""
    return s.groupby([df.sample_id, df.cs]).transform(how)


# ----------------------------------------------------------------------------- grupo coord
def fe_coord(df):
    """Urgencia de la tarea frente a las demás de SU tipo + estado de la coordinación de salida."""
    t = df.task_index
    is_c = t.isin(COORD_TYPES)
    # deadline por tipo: entrada -> time_to_sector, salida -> tiempo al exit point
    dl = pd.Series(np.nan, index=df.index)
    dl[t == 2] = df.time_to_sector[t == 2]
    m = t.isin([3, 10, 13])
    dl[m] = df.k_timeToExitPoint.fillna(df.time_to_sector.abs())[m]
    g = dl.groupby([df.sample_id, t])
    out = {
        "c3_dl": dl,
        "c3_dl_rank": g.rank(method="min"),
        "c3_dl_is_min": (dl == g.transform("min")).astype(float).where(is_c),
        "c3_dl_minus_min": dl - g.transform("min"),
        "c3_age_rank_type": df.x_task_age.where(is_c).groupby([df.sample_id, t]).rank(method="min",
                                                                                      ascending=False),
        # niveles: salida no acordada con lo autorizado
        "c3_exit_minus_cfl": (df.exit_FL - df.k_clearedFL).where(t.isin([3, 10])),
        "c3_exit_minus_fl": (df.exit_FL - df.current_FL).where(t.isin([3, 10])),
        "c3_ec_mismatch": ((t == 3) & (df.k_outCoordMismatch == 1)).astype(int),
        "c3_ec_condmet": df.k_exitConditionsAreMet.where(t.isin([3, 10])),
        "c3_ec_tool": ((t.isin([3, 10])) & (df.k_coordinationToolActivation == 1)).astype(int),
        "c3_ec_notdue": ((t == 3) & (df.k_aircraftNeedsToBeTransferred != 1)).astype(int),
        "c3_en_coordinated": ((t == 2) & (df.x_isc_coordinated == 1)).astype(int),
        "c3_en_offered": ((t == 2) & (df.x_transfer_in == 1)).astype(int),
        "c3_en_notdone": ((t == 2) & (df.x_entry_coord_not_done == 1)).astype(int),
        "c3_rtr_dev": df.non_conformance_agreed_route_deviation_horizontally.where(t == 13),
    }
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------------- grupo avion2
def fe_avion2(df):
    """Estado del avión: qué otras tareas tiene en el snapshot y si es el avión 'activo'."""
    t = df.task_index
    asf = _ac(df, df.a_since_flight, "min")
    ac_first = ~df.duplicated(["sample_id", "cs"])
    # rank del avión por a_since_flight (último input) entre los aviones del snapshot
    ac_rank = asf.where(ac_first).groupby(df.sample_id).rank(method="min")
    ac_rank = ac_rank.groupby([df.sample_id, df.cs]).transform("max")
    out = {
        "a3_n_tasks": df.groupby(["sample_id", "cs"]).task_index.transform("size"),
        "a3_has_assume_pend": _ac(df, ((t == 4) & (df.x_transfer_in == 1)).astype(int), "max"),
        "a3_has_transfer": _ac(df, (t == 5).astype(int), "max"),
        "a3_has_coord": _ac(df, t.isin(COORD_TYPES).astype(int), "max"),
        "a3_n_coord": _ac(df, t.isin(COORD_TYPES).astype(int), "sum"),
        "a3_is_newest": (df.x_task_age == _ac(df, df.x_task_age, "min")).astype(int),
        "a3_is_oldest": (df.x_task_age == _ac(df, df.x_task_age, "max")).astype(int),
        "a3_ac_rank_input": ac_rank,
        "a3_age_minus_since": df.x_task_age - df.a_since_flight,
    }
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------------- grupo carga
def fe_carga(df):
    """Carga urgente del snapshot: si no hay nada urgente, el controlador hace coordinaciones."""
    t = df.task_index
    urg = (((t == 4) & (df.x_transfer_in == 1)) | ((t == 5) & (df.k_aircraftNeedsToBeTransferred == 1))
           | (df.HMI_conflict_safety_alert == 1)).astype(int)
    n_urg = urg.groupby(df.sample_id).transform("sum")
    min_tts_ass = df.time_to_sector.where((t == 4) & (df.x_transfer_in == 1)).groupby(df.sample_id).transform("min")
    is_c = t.isin(COORD_TYPES)
    out = {
        "k3_n_urgent": n_urg,
        "k3_none_urgent": (n_urg == 0).astype(int),
        "k3_min_tts_offer": min_tts_ass,
        "k3_n_coord": is_c.astype(int).groupby(df.sample_id).transform("sum"),
        "k3_n_exitc": (t == 3).astype(int).groupby(df.sample_id).transform("sum"),
        "k3_coord_x_nurg": n_urg.where(is_c),
    }
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------------- grupo inter
def fe_inter(df):
    """Interacciones explícitas tipo x columna clave."""
    t = df.task_index
    out = {}
    for code, nm in [(2, "en"), (3, "ex"), (10, "xc"), (13, "rr")]:
        m = t == code
        out[f"i3_age_{nm}"] = df.x_task_age.where(m)
        out[f"i3_since_{nm}"] = df.a_since_flight.where(m)
    out["i3_tts_en"] = df.time_to_sector.where(t == 2)
    out["i3_tte_ex"] = df.k_timeToExitPoint.where(t.isin([3, 10]))
    out["i3_cflmfl_xc"] = df.k_cfl_minus_fl.where(t == 10)
    out["i3_lvl_ex"] = df.a_last_assign_level_clearance.where(t.isin([3, 10]))
    return pd.DataFrame(out, index=df.index)


H3_GROUPS = {"coord": fe_coord, "avion2": fe_avion2, "carga": fe_carga, "inter": fe_inter}
ALL = {**FEATURE_GROUPS, **H3_GROUPS}


def build_h3(df, extra):
    d, F = build(df, BEST_GROUPS)
    new = [ALL[g](d) for g in extra]
    if new:
        d = pd.concat([d] + new, axis=1)
        F = F + [c for n in new for c in n.columns]
    return d, F


def main():
    arg = sys.argv[1]
    best = "--best" in sys.argv
    extra = [] if arg == "ref" else arg.split("+")
    nombre = "h3_coord_best" if best else ("h3_ref" if arg == "ref" else "h3_" + "_".join(extra))
    t0 = time.time()
    lab, test = lab_test()
    d, F = build_h3(lab, extra)
    dt = build_h3(test, extra)[0] if best else None
    seeds = (0, 1) if best else (0,)
    oof, pred = lgb_cv(d, F, seeds=seeds, test=dt)
    guardar(nombre, d, oof, dt, pred)
    registrar(nombre, hilo="h3_coord", descripcion=f"lgb_adri + {extra or 'nada'} ({len(seeds)} semillas)",
              segundos=time.time() - t0, n_features=len(F))
    import json
    fila = [json.loads(line) for line in open("/home/jm/proyectos/FlowDataLab/pruebas/resultados.jsonl")][-1]
    pt = fila["por_tipo"]
    print("   por tipo:", {k: pt[k] for k in ["exit_coordination", "entry_coordination", "return_to_route",
                                               "exit_conditions", "conflict_resolution", "qos_improvement",
                                               "assume_aircraft", "transfer_aircraft", "zone_conflict"]},
          flush=True)


if __name__ == "__main__":
    main()

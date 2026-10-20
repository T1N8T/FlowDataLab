"""
f2_coord.py · hilo f2_coord · features para coordinaciones y tipos raros, SOBRE la línea con futuro.

Grupos:
- adri:  alerta + contexto (pruebas/base.py, los de lgb_adri)
- adri2: exit_coord + deadline + tipo_estado (pruebas/base.py)
- coord: fe_coord de pruebas/h3_coord.py (c3_*, solo snapshot)
- cfut:  NUEVAS de sesión (s2_*): cambios futuros/pasados del estado de coordinación del avión
         (k_out_*, k_outCoordRevision, x_isc_*, herramienta de coordinación), siguiente input que "resuelve"
         la coordinación, avión activo (input más próximo) y fin de la tarea frente a su tipo/avión.
         Ninguna usa target_index; agregan por (session, cs) en el tiempo y por sample_id.

Uso:
    python futuro/pruebas/f2_coord.py ref
    python futuro/pruebas/f2_coord.py adri+cfut
    python futuro/pruebas/f2_coord.py adri+cfut --best      # 3 semillas + preds de test -> f2_best
"""
import sys, time, json
import numpy as np, pandas as pd
sys.path[:0] = ["/home/jm/proyectos/FlowDataLab/futuro/pruebas", "/home/jm/proyectos/FlowDataLab/pruebas"]
from base_fut import datos, lgb_cv, guardar, registrar, OUT  # noqa: E402
import base  # noqa: E402
from h3_coord import fe_coord  # noqa: E402

CSTATE = ["k_out_NOTIFIED", "k_out_COORDINATED", "k_outCoordRevision", "k_outCoordMismatch",
          "k_coordinationToolActivation", "x_isc_notified", "x_isc_coordinated"]
BIG = 3600.0


def _next_prev(a, ev, g):
    """Para cada fila de a (ordenada por g, t): segundos hasta el siguiente evento (estricto) y desde el último
    (incluido el propio snapshot)."""
    te = a.t.where(ev)
    nxt = te.groupby(g).shift(-1).groupby(g).bfill() - a.t
    prv = a.t - te.groupby(g).ffill()
    return nxt, prv


def fe_cfut(d):
    """Features de sesión de coordinación. d contiene lab+test (las sesiones no se mezclan entre splits)."""
    t = d.task_index
    # ---- estado de coordinación por avión y snapshot
    a = d[["sample_id", "session", "cs", "t"] + CSTATE].drop_duplicates(["sample_id", "cs"])
    a = a.sort_values(["session", "cs", "t"]).reset_index(drop=True)
    g = [a.session, a.cs]
    st = a[CSTATE].fillna(-1)
    prev = st.groupby(g).shift()
    has_prev = prev.notna().all(axis=1)
    chg = (st != prev).any(axis=1) & has_prev
    up = lambda c: (st[c] > prev[c]) & has_prev  # noqa: E731
    evs = {"any": chg, "rev": up("k_outCoordRevision"), "tool": up("k_coordinationToolActivation"),
           "outc": up("k_out_COORDINATED"), "outn": up("k_out_NOTIFIED"), "iscc": up("x_isc_coordinated"),
           "iscn": up("x_isc_notified"), "mism": (st.k_outCoordMismatch != prev.k_outCoordMismatch) & has_prev}
    out = a[["sample_id", "cs"]].copy()
    for k, ev in evs.items():
        nxt, prv = _next_prev(a, ev, g)
        out[f"s2_ac_next_{k}"] = nxt
        if k in ("any", "rev", "tool"):
            out[f"s2_ac_prev_{k}"] = prv
    x = d[["sample_id", "cs"]].merge(out, on=["sample_id", "cs"], how="left")
    x.index = d.index
    o = {}
    isx = t.isin([3, 10])          # coordinación / condiciones de salida
    isn = t.isin([2, 9])           # coordinación / condiciones de entrada
    o["s2_ac_next_any"] = x.s2_ac_next_any
    o["s2_ac_prev_any"] = x.s2_ac_prev_any
    o["s2_ex_next_rev"] = x.s2_ac_next_rev.where(isx)
    o["s2_ex_prev_rev"] = x.s2_ac_prev_rev.where(isx)
    o["s2_ex_next_tool"] = x.s2_ac_next_tool.where(isx)
    o["s2_ex_next_outc"] = x[["s2_ac_next_outc", "s2_ac_next_outn", "s2_ac_next_mism"]].min(axis=1).where(isx)
    o["s2_en_next_isc"] = x[["s2_ac_next_iscc", "s2_ac_next_iscn"]].min(axis=1).where(isn)
    # rank del próximo cambio de coordinación entre las tareas de coordinación del snapshot
    cn = x.s2_ac_next_any.where(t.isin([2, 3, 9, 10]))
    o["s2_coord_next_rk"] = cn.groupby(d.sample_id).rank(method="min")
    o["s2_coord_next_is_min"] = (cn == cn.groupby(d.sample_id).transform("min")).astype(float).where(cn.notna())
    # ---- siguiente input que resuelve la coordinación (salida <-> transfer; entrada <-> assume)
    na = d.f_next_act
    res = np.where(t == 3, na == 4, np.where(t == 2, na.isin([0, 1]), np.nan))
    o["s2_next_resolves"] = pd.Series(res, index=d.index).where(na.notna())
    o["s2_next_resolves_t"] = d.f_next_input.where(o["s2_next_resolves"] == 1)
    # ---- avión activo: input más próximo en el futuro / más reciente en el pasado
    acn = d.groupby(["sample_id", "cs"]).f_next_input.transform("min")
    first = ~d.duplicated(["sample_id", "cs"])
    rk = acn.where(first).groupby(d.sample_id).rank(method="min")
    o["s2_ac_next_rank"] = rk.groupby([d.sample_id, d.cs]).transform("max")
    o["s2_ac_next_minus_min"] = acn - acn.groupby(d.sample_id).transform("min")
    o["s2_ac_is_last"] = (d.a_since_flight == d.groupby("sample_id").a_since_flight.transform("min")).astype(int)
    # ---- fin de la tarea frente a su tipo y a su avión
    o["s2_end_rk_type"] = d.f_task_end.groupby([d.sample_id, t]).rank(method="min")
    o["s2_end_minus_cs"] = d.f_task_end - d.f_cs_end
    o["s2_end_minus_min_type"] = d.f_task_end - d.f_task_end.groupby([d.sample_id, t]).transform("min")
    return pd.DataFrame(o, index=d.index)


GROUPS = {
    "adri": lambda d: [base.fe_alerta(d), base.fe_contexto(d)],
    "adri2": lambda d: [base.fe_exit_coord(d), base.fe_deadline(d), base.fe_tipo_estado(d)],
    "coord": lambda d: [fe_coord(d)],
    "cfut": lambda d: [fe_cfut(d)],
    # subconjuntos de cfut
    "clean": lambda d: [fe_cfut(d)[LEAN]],
    "cact": lambda d: [fe_cfut(d)[ACT]],
}
LEAN = ["s2_ex_next_rev", "s2_ex_next_tool", "s2_ex_next_outc", "s2_en_next_isc", "s2_coord_next_rk"]
ACT = ["s2_ac_next_rank", "s2_ac_next_minus_min", "s2_end_rk_type", "s2_end_minus_min_type"]


def build(groups):
    lab, te, fs = datos()
    F = fs["base"] + fs["logic"] + fs["past"] + fs["future"]
    d = pd.concat([lab, te], ignore_index=True)
    new = [n for gname in groups for n in GROUPS[gname](d)]
    if new:
        d = pd.concat([d] + new, axis=1)
        F = F + [c for n in new for c in n.columns]
    lab = d[d.split != "test"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    return lab, te, F


def main():
    arg = sys.argv[1]
    best = "--best" in sys.argv
    groups = [] if arg == "ref" else arg.split("+")
    nombre = "f2_best" if best else ("f2_ref" if arg == "ref" else "f2_" + "_".join(groups))
    t0 = time.time()
    lab, te, F = build(groups)
    seeds = (0, 1, 2) if best else (0,)
    oof, pred = lgb_cv(lab, F, seeds=seeds, test=te if best else None)
    guardar(nombre, lab, oof, te if best else None, pred)
    registrar(nombre, hilo="f2_coord", descripcion=f"futuro (todos los bloques) + {groups or 'nada'} "
              f"({len(seeds)} semillas)", segundos=time.time() - t0, n_features=len(F))
    fila = [json.loads(line) for line in open(OUT / "resultados.jsonl")][-1]
    pt = fila["por_tipo"]
    print("   por tipo:", {k[:12]: pt[k] for k in ["exit_coordination", "entry_coordination", "exit_conditions",
                                                    "exit_conflict_resolution", "qos_improvement",
                                                    "conflict_resolution", "return_to_route", "zone_conflict"]},
          flush=True)


if __name__ == "__main__":
    main()

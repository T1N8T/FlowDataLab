"""f11_feats.py · Pablo (A) · hilo f11 · bloques de features nuevos para f11_lab.py (--add). Ninguno usa etiquetas.

    link  lk_*  diferencias entre los momentos de la evidencia de una misma fila (¿la instrucción compatible resuelve
                ESTA tarea?, ¿ocurre al final de la racha?): los árboles no construyen bien las diferencias solos
    pair  pr_*  conflictos (tipos 0, 1, 6): la tarea nombra dos aviones y f_*/si_* solo miran el primero. Aquí, la
                siguiente instrucción y el siguiente cambio de nivel autorizado / rumbo del SEGUNDO avión
    nsi   ns_*  orden en la SESIÓN: cuántas instrucciones (a cualquier avión) llegan antes de la siguiente a este avión
    io    io_*  ORÁCULO SIN ETIQUETAS: qué hacen con ese avión los demás controladores del mismo escenario en el mismo
                momento (tiempo alineado de feat_oracle.align): fracción de sesiones fuente en que el avión recibe una
                instrucción compatible con la tarea a ±30/60 s, cualquier instrucción a ±30 s, la tarea desaparece a
                ±60 s, y la mediana de cuándo la resuelven. Fuentes: TODAS las sesiones del escenario, también las de
                test, salvo la propia (no usa etiquetas: es idéntico en la CV y en test)
Uso: f11_feats.block(nombre, D) con D = lab + te concatenados (columnas de la caché de f10_instr.datos_instr()).
Se cachea en futuro/pruebas/cache_f11_<nombre>.parquet.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro")]
import features_future as ff  # noqa: E402

OUT = ROOT / "futuro" / "pruebas"
CONFLICT = [0, 1, 6]


def _events(D):
    """Instrucciones por (session, cs): tiempos (misma extracción que features_future.add_future)."""
    ev = np.where(D.a_since_flight < 3600, np.round(D.t - D.a_since_flight), np.nan)
    k = D.assign(ev=ev).dropna(subset=["ev"]).drop_duplicates(["session", "cs", "ev"])
    return k[["session", "cs", "ev"]].sort_values("ev")


def _next_after(times_by_key, keys, t):
    """Para cada fila (key, t): segundos hasta el primer tiempo de times_by_key[key] > t + 1.5 (NaN si no hay)."""
    out = np.full(len(t), np.nan)
    df = pd.DataFrame(dict(k=keys, t=t, i=np.arange(len(t))))
    for key, g in df.groupby("k", sort=False):
        e = times_by_key.get(key)
        if e is None or not len(e):
            continue
        j = np.searchsorted(e, g.t.values + 1.5)
        ok = j < len(e)
        out[g.i.values[ok]] = e[j[ok]] - g.t.values[ok]
    return out


def link(D):
    t_end = D.groupby("block").t.transform("max") - D.t
    o = pd.DataFrame(index=D.index)
    o["lk_end_m_match"] = D.f_task_end - D.si_match
    o["lk_end_m_next"] = D.f_task_end - D.f_next_input
    o["lk_sx_m_match"] = D.sx_task - D.si_match
    o["lk_csend_m_end"] = D.f_cs_end - D.f_task_end
    o["lk_match_m_toend"] = D.si_match - t_end
    o["lk_next_m_toend"] = D.f_next_input - t_end
    o["lk_end_m_toend"] = D.f_task_end - t_end
    o["lk_sx_m_toend"] = D.sx_task - t_end
    o["lk_resolved"] = ((D.f_task_end - D.si_match).abs() <= 10).astype(float).where(D.si_match.notna())
    o["lk_resolved_rk"] = (-o.lk_resolved.fillna(0)).groupby(D.sample_id).rank(method="min")
    return o


def pair(D):
    ti = D.task_index.values
    names = D.task_name.values
    cs2 = np.array([n[len(ff.TYPES[t]) + 1:].split("_")[1] if t in CONFLICT and n.count("_") >= 1 + ff.TYPES[t].count("_") + 1
                    else "" for n, t in zip(names, ti)], dtype=object)
    E = _events(D)
    tb = {k: g.ev.values for k, g in E.groupby(["session", "cs"])}
    keys = list(zip(D.session.values, cs2))
    nxt2 = _next_after(tb, keys, D.t.values)
    # cambios del nivel autorizado / rumbo del segundo avión
    a = D.drop_duplicates(["session", "cs", "t"])[["session", "cs", "t", "k_clearedFL", "k_isClearedHeadingAssigned"]] \
        .sort_values(["session", "cs", "t"])
    ch = np.zeros(len(a), bool)
    for c in ["k_clearedFL", "k_isClearedHeadingAssigned"]:
        v = a[c].fillna(-1.0)
        p = v.groupby([a.session, a.cs]).shift()
        ch |= ((v != p) & p.notna()).values
    c = a[ch]
    tc = {k: g.t.values for k, g in c.groupby(["session", "cs"])}
    clr2 = _next_after(tc, keys, D.t.values - 1.5 + 0.01)
    is_c = np.isin(ti, CONFLICT)
    o = pd.DataFrame(index=D.index)
    o["pr_next2"] = np.where(is_c, nxt2, np.nan)
    o["pr_next_min"] = np.where(is_c, np.fmin(nxt2, D.f_next_input.values), np.nan)
    o["pr_clear2"] = np.where(is_c, clr2, np.nan)
    o["pr_next_min_rk"] = pd.Series(np.where(is_c, o.pr_next_min, np.nan)).groupby(D.sample_id.values).rank(method="min").values
    return o


def nsi(D):
    E = _events(D)
    se = {s: g.ev.values for s, g in E.groupby("session")}
    n_before = np.full(len(D), np.nan)
    n60 = np.zeros(len(D))
    for s, idx in D.groupby("session").indices.items():
        e = se.get(s, np.array([]))
        t = D.t.values[idx]
        lo = np.searchsorted(e, t + 1.5)
        hi = np.searchsorted(e, t + D.f_next_input.values[idx] - 0.5)
        v = (hi - lo).astype(float)
        v[np.isnan(D.f_next_input.values[idx])] = np.nan
        n_before[idx] = v
        n60[idx] = np.searchsorted(e, t + 60) - lo
    o = pd.DataFrame(index=D.index)
    o["ns_n_before"] = n_before
    o["ns_first"] = (n_before == 0).astype(float)
    o["ns_n60_session"] = n60
    o["ns_match_first"] = ((n_before == 0) & (np.abs(D.si_match.values - D.f_next_input.values) < 1)).astype(float)
    return o


COMPAT = {4: [0, 1], 2: [0, 1], 5: [4], 12: [3], 13: [3], 10: [2], 3: [2, 3], 6: [2, 3], 1: [2, 3], 0: [0, 1, 2, 3],
          7: [2, 3], 8: [2, 3]}


def io(D):
    import feat_oracle
    tau = feat_oracle.align(D)
    off = pd.Series(D.t.values - tau).groupby(D.session.values).first()
    ev = np.where(D.a_since_flight < 3600, np.round(D.t - D.a_since_flight), np.nan)
    k = D.assign(ev=ev).dropna(subset=["ev"]).sort_values("t").drop_duplicates(["session", "cs", "ev"])
    acts = k[["a_last_" + a for a in ff.ACTS]].values
    k = k.assign(act=np.where(acts.max(1) == 1, acts.argmax(1), len(ff.ACTS)))
    k["tau"] = k.ev - k.session.map(off).values
    cs_id = {c: i for i, c in enumerate(pd.unique(D.cs))}
    tk = D.groupby(["session", "task_name"]).t.max().rename("t_end").reset_index()
    tk["tau_end"] = tk.t_end - tk.session.map(off).values
    pres = D.groupby("session").cs.apply(set)
    n = len(D)
    S = 20000.0                                            # tau en [-5000, 15000) -> clave * 1e5 + tau + 5000
    cols = ["m30", "m60", "a30", "e60", "src", "tsrc"]
    acc = {c: np.zeros(n) for c in cols}
    dts = []
    ti = D.task_index.values
    comp = np.zeros((n, len(ff.ACTS) + 1), bool)
    for typ, cm in COMPAT.items():
        comp[np.ix_(ti == typ, cm)] = True
    for sc, g in D.groupby("scenario"):
        idx = g.index.values
        sess = pd.unique(g.session)
        rcs = np.array([cs_id[c] for c in g.cs.values])
        rt = tau[idx]
        rses = g.session.values
        dt_mat = np.full((len(idx), len(sess)), np.nan)
        for j, s2 in enumerate(sess):
            ok_row = rses != s2
            e = k[k.session == s2]
            ekey = np.array([cs_id[c] for c in e.cs.values]) * (len(ff.ACTS) + 1) + e.act.values
            val = np.sort(ekey * 1e5 + e.tau.values + 5000)
            dmin = np.full((len(idx), len(ff.ACTS) + 1), np.inf)
            for a in range(len(ff.ACTS) + 1):
                q = (rcs * (len(ff.ACTS) + 1) + a) * 1e5 + rt + 5000
                p = np.searchsorted(val, q)
                for pp in (p - 1, np.minimum(p, len(val) - 1)):
                    if not len(val):
                        break
                    v = val[np.clip(pp, 0, len(val) - 1)]
                    same = np.floor(v / 1e5) == np.floor(q / 1e5)
                    dmin[:, a] = np.where(same & (pp >= 0), np.minimum(dmin[:, a], np.abs(v - q)), dmin[:, a])
            present = np.isin(g.cs.values, list(pres[s2])) & ok_row
            dm = np.where(comp[idx], dmin, np.inf).min(1)
            da = dmin.min(1)
            acc["src"][idx] += present
            acc["m30"][idx] += present & (dm <= 30)
            acc["m60"][idx] += present & (dm <= 60)
            acc["a30"][idx] += present & (da <= 30)
            te_ = tk[tk.session == s2].set_index("task_name").tau_end
            tend = g.task_name.map(te_).values
            hast = ~np.isnan(tend) & ok_row
            acc["tsrc"][idx] += hast
            acc["e60"][idx] += hast & (np.abs(tend - rt) <= 60)
            dt_mat[:, j] = np.where(hast, tend - rt, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(dt_mat, axis=1)
        dts.append((idx, med))
    o = pd.DataFrame(index=D.index)
    with np.errstate(all="ignore"):
        o["io_match30"] = acc["m30"] / acc["src"]
        o["io_match60"] = acc["m60"] / acc["src"]
        o["io_any30"] = acc["a30"] / acc["src"]
        o["io_end60"] = acc["e60"] / acc["tsrc"]
        o["io_task_frac"] = acc["tsrc"] / acc["src"]
    o["io_dt_end"] = np.nan
    for idx, med in dts:
        o.loc[idx, "io_dt_end"] = med
    o["io_dt_end_abs"] = o.io_dt_end.abs()
    for c in ["io_match30", "io_match60", "io_end60"]:
        o[c + "_rk"] = o[c].round(6).groupby(D.sample_id.values).rank(ascending=False, method="min")
    return o


def causal(D):
    """cz_* de realista/feat_causal.py (segundos desde el ÚLTIMO cambio del estado que resuelve cada tarea): en el modelo
    con futuro no se había probado; puede marcar qué dispara el inicio de una racha (p. ej. una coordinación entrante)."""
    sys.path.insert(0, str(ROOT / "realista"))
    from feat_causal import add_causal
    o = add_causal(D[["session", "cs", "t", "sample_id", "task_index", "x_isc_notified", "x_isc_coordinated",
                      "x_entry_coord_not_done", "x_transfer_in", "is_assumed", "k_out_NOTIFIED", "k_out_COORDINATED",
                      "k_outCoordMismatch", "k_outCoordRevision", "k_clearedFL", "k_isClearedHeadingAssigned",
                      "k_isLocked", "exit_FL", "k_isAtExitFL", "k_hasToClimbToExitFL", "k_hasToDescendToExitFL",
                      "zone_conflict_ind", "conflict_ind", "monitoring_rule_activated"]].copy())
    return o[[c for c in o.columns if c.startswith("cz_")]]


def cfl(D):
    """QUÉ hace el siguiente cambio de nivel autorizado (sx_cfl solo dice cuándo): si el nuevo nivel es el de salida
    (resuelve exit_conditions), si acerca al nivel de salida, y el salto en FL. También el siguiente cambio de nivel de
    salida (exit_FL) y si después coinciden. Sin etiquetas."""
    a = D.drop_duplicates(["session", "cs", "t"])[["session", "cs", "t", "k_clearedFL", "exit_FL"]] \
        .sort_values(["session", "cs", "t"]).reset_index(drop=True)
    g = a.groupby(["session", "cs"], sort=False)
    prev = g.k_clearedFL.shift()
    ch = (a.k_clearedFL != prev) & prev.notna() & a.k_clearedFL.notna()
    ev = a[ch].assign(new=a.k_clearedFL[ch], old=prev[ch], ex=a.exit_FL[ch])
    out = {k: np.full(len(a), np.nan) for k in ("dt", "to_exit", "toward", "delta")}
    for key, idx in a.groupby(["session", "cs"], sort=False).indices.items():
        e = ev[(ev.session == key[0]) & (ev.cs == key[1])] if False else None
    evg = {k: v for k, v in ev.groupby(["session", "cs"], sort=False)}
    for key, idx in a.groupby(["session", "cs"], sort=False).indices.items():
        e = evg.get(key)
        if e is None:
            continue
        t = a.t.values[idx]
        j = np.searchsorted(e.t.values, t + 0.5)
        ok = j < len(e)
        jj = j[ok]
        ii = idx[ok]
        out["dt"][ii] = e.t.values[jj] - t[ok]
        nw, od, ex = e.new.values[jj], e.old.values[jj], e.ex.values[jj]
        out["to_exit"][ii] = np.where(np.isnan(ex), np.nan, (np.abs(nw - ex) < 1).astype(float))
        out["toward"][ii] = np.where(np.isnan(ex), np.nan, (np.abs(nw - ex) < np.abs(od - ex)).astype(float))
        out["delta"][ii] = nw - od
    A = a[["session", "cs", "t"]].assign(**{f"cf_{k}": v for k, v in out.items()})
    o = D[["session", "cs", "t"]].merge(A, on=["session", "cs", "t"], how="left").drop(columns=["session", "cs", "t"])
    o.index = D.index
    ti = D.task_index.values
    o["cf_exitcond_soon"] = np.where(ti == 10, (o.cf_to_exit == 1) & (o.cf_dt <= 60), np.nan).astype(float)
    o["cf_dt_if_exit"] = np.where(o.cf_to_exit == 1, o.cf_dt, np.nan)
    return o


BLOCKS = dict(link=link, pair=pair, nsi=nsi, io=io, causal=causal, cfl=cfl)


def block(name, D):
    p = OUT / f"cache_f11_{name}.parquet"
    if p.exists():
        B = pd.read_parquet(p)
        if len(B) == len(D):
            return B
    B = BLOCKS[name](D).reset_index(drop=True)
    B.to_parquet(p)
    return B


if __name__ == "__main__":
    from f10_instr import datos_instr
    lab, te, fs = datos_instr()
    D = pd.concat([lab, te], ignore_index=True)
    for b in sys.argv[1:] or BLOCKS:
        B = block(b, D)
        y = D.target_index.values[:len(B)] == 1
        print(b, B.shape)
        print(B[y].describe().T[["count", "mean", "50%"]].join(B[~y & (D.split != "test").values].describe().T[["count", "mean", "50%"]], rsuffix="_no"))

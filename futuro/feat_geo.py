"""
feat_geo.py · Pablo (A) · RAMA FUTURO · Geometría del tráfico (sección 18 de LGB_futuro.ipynb). Columnas gx*_.

Los datos no traen posiciones (ni lat/lon ni rumbo), así que no hay CPA/TCPA entre parejas. Sí hay, por avión, distancia
y tiempo al borde y al punto de salida, velocidad, nivel actual, nivel de salida y nivel autorizado. Tres bloques:
    geoA (gxa_)  borde del sector: velocidad real de acercamiento al borde (últimos ~30 s), tiempo hasta cruzar con esa
                 velocidad, orden de entrada / salida del avión entre los aviones del snapshot.
    geoB (gxb_)  tráfico que cruza el mismo borde a la vez: nº de otros aviones que entran (salen) a ±60 / ±120 s, de ellos
                 cuántos a ≤ 10 / 20 FL (entrada: nivel actual; salida: nivel de salida), mínima diferencia de tiempo y de
                 nivel con ellos. Es la interacción física que se puede calcular sin posiciones.
    geoC (gxc_)  niveles: velocidad vertical (FL/min), FL que faltan hasta el de salida y ritmo necesario, nivel
                 autorizado frente al de salida, segundos desde el último cambio del nivel de salida y hasta el siguiente
                 (el nivel de salida cambia ~1.6 veces por avión: revisiones de la coordinación de salida).
No usan etiquetas. Usan otras filas del mismo snapshot (otros aviones) y snapshots pasados / futuros del mismo avión.
"""
import numpy as np, pandas as pd
from feat_coord import next_change

KEY = ["session", "cs", "t"]


def _aircraft(d):
    """Una fila por (snapshot, avión)."""
    return d.drop_duplicates(["sample_id", "cs"])[["sample_id"] + KEY + [
        "distance_to_sector", "time_to_sector", "k_timeToExitPoint", "current_FL", "exit_FL", "k_clearedFL",
        "within_sector", "approaching_sector"]].reset_index(drop=True)


def _lag(a, cols, back=30.0):
    """Valores del mismo avión ~back s antes (el snapshot más reciente con t <= t - back) y su Δt."""
    s = a[KEY + cols].drop_duplicates(KEY).sort_values("t")
    q = a[KEY].assign(tq=a.t - back).sort_values("tq")
    m = pd.merge_asof(q, s.rename(columns={"t": "tl"}), left_on="tq", right_on="tl", by=["session", "cs"],
                      direction="backward", tolerance=60.0)
    m = m.set_index(KEY).reindex(pd.MultiIndex.from_frame(a[KEY]))
    return m[cols].values, (a.t.values - m.tl.values)


def _since_change(d, col):
    """Segundos desde el último cambio de col en el mismo avión (NaN si no ha cambiado)."""
    a = d.drop_duplicates(KEY)[KEY + [col]].sort_values(KEY).reset_index(drop=True)
    g = [a.session, a.cs]
    v = a[col].fillna(-1.0)
    ch = (v != v.groupby(g).shift()) & (a.groupby(["session", "cs"]).cumcount() > 0)
    last = pd.Series(np.where(ch, a.t, np.nan)).groupby(g).ffill()
    a["_v"] = a.t.values - last.values
    return d[KEY].merge(a[KEY + ["_v"]], on=KEY, how="left")["_v"].values


def _crossing(a, tcol, flcol, mask, pre):
    """Interacciones entre aviones del mismo snapshot que cruzan el mismo borde (mask) en tiempos parecidos."""
    x = a.loc[mask, ["sample_id", tcol, flcol]].reset_index()
    p = x.merge(x, on="sample_id", suffixes=("", "_o"))
    p = p[p["index"] != p["index_o"]]
    dt = (p[tcol] - p[tcol + "_o"]).abs()
    dfl = (p[flcol] - p[flcol + "_o"]).abs()
    out = pd.DataFrame(index=a.index)
    g = p["index"]
    for w in (60, 120):
        out[f"{pre}_n{w}"] = (dt <= w).groupby(g).sum()
    for f in (10, 20):
        out[f"{pre}_n120_fl{f}"] = ((dt <= 120) & (dfl <= f)).groupby(g).sum()
    out[f"{pre}_min_dt"] = dt.groupby(g).min()
    out[f"{pre}_min_dfl120"] = dfl.where(dt <= 120).groupby(g).min()
    # combinación: el más cercano en tiempo y nivel a la vez (s + 6 s por FL)
    out[f"{pre}_min_tfl"] = (dt + 6 * dfl).groupby(g).min()
    cnt = [c for c in out.columns if "_n" in c[len(pre):]]
    out.loc[mask.values, cnt] = out.loc[mask.values, cnt].fillna(0)
    return out


def add_geo(d):
    d = d.copy()
    a = _aircraft(d)
    # ---- A · borde del sector
    lagv, dtl = _lag(a, ["distance_to_sector", "current_FL", "exit_FL"])
    dist = a.distance_to_sector.abs().values
    rate = (np.abs(lagv[:, 0]) - dist) / dtl                         # m/s hacia el borde (entrada o salida)
    A = pd.DataFrame({"sample_id": a.sample_id, "cs": a.cs})
    A["gxa_close_rate"] = rate
    outside = a.distance_to_sector.values > 0
    A["gxa_tcross"] = np.clip(dist / np.where(rate > 1, rate, np.nan), 0, 3600)
    A["gxa_tcross_minus_tts"] = A.gxa_tcross - a.time_to_sector.abs().values
    ent = a.time_to_sector.where(outside)
    ext = a.k_timeToExitPoint.where(~outside)
    A["gxa_entry_order"] = ent.groupby(a.sample_id).rank(method="min")
    A["gxa_exit_order"] = ext.groupby(a.sample_id).rank(method="min")
    # ---- B · tráfico que cruza el mismo borde
    a["_ent"], a["_ext"] = ent, ext
    B = pd.concat([_crossing(a, "_ent", "current_FL", pd.Series(outside), "gxb_entry"),
                   _crossing(a, "_ext", "exit_FL", pd.Series(~outside), "gxb_exit")], axis=1)
    # ---- C · niveles
    C = pd.DataFrame(index=a.index)
    C["gxc_vrate"] = (a.current_FL.values - lagv[:, 1]) / dtl * 60
    C["gxc_fl_to_exit"] = a.exit_FL - a.current_FL
    tx = np.where(outside, a.time_to_sector.values + a.k_timeToExitPoint.values, a.k_timeToExitPoint.values)
    C["gxc_need_rate"] = C.gxc_fl_to_exit / np.clip(tx, 30, None) * 60
    C["gxc_need_vs_rate"] = C.gxc_need_rate - C.gxc_vrate
    C["gxc_cfl_vs_exit"] = a.k_clearedFL - a.exit_FL
    C["gxc_exitfl_changed30"] = (a.exit_FL.values != lagv[:, 2]).astype(float)
    C.loc[np.isnan(lagv[:, 2]), "gxc_exitfl_changed30"] = np.nan
    G = pd.concat([A, B, C], axis=1).drop_duplicates(["sample_id", "cs"])
    d = d.merge(G, on=["sample_id", "cs"], how="left")
    d["gxc_exitfl_since"] = _since_change(d, "exit_FL")
    d["gxc_exitfl_next"] = next_change(d, ["exit_FL"])
    return d

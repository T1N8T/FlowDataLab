"""
features_realista.py · LÍNEA REALISTA · Features de un modelo que funcionaría en tiempo real: el snapshot actual y
snapshots PASADOS de la sesión. Nada del futuro y nada de etiquetas (ni oráculo).

Reproduce exactamente realista/features_real.build() + feature_sets() del repo con los bloques de la entrega
(real_adri_noche): base + logic + past + causal + geo. Lo común con la línea futuro (carga, rachas, reglas, pasado,
rangos) se importa de features_futuro.py; aquí viven el bloque causal (antes realista/feat_causal.py) y la parte
realista del bloque geo (antes futuro/feat_geo.py, sin gxc_exitfl_next, que mira al siguiente cambio).

    base    columnas originales (las a_*, x_*, k_* del reto ya miran solo atrás)
    logic   reglas dentro del snapshot (logic_*): deadline por tipo, factibilidad, ventana típica, EDF
    past    p_*, blk_pos, gap_prev: antigüedad de la tarea, deltas frente al snapshot anterior, posición en la racha
            (sin blk_len ni gap_next: miran al resto de la racha)
    causal  cz_*: segundos desde el último cambio del estado que resuelve cada tarea
    geo     gxa_*, gxb_*, gxc_*: acercamiento al borde, tráfico que cruza el mismo borde a la vez y niveles

Uso:  from features_realista import construir;  lab, te, fs = construir()
      (la primera vez ~15 s; luego se lee de entrega/cache/features_realista.parquet)
"""
import numpy as np, pandas as pd
import comun
import features_futuro as ff

CACHE_FEATS = comun.CACHE / "features_realista.parquet"
BLOQUES = ["base", "logic", "past", "causal", "geo"]
FUTURE_COLS = {"blk_len", "gap_next", "gxc_exitfl_next"}           # miran al futuro: nunca entre las features
META = ['sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index',
        'cs', 't', 'split', 'fold', 'block']
KEY = ["session", "cs", "t"]


# ----------------------------------------------------------------------------------------------------- bloque causal
def since_change(d, cols, kind="any"):
    """Segundos desde el último snapshot (actual incluido) de la sesión en que cambió alguna de cols para el avión.
    kind: 'any', 'up' (sube) o 'down' (baja). NaN = no ha cambiado aún. Solo mira atrás."""
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
    """cz_*: segundos desde el último cambio de cada estado y cz_task = el que corresponde al tipo de la tarea (+ rangos)."""
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
    return ff.add_ranks(d, "cz_task")


# ----------------------------------------------------------------------------------------------------- bloque geo
def _aircraft(d):
    """Una fila por (snapshot, avión)."""
    return d.drop_duplicates(["sample_id", "cs"])[["sample_id"] + KEY + [
        "distance_to_sector", "time_to_sector", "k_timeToExitPoint", "current_FL", "exit_FL", "k_clearedFL",
        "within_sector", "approaching_sector"]].reset_index(drop=True)


def _lag(a, cols, back=30.0):
    """Valores del mismo avión ~back s ANTES (el snapshot más reciente con t <= t - back) y su Δt."""
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
    out[f"{pre}_min_tfl"] = (dt + 6 * dfl).groupby(g).min()           # el más cercano en tiempo y nivel (s + 6 s por FL)
    cnt = [c for c in out.columns if "_n" in c[len(pre):]]
    out.loc[mask.values, cnt] = out.loc[mask.values, cnt].fillna(0)
    return out


def add_geo(d):
    """gxa_ (borde del sector), gxb_ (tráfico que cruza el mismo borde a la vez), gxc_ (niveles). Los datos no traen
    posiciones, así que la geometría se aproxima con distancia / tiempo al borde, velocidad y niveles. Usa otros aviones
    del mismo snapshot y snapshots ANTERIORES del mismo avión; no calcula gxc_exitfl_next (futuro)."""
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
    return d


# ----------------------------------------------------------------------------------------------------- todo junto
def feature_sets(d):
    """Columnas de cada bloque (mismo orden que realista/features_real.feature_sets)."""
    pref = ("logic_", "p_", "cz_", "gx", "h_", "ctx_")
    base = [c for c in d.columns if c not in META and not c.startswith(pref)
            and c not in ("blk_pos", "blk_len", "gap_prev", "gap_next")]
    fs = dict(base=base,
              logic=[c for c in d.columns if c.startswith("logic_")],
              past=[c for c in d.columns if c.startswith("p_")] + ["blk_pos", "gap_prev"],
              causal=[c for c in d.columns if c.startswith("cz_")],
              geo=[c for c in d.columns if c.startswith("gx") and c not in FUTURE_COLS])
    assert not FUTURE_COLS & set(sum(fs.values(), [])), "columna que mira al futuro entre las features"
    return fs


def _calcular():
    d = ff.cargar()
    if "fold" not in d:                                    # folds comunes (entrega/datos/folds.csv); test -> -1
        folds = pd.read_csv(comun.DATOS / "folds.csv")
        d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold))).fillna(-1).astype(int)
        d.loc[d.split == "test", "fold"] = -1
    d = ff.add_blocks(d)
    d = ff.add_logic(d)
    d = ff.add_past(d)
    d = add_geo(d)
    d = add_causal(d)
    d = d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    f64 = [c for c in d.columns if d[c].dtype == "float64" and c != "t"]
    d[f64] = d[f64].astype("float32")                       # como la caché de realista/noche_adri.py
    return d


def construir(recalcular=False):
    """(lab, te, fs): filas etiquetadas (train + validation) y de test, ordenadas por sample_id, row_index, y las columnas
    de cada bloque (base, logic, past, causal, geo)."""
    if CACHE_FEATS.exists() and not recalcular:
        d = pd.read_parquet(CACHE_FEATS)
    else:
        d = _calcular()
        CACHE_FEATS.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(CACHE_FEATS)
    fs = feature_sets(d)
    lab = d[d.split != "test"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    return lab, te, fs


if __name__ == "__main__":
    lab, te, fs = construir(recalcular=True)
    print(f"lab {lab.sample_id.nunique()} snapshots ({len(lab)} filas) · test {te.sample_id.nunique()} · bloques " +
          ", ".join(f"{k}={len(v)}" for k, v in fs.items()))

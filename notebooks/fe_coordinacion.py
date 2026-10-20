"""
Features de "cuándo toca coordinar" (exit_coordination) y de return_to_route.
Pensadas para pegarse en LGB_adri.ipynb (sección 3) o importarse tal cual.

Reglas del equipo respetadas:
  * Todo se calcula con groupby("sample_id") (o "sample_id" + "cs"): cero info de otros snapshots.
  * Los vacíos se dejan como NaN (nunca 0). Donde hay que comparar, se usa fillna SOLO dentro de la comparación.
  * Si una columna no existe en el CSV, la feature sale NaN y se avisa (no rompe).

Índices de tipo (TYPES de los notebooks):
  2 entry_coordination · 3 exit_coordination · 4 assume · 5 transfer · 7 non_conformance · 12 zone · 13 return_to_route
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

EC, TR, NC, RTR = 3, 5, 7, 13

NEEDED = ["k_out_NOTIFIED", "k_out_COORDINATED", "k_outCoordMismatch", "k_coordinationToolActivation",
          "k_timeToExitPoint", "k_aircraftNeedsToBeTransferred", "x_task_age", "a_since_flight", "time_to_sector",
          "non_conformance_agreed_route_deviation_horizontally", "non_conformance_agreed_route_deviation_fl"]


def _c(df, name):
    """Columna como float; si no existe, NaN (y queda avisado en check_cols)."""
    return df[name].astype(float) if name in df.columns else pd.Series(np.nan, index=df.index)


def check_cols(df):
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:
        print("AVISO, columnas que no existen (sus features saldrán NaN):", miss)
    return miss


def _rank_in(df, v, mask, ascending=True):
    """Rango (1 = primero) dentro del snapshot, solo entre las filas de `mask`. El resto queda NaN."""
    return v.where(mask).groupby(df.sample_id).rank(method="min", ascending=ascending)


def _snap(df, v, mask, how):
    """min/max/sum/count de v entre las filas de `mask`, repartido a todas las filas del snapshot."""
    return v.where(mask).groupby(df.sample_id).transform(how)


def _ac(df, m):
    """Indicador a nivel de avión (dentro del snapshot), repartido a todas sus tareas."""
    return m.astype(int).groupby([df.sample_id, df.cs]).transform("max")


def fe_coordinacion(df):
    t = df.task_index
    ec, tr = t == EC, t == TR
    notif, coord = _c(df, "k_out_NOTIFIED"), _c(df, "k_out_COORDINATED")
    mism, tool = _c(df, "k_outCoordMismatch"), _c(df, "k_coordinationToolActivation")
    tte, due = _c(df, "k_timeToExitPoint"), _c(df, "k_aircraftNeedsToBeTransferred")
    age, since = _c(df, "x_task_age"), _c(df, "a_since_flight")
    abs_tts = _c(df, "time_to_sector").abs()

    pend = ec & (coord.fillna(0) != 1)          # coordinación de salida todavía sin cerrar
    out = {}

    # 1) Señales de "toca coordinar", SOLO en las filas exit_coordination (resto NaN: así el árbol no las mezcla)
    out["ec_pending"] = pend.astype(int)
    out["ec_mismatch"] = mism.where(ec)
    out["ec_tool"] = tool.where(ec)
    out["ec_notified"] = notif.where(ec)
    out["ec_need"] = (mism.fillna(0) + tool.fillna(0) + (1 - notif.fillna(0))).where(ec)   # 0..3, heurística

    # 2) Cuál de las pendientes: ranking SOLO entre exit_coordination pendientes
    out["ec_tte"] = tte.where(pend)
    out["ec_rank_tte"] = _rank_in(df, tte, pend, True)            # 1 = la que antes llega a la salida
    out["ec_gap_tte"] = tte.where(pend) - _snap(df, tte, pend, "min")
    out["ec_rank_age"] = _rank_in(df, age, pend, False)           # 1 = la más antigua
    out["ec_rank_since"] = _rank_in(df, since, pend, False)       # 1 = lleva más tiempo sin que el controlador la toque
    out["ec_rank_need"] = _rank_in(df, out["ec_need"], pend, False)

    # 3) Gemelo exit_coordination <-> transfer del mismo avión (análogo al twin assume/entry)
    ac_pend = _ac(df, pend)
    ac_tr = _ac(df, tr)
    out["ac_pend_ec"] = ac_pend
    out["ac_has_transfer"] = ac_tr
    out["tx_blocked_by_ec"] = (tr & (ac_pend == 1)).astype(int)                       # transfer con coordinación aún abierta
    out["ec_before_transfer"] = (ec & (ac_tr == 1) & (due.fillna(0) != 1)).astype(int)  # transfer aún no debido: toca coordinar

    # 4) Holgura del snapshot: ¿hay algo más urgente que impida coordinar?
    assume = t == 4
    out["cx_min_tts_assume"] = _snap(df, abs_tts, assume, "min")
    out["cx_min_tts_transfer"] = _snap(df, abs_tts, tr, "min")
    out["cx_n_pending_ec"] = pend.astype(int).groupby(df.sample_id).transform("sum")
    out["ec_vs_assume_gap"] = abs_tts.where(pend) - out["cx_min_tts_assume"]            # cuánto "después" que el assume más cercano
    return pd.DataFrame(out, index=df.index)


def fe_ruta(df):
    t = df.task_index
    rtr, nc = t == RTR, t == NC
    dev_h = _c(df, "non_conformance_agreed_route_deviation_horizontally").abs()
    dev_fl = _c(df, "non_conformance_agreed_route_deviation_fl").abs()
    age = _c(df, "x_task_age")
    out = {}
    out["rr_dev_h"] = dev_h.where(rtr)
    out["rr_dev_fl"] = dev_fl.where(rtr)
    out["rr_rank_dev"] = _rank_in(df, dev_h, rtr, False)          # 1 = la que más se desvía
    out["rr_dev_over_max"] = dev_h.where(rtr) / _snap(df, dev_h, rtr, "max").replace(0, np.nan)
    out["rr_rank_age"] = _rank_in(df, age, rtr, False)
    out["rr_has_nc"] = (rtr & (_ac(df, nc) == 1)).astype(int)     # el mismo avión ya tiene non_conformance abierta
    out["rr_n"] = rtr.astype(int).groupby(df.sample_id).transform("sum")
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- diagnóstico (sin modelo)
def _auc(y, x):
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 < 20 or n0 < 20:
        return np.nan
    r = rankdata(x)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def diag_tipo(d, tipo_idx, cols, folds=range(5)):
    """
    ¿Separa cada columna (a) CUÁNDO se elige una tarea de este tipo y (b) CUÁL, sin entrenar nada?
      auc_f  : AUC por fold sobre las filas de ese tipo (elegida vs no elegida). 0.5 = nada.
      sig    : nº de folds donde el AUC se aleja >0.03 de 0.5 en el MISMO sentido que la media (máx 5).
    Una columna es fiable si `sig` = 5. Si ninguna lo es, la señal no está en el snapshot.
    """
    x = d[d.task_index == tipo_idx]
    rows = []
    for c in cols:
        if c not in x.columns:
            continue
        a = [_auc(g.target_index.values.astype(int), g[c].astype(float).values) for _, g in x.groupby("fold")]
        a = np.array(a, float)
        m = np.nanmean(a)
        sg = np.sign(m - 0.5)
        rows.append(dict(col=c, auc=round(m, 3), min_fold=round(np.nanmin(a), 3), max_fold=round(np.nanmax(a), 3),
                         sig=int(np.sum(sg * (a - 0.5) > 0.03)), pct_nan=round(x[c].isna().mean(), 2)))
    r = pd.DataFrame(rows)
    return r.reindex(r.auc.sub(0.5).abs().sort_values(ascending=False).index).reset_index(drop=True)
"""
extra_sty.py · bloque sty_*: ESTILO del controlador estimado con sus propios inputs ANTERIORES en la sesión (causal).
Un input se reconstruye como en futuro/features_future.add_future pero solo hacia atrás: ev = t - a_since_flight (instante
exacto), su tipo por a_last_*, y se conoce desde el primer snapshot que lo ve (t_reg <= t). Solo inputs del propio
controlador: assume (assume / force assume con is_assumed = 1 después) y órdenes de nivel / laterales a aviones asumidos.
Para cada snapshot, con los inputs con t_reg <= t (medianas acumuladas de la sesión):
  sty_ass_tts    time_to_sector en el momento de aceptar (≈ el del primer snapshot que lo ve + el retraso)
  sty_ass_wait   segundos que dejó esperar la oferta (oferta vista antes: t_vista - x_state_age) antes de aceptar
  sty_lvl_after  segundos entre aceptar un avión y su primera orden de nivel
  sty_lat_after  ídem con la primera orden lateral
y por fila la distancia a ese estilo (sty_*_diff). Nada de etiquetas ni de snapshots posteriores.
"""
import numpy as np, pandas as pd

ACTS = ["a_last_assume_control", "a_last_force_assume_control", "a_last_assign_level_clearance",
        "a_last_assign_lateral_clearance", "a_last_initiate_transfer_of_control"]
KEY = ["session", "cs", "t"]


def _events(d):
    a = d.drop_duplicates(KEY)[KEY + ["a_since_flight", "time_to_sector", "x_state_age", "x_transfer_in", "is_assumed"]
                               + ACTS].sort_values(KEY).reset_index(drop=True)
    a["ev"] = np.where(a.a_since_flight < 3600, np.round(a.t - a.a_since_flight), np.nan)
    acts = a[ACTS].values
    a["act"] = np.where(acts.max(1) == 1, acts.argmax(1), len(ACTS))
    # instante de la oferta, visto en algún snapshot anterior o actual (t - x_state_age con x_transfer_in = 1)
    off = a[a.x_transfer_in == 1].assign(t_off=lambda x: x.t - x.x_state_age)
    off = off.groupby(["session", "cs"]).agg(t_off=("t_off", "min"), t_off_seen=("t", "min")).reset_index()
    e = a.dropna(subset=["ev"]).drop_duplicates(["session", "cs", "ev"], keep="first").copy()
    e = e.merge(off, on=["session", "cs"], how="left")
    ass = e.act.isin([0, 1]) & (e.is_assumed == 1)
    e["ass_tts"] = (e.time_to_sector + (e.t - e.ev)).where(ass)
    e["ass_wait"] = (e.ev - e.t_off).where(ass & (e.t_off_seen <= e.t) & (e.t_off <= e.ev))
    ass_ev = e[ass].groupby(["session", "cs"]).ev.min().rename("ass_ev")
    e = e.merge(ass_ev, on=["session", "cs"], how="left")
    for k, act in (("lvl_after", 2), ("lat_after", 3)):
        m = (e.act == act) & (e.ev > e.ass_ev)
        first = e[m].groupby(["session", "cs"]).ev.transform("min")
        e[k] = np.nan
        e.loc[m, k] = (e.loc[m, "ev"] - e.loc[m, "ass_ev"]).where(e.loc[m, "ev"] == first)
    return e.rename(columns={"t": "t_reg"})[["session", "t_reg", "ass_tts", "ass_wait", "lvl_after", "lat_after"]]


def add(d):
    d = d.copy()
    e = _events(d).sort_values(["session", "t_reg"]).reset_index(drop=True)
    cols = ["ass_tts", "ass_wait", "lvl_after", "lat_after"]
    st = e[["session", "t_reg"]].copy()
    for c in cols:
        st[f"sty_{c}"] = e.groupby("session")[c].transform(lambda s: s.expanding().median())
        st[f"sty_n_{c}"] = e[c].notna().groupby(e.session).cumsum()
    st = st.drop_duplicates(["session", "t_reg"], keep="last")
    snap = d.drop_duplicates("sample_id")[["sample_id", "session", "t"]].sort_values("t")
    m = pd.merge_asof(snap, st.sort_values("t_reg"), left_on="t", right_on="t_reg", by="session",
                      direction="backward").drop(columns=["session", "t", "t_reg"])
    d = d.merge(m, on="sample_id", how="left")
    out = [f"sty_{c}" for c in cols] + [f"sty_n_{c}" for c in cols]
    d["sty_ass_tts_diff"] = d.time_to_sector - d.sty_ass_tts
    d["sty_ass_wait_diff"] = (d.x_state_age - d.sty_ass_wait).where(d.x_transfer_in == 1)
    d["sty_lvl_diff"] = (d.x_state_age - d.sty_lvl_after).where(d.is_assumed == 1)
    d["sty_lat_diff"] = (d.x_state_age - d.sty_lat_after).where(d.is_assumed == 1)
    out += ["sty_ass_tts_diff", "sty_ass_wait_diff", "sty_lvl_diff", "sty_lat_diff"]
    d[out] = d[out].astype("float32")
    return d, out

"""features_futuro.py · línea "futuro" · features del modelo N (usan OTROS snapshots de la sesión; ninguna usa etiquetas).

Bloques (fs = dict bloque -> columnas):
    base    columnas originales del CSV (sin identificadores ni tiempo)
    logic   logic_*: reglas físicas / de decisión dentro del snapshot (deadline por tipo, factibilidad, ventanas, EDF)
    past    p_* + blk_pos, blk_len, gap_prev, gap_next: pasado de la sesión (tarea nueva, deltas, posición en la racha)
    future  f_*: futuro de la sesión (siguiente instrucción al avión y su tipo, cuándo desaparece la tarea / el avión)
    clear   sx_*: segundos hasta el siguiente cambio de nivel, rumbo, nivel de salida, zona, conflicto o ruta del avión
    instr   si_*: segundos hasta / desde la siguiente / última instrucción de cada tipo y la que ENCAJA con la tarea

Reproduce exactamente futuro/features_future.py (load, add_folds, add_blocks, add_logic, add_future, add_past),
futuro/feat_clear.py, futuro/feat_coord.py (next_change, add_ranks) y futuro/pruebas/f10_instr.py (add_instr), en el
mismo orden de filas y columnas que futuro/pruebas/cache.parquet + cache_instr.parquet.

Uso:
    from features_futuro import construir
    lab, te, fs = construir()        # lab = train+validation (con fold), te = test (fold = -1); orden sample_id, row_index
    F = [c for b in ["base", "logic", "past", "future", "clear", "instr"] for c in fs[b]]
La primera vez calcula todo (~1–3 min) y lo guarda en entrega/cache/features_futuro.parquet.
    python entrega/codigo/features_futuro.py          # calcula (o lee) la caché e imprime el tamaño de cada bloque
"""
import time, warnings
import numpy as np, pandas as pd
from comun import DATOS, CACHE, TIPOS

TYPES = TIPOS
ENTRY_TYPES, EXIT_TYPES = [0, 2, 4, 9], [1, 3, 5, 10]
NEEDS_CONTROL = [1, 3, 6, 7, 8, 10, 12, 13]          # requieren dar instrucciones -> avión asumido
ACTS = ["assume_control", "force_assume_control", "assign_level_clearance", "assign_lateral_clearance",
        "initiate_transfer_of_control"]
ACT_N = ["assume", "force", "nivel", "lateral", "transfer", "otro"]     # nombres cortos (ACTS + "otro") del bloque instr
# instrucciones (índice en ACT_N) que "encajan" con cada tipo de tarea (compatibilidad lógica, no aprendida)
COMPAT = {4: [0, 1], 2: [0, 1], 5: [4], 12: [3], 13: [3], 10: [2], 3: [2, 3], 6: [2, 3], 1: [2, 3], 0: [0, 1, 2, 3],
          7: [2, 3], 8: [2, 3]}
META = ['sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index',
        'cs', 't', 'split', 'fold', 'block']
PAST_EXTRA = ["blk_pos", "blk_len", "gap_prev", "gap_next"]
BLOQUES = ["base", "logic", "past", "future", "clear", "instr"]
CACHE_FEATS = CACHE / "features_futuro.parquet"
KEY = ["session", "cs", "t"]


# ----------------------------------------------------------------------------- datos
def cargar():
    """train + validation + test en una tabla: split, cs (indicativo del avión), t (segundos) y fold (test -> -1).
    Ordenada por session, t, row_index."""
    tr = pd.read_csv(DATOS / "train.csv").assign(split="train")
    va = pd.read_csv(DATOS / "validation.csv").assign(split="validation")
    te = pd.read_csv(DATOS / "test.csv").assign(split="test", target_index=-1)
    d = pd.concat([tr, va, te], ignore_index=True)
    d["cs"] = [n[len(TYPES[t]) + 1:].split("_")[0] for n, t in zip(d.task_name, d.task_index)]
    d["t"] = pd.to_timedelta(d.timestamp).dt.total_seconds()
    d = d.sort_values(["session", "t", "row_index"]).reset_index(drop=True)
    folds = pd.read_csv(DATOS / "folds.csv")
    d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold))).fillna(-1).astype(int)
    d.loc[d.split == "test", "fold"] = -1
    return d


# ----------------------------------------------------------------------------- bloques
def add_blocks(d):
    """Racha (block) = snapshots contiguos (hueco <= 4.5 s) de una sesión; posición y longitud en la racha y huecos."""
    st = d.drop_duplicates("sample_id")[["sample_id", "session", "t"]].sort_values(["session", "t"])
    gap = st.groupby("session").t.diff()
    st["block"] = (gap.isna() | (gap > 4.5)).cumsum()
    st["blk_pos"] = st.groupby("block").cumcount()
    st["blk_len"] = st.groupby("block").t.transform("size")
    st["gap_prev"] = gap.fillna(999).clip(upper=999)
    st["gap_next"] = st.groupby("session").t.diff(-1).abs().fillna(999).clip(upper=999)
    return d.merge(st.drop(columns=["session", "t"]), on="sample_id")


def add_logic(d):
    """logic_*: reglas físicas / de decisión, interpretables (solo filas del mismo snapshot)."""
    d = d.copy()
    tts, ty = d.time_to_sector, d.task_index
    gs = d.k_groundSpeed.where(d.k_groundSpeed > 50)
    t_zone = d.distance_to_zone * 1852 / gs
    # deadline: segundos que quedan para el momento que define la tarea
    dl = np.select(
        [ty.isin(ENTRY_TYPES), ty.isin(EXIT_TYPES), ty == 12, ty == 6],
        [tts, -tts, t_zone, d.time_to_MD], default=-tts)
    d["logic_deadline"] = dl
    d["logic_deadline_abs"] = np.abs(dl)
    # factibilidad: no se dan instrucciones a un avión que no controlas; entry_coordination solo con su assume
    d["logic_feasible"] = (~(ty.isin(NEEDS_CONTROL) & (d.is_assumed == 0)) & ~((ty == 2) & (d.l_twin_AE == 0))).astype(int)
    # ventanas típicas (del EDA): dentro de la ventana -> 1
    win = {4: (-60, 120), 5: (-60, 60), 2: (0, 400), 12: (120, 480), 6: (60, 240), 1: (-300, 0), 3: (-300, 0),
           10: (-600, 0), 13: (-900, 0), 0: (0, 300)}
    lo = ty.map({k: v[0] for k, v in win.items()}); hi = ty.map({k: v[1] for k, v in win.items()})
    # la ventana se evalúa sobre el deadline en el sentido del tipo (salida: time_to_sector negativo)
    x = np.where(ty.isin(EXIT_TYPES + [13]), tts, dl)
    d["logic_in_window"] = ((x >= lo) & (x <= hi)).astype(float).where(lo.notna())
    d["logic_window_dist"] = np.maximum(lo - x, x - hi).clip(lower=0).where(lo.notna())
    # Earliest-deadline-first entre tareas factibles con deadline >= -60 s
    edf = d.logic_deadline.where((d.logic_feasible == 1) & (d.logic_deadline >= -60))
    d["logic_edf_rank"] = edf.groupby(d.sample_id).rank(method="min")
    d["logic_edf_rank_type"] = edf.groupby([d.sample_id, d.task_index]).rank(method="min")
    # urgencias concretas
    d["logic_offered"] = ((ty == 4) & (d.x_transfer_in == 1)).astype(int)                      # assume ofrecido y esperando
    d["logic_due_transfer"] = ((ty == 5) & (d.k_aircraftNeedsToBeTransferred == 1)).astype(int)
    d["logic_stca"] = ((ty.isin([0, 1, 6])) & (d.HMI_conflict_safety_alert == 1)).astype(int)
    d["logic_twin_assume"] = ((d.l_twin_AE == 1) & (ty == 4)).astype(int)
    d["logic_t_zone"] = t_zone
    # nº de tareas "en ventana" que compiten en el snapshot
    d["logic_n_window"] = d.logic_in_window.fillna(0).groupby(d.sample_id).transform("sum")
    d["logic_n_window_type"] = d.logic_in_window.fillna(0).groupby([d.sample_id, d.task_index]).transform("sum")
    return d


def add_future(d):
    """f_*: señales de snapshots posteriores de la misma sesión (siguiente input al avión, desaparición de la tarea)."""
    d = d.copy()
    d["ev"] = np.where(d.a_since_flight < 3600, np.round(d.t - d.a_since_flight), np.nan)
    # tipo de cada input: a_last_* del primer snapshot que lo registra
    k = d.dropna(subset=["ev"]).sort_values("t").drop_duplicates(["session", "cs", "ev"])
    acts = k[["a_last_" + a for a in ACTS]].values
    k = k.assign(act=np.where(acts.max(1) == 1, acts.argmax(1), len(ACTS)))[["session", "cs", "ev", "act"]]
    nxt, nxt2, nact = (np.full(len(d), np.nan) for _ in range(3))
    for s, g in d.groupby("session"):
        ks = k[k.session == s]
        evs = {cs: (gg.ev.values, gg.act.values) for cs, gg in ks.sort_values("ev").groupby("cs")}
        for cs, gi in g.groupby("cs").groups.items():
            if cs not in evs: continue
            e, a = evs[cs]
            t = d.t.values[gi]
            j = np.searchsorted(e, t + 1.5)
            ok = j < len(e); ok2 = j + 1 < len(e)
            idx = np.asarray(gi)
            nxt[idx[ok]] = e[j[ok]] - t[ok]
            nact[idx[ok]] = a[j[ok]]
            nxt2[idx[ok2]] = e[j[ok2] + 1] - t[ok2]
    d["f_next_input"], d["f_next_input2"], d["f_next_act"] = nxt, nxt2, nact
    d["f_next_input_rk"] = d.groupby("sample_id").f_next_input.rank(method="min")
    # ¿el siguiente input "encaja" con el tipo de tarea?
    d["f_next_match"] = [float(a in COMPAT.get(t, [])) if a == a else np.nan for t, a in zip(d.task_index, d.f_next_act)]
    d["f_next_match_soon"] = d.f_next_match * (d.f_next_input <= 60)
    # desaparición de la tarea / avión de la lista
    d["f_task_end"] = d.groupby(["session", "task_name"]).t.transform("max") - d.t
    d["f_cs_end"] = d.groupby(["session", "cs"]).t.transform("max") - d.t
    d["f_task_end_rk"] = d.groupby("sample_id").f_task_end.rank(method="min")
    d["f_task_end_rk_flight"] = d.groupby(["sample_id", "cs"]).f_task_end.rank(method="min")
    # ¿sigue en el siguiente snapshot contiguo? ¿termina la tarea dentro de esta racha?
    d["f_gone_next"] = ((d.f_task_end < d.gap_next - 0.5) & (d.gap_next <= 4.5)).astype(int)
    d["f_end_in_block"] = (d.f_task_end <= (d.blk_len - 1 - d.blk_pos) * 3.5).astype(int)
    return d.drop(columns="ev")


def add_past(d):
    """p_*: pasado de la sesión (desde cuándo existe la tarea, deltas respecto al snapshot anterior)."""
    d = d.copy()
    d["p_task_start"] = d.t - d.groupby(["session", "task_name"]).t.transform("min")
    d["p_task_start_rk"] = d.groupby("sample_id").p_task_start.rank(method="min")
    d["p_new_in_block"] = (d.p_task_start <= d.blk_pos * 3.5).astype(int)
    for c in ["time_to_sector", "x_task_age", "a_since_flight"]:
        d["p_d_" + c] = d[c] - d.groupby(["session", "task_name"])[c].shift()
    d["p_since_any_rk"] = d.groupby("sample_id").a_since_flight.rank(method="min")
    return d


def next_change(d, cols, kind="any"):
    """Segundos hasta el siguiente snapshot (estrictamente posterior) de la sesión en que cambia alguna de `cols` para el
    mismo avión. kind: 'any' = cualquier cambio, 'up' = sube (0->1, NaN->valor), 'down' = baja (1->0).
    Devuelve un array alineado con las filas de d (NaN = no vuelve a cambiar)."""
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
    """Rangos de `col` en el snapshot, entre las tareas del mismo tipo y entre las del mismo avión (en el sitio)."""
    d[col + "_rk"] = d.groupby("sample_id")[col].rank(method="min")
    d[col + "_rk_type"] = d.groupby(["sample_id", "task_index"])[col].rank(method="min")
    d[col + "_rk_flight"] = d.groupby(["sample_id", "cs"])[col].rank(method="min")
    return d


def add_clear(d):
    """sx_*: cambios FUTUROS del estado del avión ligados a instrucciones (nivel, rumbo, nivel de salida, zona, conflicto,
    ruta) y sx_task = el cambio que resuelve el tipo de la tarea, con sus rangos."""
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


def add_instr(d):
    """si_* (DataFrame aparte, alineado con d): con la lista de inputs de cada avión en la sesión (a_since_flight + a_last_*)
        si_next_<act> / si_prev_<act>  s hasta la siguiente / desde la última instrucción de cada tipo (ACT_N)
        si_match / si_prevmatch        lo mismo para la instrucción COMPATIBLE con la tarea (COMPAT), aunque no sea la primera
        si_n60, si_n180                nº de instrucciones al avión en los próximos 60 / 180 s
        si_match_rk / si_match_rk_flight  rango de si_match en el snapshot / entre las tareas del mismo avión"""
    ev = np.where(d.a_since_flight < 3600, np.round(d.t - d.a_since_flight), np.nan)
    k = d.assign(ev=ev).dropna(subset=["ev"]).sort_values("t").drop_duplicates(["session", "cs", "ev"])
    acts = k[["a_last_" + a for a in ACTS]].values
    k = k.assign(act=np.where(acts.max(1) == 1, acts.argmax(1), len(ACTS)))[["session", "cs", "ev", "act"]]
    n = len(d)
    nxt = np.full((n, len(ACT_N)), np.nan); prv = np.full((n, len(ACT_N)), np.nan)
    n60, n180 = np.zeros(n), np.zeros(n)
    t_all = d.t.values
    ev_g = {key: g for key, g in k.sort_values("ev").groupby(["session", "cs"])}
    for key, idx in d.groupby(["session", "cs"]).indices.items():
        if key not in ev_g:
            continue
        g = ev_g[key]; e, a = g.ev.values, g.act.values
        t = t_all[idx]
        lo = np.searchsorted(e, t + 1.5)
        n60[idx] = np.searchsorted(e, t + 60, side="right") - lo
        n180[idx] = np.searchsorted(e, t + 180, side="right") - lo
        for j in range(len(ACT_N)):
            ej = e[a == j]
            if not len(ej):
                continue
            p = np.searchsorted(ej, t + 1.5); ok = p < len(ej)
            nxt[idx[ok], j] = ej[p[ok]] - t[ok]
            q = np.searchsorted(ej, t + 1.5) - 1; ok = q >= 0
            prv[idx[ok], j] = t[ok] - ej[q[ok]]
    out = pd.DataFrame(index=d.index)
    for j, nm in enumerate(ACT_N):
        out["si_next_" + nm], out["si_prev_" + nm] = nxt[:, j], prv[:, j]
    ti = d.task_index.values
    match, pmatch = np.full(n, np.nan), np.full(n, np.nan)
    for typ, comp in COMPAT.items():
        m = ti == typ
        with warnings.catch_warnings():          # filas sin ninguna instrucción compatible -> NaN (sin aviso)
            warnings.simplefilter("ignore", RuntimeWarning)
            match[m] = np.nanmin(nxt[m][:, comp], axis=1) if m.any() else np.nan
            pmatch[m] = np.nanmin(prv[m][:, comp], axis=1) if m.any() else np.nan
    out["si_match"], out["si_prevmatch"], out["si_n60"], out["si_n180"] = match, pmatch, n60, n180
    out["si_match_rk"] = out.si_match.groupby(d.sample_id.values).rank(method="min")
    out["si_match_rk_flight"] = out.si_match.groupby([d.sample_id.values, d.cs.values]).rank(method="min")
    return out


# ----------------------------------------------------------------------------- todo junto
def feature_sets(d):
    """Bloques -> columnas, en el orden de las columnas de d (el orden importa: LightGBM muestrea columnas por semilla)."""
    pref = ("logic_", "f_", "p_", "sx_", "si_")
    return dict(base=[c for c in d.columns if c not in META and not c.startswith(pref) and c not in PAST_EXTRA],
                logic=[c for c in d.columns if c.startswith("logic_")],
                past=[c for c in d.columns if c.startswith("p_")] + PAST_EXTRA,
                future=[c for c in d.columns if c.startswith("f_")],
                clear=[c for c in d.columns if c.startswith("sx_")],
                instr=[c for c in d.columns if c.startswith("si_")])


def construir(forzar=False):
    """(lab, te, fs). lab = train+validation, te = test (fold = -1), cada uno ordenado por sample_id, row_index.
    Cacheado en entrega/cache/features_futuro.parquet (filas de lab y luego las de te)."""
    if CACHE_FEATS.exists() and not forzar:
        D = pd.read_parquet(CACHE_FEATS)
    else:
        t0 = time.time()
        d = cargar()
        d = add_blocks(d)
        d = add_logic(d)
        d = add_future(d)
        d = add_past(d)
        d = add_clear(d)
        d = d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
        D = pd.concat([d[d.split != "test"], d[d.split == "test"]], ignore_index=True)
        D = pd.concat([D, add_instr(D)], axis=1)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FEATS.with_suffix(f".{time.time_ns()}.tmp")      # escritura atómica
        D.to_parquet(tmp); tmp.rename(CACHE_FEATS)
        print(f"features de futuro calculadas en {time.time() - t0:.0f}s -> {CACHE_FEATS}", flush=True)
    fs = feature_sets(D)
    lab = D[D.split != "test"].reset_index(drop=True)
    te = D[D.split == "test"].reset_index(drop=True).assign(fold=-1)
    assert (lab.groupby("sample_id").target_index.sum() == 1).all(), "≠ 1 etiqueta por snapshot"
    assert not set(sum(fs.values(), [])) & set(META), "columna prohibida entre las features"
    return lab, te, fs


if __name__ == "__main__":
    lab, te, fs = construir()
    print(f"lab {lab.shape}, te {te.shape}; bloques: " + ", ".join(f"{k}={len(v)}" for k, v in fs.items()))

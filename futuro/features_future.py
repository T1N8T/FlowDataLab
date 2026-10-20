"""
features_future.py · RAMA "futuro" · Features que usan OTROS snapshots de la sesión (rompen la regla 1 a propósito).
Es la versión antigua de src/features.py (antes de la limpieza del equipo), con los folds de Data/folds.csv.
La usan futuro/m_nn_future.py, futuro/m_lgb_future.py y futuro/temporal.py. No usar en los modelos de main.

Bloques de features:
- base: columnas originales
- logic_*: reglas físicas / de decisión (deadline por tipo, factibilidad, EDF)
- f_*: información futura de la sesión (siguiente input, tipo de ese input, desaparición de la tarea)
- p_*: información pasada de la sesión (tarea nueva, deltas respecto al snapshot anterior)
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "Data", ROOT / "models"
TYPES = ['entry_conflict_resolution', 'exit_conflict_resolution', 'entry_coordination', 'exit_coordination',
         'assume_aircraft', 'transfer_aircraft', 'conflict_resolution', 'non_conformance_resolution',
         'qos_improvement', 'entry_conditions', 'exit_conditions', 'aircraft_request', 'zone_conflict',
         'return_to_route']
ENTRY_TYPES, EXIT_TYPES = [0, 2, 4, 9], [1, 3, 5, 10]
NEEDS_CONTROL = [1, 3, 6, 7, 8, 10, 12, 13]          # requieren dar instrucciones -> avión asumido
ACTS = ["assume_control", "force_assume_control", "assign_level_clearance", "assign_lateral_clearance",
        "initiate_transfer_of_control"]
# Folds fijos por controlador (020 tiene validación -> va a un fold como cualquier otro)
N_FOLDS = 5
META = ['sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index',
        'cs', 't', 'split', 'fold', 'block']


def load():
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    va = pd.read_csv(DATA / "validation.csv").assign(split="validation")
    te = pd.read_csv(DATA / "test.csv").assign(split="test", target_index=-1)
    d = pd.concat([tr, va, te], ignore_index=True)
    d["cs"] = [n[len(TYPES[t]) + 1:].split("_")[0] for n, t in zip(d.task_name, d.task_index)]
    d["t"] = pd.to_timedelta(d.timestamp).dt.total_seconds()
    return d.sort_values(["session", "t", "row_index"]).reset_index(drop=True)


def add_folds(d):
    """Folds comunes del equipo (Data/folds.csv). Test -> -1."""
    folds = pd.read_csv(DATA / "folds.csv")
    d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold))).fillna(-1).astype(int)
    d.loc[d.split == "test", "fold"] = -1
    return d


def add_blocks(d):
    """Bloque = racha de snapshots contiguos (hueco <= 4.5 s) de una sesión."""
    st = d.drop_duplicates("sample_id")[["sample_id", "session", "t"]].sort_values(["session", "t"])
    gap = st.groupby("session").t.diff()
    st["block"] = (gap.isna() | (gap > 4.5)).cumsum()
    st["blk_pos"] = st.groupby("block").cumcount()
    st["blk_len"] = st.groupby("block").t.transform("size")
    st["gap_prev"] = gap.fillna(999).clip(upper=999)
    st["gap_next"] = st.groupby("session").t.diff(-1).abs().fillna(999).clip(upper=999)
    return d.merge(st.drop(columns=["session", "t"]), on="sample_id")


def add_logic(d):
    """Reglas físicas/de decisión, interpretables."""
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
    # ventanas típicas (de EDA): dentro de la ventana -> 1
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
    # nº de tareas factibles y "en ventana" que compiten en el snapshot
    d["logic_n_window"] = d.logic_in_window.fillna(0).groupby(d.sample_id).transform("sum")
    d["logic_n_window_type"] = d.logic_in_window.fillna(0).groupby([d.sample_id, d.task_index]).transform("sum")
    return d


def add_future(d):
    """Señales de snapshots posteriores de la misma sesión (no usan etiquetas)."""
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
    # ¿el siguiente input "encaja" con el tipo de tarea? (compatibilidad lógica, no aprendida)
    compat = {4: [0, 1], 2: [0, 1], 5: [4], 12: [3], 13: [3], 10: [2], 3: [2, 3], 6: [2, 3], 1: [2, 3], 0: [0, 1, 2, 3],
              7: [2, 3], 8: [2, 3]}
    d["f_next_match"] = [float(a in compat.get(t, [])) if a == a else np.nan for t, a in zip(d.task_index, d.f_next_act)]
    d["f_next_match_soon"] = d.f_next_match * (d.f_next_input <= 60)
    # desaparición de la tarea / avión de la lista
    d["f_task_end"] = d.groupby(["session", "task_name"]).t.transform("max") - d.t
    d["f_cs_end"] = d.groupby(["session", "cs"]).t.transform("max") - d.t
    d["f_task_end_rk"] = d.groupby("sample_id").f_task_end.rank(method="min")
    d["f_task_end_rk_flight"] = d.groupby(["sample_id", "cs"]).f_task_end.rank(method="min")
    # ¿sigue en el siguiente snapshot contiguo?
    d["f_gone_next"] = ((d.f_task_end < d.gap_next - 0.5) & (d.gap_next <= 4.5)).astype(int)
    # ¿termina la tarea dentro de este bloque contiguo (episodio)?
    d["f_end_in_block"] = (d.f_task_end <= (d.blk_len - 1 - d.blk_pos) * 3.5).astype(int)
    return d.drop(columns="ev")


def add_past(d):
    d = d.copy()
    d["p_task_start"] = d.t - d.groupby(["session", "task_name"]).t.transform("min")
    d["p_task_start_rk"] = d.groupby("sample_id").p_task_start.rank(method="min")
    d["p_new_in_block"] = (d.p_task_start <= d.blk_pos * 3.5).astype(int)
    for c in ["time_to_sector", "x_task_age", "a_since_flight"]:
        d["p_d_" + c] = d[c] - d.groupby(["session", "task_name"])[c].shift()
    d["p_since_any_rk"] = d.groupby("sample_id").a_since_flight.rank(method="min")
    return d


def build():
    d = load()
    d = add_folds(d)
    d = add_blocks(d)
    d = add_logic(d)
    d = add_future(d)
    d = add_past(d)
    return d


def feature_sets(d):
    base = [c for c in d.columns if c not in META and not c.startswith(("logic_", "f_", "p_"))
            and c not in ("blk_pos", "blk_len", "gap_prev", "gap_next")]
    logic = [c for c in d.columns if c.startswith("logic_")]
    fut = [c for c in d.columns if c.startswith("f_")]
    past = [c for c in d.columns if c.startswith("p_")] + ["blk_pos", "blk_len", "gap_prev", "gap_next"]
    return dict(base=base, logic=logic, future=fut, past=past)


if __name__ == "__main__":
    d = build()
    print(d.shape, {k: len(v) for k, v in feature_sets(d).items()})

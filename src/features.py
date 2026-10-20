"""
features.py · común (dueño: Adri, B) · Feature engineering compartido.
Uso:  python src/features.py   -> escribe models/features.parquet (train+validation+test, con fold de Data/folds.csv)
      from features import load_feats   -> lo carga ordenado por sample_id, row_index (el orden de oof/ y preds/)

Bloques de features:
- base: columnas originales
- logic_*: reglas físicas / de decisión (deadline por tipo, factibilidad, EDF)
Regla 1: todo se calcula solo con filas del mismo sample_id. Las features con snapshots pasados/futuros (f_*, p_*)
se quitaron; si se permiten, están en el commit 67a15bc (add_blocks, add_future, add_past).
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
META = ['sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index',
        'cs', 'split', 'fold']


def load():
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    va = pd.read_csv(DATA / "validation.csv").assign(split="validation")
    te = pd.read_csv(DATA / "test.csv").assign(split="test", target_index=-1)
    d = pd.concat([tr, va, te], ignore_index=True)
    d["cs"] = [n[len(TYPES[t]) + 1:].split("_")[0] for n, t in zip(d.task_name, d.task_index)]
    return d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def add_folds(d):
    """Folds comunes del equipo (Data/folds.csv, python src/folds.py). Test -> -1."""
    folds = pd.read_csv(DATA / "folds.csv")
    d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold))).fillna(-1).astype(int)
    assert (d.loc[d.split != "test", "fold"] >= 0).all(), "hay controladores etiquetados sin fold en Data/folds.csv"
    d.loc[d.split == "test", "fold"] = -1
    return d




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






def build():
    d = load()
    d = add_folds(d)
    d = add_logic(d)
    return d


def load_feats():
    """models/features.parquet ordenado por sample_id, row_index. Comprueba que sus folds son los de Data/folds.csv."""
    d = pd.read_parquet(MODELS / "features.parquet")
    folds = pd.read_csv(DATA / "folds.csv")
    lab = d.split != "test"
    assert (d.loc[lab, "fold"].values == d.loc[lab, "atco"].map(dict(zip(folds.atco, folds.fold))).values).all(), \
        "features.parquet tiene otros folds: regenerar con python src/features.py"
    return d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def feature_sets(d):
    base = [c for c in d.columns if c not in META and not c.startswith("logic_")]
    logic = [c for c in d.columns if c.startswith("logic_")]
    return dict(base=base, logic=logic)


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    d = build()
    d.to_parquet(MODELS / "features.parquet")
    print(d.shape, {k: len(v) for k, v in feature_sets(d).items()})

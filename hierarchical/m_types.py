"""
m_types.py · Juanma (C) · Especialistas por tipo: score(tarea) = P(tipo | snapshot) × P(tarea | tipo, snapshot).

- Etapa 1: LightGBM multiclase sobre una fila por snapshot con features agregadas de la lista por tipo
  (cuántas tareas de cada tipo, flags de urgencia, mínimos de edad/tiempo...). Los tipos ausentes en el snapshot
  se anulan y se renormaliza.
- Etapa 2: LightGBM binario sobre las tareas del tipo correcto de cada snapshot (¿cuál de ellas es?), con features de
  la fila + rankings dentro del (snapshot, tipo). Se normaliza dentro de cada (snapshot, tipo).
- Regla 1: todas las features usan solo filas del mismo sample_id. task_name solo se usa para agrupar por avión.
- CV con Data/folds.csv (GroupKFold por atco). Test: entrenado con train+validation.

Uso:  python hierarchical/m_types.py            # CV + salidas oof/types.csv, preds/types.csv, submissions/sub_types.csv
      python hierarchical/m_types.py --quick    # solo CV, sin entrenar para test ni guardar
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cv"))
from metric import report, save_outputs

DATA = ROOT / "Data"
NAME = "types"
N_TYPES = 14
NOT_FEATS = ["sample_id", "target_index", "session", "atco", "scenario", "timestamp", "task_name", "row_index",
             "split", "fold", "cs"]
# columnas que se rankean dentro del (snapshot, tipo): menor valor = rank 1
RANK_COLS = ["x_task_age", "x_state_age", "a_since_flight", "abs_tts", "k_timeToExitPoint", "distance_to_zone",
             "time_to_MD", "distance_to_MD"]
TYPES = ['entry_conflict_resolution', 'exit_conflict_resolution', 'entry_coordination', 'exit_coordination',
         'assume_aircraft', 'transfer_aircraft', 'conflict_resolution', 'non_conformance_resolution',
         'qos_improvement', 'entry_conditions', 'exit_conditions', 'aircraft_request', 'zone_conflict',
         'return_to_route']


# ----------------------------------------------------------------------------- datos y features (dentro del snapshot)
def load():
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    va = pd.read_csv(DATA / "validation.csv").assign(split="validation")
    te = pd.read_csv(DATA / "test.csv").assign(split="test", target_index=0)
    d = pd.concat([tr, va, te], ignore_index=True).sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    folds = pd.read_csv(DATA / "folds.csv")
    d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold))).fillna(-1).astype(int)
    d.loc[d.split == "test", "fold"] = -1
    d["cs"] = [n[len(TYPES[t]) + 1:].split("_")[0] for n, t in zip(d.task_name, d.task_index)]   # avión, solo para agrupar
    num = d.columns.difference(NOT_FEATS + ["task_index"])
    d[num] = d[num].astype(np.float32)
    return d


def add_row_feats(d):
    d["abs_tts"] = d.time_to_sector.abs()
    n = {}
    g = d.groupby(["sample_id", "task_index"])
    for c in RANK_COLS:
        n[f"rt_{c}"] = g[c].rank(method="min")
        n[f"dt_{c}"] = d[c] - g[c].transform("min")
    n["rt_n"] = g.row_index.transform("size")
    ga = d.groupby(["sample_id", "cs"])
    n["n_tasks_flight"] = ga.row_index.transform("size")
    n["oldest_in_flight"] = d.x_task_age == ga.x_task_age.transform("max")
    n["rank_since_flight_all"] = d.groupby("sample_id").a_since_flight.rank(method="min")
    # tipo × estado
    ty = d.task_index
    n["assume_offered"] = (ty == 4) & (d.x_transfer_in == 1)
    n["transfer_due"] = (ty == 5) & (d.k_aircraftNeedsToBeTransferred == 1)
    n["entry_coord_pending"] = (ty == 2) & (d.x_isc_coordinated != 1)
    return pd.concat([d, pd.DataFrame(n).astype(np.float32)], axis=1)


def snapshot_feats(d):
    """Una fila por snapshot: agregados por tipo (la etapa 1 decide el tipo con esto)."""
    agg = d.groupby(["sample_id", "task_index"]).agg(
        n=("row_index", "size"), age_min=("x_task_age", "min"), age_max=("x_task_age", "max"),
        state_age_min=("x_state_age", "min"), since_min=("a_since_flight", "min"), tts_min=("abs_tts", "min"),
        offered=("assume_offered", "sum"), due=("transfer_due", "sum"), last_cs=("a_is_last_cs", "max"),
        last2_cs=("a_is_last2_cs", "max"), assumed=("is_assumed", "sum"), zone_min=("distance_to_zone", "min"),
        md_min=("time_to_MD", "min"), conflict=("conflict_ind", "sum"), stca=("HMI_conflict_safety_alert", "sum"),
        twin=("l_twin_AE", "sum"))
    S = agg.unstack("task_index")
    S.columns = [f"s_{a}_{t}" for a, t in S.columns]
    S = S.reindex(columns=[f"s_{a}_{t}" for a in agg.columns for t in range(N_TYPES)])
    for t in range(N_TYPES):
        S[f"s_n_{t}"] = S[f"s_n_{t}"].fillna(0)
    g = d.groupby("sample_id")
    S["seq_len"] = g.seq_len.first()
    S["n_flights"] = g.cs.nunique()
    S["since_min_all"] = g.a_since_flight.min()
    S["a_n60"] = g.a_n60.max()
    S["a_n300"] = g.a_n300.max()
    meta = g.agg(fold=("fold", "first"), split=("split", "first"))
    tgt = d.loc[d.target_index == 1].set_index("sample_id").task_index
    meta["y"] = tgt.reindex(meta.index).fillna(-1).astype(int)
    present = pd.crosstab(d.sample_id, d.task_index).reindex(columns=range(N_TYPES), fill_value=0).gt(0)
    return S.astype(np.float32), meta, present.reindex(S.index).values


# ----------------------------------------------------------------------------- modelos
def type_model(seed=0):
    return lgb.LGBMClassifier(objective="multiclass", num_class=N_TYPES, n_estimators=300, learning_rate=0.04,
                              num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=2.0, random_state=seed, verbose=-1, n_jobs=12)


def task_model(seed=0):
    return lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.04, num_leaves=31,
                              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                              reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=12)


def fit_predict(d, S, meta, present, F, tr_mask_s, te_mask_s, seed=0):
    """Entrena con los snapshots tr_mask_s y devuelve (score por fila, P(tipo) por fila) para los te_mask_s."""
    # etapa 1: P(tipo)
    m1 = type_model(seed).fit(S[tr_mask_s], meta.y[tr_mask_s])
    P = np.zeros((te_mask_s.sum(), N_TYPES), np.float32)
    P[:, m1.classes_] = m1.predict_proba(S[te_mask_s])
    P = P * present[te_mask_s]
    P /= P.sum(1, keepdims=True)
    # etapa 2: P(tarea | tipo) entre las tareas del tipo correcto
    sid_tr = set(meta.index[tr_mask_s])
    row_tr = d.sample_id.isin(sid_tr).values & (d.task_index.values == d.sample_id.map(meta.y).values)
    m2 = task_model(seed).fit(d.loc[row_tr, F], d.target_index[row_tr], categorical_feature=["task_index"])
    row_te = d.sample_id.isin(set(meta.index[te_mask_s])).values
    q = pd.Series(m2.predict_proba(d.loc[row_te, F])[:, 1], index=d.index[row_te])
    q = q / q.groupby([d.sample_id[row_te], d.task_index[row_te]]).transform("sum")
    pt = pd.DataFrame(P, index=meta.index[te_mask_s])
    p_type = pt.values[pt.index.get_indexer(d.sample_id[row_te]), d.task_index[row_te].values]
    return q.values * p_type, p_type, row_te


def main():
    quick = "--quick" in sys.argv
    d = add_row_feats(load())
    S, meta, present = snapshot_feats(d)
    F = [c for c in d.columns if c not in NOT_FEATS]
    print(f"{len(F)} features por fila, {S.shape[1]} por snapshot", flush=True)

    lab_s = (meta.split != "test").values
    oof = pd.Series(np.nan, index=d.index)
    oof_type = pd.Series(np.nan, index=d.index)
    for f in range(5):
        sc, pty, rows = fit_predict(d, S, meta, present, F, lab_s & (meta.fold != f).values, (meta.fold == f).values)
        oof[rows], oof_type[rows] = sc, pty
        # diagnóstico: ¿acierta el tipo? (top-1 y top-3 de P(tipo))
        x = d[rows].assign(p=pty)
        tt = x[x.target_index == 1]
        print(f"fold {f}: P(tipo correcto) media={tt.p.mean():.3f}", flush=True)

    lab = d[d.split != "test"]
    o = pd.DataFrame({"sample_id": lab.sample_id, "row_index": lab.row_index, "score": oof[lab.index]})
    cv = report(o, NAME)
    # diagnóstico: techo si el tipo fuera perfecto vs. si la tarea dentro del tipo fuera perfecta
    t = lab.assign(pt=oof_type[lab.index])
    y = t[t.target_index == 1].set_index("sample_id")
    rank_type = t.drop_duplicates(["sample_id", "task_index"]).assign(
        r=lambda x: x.groupby("sample_id").pt.rank(ascending=False, method="first"))
    rt = rank_type.set_index(["sample_id", "task_index"]).r
    print(f"  tipo correcto top-1 {np.mean(rt.loc[list(zip(y.index, y.task_index))].values <= 1):.3f}  "
          f"top-3 {np.mean(rt.loc[list(zip(y.index, y.task_index))].values <= 3):.3f}")

    if quick:
        return
    te_s = (meta.split == "test").values
    sc, _, rows = fit_predict(d, S, meta, present, F, lab_s, te_s)
    te = d[rows]
    p = pd.DataFrame({"sample_id": te.sample_id.values, "row_index": te.row_index.values, "score": sc})
    save_outputs(NAME, o, p)
    print(f"{NAME}: CV top3 = {cv:.4f}")


if __name__ == "__main__":
    main()

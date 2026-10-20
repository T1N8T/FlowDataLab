"""
base.py · Juanma (C) · Base común de los hilos de pruebas (solo snapshot, regla 1).

- Datos y features de lgb_adri copiados de notebooks/LGB_adri.ipynb (grupos de models/lgb_adri_config.json).
- Salidas en pruebas/oof|preds/<nombre>.csv (formato común), FUERA de oof/ para no tocar el stacking de main.
- registrar(): calcula CV top-3, peor controlador, desviación entre folds y acierto por tipo, y añade una línea a
  pruebas/resultados.jsonl (lo lee el panel de comparación).

Uso:
    import sys; sys.path.insert(0, "pruebas")
    from base import lab_test, build, BEST_GROUPS, PARAMS, lgb_cv, guardar, registrar
    lab, test = lab_test()
    d, F = build(lab, BEST_GROUPS)
"""
import json, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
OUT = ROOT / "pruebas"
sys.path.insert(0, str(ROOT / "cv"))
from metric import hits, labels  # noqa: E402

TYPES = ['entry_conflict_resolution', 'exit_conflict_resolution', 'entry_coordination', 'exit_coordination',
         'assume_aircraft', 'transfer_aircraft', 'conflict_resolution', 'non_conformance_resolution',
         'qos_improvement', 'entry_conditions', 'exit_conditions', 'aircraft_request', 'zone_conflict',
         'return_to_route']
_CFG = json.loads((ROOT / "models" / "lgb_adri_config.json").read_text())
BEST_GROUPS = _CFG["groups"]                      # ["alerta", "contexto"] -> lgb_adri, CV 0.6569 con 2 semillas
PARAMS = {**_CFG["params"], "n_jobs": 2}          # 2 hilos por proceso: varios hilos de pruebas a la vez
META = ['sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index',
        'fold', 'cs']


# ----------------------------------------------------------------------------- datos (celda 4 del notebook)
def _load(name):
    d = pd.read_csv(DATA / f"{name}.csv")
    pref = d.task_index.map(lambda t: len(TYPES[t]) + 1)
    d["cs"] = [n[p:].split("_")[0] for n, p in zip(d.task_name, pref)]   # indicativo del avión
    return d


def lab_test():
    """(lab, test): train+validation con fold de Data/folds.csv, y test. Ordenados por sample_id, row_index."""
    lab = pd.concat([_load("train"), _load("validation")], ignore_index=True)
    lab = lab.merge(pd.read_csv(DATA / "folds.csv"), on="atco", how="left")
    test = _load("test").assign(fold=-1)
    lab = lab.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    test = test.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    assert lab.fold.notna().all() and (lab.groupby("sample_id").target_index.sum() == 1).all()
    return lab, test


# ----------------------------------------------------------------------------- features (celda 6 del notebook)
RANK_COLS = ["a_since_flight", "x_task_age", "x_state_age", "time_to_MD", "k_timeToExitPoint",
             "distance_to_zone", "abs_tts"]
ENTRY_TYPES, EXIT_TYPES, CONFLICT_TYPES = [0, 2, 4, 9], [1, 3, 5, 10], [0, 1, 6]


def base_cols(df):
    return [c for c in df.columns if c not in META]


def _with_abs(df):
    return df.assign(abs_tts=df.time_to_sector.abs())


def _ac_any(df, mask):
    """1 si alguna tarea del mismo avión (en el mismo snapshot) cumple mask."""
    return mask.astype(int).groupby([df.sample_id, df.cs]).transform("max")


def fe_snapshot(df):
    x = _with_abs(df)
    g = x.groupby("sample_id")
    out = {}
    for c in RANK_COLS:
        out[f"rs_{c}"] = g[c].rank(method="min")
        out[f"ds_{c}"] = x[c] - g[c].transform("min")
    return pd.DataFrame(out, index=df.index)


def fe_tipo(df):
    x = _with_abs(df)
    g = x.groupby(["sample_id", "task_index"])
    out = {}
    for c in RANK_COLS:
        out[f"rt_{c}"] = g[c].rank(method="min")
        out[f"dt_{c}"] = x[c] - g[c].transform("min")
    return pd.DataFrame(out, index=df.index)


def fe_avion(df):
    g = df.groupby(["sample_id", "cs"])
    out = {
        "ac_age_rank": g.x_task_age.rank(method="min", ascending=False),
        "ac_n_types": g.task_index.transform("nunique"),
        "ac_has_assume": _ac_any(df, df.task_index == 4),
    }
    return pd.DataFrame(out, index=df.index)


def fe_tipo_estado(df):
    t = df.task_index
    out = {
        "ts_assume_offered": ((t == 4) & (df.x_transfer_in == 1)).astype(int),
        "ts_transfer_due": ((t == 5) & (df.k_aircraftNeedsToBeTransferred == 1)).astype(int),
        "ts_entry_coord_done": ((t == 2) & (df.x_isc_coordinated == 1)).astype(int),
        "ts_entry_notified": ((t == 2) & (df.x_isc_notified == 1)).astype(int),
        "ts_zone_dist": df.distance_to_zone.where(t == 12),
        "ts_conflict_ttmd": df.time_to_MD.where(t.isin(CONFLICT_TYPES)),
        "ts_twin_assume": ((t == 4) & (df.l_twin_AE == 1)).astype(int),
        "ts_twin_entry": ((t == 2) & (df.l_twin_AE == 1)).astype(int),
    }
    return pd.DataFrame(out, index=df.index)


def fe_contexto(df):
    g = df.groupby("sample_id")
    tmp = pd.DataFrame({
        "sample_id": df.sample_id,
        "is_assume_offer": ((df.task_index == 4) & (df.x_transfer_in == 1)).astype(int),
        "is_transfer_due": ((df.task_index == 5) & (df.k_aircraftNeedsToBeTransferred == 1)).astype(int),
        "is_zone": (df.task_index == 12).astype(int),
        "is_conflict": df.task_index.isin(CONFLICT_TYPES).astype(int),
        "is_entry_coord": (df.task_index == 2).astype(int),
    })
    gt = tmp.groupby("sample_id")
    out = {"cx_n_aircraft": g.cs.transform("nunique")}
    for c in tmp.columns[1:]:
        out["cx_n_" + c[3:]] = gt[c].transform("sum")
    return pd.DataFrame(out, index=df.index)


def fe_exit_coord(df):
    t = df.task_index
    ec = t == 3
    out = {
        "ec_notified": (ec & (df.k_out_NOTIFIED == 1)).astype(int),
        "ec_coordinated": (ec & (df.k_out_COORDINATED == 1)).astype(int),
        "ec_mismatch": (ec & (df.k_outCoordMismatch == 1)).astype(int),
        "ec_tool": (ec & (df.k_coordinationToolActivation == 1)).astype(int),
        "ec_revision": df.k_outCoordRevision.where(ec),
        "ec_with_transfer": (ec & (_ac_any(df, t == 5) == 1)).astype(int),
        "tr_with_exit_coord": ((t == 5) & (_ac_any(df, ec) == 1)).astype(int),
        "age_vs_newest_ac": df.x_task_age - df.groupby(["sample_id", "cs"]).x_task_age.transform("min"),
    }
    return pd.DataFrame(out, index=df.index)


def fe_alerta(df):
    conf = df.task_index.isin(CONFLICT_TYPES)
    stca = df.HMI_conflict_safety_alert == 1
    ttmd = df.time_to_MD.where(conf)
    out = {
        "al_conflict_stca": (conf & stca).astype(int),
        "al_n_stca": stca.astype(int).groupby(df.sample_id).transform("sum"),
        "al_ttmd_rank": ttmd.groupby(df.sample_id).rank(method="min"),
        "al_conflict_multi": (conf & (df.multiple_conflicts_ind == 1)).astype(int),
        "al_conflict_dmd": df.distance_to_MD.where(conf),
    }
    return pd.DataFrame(out, index=df.index)


def fe_deadline(df):
    t = df.task_index
    dl = pd.Series(np.nan, index=df.index)
    entry, exit_, zone = t.isin(ENTRY_TYPES), t.isin(EXIT_TYPES), t == 12
    dl[entry] = df.time_to_sector[entry]
    dl[exit_] = df.k_timeToExitPoint.fillna(df.time_to_sector.abs())[exit_]
    dl[t == 6] = df.time_to_MD[t == 6]
    dl[zone] = (df.distance_to_zone * 1852 / df.k_groundSpeed.fillna(230))[zone]
    out = {
        "dl": dl,
        "dl_rank": dl.groupby(df.sample_id).rank(method="min"),
        "dl_rank_type": dl.groupby([df.sample_id, t]).rank(method="min"),
        "dl_minus_min": dl - dl.groupby(df.sample_id).transform("min"),
    }
    return pd.DataFrame(out, index=df.index)


FEATURE_GROUPS = {"snapshot": fe_snapshot, "tipo": fe_tipo, "avion": fe_avion,
                  "tipo_estado": fe_tipo_estado, "contexto": fe_contexto,
                  "exit_coord": fe_exit_coord, "alerta": fe_alerta, "deadline": fe_deadline}


def build(df, groups=BEST_GROUPS):
    """(df con las columnas nuevas, lista de features). groups=BEST_GROUPS reproduce lgb_adri."""
    new = [FEATURE_GROUPS[g](df) for g in groups]
    out = pd.concat([df] + new, axis=1)
    return out, base_cols(df) + [c for n in new for c in n.columns]


# ----------------------------------------------------------------------------- LightGBM de referencia
def lgb_fit(A, F, params):
    m = lgb.LGBMRanker(**params)
    m.fit(A[F], A.target_index, group=A.groupby("sample_id", sort=False).size().values,
          categorical_feature=["task_index"])
    return m


def lgb_cv(d, F, params=None, seeds=(0,), test=None):
    """CV con Data/folds.csv. Devuelve (oof, pred_test o None). d y test ordenados por sample_id."""
    params = {**PARAMS, **(params or {})}
    oof = np.zeros(len(d))
    for f in range(5):
        tr, va = (d.fold != f).values, (d.fold == f).values
        for s in seeds:
            oof[va] += lgb_fit(d[tr], F, {**params, "random_state": s}).predict(d.loc[va, F]) / len(seeds)
    pred = None
    if test is not None:
        pred = np.mean([lgb_fit(d, F, {**params, "random_state": s}).predict(test[F]) for s in seeds], axis=0)
    return oof, pred


# ----------------------------------------------------------------------------- salidas y registro
def guardar(nombre, lab, oof, test=None, pred=None, out=OUT):
    """<out>/oof/<nombre>.csv y <out>/preds/<nombre>.csv (sample_id,row_index,score); out = pruebas/ por defecto."""
    for sub in ("oof", "preds"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    lab[["sample_id", "row_index"]].assign(score=np.asarray(oof)).to_csv(out / "oof" / f"{nombre}.csv", index=False)
    if pred is not None:
        test[["sample_id", "row_index"]].assign(score=np.asarray(pred)).to_csv(out / "preds" / f"{nombre}.csv",
                                                                               index=False)


def registrar(nombre, hilo, descripcion, oof_df=None, segundos=None, n_features=None, extra=None, out=OUT):
    """Mide un OOF (sample_id,row_index,score; por defecto pruebas/oof/<nombre>.csv) y lo apunta en
    <out>/resultados.jsonl. Devuelve la CV top-3."""
    if oof_df is None:
        oof_df = pd.read_csv(out / "oof" / f"{nombre}.csv")
    h = hits(oof_df)
    lab = labels().set_index("sample_id").loc[h.index]
    tgt = pd.concat([pd.read_csv(DATA / f"{f}.csv", usecols=["sample_id", "target_index", "task_index"])
                     for f in ("train", "validation")]).query("target_index == 1").set_index("sample_id").task_index
    by_atco = h.groupby(lab.atco).mean()
    by_fold = h.groupby(lab.fold).mean()
    by_type = h.groupby(tgt.reindex(h.index).map(lambda t: TYPES[t])).mean()
    fila = dict(hora=time.strftime("%Y-%m-%d %H:%M"), nombre=nombre, hilo=hilo, descripcion=descripcion,
                cv=round(float(h.mean()), 4), min_atco=round(float(by_atco.min()), 4),
                std_folds=round(float(by_fold.std()), 4), n_features=n_features,
                segundos=None if segundos is None else round(segundos),
                tiene_preds=(out / "preds" / f"{nombre}.csv").exists(),
                por_atco={str(k): round(float(v), 4) for k, v in by_atco.items()},
                por_tipo={k: round(float(v), 4) for k, v in by_type.items()}, **(extra or {}))
    with open(out / "resultados.jsonl", "a") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"[{hilo}] {nombre:30s} CV top3 = {fila['cv']:.4f}  min atco {fila['min_atco']:.3f}  "
          f"std folds {fila['std_folds']:.3f}", flush=True)
    return fila["cv"]

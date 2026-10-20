"""
m_lgb_future.py · Pablo (A) · RAMA "futuro" · LightGBM LambdaRank con información de otros snapshots de la sesión.

!! ROMPE LA REGLA 1 DEL EQUIPO A PROPÓSITO. Vive en futuro/ y guarda en futuro/oof|preds|submissions (fuera del stacking de main). !!

- Features por bloques (futuro/features_future.py, ninguna usa la etiqueta):
    base   = columnas originales            logic  = reglas dentro del snapshot (logic_*)
    past   = p_*, blk_*, gap_*              future = f_* (siguiente instrucción, desaparición de la tarea...)
- Parámetros base de EQUIPO.md (modelo de referencia), task_index categórica. Media de varias semillas.
- CV con Data/folds.csv; test entrenado con train+validation. Salidas con cv/metric.save_outputs (formato común).
- Cada ejecución se apunta en experiments/lgb_future_runs.csv. Notebook: futuro/LGB_futuro.ipynb.

Uso:
    python futuro/m_lgb_future.py                                   # 3 semillas, guarda oof/preds/submissions
    python futuro/m_lgb_future.py --quick --leaves 63               # prueba rápida (1 semilla, no guarda)
    python futuro/m_lgb_future.py --quick --feats base,logic,past   # quitar un bloque de features
    python futuro/m_lgb_future.py --quick --feats base,logic,past,future,clear --sesion --peso tramo  # peso 1/sqrt(tramo)
    python futuro/m_lgb_future.py --feats base,logic,past,future,clear --sesion --name lgb_future_clear
        # + segundo paso de sesión (futuro/feat_session.py): guarda también <name>_ses y lo apunta en el registro
"""
import sys, time, argparse, fcntl, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUT = ROOT / "futuro"
sys.path.insert(0, str(ROOT / "cv"))
import features_future as ff
import feat_session
from metric import report, save_outputs, hits, labels

EXP = ROOT / "experiments"
LOG = EXP / "lgb_future_runs.csv"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgb_future")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--feats", default="base,logic,past,future", help="bloques de features separados por comas")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-child", type=int, default=50)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.7)
    ap.add_argument("--trunc", type=int, default=5, help="lambdarank_truncation_level")
    ap.add_argument("--reg-lambda", type=float, default=0.0, help="regularización L2")
    ap.add_argument("--objective", default="lambdarank", help="lambdarank o rank_xendcg")
    ap.add_argument("--folds", default="", help="solo estos folds, p.ej. '0' (pruebas rápidas; no guarda)")
    ap.add_argument("--quick", action="store_true", help="1 semilla y sin guardar oof/preds/submission")
    ap.add_argument("--peso", default="", help="peso por snapshot en el 1.er pase: racha o tramo (1/longitud^peso-pot)")
    ap.add_argument("--peso-pot", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sesion", action="store_true", help="segundo paso de sesión (feat_session.py) -> <name>_ses")
    ap.add_argument("--tag", default="", help="nota libre para el registro de experimentos")
    a = ap.parse_args(argv)
    if a.quick: a.seeds = 1
    return a


def build_data():
    """Carga y calcula las features una vez (el notebook lo reutiliza entre ejecuciones)."""
    t0 = time.time()
    d = ff.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    fs = ff.feature_sets(d)
    assert not set(sum(fs.values(), [])) & set(ff.META), "columna prohibida entre las features"
    lab, te = d[d.split != "test"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    lab = add_lengths(lab)
    print(f"{len(lab)} filas etiquetadas, {len(te)} de test; bloques: " +
          ", ".join(f"{k}={len(v)}" for k, v in fs.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return dict(lab=lab, te=te, fs=fs)


def model(a, seed):
    extra = dict(lambdarank_truncation_level=a.trunc) if a.objective == "lambdarank" else {}
    return lgb.LGBMRanker(objective=a.objective, n_estimators=a.n_estimators, learning_rate=a.lr,
                          num_leaves=a.leaves, min_child_samples=a.min_child, subsample=a.subsample, subsample_freq=1,
                          colsample_bytree=a.colsample, reg_lambda=a.reg_lambda, random_state=seed,
                          verbose=-1, n_jobs=a.threads, **extra)


def fit(a, A, F, seed):
    w = None if not a.peso else 1.0 / A["n_" + a.peso].values ** a.peso_pot
    return model(a, seed).fit(A[F], A.target_index, group=A.groupby("sample_id", sort=False).size().values,
                              sample_weight=w, categorical_feature=["task_index"])


def add_lengths(lab):
    """n_racha = snapshots de la racha; n_tramo = snapshots del tramo (racha + misma respuesta seguida). Solo para pesos."""
    y = lab[lab.target_index == 1].sort_values(["session", "t"])
    seg = ((y.task_name != y.groupby("session").task_name.shift()) | (y.blk_pos == 0)).cumsum()
    n = seg.map(seg.value_counts())
    lab["n_racha"] = lab.blk_len
    lab["n_tramo"] = lab.sample_id.map(dict(zip(y.sample_id, n)))
    return lab


def log_run(a, cv, fold_cv, min_atco, n_feats, secs, saved):
    EXP.mkdir(exist_ok=True)
    row = dict(when=time.strftime("%Y-%m-%d %H:%M"), name=a.name, tag=a.tag, cv=round(cv, 4), min_atco=round(min_atco, 4),
               **{f"fold{k}": round(v, 4) for k, v in fold_cv.items()}, n_feats=n_feats, secs=int(secs), saved=saved,
               **{k: v for k, v in vars(a).items() if k not in ("name", "tag")})
    with open(EXP / ".lgb_future_runs.lock", "w") as lock:        # varias pruebas a la vez en gpu.dsic
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)


def run(a, data):
    t0 = time.time()
    print(f"\n=== {a.name} {a.tag}  " + " ".join(f"{k}={v}" for k, v in vars(a).items() if k not in ("name", "tag")),
          flush=True)
    lab, te, fs = data["lab"], data["te"], data["fs"]
    F = [c for b in a.feats.split(",") for c in fs[b]]
    folds = [int(f) for f in a.folds.split(",")] if a.folds else sorted(lab.fold.unique())
    oof = np.full(len(lab), np.nan)
    for f in folds:
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        oof[va] = np.mean([fit(a, lab[tr], F, s).predict(lab.loc[va, F]) for s in range(a.seeds)], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    done = ~np.isnan(oof)
    oof_df = lab.loc[done, ["sample_id", "row_index"]].assign(score=oof[done])
    cv, fold_cv, min_atco = evaluate(a.name, oof_df, len(F))
    saved = not (a.quick or a.folds)
    if saved:
        pred = np.mean([fit(a, lab, F, s).predict(te[F]) for s in range(a.seeds)], axis=0)
        save_outputs(a.name, oof_df, te[["sample_id", "row_index"]].assign(score=pred), out=FUT)
    log_run(a, cv, fold_cv, min_atco, len(F), time.time() - t0, saved)
    if a.sesion and not a.folds:
        so, sp, w = feat_session.sesion(lab, oof, te, pred) if saved else feat_session.sesion(lab, oof)
        print(f"  pesos del segundo paso: {w}")
        ses = argparse.Namespace(**{**vars(a), "name": a.name + "_ses", "tag": (a.tag + " +sesion").strip()})
        oof_ses = lab[["sample_id", "row_index"]].assign(score=so)
        cvs, fold_s, min_s = evaluate(ses.name, oof_ses, len(F))
        if saved:
            save_outputs(ses.name, oof_ses, te[["sample_id", "row_index"]].assign(score=sp), out=FUT)
        log_run(ses, cvs, fold_s, min_s, len(F), time.time() - t0, saved)
        return cvs
    return cv


def evaluate(name, oof_df, n_feats):
    h = hits(oof_df).rename("hit").reset_index().merge(labels(), on="sample_id")
    cv, fold_cv, min_atco = h.hit.mean(), h.groupby("fold").hit.mean().to_dict(), h.groupby("atco").hit.mean().min()
    print(f"{name}: CV top3 = {cv:.4f}   (min atco {min_atco:.3f})   " +
          "  ".join(f"f{k}:{v:.3f}" for k, v in fold_cv.items()) + f"   {n_feats} features", flush=True)
    return cv, fold_cv, min_atco


def main():
    run(parse(), build_data())


if __name__ == "__main__":
    main()

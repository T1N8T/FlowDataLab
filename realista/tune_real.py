"""
tune_real.py · Pablo (A) · RAMA "REALISTA" · Búsqueda de hiperparámetros del LightGBM realista con Optuna.

Cada prueba = CV de 5 folds (Data/folds.csv) con 1 semilla; objetivo = top-3. Guarda también mediana por sesión y peor
controlador en los user_attrs. Varios procesos comparten el estudio (JournalStorage en realista/tune/), así que se
puede lanzar N veces a la vez. Los mejores se confirman después con m_lgb_real.py y 3+ semillas.

Uso:
    python realista/tune_real.py --study t1 --feats base,logic,past,causal,geo --trials 40 --threads 8
"""
import sys, argparse, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import metricas  # noqa: E402


def load_data(cache=""):
    import features_real as fr
    if cache and Path(cache).exists():
        d = pd.read_pickle(cache)
    else:
        d = fr.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    return d[d.split != "test"].reset_index(drop=True), fr.feature_sets(d)


def params(tr):
    obj = tr.suggest_categorical("objective", ["lambdarank", "rank_xendcg"])
    p = dict(objective=obj, n_estimators=tr.suggest_int("n_estimators", 150, 1500, log=True),
             learning_rate=tr.suggest_float("lr", 0.01, 0.12, log=True),
             num_leaves=tr.suggest_int("leaves", 7, 127, log=True),
             min_child_samples=tr.suggest_int("min_child", 10, 400, log=True),
             subsample=tr.suggest_float("subsample", 0.4, 1.0), subsample_freq=1,
             colsample_bytree=tr.suggest_float("colsample", 0.2, 1.0),
             reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 30, log=True),
             min_split_gain=tr.suggest_float("min_gain", 1e-4, 1.0, log=True),
             max_bin=tr.suggest_categorical("max_bin", [63, 127, 255]),
             cat_smooth=tr.suggest_float("cat_smooth", 1, 50, log=True),
             extra_trees=tr.suggest_categorical("extra_trees", [False, True]))
    if obj == "lambdarank":
        p["lambdarank_truncation_level"] = tr.suggest_int("trunc", 3, 40, log=True)
    return p


def cv(lab, F, p, seed=0, threads=8):
    grp = lambda A: A.groupby("sample_id", sort=False).size().values  # noqa: E731
    oof = np.full(len(lab), np.nan)
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        A = lab[tr]
        m = lgb.LGBMRanker(**p, random_state=seed, verbose=-1, n_jobs=threads)
        m.fit(A[F], A.target_index, group=grp(A), categorical_feature=["task_index"])
        oof[va] = m.predict(lab.loc[va, F])
    return lab[["sample_id", "row_index"]].assign(score=oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="t1")
    ap.add_argument("--feats", default="base,logic,past,causal,geo")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache", default="")
    ap.add_argument("--enqueue-default", action="store_true", help="primera prueba = parámetros actuales")
    a = ap.parse_args()
    lab, fs = load_data(a.cache)
    F = [c for b in a.feats.split(",") for c in fs[b]]
    (REAL / "tune").mkdir(exist_ok=True)
    st = JournalStorage(JournalFileBackend(str(REAL / "tune" / f"{a.study}.log")))
    study = optuna.create_study(study_name=a.study, storage=st, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=16, multivariate=True))
    if a.enqueue_default:
        study.enqueue_trial(dict(objective="lambdarank", n_estimators=400, lr=0.05, leaves=31, min_child=50,
                                 subsample=0.8, colsample=0.7, reg_lambda=1e-3, min_gain=1e-4, max_bin=255,
                                 cat_smooth=10, extra_trees=False, trunc=5))

    def obj(tr):
        t0 = time.time()
        m = metricas.metricas(cv(lab, F, params(tr), threads=a.threads))
        for k in ("ses_median", "min_atco", "sc1", "sc2", "sc3"):
            tr.set_user_attr(k, float(m[k]))
        tr.set_user_attr("secs", int(time.time() - t0))
        print(f"trial {tr.number}: {m['cv']:.4f} ses {m['ses_median']:.3f} min {m['min_atco']:.3f} "
              f"({time.time() - t0:.0f}s) {tr.params}", flush=True)
        return m["cv"]

    study.optimize(obj, n_trials=a.trials)


if __name__ == "__main__":
    main()

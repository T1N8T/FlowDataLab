"""
m_gbm_future.py · RAMA FUTURO · XGBoost (rank:ndcg) o CatBoost (YetiRank) con las mismas features que m_lgb_future.py.

Mismos datos (features_future.build), mismos folds (Data/folds.csv), mismo formato de salida que lgb_future, para
compararlo con el LightGBM y meterlo en el stacking (stack.py --futuro los recoge solo).
Cada ejecución se apunta en experiments/gbm_future_runs.csv.

Uso (desde la raíz; en GPU, elegir la tarjeta con CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0 python futuro/m_gbm_future.py --algo xgb             # -> futuro/oof|preds|submissions/xgb_future
    CUDA_VISIBLE_DEVICES=1 python futuro/m_gbm_future.py --algo cat             # -> cat_future
    python futuro/m_gbm_future.py --algo xgb --quick                            # 1 semilla, no guarda
    python futuro/m_gbm_future.py --algo cat --feats base,logic,past,future,coord --name cat_future_coord
    --cpu  para lanzarlo sin GPU
"""
import sys, time, argparse, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUT = ROOT / "futuro"
sys.path.insert(0, str(ROOT / "cv"))
sys.path.insert(0, str(FUT))
from m_lgb_future import build_data
from metric import save_outputs, hits, labels

LOG = ROOT / "experiments" / "gbm_future_runs.csv"
N_TYPES = 14


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["xgb", "cat"], required=True)
    ap.add_argument("--name", default="", help="por defecto xgb_future / cat_future")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--feats", default="base,logic,past,future")
    ap.add_argument("--n-estimators", type=int, default=0, help="0 = 600 (xgb) / 1500 (cat)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="1 semilla y sin guardar")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    a.name = a.name or f"{a.algo}_future"
    a.n_estimators = a.n_estimators or (600 if a.algo == "xgb" else 1500)
    if a.quick: a.seeds = 1
    return a


# ----------------------------------------------------------------------------- modelos
def xgb_fit_predict(a, A, F, preds_on, seed):
    import xgboost as xgb
    def X(df):   # task_index en one-hot (evita el camino categórico de xgboost 2.0, roto con numpy 2)
        x = df[F].drop(columns="task_index").astype(np.float32)
        for k in range(N_TYPES):
            x[f"ti_{k}"] = (df.task_index.values == k).astype(np.float32)
        return x
    dev = dict(device="cpu" if a.cpu else "cuda") if int(xgb.__version__.split(".")[0]) >= 2 else \
        dict(tree_method="hist" if a.cpu else "gpu_hist")
    m = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@3", n_estimators=a.n_estimators, learning_rate=a.lr,
                      max_depth=a.depth, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                      tree_method="hist", random_state=seed, **dev)
    m.fit(X(A), A.target_index, qid=A.sample_id.values)
    return [m.predict(X(p)) for p in preds_on]


def cat_fit_predict(a, A, F, preds_on, seed):
    from catboost import CatBoost, Pool
    def P(df, y=None):
        x = df[F].copy()
        x["task_index"] = x.task_index.astype(int)
        return Pool(x, label=y, group_id=df.sample_id.values, cat_features=["task_index"])
    params = dict(loss_function="YetiRank", iterations=a.n_estimators, learning_rate=a.lr, depth=a.depth,
                  random_seed=seed, verbose=0, thread_count=-1, metric_period=a.n_estimators)
    if not a.cpu:
        params.update(task_type="GPU", devices="0")
    m = CatBoost(params)
    m.fit(P(A, A.target_index.values))
    return [m.predict(P(p)) for p in preds_on]


# ----------------------------------------------------------------------------- CV + salidas
def run(a, data):
    t0 = time.time()
    fit_predict = xgb_fit_predict if a.algo == "xgb" else cat_fit_predict
    lab, te, fs = data["lab"], data["te"], data["fs"]
    F = [c for b in a.feats.split(",") for c in fs[b]]
    print(f"\n=== {a.name} {a.tag}  algo={a.algo} feats={a.feats} ({len(F)}) seeds={a.seeds} "
          f"n_estimators={a.n_estimators} lr={a.lr} depth={a.depth}", flush=True)
    oof = np.zeros(len(lab))
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        V = lab[va]
        oof[va] = np.mean([fit_predict(a, lab[tr], F, [V], s)[0] for s in range(a.seeds)], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    oof_df = lab[["sample_id", "row_index"]].assign(score=oof)
    h = hits(oof_df).rename("hit").reset_index().merge(labels(), on="sample_id")
    cv, fold_cv, min_atco = h.hit.mean(), h.groupby("fold").hit.mean().to_dict(), h.groupby("atco").hit.mean().min()
    print(f"{a.name}: CV top3 = {cv:.4f}   (min atco {min_atco:.3f})   " +
          "  ".join(f"f{k}:{v:.3f}" for k, v in fold_cv.items()), flush=True)
    if not a.quick:
        pred = np.mean([fit_predict(a, lab, F, [te], s)[0] for s in range(a.seeds)], axis=0)
        save_outputs(a.name, oof_df, te[["sample_id", "row_index"]].assign(score=pred), out=FUT)
    row = dict(when=time.strftime("%Y-%m-%d %H:%M"), name=a.name, tag=a.tag, cv=round(cv, 4),
               min_atco=round(min_atco, 4), **{f"fold{k}": round(v, 4) for k, v in fold_cv.items()},
               secs=int(time.time() - t0), saved=not a.quick,
               **{k: v for k, v in vars(a).items() if k not in ("name", "tag")})
    LOG.parent.mkdir(exist_ok=True)
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)
    print(f"Total {time.time() - t0:.0f}s", flush=True)
    return cv


if __name__ == "__main__":
    run(parse(), build_data())

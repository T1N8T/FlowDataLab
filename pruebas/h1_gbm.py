"""h1_gbm · CatBoost ranker y XGBoost ranker sobre las features de lgb_adri (solo snapshot, regla 1).

Uso: python pruebas/h1_gbm.py <variante> [--test]
  variantes: cat_yeti, cat_yetipair, cat_qsm, xgb_ndcg, xgb_pairwise
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/pruebas")
from base import lab_test, build, guardar, registrar  # noqa: E402

VAR = sys.argv[1]
TEST = "--test" in sys.argv
lab, test = lab_test()
d, F = build(lab)
dt, _ = build(test)
print(VAR, len(d), "filas", len(F), "features", flush=True)


def cat_fit(A, loss, it=800, lr=0.05):
    from catboost import CatBoostRanker, Pool
    X = A[F].copy(); X["task_index"] = X.task_index.astype(int)
    m = CatBoostRanker(loss_function=loss, iterations=it, learning_rate=lr, depth=6, thread_count=2,
                       random_seed=0, verbose=0)
    m.fit(Pool(X, A.target_index.values, group_id=A.sample_id.values, cat_features=["task_index"]))
    return lambda B: m.predict(Pool(B[F].assign(task_index=B.task_index.astype(int)),
                                    cat_features=["task_index"]))


def xgb_fit(A, obj, it=600):
    import xgboost as xgb
    cats = pd.CategoricalDtype(range(14))
    X = A[F].assign(task_index=A.task_index.astype(cats))
    m = xgb.XGBRanker(objective=obj, n_estimators=it, learning_rate=0.05, max_depth=6, min_child_weight=1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                      nthread=2, random_state=0, lambdarank_pair_method="topk" if obj == "rank:ndcg" else "mean",
                      lambdarank_num_pair_per_sample=8 if obj == "rank:ndcg" else 1, ndcg_exp_gain=False)
    m.fit(X, A.target_index.values, qid=A.sample_id.values)
    return lambda B: m.predict(B[F].assign(task_index=B.task_index.astype(cats)))


FITS = {"cat_yeti": lambda A: cat_fit(A, "YetiRank"),
        "cat_fast": lambda A: cat_fit(A, "YetiRank", it=300, lr=0.1),
        "cat_yetipair": lambda A: cat_fit(A, "YetiRankPairwise", it=500),
        "cat_qsm": lambda A: cat_fit(A, "QuerySoftMax"),
        "xgb_ndcg": lambda A: xgb_fit(A, "rank:ndcg"),
        "xgb_pairwise": lambda A: xgb_fit(A, "rank:pairwise")}
DESC = {"cat_yeti": "CatBoostRanker YetiRank 800it lr0.05 d6, task_index cat, feats lgb_adri",
        "cat_fast": "CatBoostRanker YetiRank 300it lr0.1 d6 (versión ligera), task_index cat",
        "cat_yetipair": "CatBoostRanker YetiRankPairwise 500it lr0.05 d6, task_index cat",
        "cat_qsm": "CatBoostRanker QuerySoftMax 800it lr0.05 d6, task_index cat",
        "xgb_ndcg": "XGBRanker rank:ndcg topk 8 pares, 600it lr0.05 d6 ss0.8 cs0.8, task_index category",
        "xgb_pairwise": "XGBRanker rank:pairwise 600it lr0.05 d6 ss0.8 cs0.8, task_index category"}

t0 = time.time()
fit = FITS[VAR]
oof = np.zeros(len(d))
for f in range(5):
    tr, va = (d.fold != f).values, (d.fold == f).values
    oof[va] = fit(d[tr])(d[va])
    print(f"  fold {f} {time.time() - t0:.0f}s", flush=True)
pred = fit(d)(dt) if TEST else None
seg = time.time() - t0
nombre = "h1_" + VAR
guardar(nombre, d, oof, dt, pred)
registrar(nombre, hilo="h1_gbm", descripcion=DESC[VAR], segundos=seg, n_features=len(F))

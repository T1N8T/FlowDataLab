"""
h2_lgb.py · hilo h2_lgb · Variantes de LightGBM sobre las 99 features de lgb_adri (solo snapshot, regla 1).

Uso:
    .venv/bin/python pruebas/h2_lgb.py ref xendcg binary ...      # variantes con 1 semilla, solo OOF
    .venv/bin/python pruebas/h2_lgb.py --final ref xendcg          # 3 semillas + preds de test (sufijo _s3)
"""
import sys, time
import numpy as np, lightgbm as lgb
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/pruebas")
import base
from base import lab_test, build, PARAMS, lgb_cv, guardar, registrar

VARIANTES = {
    "ref": ({}, "referencia lgb_adri (lambdarank, PARAMS de base)"),
    "xendcg": ({"objective": "rank_xendcg"}, "objective rank_xendcg"),
    "binary": ({"objective": "binary"}, "objective binary (clasificador por fila, sin group)"),
    "trunc3": ({"lambdarank_truncation_level": 3}, "lambdarank truncation_level 3"),
    "nonorm": ({"lambdarank_norm": False}, "lambdarank_norm False"),
    "reg": ({"num_leaves": 15, "min_child_samples": 200, "reg_lambda": 5, "colsample_bytree": 0.5},
            "num_leaves 15, min_child 200, reg_lambda 5, colsample 0.5"),
    "slow": ({"learning_rate": 0.03, "n_estimators": 800}, "lr 0.03, 800 árboles"),
    "big": ({"num_leaves": 63, "min_child_samples": 100}, "num_leaves 63, min_child 100"),
    "extra": ({"extra_trees": True}, "extra_trees True"),
    "smooth": ({"path_smooth": 10}, "path_smooth 10"),
    # combinaciones (se deciden según los resultados de arriba)
    "reg_xendcg": ({"objective": "rank_xendcg", "num_leaves": 15, "min_child_samples": 200, "reg_lambda": 5,
                    "colsample_bytree": 0.5}, "rank_xendcg + regularizado"),
    "reg_slow": ({"num_leaves": 15, "min_child_samples": 200, "reg_lambda": 5, "colsample_bytree": 0.5,
                  "learning_rate": 0.03, "n_estimators": 800}, "regularizado + lr 0.03, 800 árboles"),
    "extra_slow": ({"extra_trees": True, "learning_rate": 0.03, "n_estimators": 800},
                   "extra_trees + lr 0.03, 800 árboles"),
    "extra_reg": ({"extra_trees": True, "num_leaves": 15, "min_child_samples": 200, "reg_lambda": 5,
                   "colsample_bytree": 0.5}, "extra_trees + regularizado"),
}


def _fit_binary(A, F, params):
    p = {k: v for k, v in params.items() if not k.startswith("lambdarank")}
    m = lgb.LGBMClassifier(**p)
    m.fit(A[F], A.target_index, categorical_feature=["task_index"])
    m.predict = lambda X, _m=m: _m.predict_proba(X)[:, 1]
    return m


def correr(nombre, d, F, dt, seeds, con_test):
    params, desc = VARIANTES[nombre]
    orig = base.lgb_fit
    if params.get("objective") == "binary":
        base.lgb_fit = _fit_binary
    t0 = time.time()
    try:
        oof, pred = lgb_cv(d, F, params, seeds=seeds, test=dt if con_test else None)
    finally:
        base.lgb_fit = orig
    tag = f"h2_{nombre}" + ("_s3" if len(seeds) == 3 else "")
    guardar(tag, d, oof, dt, pred)
    registrar(tag, hilo="h2_lgb", descripcion=f"{desc}; semillas {list(seeds)}", segundos=time.time() - t0,
              n_features=len(F))


if __name__ == "__main__":
    args = sys.argv[1:]
    final = "--final" in args
    nombres = [a for a in args if a != "--final"]
    lab, test = lab_test()
    d, F = build(lab)
    dt, _ = build(test)
    for n in nombres:
        correr(n, d, F, dt, seeds=(0, 1, 2) if final else (0,), con_test=final)

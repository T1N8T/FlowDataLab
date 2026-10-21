"""f11_xgb.py · Pablo (A) · hilo f11 · XGBoost rank:ndcg en GPU con las features de N (base, logic, past, future, clear,
instr), para tener un primer pase distinto en el segundo paso conjunto (f11_ses2.py). Mismos folds y formato.
Uso: CUDA_VISIBLE_DEVICES=0 python futuro/pruebas/f11_xgb.py NOMBRE semillas [orc]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import f11_lab, feat_session  # noqa: E402
from metric import hits  # noqa: E402


def X(df, F):
    x = df[F].drop(columns="task_index").astype(np.float32)
    for k in range(14):
        x[f"ti_{k}"] = (df.task_index.values == k).astype(np.float32)
    return x


def fit(A, F, seed):
    m = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@3", n_estimators=600, learning_rate=0.05, max_depth=6,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
                      device="cuda", random_state=seed, nthread=2)
    m.fit(X(A, F), A.target_index, qid=A.sample_id.values)
    return m


if __name__ == "__main__":
    nombre, seeds, orc = sys.argv[1], int(sys.argv[2]), len(sys.argv) > 3
    t0 = time.time()
    lab, te, fs = datos_instr()
    F = [c for b in f11_lab.N_FEATS for c in fs[b]]
    if orc:
        Xo, ocols = f11_lab.orc_cache(lab, te, "coord")
        F = F + ocols
    oof = np.zeros(len(lab))
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        L = lab if not orc else pd.concat([lab, pd.DataFrame({c: Xo[f"{c}__f{f}"].values[:len(lab)] for c in ocols})], axis=1)
        oof[va] = np.mean([fit(L[tr], F, s).predict(X(L[va], F)) for s in range(seeds)], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    L, T = lab, te
    if orc:
        L = pd.concat([lab, pd.DataFrame({c: Xo[f"{c}__full"].values[:len(lab)] for c in ocols})], axis=1)
        T = pd.concat([te, pd.DataFrame({c: Xo[f"{c}__full"].values[len(lab):] for c in ocols})], axis=1)
    pred = np.mean([fit(L, F, s).predict(X(T, F)) for s in range(seeds)], axis=0)
    nm = f"f11_{nombre}_s{seeds}"
    so, sp, w = feat_session.sesion(lab, oof, te, pred)
    guardar(nm, lab, oof, te, pred)
    guardar(nm + "_ses", lab, so, te, sp)
    for suf, sc in (("", oof), ("_ses", so)):
        print(f"{nm}{suf}: CV {hits(lab[['sample_id', 'row_index']].assign(score=sc)).mean():.4f}", flush=True)
    print(f"RESUMEN {nm} {time.time() - t0:.0f}s", flush=True)

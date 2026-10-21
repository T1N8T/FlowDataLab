"""
xgb_lab.py · XGBoost en GPU con las features realistas (mismos folds y formato que lab.py). Pensado para mezclar.
    CUDA_VISIBLE_DEVICES=0 python realista/lab/xgb_lab.py --name xgb1 --seeds 0 --test
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import lab as L  # noqa: E402
import xgboost as xgb  # noqa: E402


def X(df, F):
    x = df[[c for c in F if c != "task_index"]].astype(np.float32)
    for k in range(14):
        x[f"ti_{k}"] = (df.task_index.values == k).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--obj", default="rank:ndcg")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=37)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--mcw", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--extra", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    extra = [x for x in a.extra.split(",") if x]
    lab, te, fs = L.load(extra)
    F = [c for b in L.FEATS.split(",") + extra for c in fs[b]]
    p = dict(objective=a.obj, n_estimators=a.n, learning_rate=a.lr, max_depth=a.depth, max_leaves=a.leaves,
             grow_policy="lossguide", min_child_weight=a.mcw, subsample=0.94, colsample_bytree=0.77, reg_lambda=a.lam,
             max_bin=63, tree_method="hist", device="cuda", nthread=2)
    if a.obj.startswith("rank:"):
        p.update(lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=a.pairs)
    print(f"=== {a.name} {p} feats={len(F)}", flush=True)
    ref = L.ref_noche(lab)
    Xl, Xt = X(lab, F), X(te, F)
    y, sid = lab.target_index.values, lab.sample_id.values
    oofs = []
    for seed in map(int, a.seeds.split(",")):
        po = L.OUT / "oof" / f"{a.name}__s{seed}.npy"
        oof = np.zeros(len(lab))
        for f in sorted(lab.fold.unique()):
            tr, va = (lab.fold != f).values, (lab.fold == f).values
            M = (xgb.XGBRanker if a.obj.startswith("rank:") else xgb.XGBClassifier)(random_state=seed, **p)
            if a.obj.startswith("rank:"):
                M.fit(Xl[tr], y[tr], qid=sid[tr])
                oof[va] = M.predict(Xl[va])
            else:
                M.fit(Xl[tr], y[tr])
                oof[va] = M.predict_proba(Xl[va])[:, 1]
        np.save(po, oof)
        oofs.append(oof)
        print(L.linea(f"{a.name}__s{seed}", L.medir(lab, oof, ref)), f"({time.time() - t0:.0f}s)", flush=True)
        if a.test:
            M = (xgb.XGBRanker if a.obj.startswith("rank:") else xgb.XGBClassifier)(random_state=seed, **p)
            if a.obj.startswith("rank:"):
                M.fit(Xl, y, qid=sid)
                pt = M.predict(Xt)
            else:
                M.fit(Xl, y)
                pt = M.predict_proba(Xt)[:, 1]
            np.save(L.OUT / "pred" / f"{a.name}__s{seed}.npy", pt)
    print(f"fin {a.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

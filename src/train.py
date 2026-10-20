"""
Entrenamiento y CV común (FlowDataLab).
    python src/train.py ablation          -> CV por bloques de features (+ suavizado temporal)
    python src/train.py submit [nombre]   -> entrena con train+val y escribe submissions/<nombre>.csv
Requiere models/features.parquet (python src/features.py).
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from features import MODELS, feature_sets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions"


def load_feats():
    d = pd.read_parquet(MODELS / "features.parquet")
    return d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def hit3(df, score):
    r = pd.Series(score, index=df.index).fillna(-np.inf).groupby(df.sample_id.values).rank(ascending=False, method="first")
    return float((r[df.target_index.values == 1] <= 3).mean())


def softmax_by_sample(df, score):
    s = pd.Series(score, index=df.index)
    e = np.exp(s - s.groupby(df.sample_id.values).transform("max"))
    return (e / e.groupby(df.sample_id.values).transform("sum")).values


def smooth(df, p, k=3, decay=0.6):
    """Media ponderada de la probabilidad de la misma tarea en los +-k snapshots contiguos (mismo bloque)."""
    x = pd.DataFrame({"p": p, "block": df.block.values, "tn": df.task_name.values, "pos": df.blk_pos.values},
                     index=df.index).sort_values(["block", "tn", "pos"])
    g = x.groupby(["block", "tn"]).p
    num, den = x.p.copy(), pd.Series(1.0, index=x.index)
    for i in range(1, k + 1):
        w = decay ** i
        for sh in (g.shift(i), g.shift(-i)):
            num += w * sh.fillna(0); den += w * sh.notna()
    return (num / den).reindex(df.index).values


def lgb_ranker(seed=0, **kw):
    p = dict(objective="lambdarank", n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=50,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, lambdarank_truncation_level=5,
             random_state=seed, verbose=-1, n_jobs=12)
    p.update(kw)
    return lgb.LGBMRanker(**p)


def fit_rank(m, A, F):
    m.fit(A[F], A.target_index, group=A.groupby("sample_id", sort=False).size().values, categorical_feature=["task_index"])
    return m


def cv_oof(lab, F, make=lgb_ranker, **kw):
    oof = np.zeros(len(lab))
    for f in sorted(lab.fold.unique()):
        tr, va = lab.fold != f, lab.fold == f
        oof[va.values] = fit_rank(make(**kw), lab[tr], F).predict(lab.loc[va, F])
    return oof


def report(lab, oof, name):
    p = softmax_by_sample(lab, oof)
    ps = smooth(lab, p)
    by_atco = pd.Series({a: hit3(g, ps[g.index]) for a, g in lab.groupby("atco")})
    print(f"{name:45s} top3={hit3(lab, oof):.4f}  +suavizado={hit3(lab, ps):.4f}  (min atco {by_atco.min():.3f})", flush=True)
    return ps


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    d = load_feats()
    fs = feature_sets(d)
    lab, te = d[d.split != "test"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    sets = {
        "base": fs["base"],
        "base+logic": fs["base"] + fs["logic"],
        "base+logic+past": fs["base"] + fs["logic"] + fs["past"],
        "base+future": fs["base"] + fs["future"],
        "base+logic+past+future (todo)": fs["base"] + fs["logic"] + fs["past"] + fs["future"],
    }
    if mode == "ablation":
        for name, F in sets.items():
            oof = cv_oof(lab, F)
            report(lab, oof, name)
            np.save(MODELS / f"oof_lgb_{name.split(' ')[0]}.npy", oof)
    elif mode == "submit":
        name = sys.argv[2] if len(sys.argv) > 2 else "sub_lgb_all"
        F = sets["base+logic+past+future (todo)"]
        s = np.mean([fit_rank(lgb_ranker(seed), lab, F).predict(te[F]) for seed in range(3)], axis=0)
        ps = smooth(te, softmax_by_sample(te, s))
        write_sub(te, ps, name)


def write_sub(te, score, name):
    x = te.assign(s=score).sort_values(["sample_id", "s"], ascending=[True, False])
    sub = x.groupby("sample_id").row_index.apply(lambda r: " ".join(map(str, r.iloc[:3]))).rename("prediction").reset_index()
    OUT.mkdir(exist_ok=True)
    sub.to_csv(OUT / f"{name}.csv", index=False)
    print("Escrito", OUT / f"{name}.csv", len(sub))


if __name__ == "__main__":
    main()

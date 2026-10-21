"""
lab.py · banco de pruebas del modelo realista (snapshot actual + snapshots pasados; nada del futuro, sin oráculo).
Lee la caché de features (build_cache.py = realista/features_real.build, con el bloque ctx) y entrena variantes con
los folds de Data/folds.csv. Guarda el OOF y la predicción de test de cada semilla en
models/real_lab/oof|pred/<nombre>__s<semilla>.npy (si ya existen, se leen). Compara con real_adri_noche.

    python realista/lab/build_cache.py                                          # una vez (~20 s)
    python realista/lab/lab.py --name b045 --seeds 0,1 --threads 4
    python realista/lab/lab.py --name ctx --feats base,logic,past,causal,geo,ctx --seeds 0 --test
    python realista/lab/lab.py --name sm --obj softmax --set lr=0.03,reg_lambda=2 --seeds 0
    python realista/lab/lab.py --name sty --extra sty --seeds 0                 # bloque nuevo: extra_sty.py
"""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OUT = ROOT / "models" / "real_lab"
sys.path[:0] = [str(EXP), str(ROOT / "realista"), str(ROOT / "futuro"), str(ROOT / "cv")]
import lightgbm as lgb  # noqa: E402
import features_real as fr  # noqa: E402
import metricas  # noqa: E402
from metric import hits  # noqa: E402

FEATS = "base,logic,past,causal,geo"
CV023 = [2, 11, 10, 6, 17, 15]
B045 = dict(n_estimators=756, lr=0.01864, leaves=37, min_child=25, subsample=0.93956, colsample=0.77206,
            reg_lambda=21.55722, min_gain=0.0005, max_bin=63, cat_smooth=1.50382, extra_trees=False, trunc=8)
CFGS = dict(b045=B045,
            b030=dict(B045, n_estimators=1010, min_child=16, trunc=7),
            b056=dict(B045, leaves=36, lr=0.02564, n_estimators=539),
            b059=dict(B045, lr=0.02312, min_child=16, n_estimators=1010, trunc=5),
            b060=dict(B045, lr=0.01985, max_bin=127, min_child=16, n_estimators=1010, trunc=7))


def load(extra=()):
    d = pd.read_pickle(OUT / "cache_features.pkl")
    fs = fr.feature_sets(d)
    for name in extra:                      # bloques nuevos: extra_<name>.py con add(d) -> (d, columnas)
        mod = __import__(f"extra_{name}")
        d, cols = mod.add(d)
        fs[name] = cols
    lab = d[d.split != "test"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    return lab, te, fs


def starts_of(sid):
    s = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    return s, np.diff(np.r_[s, len(sid)])


def softmax_obj(group, eps=0.0):
    starts = np.r_[0, np.cumsum(group)[:-1]]
    gid = np.repeat(np.arange(len(group)), group)

    def f(preds, data):
        y = data.get_label()
        if eps:
            y = y * (1 - eps) + eps / group[gid]
        mx = np.maximum.reduceat(preds, starts)
        e = np.exp(preds - mx[gid])
        p = e / np.add.reduceat(e, starts)[gid]
        w = data.get_weight()
        g, h = p - y, np.maximum(p * (1 - p), 1e-6)
        if w is not None:
            g, h = g * w, h * w
        return g, h
    return f


def params_of(c, obj, seed, threads):
    p = dict(learning_rate=c["lr"], num_leaves=int(c["leaves"]), min_data_in_leaf=int(c["min_child"]),
             bagging_fraction=c["subsample"], bagging_freq=1, feature_fraction=c["colsample"],
             lambda_l2=c["reg_lambda"], min_gain_to_split=c["min_gain"], max_bin=int(c["max_bin"]),
             cat_smooth=c["cat_smooth"], extra_trees=bool(c["extra_trees"]), seed=seed, verbose=-1,
             num_threads=threads, min_sum_hessian_in_leaf=c.get("min_hess", 1e-3))
    for k in ("lambda_l1", "path_smooth", "max_cat_threshold", "cat_l2", "min_data_per_group", "max_depth"):
        if k in c:
            p[k] = c[k]
    if obj == "lambdarank":
        p.update(objective="lambdarank", lambdarank_truncation_level=int(c["trunc"]))
        if "norm" in c:
            p["lambdarank_norm"] = bool(c["norm"])
    elif obj == "xendcg":
        p.update(objective="rank_xendcg")
    elif obj == "binary":
        p.update(objective="binary")
    return p


def fit_predict(c, obj, A, F, seed, threads, Xs, w=None, eps=0.0):
    grp = starts_of(A.sample_id.values)[1]
    ds = lgb.Dataset(A[F], A.target_index.values.astype(float), group=grp, weight=w,
                     categorical_feature=["task_index"], free_raw_data=True)
    p = params_of(c, obj, seed, threads)
    if obj == "softmax":
        p["objective"] = softmax_obj(grp, eps)
    m = lgb.train(p, ds, num_boost_round=int(c["n_estimators"]))
    return [m.predict(X[F]) for X in Xs], m


def zscore(score, sid):
    s = pd.Series(np.asarray(score, dtype="float64"))
    g = s.groupby(sid)
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values


def medir(lab, score, ref=None):
    df = lab[["sample_id", "row_index"]].assign(score=np.asarray(score))
    m = metricas.metricas(df, None if ref is None else lab[["sample_id", "row_index"]].assign(score=np.asarray(ref)))
    h = hits(df)
    atco = lab.drop_duplicates("sample_id").set_index("sample_id").atco
    m["cv023"] = h[atco.reindex(h.index).isin(CV023).values].mean()
    return m


def ref_noche(lab):
    r = pd.read_csv(ROOT / "realista" / "oof" / "real_adri_noche.csv")
    return lab[["sample_id", "row_index"]].merge(r, on=["sample_id", "row_index"], how="left").score.values


def linea(nombre, m):
    s = (f"{nombre:28s} cv {m['cv']:.4f}  cv023 {m['cv023']:.4f}  min_atco {m['min_atco']:.3f}  "
         f"ses_med {m['ses_median']:.4f}")
    if "d_cv" in m:
        s += f"  Δ {m['d_cv'] * 100:+.2f}  ses {m['ses_up']}/{m['ses_down']}  atco {m['atco_up']}/{m['atco_down']}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="b045")
    ap.add_argument("--set", default="", help="k=v,k=v sobre la configuración")
    ap.add_argument("--obj", default="lambdarank", help="lambdarank | softmax | binary | xendcg")
    ap.add_argument("--eps", type=float, default=0.0, help="label smoothing (softmax)")
    ap.add_argument("--feats", default=FEATS)
    ap.add_argument("--extra", default="", help="bloques nuevos (extra_<x>.py), separados por comas")
    ap.add_argument("--drop", default="")
    ap.add_argument("--weight", default="", help="atco = cada controlador pesa lo mismo")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--test", action="store_true", help="también entrena con todo y predice test")
    a = ap.parse_args()
    c = dict(CFGS[a.cfg])
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        c[k] = type(c[k])(float(v)) if k in c and not isinstance(c[k], bool) else (v == "1" if k in c else float(v))
    extra = [x for x in a.extra.split(",") if x]
    t0 = time.time()
    lab, te, fs = load(extra)
    drop = set(a.drop.split(","))
    F = [x for b in a.feats.split(",") + extra for x in fs[b] if x not in drop]
    w_all = None
    if a.weight == "atco":
        n = lab.groupby("atco").sample_id.transform("nunique").values
        w_all = (1.0 / n) / np.mean(1.0 / n)
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    print(f"=== {a.name}  obj={a.obj} cfg={a.cfg} {json.dumps(c)} feats={len(F)} extra={extra} "
          f"weight={a.weight} ({time.time() - t0:.0f}s)", flush=True)
    ref = ref_noche(lab)
    oofs = []
    for seed in map(int, a.seeds.split(",")):
        po = OUT / "oof" / f"{a.name}__s{seed}.npy"
        if po.exists():
            oof = np.load(po)
        else:
            oof = np.zeros(len(lab))
            for f in sorted(lab.fold.unique()):
                tr, va = (lab.fold != f).values, (lab.fold == f).values
                (pv,), _ = fit_predict(c, a.obj, lab[tr], F, seed, a.threads, [lab[va]],
                                       None if w_all is None else w_all[tr], a.eps)
                oof[va] = pv
            np.save(po, oof)
        oofs.append(oof)
        print(linea(f"{a.name}__s{seed}", medir(lab, oof, ref)), f"({time.time() - t0:.0f}s)", flush=True)
        if a.test:
            pp = OUT / "pred" / f"{a.name}__s{seed}.npy"
            if not pp.exists():
                (pt,), m = fit_predict(c, a.obj, lab, F, seed, a.threads, [te], w_all, a.eps)
                np.save(pp, pt)
                if seed == 0:
                    imp = pd.Series(m.feature_importance("gain"), index=F).sort_values(ascending=False)
                    imp.to_csv(OUT / f"imp_{a.name}.csv")
    if len(oofs) > 1:
        z = np.mean([zscore(o, lab.sample_id.values) for o in oofs], axis=0)
        print(linea(f"{a.name}__media{len(oofs)}", medir(lab, z, ref)), flush=True)
    print(f"fin {a.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

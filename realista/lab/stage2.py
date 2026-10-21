"""
stage2.py · segunda etapa DENTRO del snapshot: reordena con la puntuación OOF de la etapa 1 y el contexto de la lista
(solo filas del mismo sample_id; nada de otros snapshots). CV con los mismos folds de Data/folds.csv.

    python realista/lab/stage2.py --s1 noche                 # etapa 1 = realista/oof|preds/real_adri_noche.csv
    python realista/lab/stage2.py --s1 exp:b045,exp:sm_b045  # etapa 1 = media de z de OOF guardados en models/real_lab/oof (semillas __s*)
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import lab as L  # noqa: E402
import lightgbm as lgb  # noqa: E402


def s1_scores(spec, lab, te):
    """Etapa 1 alineada con lab/te: media de z-scores de los modelos indicados."""
    oo, pp = [], []
    for s in spec.split(","):
        if s == "noche":
            o = pd.read_csv(L.ROOT / "realista" / "oof" / "real_adri_noche.csv")
            p = pd.read_csv(L.ROOT / "realista" / "preds" / "real_adri_noche.csv")
            oo.append(lab[["sample_id", "row_index"]].merge(o, how="left").score.values)
            pp.append(te[["sample_id", "row_index"]].merge(p, how="left").score.values)
        else:
            name = s.split(":")[1]
            fo = sorted((L.OUT / "oof").glob(f"{name}__s*.npy"))
            oo.append(np.mean([L.zscore(np.load(f), lab.sample_id.values) for f in fo], axis=0))
            fp = [L.OUT / "pred" / f.name for f in fo if (L.OUT / "pred" / f.name).exists()]
            pp.append(np.mean([L.zscore(np.load(f), te.sample_id.values) for f in fp], axis=0) if fp else None)
    o = np.mean([L.zscore(x, lab.sample_id.values) for x in oo], axis=0)
    p = None if any(x is None for x in pp) else np.mean([L.zscore(x, te.sample_id.values) for x in pp], axis=0)
    return o, p


def kth(keys, z, k):
    """Valor del k-ésimo mayor z (k = 1, 2, ...) dentro de cada grupo `keys`, alineado con las filas (NaN si no hay)."""
    df = pd.DataFrame({"z": z, **{f"k{i}": v for i, v in enumerate(keys)}})
    kc = [f"k{i}" for i in range(len(keys))]
    s = df.sort_values(kc + ["z"], ascending=[True] * len(kc) + [False])
    pos = s.groupby(kc, sort=False).cumcount()
    val = s[pos == k - 1].set_index(kc).z
    return df[kc].merge(val.rename("_v").reset_index(), on=kc, how="left")["_v"].values


def ctx(D, z):
    """Columnas de la etapa 2 a partir de la puntuación z (ya normalizada por snapshot)."""
    X = pd.DataFrame({"z": z}, index=D.index)
    sid, cs, ty = D.sample_id.values, D.cs.values, D.task_index.values
    g = X.groupby(sid).z
    X["rk"] = g.rank(ascending=False, method="first")
    top = g.transform("max")
    X["gap_top"] = X.z - top
    k3, k4 = kth([sid], z, 3), kth([sid], z, 4)
    X["gap_3"], X["gap_4"] = X.z - k3, X.z - k4
    X["gap_3_4"] = k3 - k4
    X["n_close"] = (X.z >= top - 0.5).groupby(sid).transform("sum")
    # avión: mejor z de las OTRAS tareas del mismo avión y nº de tareas del avión en el top-5
    gc = X.groupby([sid, cs]).z
    mx, n = gc.transform("max"), gc.transform("size")
    second = kth([sid, cs], z, 2)
    X["cs_other_max"] = np.where(X.z >= mx, second, mx)
    X["cs_other_max"] = X.cs_other_max.where(n > 1)
    X["cs_rk"] = gc.rank(ascending=False, method="first")
    X["cs_n_top5"] = (X.rk <= 5).groupby([sid, cs]).transform("sum")
    # tipo: rango dentro del tipo, mejor z del tipo y nº del tipo en el top-5
    gt = X.groupby([sid, ty]).z
    X["ty_rk"] = gt.rank(ascending=False, method="first")
    X["ty_gap"] = X.z - gt.transform("max")
    X["ty_n_top5"] = (X.rk <= 5).groupby([sid, ty]).transform("sum")
    X["ty_n"] = gt.transform("size")
    for c in ("task_index", "seq_len", "l_twin_AE", "l_n_flight", "is_assumed"):
        X[c] = D[c].values
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="noche")
    ap.add_argument("--name", default="")
    ap.add_argument("--raw", type=int, default=0, help="añade las N features originales más importantes de b045")
    ap.add_argument("--obj", default="lambdarank")
    ap.add_argument("--blend", type=float, default=0.0, help="score = etapa2_z + blend · etapa1_z")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    t0 = time.time()
    lab, te, fs = L.load()
    o1, p1 = s1_scores(a.s1, lab, te)
    Xl, Xt = ctx(lab, o1), (ctx(te, p1) if p1 is not None else None)
    F = list(Xl.columns)
    if a.raw:
        imp = pd.read_csv(L.OUT / "imp_b045.csv", index_col=0).iloc[:, 0]
        raw = [c for c in imp.index[:a.raw] if c not in F]
        for c in raw:
            Xl[c] = lab[c].values
            if Xt is not None:
                Xt[c] = te[c].values
        F += raw
    Xl["sample_id"], Xl["target_index"] = lab.sample_id.values, lab.target_index.values
    if Xt is not None:
        Xt["sample_id"] = te.sample_id.values
    c = dict(n_estimators=300, lr=0.03, leaves=15, min_child=100, subsample=0.8, colsample=0.8, reg_lambda=10.0,
             min_gain=0.0, max_bin=255, cat_smooth=10.0, extra_trees=False, trunc=8)
    print(f"=== stage2 s1={a.s1} feats={len(F)} obj={a.obj} ({time.time() - t0:.0f}s)", flush=True)
    m1 = L.medir(lab, o1)
    print(L.linea("etapa1", m1), flush=True)
    oofs, preds = [], []
    for seed in map(int, a.seeds.split(",")):
        oof = np.zeros(len(lab))
        for f in sorted(lab.fold.unique()):
            tr, va = (lab.fold != f).values, (lab.fold == f).values
            (pv,), _ = L.fit_predict(c, a.obj, Xl[tr], F, seed, a.threads, [Xl[va]])
            oof[va] = pv
        oofs.append(L.zscore(oof, lab.sample_id.values))
        print(L.linea(f"stage2__s{seed}", L.medir(lab, oof, o1)), f"({time.time() - t0:.0f}s)", flush=True)
        if Xt is not None and a.name:
            (pt,), _ = L.fit_predict(c, a.obj, Xl, F, seed, a.threads, [Xt])
            preds.append(L.zscore(pt, te.sample_id.values))
    z = np.mean(oofs, axis=0)
    print(L.linea("stage2__media", L.medir(lab, z, o1)), flush=True)
    for b in (0.5, 1.0, 2.0):
        print(L.linea(f"stage2+{b}·s1", L.medir(lab, z + b * o1, o1)), flush=True)
    if a.name:
        zz = z + a.blend * o1
        np.save(L.OUT / "oof" / f"{a.name}__s0.npy", zz)
        if preds:
            np.save(L.OUT / "pred" / f"{a.name}__s0.npy", np.mean(preds, axis=0) + a.blend * p1)
        print("guardado", a.name)


if __name__ == "__main__":
    main()

"""
m_mix_real.py · RAMA "REALISTA" · LightGBM con el bloque ctx y dos objetivos (lambdarank y softmax sobre la lista),
mezclado con real_adri_noche. Solo el snapshot actual y snapshots PASADOS (features_real.py): nada del futuro ni oráculo.

Miembros (parámetros b045 de la búsqueda nocturna de Adri; bloques base + logic + past + causal + geo [+ ctx]):
    b045     lambdarank, sin ctx                      (referencia con la misma semilla)
    ctx      lambdarank + ctx                         (+0.2 sobre b045 con las mismas semillas)
    sm       softmax sobre la lista, sin ctx          (objetivo propio: grad = p - y, hess = p (1 - p) por snapshot)
    sm_ctx   softmax + ctx
Cada miembro = media de z-scores (en el snapshot) de sus semillas. Mezcla por defecto: real_adri_noche + ctx, pesos
iguales -> realista/oof|preds|submissions/<name>.csv. Cada miembro también se guarda como real_<miembro>.
Con 1 semilla softmax parecía sumar en mezcla; con 3 semillas no aporta de forma fiable (sección 5 de LGB_realista.ipynb).
Las semillas se guardan en models/real_mix/ (no versionado): si ya existen, se leen y no se reentrena.

Uso (desde la raíz; ~4 min por semilla y miembro con 4 hilos):
    python realista/m_mix_real.py                                   # ctx con 4 semillas + mezcla real_mix
    python realista/m_mix_real.py --miembros ctx --seeds 5          # añade semillas a un miembro
    python realista/m_mix_real.py --miembros sm_ctx --seeds 3       # otro miembro (sin cambiar la mezcla)
    python realista/m_mix_real.py --miembros "" --mezcla noche,ctx,sm_ctx --name real_mix3   # solo mezcla
"""
import sys, time, argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
MIX = ROOT / "models" / "real_mix"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import features_real as fr  # noqa: E402
import m_lgb_real as R  # noqa: E402
import metricas  # noqa: E402
from metric import hits, save_outputs  # noqa: E402

FEATS = ["base", "logic", "past", "causal", "geo"]
CV023 = [2, 11, 10, 6, 17, 15]
B045 = dict(n_estimators=756, lr=0.01864, leaves=37, min_child=25, subsample=0.93956, colsample=0.77206,
            reg_lambda=21.55722, min_gain=0.0005, max_bin=63, cat_smooth=1.50382, trunc=8)
MIEMBROS = dict(b045=dict(obj="lambdarank", feats=FEATS), ctx=dict(obj="lambdarank", feats=FEATS + ["ctx"]),
                sm=dict(obj="softmax", feats=FEATS), sm_ctx=dict(obj="softmax", feats=FEATS + ["ctx"]))


def datos():
    cache = MIX / "cache_features.pkl"
    t0 = time.time()
    if cache.exists():
        d = pd.read_pickle(cache)
    else:
        d = fr.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
        f64 = [c for c in d.columns if d[c].dtype == "float64" and c != "t"]
        d[f64] = d[f64].astype("float32")
        MIX.mkdir(parents=True, exist_ok=True)
        d.to_pickle(cache)
    print(f"features listas ({time.time() - t0:.0f}s)", flush=True)
    return d[d.split != "test"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True), fr.feature_sets(d)


def softmax_obj(group):
    """Cross-entropy de la softmax sobre las tareas del snapshot (una sola tarea correcta por snapshot)."""
    starts = np.r_[0, np.cumsum(group)[:-1]]
    gid = np.repeat(np.arange(len(group)), group)

    def f(preds, data):
        e = np.exp(preds - np.maximum.reduceat(preds, starts)[gid])
        p = e / np.add.reduceat(e, starts)[gid]
        return p - data.get_label(), np.maximum(p * (1 - p), 1e-6)
    return f


def fit_predict(obj, A, F, seed, threads, Xs):
    sid = A.sample_id.values
    group = np.diff(np.r_[np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]), len(sid)])
    c = B045
    p = dict(learning_rate=c["lr"], num_leaves=c["leaves"], min_data_in_leaf=c["min_child"],
             bagging_fraction=c["subsample"], bagging_freq=1, feature_fraction=c["colsample"], lambda_l2=c["reg_lambda"],
             min_gain_to_split=c["min_gain"], max_bin=c["max_bin"], cat_smooth=c["cat_smooth"], extra_trees=False,
             seed=seed, verbose=-1, num_threads=threads, min_sum_hessian_in_leaf=1e-3)
    if obj == "lambdarank":
        p.update(objective="lambdarank", lambdarank_truncation_level=c["trunc"])
    else:
        p["objective"] = softmax_obj(group)
    ds = lgb.Dataset(A[F], A.target_index.values.astype(float), group=group, categorical_feature=["task_index"])
    m = lgb.train(p, ds, num_boost_round=c["n_estimators"])
    return [m.predict(X[F]) for X in Xs]


def zscore(score, sid):
    s = pd.Series(np.asarray(score, dtype="float64"))
    g = s.groupby(sid)
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values


def semillas(nombre, lab, te, fs, n, threads):
    """OOF y test de las semillas 0..n-1 del miembro (las ya guardadas se leen)."""
    cfg = MIEMBROS[nombre]
    F = [c for b in cfg["feats"] for c in fs[b]]
    for sub in ("oof", "pred"):
        (MIX / sub).mkdir(parents=True, exist_ok=True)
    for seed in range(n):
        po, pp = MIX / "oof" / f"{nombre}__s{seed}.npy", MIX / "pred" / f"{nombre}__s{seed}.npy"
        t0 = time.time()
        if not po.exists():
            oof = np.zeros(len(lab))
            for f in sorted(lab.fold.unique()):
                tr, va = (lab.fold != f).values, (lab.fold == f).values
                oof[va] = fit_predict(cfg["obj"], lab[tr], F, seed, threads, [lab[va]])[0]
            np.save(po, oof)
        if not pp.exists():
            np.save(pp, fit_predict(cfg["obj"], lab, F, seed, threads, [te])[0])
        print(f"  {nombre} semilla {seed} lista ({time.time() - t0:.0f}s)", flush=True)


def miembro(nombre, lab, te):
    """Media de z-scores de las semillas guardadas (o real_adri_noche si nombre = 'noche')."""
    if nombre == "noche":
        o = pd.read_csv(REAL / "oof" / "real_adri_noche.csv")
        p = pd.read_csv(REAL / "preds" / "real_adri_noche.csv")
        o = lab[["sample_id", "row_index"]].merge(o, how="left").score.values
        p = te[["sample_id", "row_index"]].merge(p, how="left").score.values
        return zscore(o, lab.sample_id.values), zscore(p, te.sample_id.values), 1
    fs = sorted(f.name for f in (MIX / "pred").glob(f"{nombre}__s*.npy") if (MIX / "oof" / f.name).exists())
    assert fs, f"{nombre}: no hay semillas en {MIX}"
    o = np.mean([zscore(np.load(MIX / "oof" / f), lab.sample_id.values) for f in fs], axis=0)
    p = np.mean([zscore(np.load(MIX / "pred" / f), te.sample_id.values) for f in fs], axis=0)
    return o, p, len(fs)


def guardar(name, lab, te, o, p, ref, feats, n, t0):
    m = metricas.metricas(lab[["sample_id", "row_index"]].assign(score=o), ref)
    h = hits(lab[["sample_id", "row_index"]].assign(score=o))
    atco = lab.drop_duplicates("sample_id").set_index("sample_id").atco
    m["cv023"] = h[atco.reindex(h.index).isin(CV023).values].mean()
    print(metricas.resumen(name, m, "real_adri_noche" if ref is not None else None) + f"   CV 023 {m['cv023']:.4f}",
          flush=True)
    save_outputs(name, lab[["sample_id", "row_index"]].assign(score=o), te[["sample_id", "row_index"]].assign(score=p),
                 out=REAL)
    R.log_run(SimpleNamespace(tag="m_mix_real", feats=feats, drop="", seeds=n, sesion=False), name, m, 0,
              time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miembros", default="ctx", help="miembros que entrenar (vacío = ninguno)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mezcla", default="noche,ctx", help="miembros de la mezcla (pesos iguales)")
    ap.add_argument("--name", default="real_mix")
    a = ap.parse_args()
    t0 = time.time()
    lab, te, fs = datos()
    ref = pd.read_csv(REAL / "oof" / "real_adri_noche.csv")
    for nombre in filter(None, a.miembros.split(",")):
        semillas(nombre, lab, te, fs, a.seeds, a.threads)
        o, p, n = miembro(nombre, lab, te)
        guardar(f"real_{nombre}", lab, te, o, p, ref, "+".join(MIEMBROS[nombre]["feats"]), n, t0)
    partes = [miembro(x, lab, te) for x in a.mezcla.split(",")]
    o, p = np.mean([x[0] for x in partes], axis=0), np.mean([x[1] for x in partes], axis=0)
    guardar(a.name, lab, te, o, p, ref, f"mezcla {a.mezcla}", sum(x[2] for x in partes), t0)


if __name__ == "__main__":
    main()

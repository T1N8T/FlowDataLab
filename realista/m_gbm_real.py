"""
m_gbm_real.py · Pablo (A) · RAMA "REALISTA" · XGBoost (rank:ndcg) o CatBoost (YetiRank) con las features realistas.

Reutiliza xgb_fit_predict / cat_fit_predict de futuro/m_gbm_future.py y los datos, métricas, registro y segundo paso
causal de realista/m_lgb_real.py. Guarda en realista/oof|preds|submissions y apunta en experiments/lgb_real_runs.csv.

Uso (desde la raíz):
    CUDA_VISIBLE_DEVICES=0 python realista/m_gbm_real.py --algo xgb --name xgb_todo --sesion --ref real_todo
    CUDA_VISIBLE_DEVICES=1 python realista/m_gbm_real.py --algo cat --name cat_todo --sesion --ref real_todo
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import m_gbm_future as G  # noqa: E402
import m_lgb_real as R  # noqa: E402
import feat_session_causal  # noqa: E402
from metric import save_outputs  # noqa: E402


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own = {"--sesion": False, "--ref": "", "--cache": "", "--feats": "base,logic,past,causal,geo"}
    for k in list(own):
        if k in argv:
            i = argv.index(k)
            if isinstance(own[k], bool):
                own[k] = True; argv.pop(i)
            else:
                own[k] = argv[i + 1]; del argv[i:i + 2]
    a = G.parse(argv + ["--feats", own["--feats"]])
    a.sesion, a.ref, a.cache, a.drop = own["--sesion"], own["--ref"], own["--cache"], ""
    if not any(x.startswith("--name") for x in argv):
        a.name = f"{a.algo}_real"
    return a


def data(cache):
    if cache and Path(cache).exists():
        import features_real as fr
        d = pd.read_pickle(cache)
        return dict(lab=d[d.split != "test"].reset_index(drop=True), te=d[d.split == "test"].reset_index(drop=True),
                    fs=fr.feature_sets(d))
    return R.build_data()


def run(a, dat):
    t0 = time.time()
    fp = G.xgb_fit_predict if a.algo == "xgb" else G.cat_fit_predict
    lab, te, fs = dat["lab"], dat["te"], dat["fs"]
    F = [c for b in a.feats.split(",") for c in fs[b]]
    print(f"\n=== {a.name} {a.tag} algo={a.algo} feats={a.feats} ({len(F)}) seeds={a.seeds} "
          f"n={a.n_estimators} lr={a.lr} depth={a.depth}", flush=True)
    oof = np.zeros(len(lab))
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        oof[va] = np.mean([fp(a, lab[tr], F, [lab[va]], s)[0] for s in range(a.seeds)], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    o = lab[["sample_id", "row_index"]].assign(score=oof)
    m = R.evaluate(a.name, o, a.ref)
    if a.quick:
        return
    pred = np.mean([fp(a, lab, F, [te], s)[0] for s in range(a.seeds)], axis=0)
    save_outputs(a.name, o, te[["sample_id", "row_index"]].assign(score=pred), out=REAL)
    R.log_run(a, a.name, m, len(F), time.time() - t0)
    if a.sesion:
        so, sp, w = feat_session_causal.sesion(lab, oof, te, pred)
        o2 = lab[["sample_id", "row_index"]].assign(score=so)
        m2 = R.evaluate(a.name + "_ses", o2, a.ref + "_ses" if a.ref else "")
        save_outputs(a.name + "_ses", o2, te[["sample_id", "row_index"]].assign(score=sp), out=REAL)
        R.log_run(a, a.name + "_ses", m2, len(F), time.time() - t0)


if __name__ == "__main__":
    a = parse()
    run(a, data(a.cache))

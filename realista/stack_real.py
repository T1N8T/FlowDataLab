"""
stack_real.py · Pablo (A) · RAMA "REALISTA" · Combina modelos realistas (realista/oof|preds) con los de main (oof|preds,
solo el snapshot, así que también valen aquí) mediante un logit condicional (softmax dentro del snapshot) sobre el
log-softmax de cada modelo. Pesos ajustados en 4 folds y medidos en el 5.º (CV anidada). --sesion aplica después el
segundo paso causal (feat_session_causal.py) al resultado.

Uso:
    python realista/stack_real.py real_todo nn_tf_v2 lgb_adri --name stk_a --sesion --ref real_todo
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import metricas  # noqa: E402
from feat_session import fit_logit  # noqa: E402
from metric import save_outputs  # noqa: E402


def load(name, kind):
    for r in (REAL, ROOT):
        p = r / kind / f"{name}.csv"
        if p.exists():
            return pd.read_csv(p).set_index(["sample_id", "row_index"]).score.rename(name)
    raise FileNotFoundError(name)


def lsm(df):
    s = df.sub(df.groupby(level=0).transform("max"))
    return s - np.log(np.exp(s).groupby(level=0).transform("sum"))


def stack(names, keys_lab, keys_te):
    O = lsm(pd.concat([load(n, "oof") for n in names], axis=1).reindex(keys_lab))
    P = lsm(pd.concat([load(n, "preds") for n in names], axis=1).reindex(keys_te))
    return O.fillna(-20).values, P.fillna(-20).values


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--name", default="stk")
    ap.add_argument("--sesion", action="store_true")
    ap.add_argument("--ref", default="real_todo")
    ap.add_argument("--nosave", action="store_true")
    a = ap.parse_args(argv)
    import features_real as fr
    d = fr.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    lab, te = d[d.split != "test"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    kl = pd.MultiIndex.from_frame(lab[["sample_id", "row_index"]])
    kt = pd.MultiIndex.from_frame(te[["sample_id", "row_index"]])
    X, XT = stack(a.models, kl, kt)
    y, g = lab.target_index.values, lab.sample_id.values
    oof = np.full(len(lab), np.nan)
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        oof[va] = X[va] @ fit_logit(X[tr], y[tr], g[tr])
    w = fit_logit(X, y, g)
    pred = XT @ w
    print("pesos:", dict(zip(a.models, w.round(3))))
    ref = pd.read_csv(REAL / "oof" / f"{a.ref}.csv")
    o = lab[["sample_id", "row_index"]].assign(score=oof)
    print(metricas.resumen(a.name, metricas.metricas(o, ref), a.ref))
    if not a.nosave:
        save_outputs(a.name, o, te[["sample_id", "row_index"]].assign(score=pred), out=REAL)
    if a.sesion:
        import feat_session_causal
        so, sp, w2 = feat_session_causal.sesion(lab, oof, te, pred)
        o2 = lab[["sample_id", "row_index"]].assign(score=so)
        rs = REAL / "oof" / f"{a.ref}_ses.csv"
        print(metricas.resumen(a.name + "_ses", metricas.metricas(o2, pd.read_csv(rs) if rs.exists() else ref),
                               a.ref + "_ses"))
        if not a.nosave:
            save_outputs(a.name + "_ses", o2, te[["sample_id", "row_index"]].assign(score=sp), out=REAL)


if __name__ == "__main__":
    main()

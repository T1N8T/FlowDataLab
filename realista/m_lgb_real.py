"""
m_lgb_real.py · Pablo (A) · RAMA "REALISTA" · LightGBM LambdaRank que solo ve el snapshot actual y snapshots PASADOS.

Mismos parámetros, folds y semillas que futuro/m_lgb_future.py (reutiliza su model()). Features de
realista/features_real.py. --sesion añade el segundo paso CAUSAL (realista/feat_session_causal.py): cada snapshot solo
usa scores de snapshots anteriores. Guarda en realista/oof|preds|submissions y apunta en experiments/lgb_real_runs.csv.

Uso:
    python realista/m_lgb_real.py --feats base,logic,past --sesion --name real_base
    python realista/m_lgb_real.py --feats base,logic,past,causal,geo --sesion --name real_todo --ref real_base
"""
import sys, time, argparse, fcntl
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import features_real as fr  # noqa: E402
import feat_session_causal  # noqa: E402
import m_lgb_future as M  # noqa: E402
import metricas  # noqa: E402
from metric import save_outputs  # noqa: E402

LOG = ROOT / "experiments" / "lgb_real_runs.csv"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="real")
    ap.add_argument("--feats", default="base,logic,past")
    ap.add_argument("--drop", default="", help="columnas a quitar, separadas por comas")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sesion", action="store_true", help="segundo paso causal -> <name>_ses")
    ap.add_argument("--ref", default="", help="OOF de realista/oof para comparar (el 2.º paso con <ref>_ses)")
    ap.add_argument("--tag", default="")
    a, rest = ap.parse_known_args(argv)
    p = M.parse(rest + ["--threads", str(a.threads)])        # parámetros del LightGBM (los de futuro por defecto)
    return a, p


def build_data():
    t0 = time.time()
    d = fr.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    fs = fr.feature_sets(d)
    lab, te = d[d.split != "test"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    print(f"{len(lab)} filas etiquetadas, {len(te)} de test; bloques: " +
          ", ".join(f"{k}={len(v)}" for k, v in fs.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return dict(lab=lab, te=te, fs=fs)


def evaluate(name, df, ref_name):
    p = REAL / "oof" / f"{ref_name}.csv"
    ref = pd.read_csv(p) if ref_name and p.exists() else None
    m = metricas.metricas(df, ref)
    print(metricas.resumen(name, m, ref_name), flush=True)
    return m


def log_run(a, name, m, n_feats, secs):
    row = dict(when=time.strftime("%Y-%m-%d %H:%M"), name=name, tag=a.tag,
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
               n_feats=n_feats, secs=int(secs), feats=a.feats, drop=a.drop, seeds=a.seeds, sesion=a.sesion)
    with open(ROOT / "experiments" / ".lgb_real_runs.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)


def run(a, p, data):
    t0 = time.time()
    lab, te, fs = data["lab"], data["te"], data["fs"]
    F = [c for b in a.feats.split(",") for c in fs[b] if c not in a.drop.split(",")]
    print(f"\n=== {a.name} {a.tag}  feats={a.feats} ({len(F)})", flush=True)
    grp = lambda A: A.groupby("sample_id", sort=False).size().values  # noqa: E731
    oof = np.full(len(lab), np.nan)
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        A = lab[tr]
        ms = [M.model(p, s).fit(A[F], A.target_index, group=grp(A), categorical_feature=["task_index"])
              for s in range(a.seeds)]
        oof[va] = np.mean([m.predict(lab.loc[va, F]) for m in ms], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    pred = np.mean([M.model(p, s).fit(lab[F], lab.target_index, group=grp(lab), categorical_feature=["task_index"])
                    .predict(te[F]) for s in range(a.seeds)], axis=0)
    oof_df = lab[["sample_id", "row_index"]].assign(score=oof)
    m = evaluate(a.name, oof_df, a.ref)
    save_outputs(a.name, oof_df, te[["sample_id", "row_index"]].assign(score=pred), out=REAL)
    log_run(a, a.name, m, len(F), time.time() - t0)
    if a.sesion:
        so, sp, w = feat_session_causal.sesion(lab, oof, te, pred)
        print(f"  pesos del segundo paso causal: {w}")
        o2 = lab[["sample_id", "row_index"]].assign(score=so)
        m2 = evaluate(a.name + "_ses", o2, a.ref + "_ses" if a.ref else "")
        save_outputs(a.name + "_ses", o2, te[["sample_id", "row_index"]].assign(score=sp), out=REAL)
        log_run(a, a.name + "_ses", m2, len(F), time.time() - t0)


if __name__ == "__main__":
    a, p = parse()
    run(a, p, build_data())

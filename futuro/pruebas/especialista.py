"""
especialista.py · Pablo (A) · RAMA FUTURO · Especialista que reordena solo los candidatos de coordinación.

La mitad de los fallos son coordinaciones, y en entry_coordination la favorita suele ser OTRA coordinación (11d): hay
que distinguir qué coordinación concreta se hace. Un LightGBM LambdaRank se entrena solo con las filas de coordinación
(entry/exit) de los snapshots cuya respuesta es una coordinación, con las features del modelo base + oráculo de
coordinaciones (por fold). Se combina con el 1.er pase solo en las filas de coordinación:
    score = base + beta · (especialista − media del especialista entre las coordinaciones del snapshot)
beta se elige con CV anidada (después del segundo paso de sesión). Base: lgb_future_clear_orcoord (3 semillas).

Uso: python futuro/pruebas/especialista.py [semillas]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "cv")]
import m_lgb_future as M, feat_session, metricas  # noqa: E402
from metric import hits  # noqa: E402

BASE = "lgb_future_clear_orcoord"
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def main(seeds=3):
    t0 = time.time()
    data = M.build_data()
    lab, fs = data["lab"], data["fs"]
    O = M.oracle_data(data, "coord")
    F = [c for b in "base,logic,past,future,clear".split(",") for c in fs[b]] + list(O["lab"].columns)
    base = lab[["sample_id", "row_index"]].merge(pd.read_csv(ROOT / "futuro" / "oof" / f"{BASE}.csv"), how="left").score.values
    is_c = lab.task_index.isin([2, 3]).values
    spec = np.full(len(lab), np.nan)
    for f in range(5):
        L = pd.concat([lab, O[f]], axis=1)
        tr = (L.fold != f).values & is_c & L.coord_grp.values.astype(bool)
        A = L[tr]
        A = A[A.groupby("sample_id").sample_id.transform("size") >= 2]
        va = (L.fold == f).values & is_c
        p = 0
        for s in range(seeds):
            m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               lambdarank_truncation_level=3, random_state=s, verbose=-1, n_jobs=4)
            m.fit(A[F], A.target_index, group=A.groupby("sample_id", sort=False).size().values,
                  categorical_feature=["task_index"])
            p = p + m.predict(L.loc[va, F]) / seeds
        spec[va] = p
        print(f"  especialista fold {f} ({time.time() - t0:.0f}s)", flush=True)
    centr = pd.Series(spec).groupby(lab.sample_id.values).transform("mean").values
    adj = np.where(is_c, spec - centr, 0.0)
    ref = pd.read_csv(ROOT / "futuro" / "oof" / f"{BASE}_ses.csv")
    H = {}
    for b in BETAS:
        so, _, _ = feat_session.sesion(lab, base + b * adj)
        H[b] = hits(lab[["sample_id", "row_index"]].assign(score=so))
        print(f"  beta={b}: + sesión {H[b].mean():.4f}", flush=True)
    info = metricas.info().reindex(H[0.0].index)
    out = pd.Series(np.nan, index=H[0.0].index); el = []
    for f in range(5):
        tr, va = (info.fold != f).values, (info.fold == f).values
        b = max(BETAS, key=lambda b: H[b][tr].mean()); el.append(b)
        out[va] = H[b][va]
    print(f"CV anidada (beta elegido por fold {el}): {out.mean():.4f}  · beta=0 (base + sesión): {H[0.0].mean():.4f}")
    bf = max(BETAS, key=lambda b: H[b].mean())
    so, _, _ = feat_session.sesion(lab, base + bf * adj)
    m = metricas.metricas(lab[["sample_id", "row_index"]].assign(score=so), ref)
    print(metricas.resumen(f"especialista beta={bf}", m, f"{BASE}_ses"), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

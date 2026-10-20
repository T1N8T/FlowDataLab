"""
oraculo.py · Pablo (A) · RAMA FUTURO · Prototipo del "oráculo" (prior colectivo, idea Pau #1 del EDA, sección 9).

Todas las sesiones de un escenario tienen el mismo tráfico (los aviones de test están al 100 % en train). Para cada
tarea se mira qué hicieron OTROS controladores en el mismo escenario y en el mismo momento (tiempo alineado):
    or_ever        fracción de las otras sesiones (que tienen la tarea) en que fue respuesta alguna vez
    or_near60/180  fracción en que fue respuesta a ±60 / ±180 s del mismo momento
    or_cs_near60   lo mismo para cualquier tarea del avión
    or_dt          segundos hasta el momento mediano en que la trabajan los demás (NaN si nadie)
    or_*_rk        rangos dentro del snapshot
USA ETIQUETAS (de otras sesiones) -> se calcula por fold: para el modelo del fold f solo cuentan sesiones etiquetadas
del mismo escenario, de folds != f y de otro controlador. Test: todas las sesiones etiquetadas.
OJO: el equipo descartó esta idea en main por la regla 4 (usa scenario y timestamp); aquí solo se mide.

Uso: python futuro/pruebas/oraculo.py [semillas] [leaves] [min_child] [solo_con|todo] [variante base|B|C|D|E]   # CV 1.er pase + segundo paso
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "cv")]
import features_future as ff, feat_session  # noqa: E402
from metric import hits, labels  # noqa: E402

from feat_oracle import align, matrices, feats  # noqa: E402


def main(seeds=1, leaves=31, min_child=50, solo_con=False, variant="base"):
    t0 = time.time()
    d = ff.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    fs = ff.feature_sets(d)
    F0 = [c for b in "base,logic,past,future,clear".split(",") for c in fs[b]]
    tau = align(d)
    mats = matrices(d, tau, extra=variant != "base")
    print(f"oráculo: matrices listas ({time.time() - t0:.0f}s)", flush=True)
    lab_m = (d.split != "test").values
    lab = d[lab_m].reset_index(drop=True)
    OR = {f: feats(d, mats, f, variant)[lab_m].reset_index(drop=True) for f in range(5)}
    cols = list(OR[0].columns)
    y = lab.target_index.values
    res = {}
    for name, use in (((("con oráculo", True),) if solo_con else (("sin oráculo", False), ("con oráculo", True)))):
        oof = np.zeros(len(lab))
        for f in range(5):
            X = pd.concat([lab[F0], OR[f]], axis=1) if use else lab[F0]
            F = F0 + cols if use else F0
            tr, va = (lab.fold != f).values, (lab.fold == f).values
            A = X[tr]
            p = 0
            for s in range(seeds):
                m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=leaves,
                                   min_child_samples=min_child, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                   lambdarank_truncation_level=5, random_state=s, verbose=-1, n_jobs=8)
                m.fit(A[F], y[tr], group=lab[tr].groupby("sample_id", sort=False).size().values,
                      categorical_feature=["task_index"])
                p = p + m.predict(X.loc[va, F]) / seeds
            oof[va] = p
            print(f"  {name} fold {f} ({time.time() - t0:.0f}s)", flush=True)
        so, _, w = feat_session.sesion(lab, oof)
        for k, sc in (("1.er pase", oof), ("+ sesión", so)):
            h = hits(lab[["sample_id", "row_index"]].assign(score=sc)).rename("hit").reset_index().merge(labels(), on="sample_id")
            print(f"{name} · {k}: CV {h.hit.mean():.4f}  min_atco {h.groupby('atco').hit.mean().min():.4f}  folds " +
                  " ".join(f"{v:.3f}" for v in h.groupby("fold").hit.mean()), flush=True)
            res[(name, k)] = h
        if use:
            imp = pd.Series(m.booster_.feature_importance("gain"), index=F)
            print("  ganancia de las or_*:", (imp[cols] / imp.sum()).round(3).sort_values(ascending=False).to_dict())
    if solo_con:
        return
    ty = lab[lab.target_index == 1].set_index("sample_id").task_index.map(dict(enumerate(ff.TYPES)))
    a, b = res[("sin oráculo", "+ sesión")].set_index("sample_id").hit, res[("con oráculo", "+ sesión")].set_index("sample_id").hit
    t = pd.DataFrame(dict(sin=a.groupby(ty).mean(), con=b.groupby(ty).mean(), n=a.groupby(ty).size()))
    print((t.assign(delta=t.con - t.sin)).sort_values("n", ascending=False).round(3).to_string())


if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 5
    main(int(a[0] or 1), int(a[1] or 31), int(a[2] or 50), a[3] == "solo_con", a[4] or "base")

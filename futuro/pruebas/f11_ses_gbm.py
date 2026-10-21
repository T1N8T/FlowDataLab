"""f11_ses_gbm.py · Pablo (A) · hilo f11 · segundo paso NO lineal y pequeño: LightGBM lambdarank con solo las 9 features
de sesión (feat_session.session_feats) + task_index (+ lsm del logit como base), árboles de pocas hojas. Por fold (los
otros 4); test con todo el OOF. La sección 9 probó un LightGBM con TODAS las features + sesión y sobreajustaba; aquí no
ve las features originales. Uso: python futuro/pruebas/f11_ses_gbm.py NOMBRE modelo [hojas árboles]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import feat_session as FS, suavizado as S  # noqa: E402


def fit(X, y, g, leaves, n, init):
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=n, learning_rate=0.05, num_leaves=leaves,
                       min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=1.0,
                       lambdarank_truncation_level=5, verbose=-1, n_jobs=4, random_state=0)
    m.fit(X, y, group=pd.Series(g).groupby(g, sort=False).size().values, init_score=init, categorical_feature=["task_index"])
    return m


def run(nombre, modelo, leaves=7, n=100):
    lab, te, _ = datos_instr()
    k = ["sample_id", "row_index"]
    o = lab[k].merge(S.leer(modelo, "oof"), on=k, how="left").score.values
    X = FS.session_feats(lab, o).assign(task_index=lab.task_index.values)
    y, g = lab.target_index.values, lab.sample_id.values
    out = np.full(len(lab), np.nan)
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        w = FS.fit_logit(X.values[tr][:, :9], y[tr], g[tr])
        base = X.values[:, :9] @ w                     # el logit de siempre como punto de partida (init_score)
        m = fit(X[tr], y[tr], g[tr], leaves, n, base[tr])
        out[va] = base[va] + m.predict(X[va])
    w = FS.fit_logit(X.values[:, :9], y, g)
    m = fit(X, y, g, leaves, n, X.values[:, :9] @ w)
    p = te[k].merge(S.leer(modelo, "preds"), on=k, how="left").score.values
    XT = FS.session_feats(te, p).assign(task_index=te.task_index.values)
    sp = XT.values[:, :9] @ w + m.predict(XT)
    guardar(nombre, lab, out, te, sp)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], *(int(x) for x in a[2:]))

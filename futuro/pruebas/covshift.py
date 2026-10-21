"""
covshift.py · Pablo (A) · RAMA FUTURO · Adaptación no supervisada al controlador objetivo (covariate shift), sección 18.

Para cada controlador c de validación (fold f): un clasificador de dominio (LightGBM binario, solo variables de entrada,
sin etiquetas) distingue las filas de entrenamiento (folds != f) de las de c. Con validación cruzada dentro del
entrenamiento (3 grupos por controlador), cada snapshot de entrenamiento recibe w = media de p / (1 − p) en sus filas,
normalizado a media 1 y recortado a [0.1, 10]. Se reentrena el ranker con pesos w^α y se predice c. α = 0 es el modelo
de siempre (mismos folds y semillas que lgb_future_clear). α se elige con CV anidada después del segundo paso de sesión.
Lo nuevo frente a Juanma (sección 15, EM de Saerens): aquello reescalaba P(tipo); esto repondera P(X).

Uso: python futuro/pruebas/covshift.py <alfa> [<alfa> ...] [--threads N]
     guarda futuro/pruebas/oof/covshift_a<alfa>.csv (1.er pase) y imprime la comparación con lgb_future_clear(_ses)
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "cv")]
import m_lgb_future as M, feat_session, metricas  # noqa: E402

FEATS = "base,logic,past,future,clear"
OUT = ROOT / "futuro" / "pruebas" / "oof"


def domain_weights(lab, F, tr, tg, threads, seed=0):
    """Peso por snapshot de entrenamiento (filas tr) hacia el dominio de las filas tg. Devuelve (w por fila de tr, AUC)."""
    X_tr, X_tg = lab.loc[tr, F], lab.loc[tg, F]
    at = lab.atco.values[tr]
    grp = {a: i % 3 for i, a in enumerate(np.random.default_rng(seed).permutation(np.unique(at)))}
    g = np.array([grp[a] for a in at])
    p = np.zeros(len(X_tr))
    for k in range(3):
        fit_tr = np.flatnonzero(g != k)
        fit_tr = np.random.default_rng(seed + k).choice(fit_tr, min(len(fit_tr), 120000), replace=False)
        X = pd.concat([X_tr.iloc[fit_tr], X_tg])
        y = np.r_[np.zeros(len(fit_tr)), np.ones(len(X_tg))]
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                                 colsample_bytree=0.7, subsample=0.8, subsample_freq=1, random_state=seed,
                                 n_jobs=threads, verbose=-1)
        clf.fit(X, y, categorical_feature=["task_index"])
        p[g == k] = clf.predict_proba(X_tr.iloc[np.flatnonzero(g == k)])[:, 1]
    # AUC del dominio: filas de entrenamiento (fuera de muestra) frente al objetivo (predicho por el último clasificador,
    auc = roc_auc_score(np.r_[np.zeros(len(p)), np.ones(len(X_tg))],
                        np.r_[p, clf.predict_proba(X_tg)[:, 1]])  # que lo vio en el ajuste: optimista)
    odds = p / np.clip(1 - p, 1e-6, None)
    w = pd.Series(odds).groupby(lab.sample_id.values[tr]).transform("mean").values
    w = np.clip(w / w.mean(), 0.1, 10)
    return w / w.mean(), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alfas", type=float, nargs="+")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    c = ap.parse_args()
    t0 = time.time()
    data = M.build_data()
    lab, fs = data["lab"], data["fs"]
    F = [x for b in FEATS.split(",") for x in fs[b]]
    a = M.parse(["--threads", str(c.threads)])
    oof = {al: np.full(len(lab), np.nan) for al in c.alfas}
    for atco in sorted(lab.atco.unique()):
        f = lab.fold.values[lab.atco.values == atco][0]
        tr, tg = (lab.fold != f).values, (lab.atco == atco).values
        w, auc = domain_weights(lab, F, tr, tg, c.threads)
        ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
        A = lab[tr]
        grp = A.groupby("sample_id", sort=False).size().values
        for al in c.alfas:
            ms = [M.model(a, s).fit(A[F], A.target_index, group=grp, sample_weight=w ** al,
                                    categorical_feature=["task_index"]) for s in range(c.seeds)]
            oof[al][tg] = np.mean([m.predict(lab.loc[tg, F]) for m in ms], axis=0)
        print(f"  atco {atco:03d} (fold {f}): AUC de dominio {auc:.3f}, tamaño efectivo {ess:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    ref1 = pd.read_csv(ROOT / "futuro" / "oof" / "lgb_future_clear.csv")
    ref2 = pd.read_csv(ROOT / "futuro" / "oof" / "lgb_future_clear_ses.csv")
    for al in c.alfas:
        o = lab[["sample_id", "row_index"]].assign(score=oof[al])
        o.to_csv(OUT / f"covshift_a{al:g}.csv", index=False)
        print(metricas.resumen(f"covshift_a{al:g}", metricas.metricas(o, ref1), "lgb_future_clear"))
        so, _, _ = feat_session.sesion(lab, oof[al])
        s = lab[["sample_id", "row_index"]].assign(score=so)
        s.to_csv(OUT / f"covshift_a{al:g}_ses.csv", index=False)
        print(metricas.resumen(f"covshift_a{al:g}_ses", metricas.metricas(s, ref2), "lgb_future_clear_ses"), flush=True)


if __name__ == "__main__":
    main()

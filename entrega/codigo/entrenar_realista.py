"""
entrenar_realista.py · LÍNEA REALISTA · Reproduce la entrega realista `real_adri_noche`.

Qué es: mezcla de 5 configuraciones de LightGBM LambdaRank (b045, b030, b056, b059, b060: las que se confirmaron con 3
semillas en la búsqueda nocturna de Adri, realista/noche_adri.py) × 4 semillas (0–3), con los bloques
base + logic + past + causal + geo (features_realista.py). Por cada configuración se promedian las 4 semillas, se pasa el
score a z-score dentro del snapshot y se hace la media de los 5 z-scores (pesos iguales, nada ajustado con el OOF).
Sin segundo paso de sesión (resta sin futuro). Folds de entrega/datos/folds.csv (por controlador).
CV de la entrega: 0.6732 · CV 023 0.6734 · peor controlador 0.595.

Salidas (comun.guardar): entrega/salidas/oof/<nombre>.csv, preds/<nombre>.csv y submissions/sub_<nombre>.csv.
Cada semilla se guarda en entrega/cache/realista/ y, si se relanza, se reutiliza.

Uso (desde la raíz o desde entrega/codigo):
    python entrega/codigo/entrenar_realista.py                       # entrega completa (~40 min con 14 hilos)
    python entrega/codigo/entrenar_realista.py --rapido              # comprobación: 1 fold, 1 semilla, sin test (~3 min)
    python entrega/codigo/entrenar_realista.py --configs b045 --semillas 1 --solo-oof --nombre real_b045_s0
"""
import argparse, time
import numpy as np, pandas as pd, lightgbm as lgb
import comun
from features_realista import construir, BLOQUES

# Parámetros exactos de realista/noche/resultados.jsonl (etapa "busqueda")
_COMUN = dict(subsample=0.9395578452213194, colsample=0.7720566531433419, reg_lambda=21.557215921190185,
              min_gain=0.0005006652310000156, cat_smooth=1.5038235350480391, extra_trees=False)
CONFIGS = {
    "b045": dict(_COMUN, n_estimators=756, lr=0.018642543216427748, leaves=37, min_child=25, max_bin=63, trunc=8),
    "b030": dict(_COMUN, n_estimators=1010, lr=0.018642543216427748, leaves=37, min_child=16, max_bin=63, trunc=7),
    "b056": dict(_COMUN, n_estimators=539, lr=0.02563945994058366, leaves=36, min_child=25, max_bin=63, trunc=8),
    "b059": dict(_COMUN, n_estimators=1010, lr=0.023116879607101234, leaves=37, min_child=16, max_bin=63, trunc=5),
    "b060": dict(_COMUN, n_estimators=1010, lr=0.019851740781837608, leaves=37, min_child=16, max_bin=127, trunc=7),
}
SEMILLAS = 4
CACHE_SEM = comun.CACHE / "realista"


def lgb_params(c):
    return dict(objective="lambdarank", n_estimators=int(c["n_estimators"]), learning_rate=c["lr"],
                num_leaves=int(c["leaves"]), min_child_samples=int(c["min_child"]), subsample=c["subsample"],
                subsample_freq=1, colsample_bytree=c["colsample"], reg_lambda=c["reg_lambda"],
                min_split_gain=c["min_gain"], max_bin=int(c["max_bin"]), cat_smooth=c["cat_smooth"],
                extra_trees=bool(c["extra_trees"]), lambdarank_truncation_level=int(c["trunc"]))


def ajustar(tr, F, c, seed, hilos):
    m = lgb.LGBMRanker(**lgb_params(c), random_state=seed, verbose=-1, n_jobs=hilos)
    return m.fit(tr[F], tr.target_index, group=tr.groupby("sample_id", sort=False).size().values,
                 categorical_feature=["task_index"])


def zscore(sample_id, score):
    """z-score del score dentro de cada snapshot (pone en la misma escala configuraciones distintas)."""
    s = pd.Series(np.asarray(score, dtype="float64"))
    g = s.groupby(np.asarray(sample_id))
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values


def oof_semilla(lab, F, nombre, c, seed, hilos, folds, cache=True):
    """Score out-of-fold de una semilla (NaN en los folds que no se calculan)."""
    p = CACHE_SEM / f"oof_{nombre}__s{seed}.npy"
    if cache and p.exists():
        return np.load(p)
    oof = np.full(len(lab), np.nan)
    for f in folds:
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        oof[va] = ajustar(lab[tr], F, c, seed, hilos).predict(lab.loc[va, F])
    if cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, oof)
    return oof


def pred_semilla(lab, te, F, nombre, c, seed, hilos):
    p = CACHE_SEM / f"pred_{nombre}__s{seed}.npy"
    if p.exists():
        return np.load(p)
    pr = ajustar(lab, F, c, seed, hilos).predict(te[F])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, pr)
    return pr


def main():
    ap = argparse.ArgumentParser(description="Entrega realista real_adri_noche (5 configuraciones × 4 semillas).")
    ap.add_argument("--hilos", type=int, default=14, help="hilos de LightGBM")
    ap.add_argument("--rapido", action="store_true", help="comprobación: solo el fold 0, 1 semilla, sin test ni guardar")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--semillas", type=int, default=SEMILLAS)
    ap.add_argument("--solo-oof", action="store_true", help="no entrena con todo ni predice test")
    ap.add_argument("--nombre", default="real_adri_noche")
    a = ap.parse_args()
    semillas = 1 if a.rapido else a.semillas
    t0 = time.time()
    lab, te, fs = construir()
    F = [c for b in BLOQUES for c in fs[b]]
    folds = [0] if a.rapido else sorted(lab.fold.unique())
    print(f"{lab.sample_id.nunique()} snapshots etiquetados, {te.sample_id.nunique()} de test · {len(F)} features (" +
          ", ".join(f"{b}={len(fs[b])}" for b in BLOQUES) + f") · {a.hilos} hilos · features en {time.time() - t0:.0f}s",
          flush=True)
    zo, zp = [], []
    for nb in a.configs:
        c = CONFIGS[nb]
        t1 = time.time()
        o = np.mean([oof_semilla(lab, F, nb, c, s, a.hilos, folds, cache=not a.rapido) for s in range(semillas)], axis=0)
        ok = ~np.isnan(o)
        cv = comun.hits(lab.loc[ok, ["sample_id", "row_index"]].assign(score=o[ok])).mean()
        print(f"  {nb}: {semillas} semilla(s), CV {'fold 0' if a.rapido else 'top-3'} {cv:.4f} ({time.time() - t1:.0f}s)",
              flush=True)
        zo.append(zscore(lab.sample_id.values, o))
        if not (a.rapido or a.solo_oof):
            p = np.mean([pred_semilla(lab, te, F, nb, c, s, a.hilos) for s in range(semillas)], axis=0)
            zp.append(zscore(te.sample_id.values, p))
    oof = np.mean(zo, axis=0)
    ok = ~np.isnan(oof)
    oof_df = lab.loc[ok, ["sample_id", "row_index"]].assign(score=oof[ok])
    m = comun.medidas(oof_df)
    print(f"\n{a.nombre}{' (rápido: solo el fold 0)' if a.rapido else ''}: " +
          " · ".join(f"{k} {v:.4f}" for k, v in m.items()) + f" · {(time.time() - t0) / 60:.1f} min")
    if a.rapido:
        return
    preds_df = None if a.solo_oof else te[["sample_id", "row_index"]].assign(score=np.mean(zp, axis=0))
    comun.guardar(a.nombre, oof=oof_df, preds=preds_df)
    print(f"guardado en {comun.SALIDAS}/(oof|preds|submissions) como {a.nombre}")


if __name__ == "__main__":
    main()

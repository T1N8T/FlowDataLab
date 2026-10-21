"""entrenar_futuro.py · línea "futuro" · entrena los modelos entregados N y C y la mezcla final M30_30.

    N  LightGBM LambdaRank (task_index categórica) con los bloques base, logic, past, future, clear e instr
       (features_futuro.py); 63 hojas, min_child_samples 100, extra_trees. Por fold de datos/folds.csv, media de las
       semillas; el modelo de test se entrena con todo train+validation. Encima, el segundo paso de sesión (sesion.py).
    C  N + oráculo de coordinaciones (oraculo.py, calculado por fold para la CV y con todo para test).
    M  mezcla en probabilidades dentro del snapshot de los dos (ya con segundo paso): p = 0.7·softmax(C) + 0.3·softmax(N).
       El peso de N (0.3) se eligió por CV anidada (4 folds -> 5.º): los 5 folds eligen 0.3; aquí se imprime para comprobarlo.

Guarda en entrega/salidas/{oof,preds,submissions} (formato común): N30, N30_ses, C30, C30_ses y M30_30 (con 30 semillas).
Reproduce futuro/pruebas/f11_lab.py (N y --orc), f11_promedio.py (30 semillas = 0–29) y f11_mezcla.py
(f11_N30_ses, f11_C30_ses, f11_M30_30).

Uso (desde la raíz del repo o desde entrega/):
    python entrega/codigo/entrenar_futuro.py N --semillas 30          # ~1 h con 14 hilos
    python entrega/codigo/entrenar_futuro.py C --semillas 30
    python entrega/codigo/entrenar_futuro.py mezcla                   # M30_30 = 0.7·C30_ses + 0.3·N30_ses
    python entrega/codigo/entrenar_futuro.py N --rapido               # comprobación: fold 0, 1 semilla, no guarda
Opciones: --hilos (14), --semilla0 (primera semilla, 0), --nombre (nombre de salida; por defecto N<semillas> / C<semillas>),
--peso (peso de N en la mezcla, 0.3).
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import K, hits, guardar, leer, lsm, medidas, labels  # noqa: E402
from features_futuro import construir, BLOQUES  # noqa: E402
from oraculo import oraculo  # noqa: E402
from sesion import sesion  # noqa: E402

# parámetros de N: los de lgb_future (PARAMS_FUT de futuro/pruebas/base_fut.py) + 63 hojas, min_child 100, extra_trees
PARAMS = dict(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, lambdarank_truncation_level=5,
              reg_lambda=0.0, extra_trees=True, verbose=-1)
W_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]     # pesos de N probados en la CV anidada


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Entrena N / C (línea futuro) o la mezcla M30_30.")
    ap.add_argument("modelo", choices=["N", "C", "mezcla"])
    ap.add_argument("--semillas", type=int, default=30)
    ap.add_argument("--semilla0", type=int, default=0)
    ap.add_argument("--hilos", type=int, default=14)
    ap.add_argument("--rapido", action="store_true", help="solo fold 0 y 1 semilla, sin test ni guardar (comprobación)")
    ap.add_argument("--nombre", default="")
    ap.add_argument("--peso", type=float, default=0.3, help="peso de N en la mezcla")
    return ap.parse_args(argv)


def ajustar(A, F, params, seed):
    """Un LGBMRanker (lambdarank) con task_index categórica; A ordenado por sample_id (grupos = snapshots)."""
    m = lgb.LGBMRanker(**{**params, "random_state": seed})
    m.fit(A[F], A.target_index, group=A.groupby("sample_id", sort=False).size().values,
          categorical_feature=["task_index"])
    return m


def con_oraculo(df, o):
    return pd.concat([df, o.reset_index(drop=True)], axis=1)


def resumen(nombre, m):
    txt = " · ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items())
    print(f"  {nombre}: {txt}", flush=True)


def entrenar(a):
    t0 = time.time()
    lab, te, fs = construir()
    F = [c for b in BLOQUES for c in fs[b]]
    O = None
    if a.modelo == "C":
        O = oraculo(lab, te)
        F = F + list(O["lab"].columns)
    params = {**PARAMS, "n_jobs": a.hilos}
    seeds = [a.semilla0] if a.rapido else list(range(a.semilla0, a.semilla0 + a.semillas))
    folds = [0] if a.rapido else sorted(lab.fold.unique())
    print(f"{a.modelo}: {len(F)} features, semillas {seeds[0]}–{seeds[-1]}, folds {folds}, {a.hilos} hilos", flush=True)
    oof = np.full(len(lab), np.nan)
    for f in folds:
        L = con_oraculo(lab, O[f]) if O is not None else lab
        tr, va = (L.fold != f).values, (L.fold == f).values
        oof[va] = np.mean([ajustar(L[tr], F, params, s).predict(L.loc[va, F]) for s in seeds], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    if a.rapido:
        ok = ~np.isnan(oof)
        h = hits(lab.loc[ok, K].assign(score=oof[ok]))
        print(f"RÁPIDO {a.modelo}: fold 0, 1 semilla, 1.er pase top-3 {h.mean():.4f} ({len(h)} snapshots; "
              f"no se guarda nada)  {time.time() - t0:.0f}s", flush=True)
        return
    L, T = (con_oraculo(lab, O["lab"]), con_oraculo(te, O["te"])) if O is not None else (lab, te)
    pred = np.mean([ajustar(L, F, params, s).predict(T[F]) for s in seeds], axis=0)
    print(f"  test ({time.time() - t0:.0f}s)", flush=True)
    so, sp, w = sesion(lab, oof, te, pred)
    nombre = a.nombre or f"{a.modelo}{a.semillas}" + (f"o{a.semilla0}" if a.semilla0 else "")
    for nm, o, p in ((nombre, oof, pred), (nombre + "_ses", so, sp)):
        guardar(nm, lab[K].assign(score=o), te[K].assign(score=p))
    print(f"pesos del segundo paso: {w}")
    ref = leer(f"N{a.semillas}_ses") if a.modelo == "C" else None
    for nm in (nombre, nombre + "_ses"):
        m = medidas(leer(nm), ref if nm.endswith("_ses") else None)
        print(f"{nm}: CV {m['cv']:.4f}" + (f"  (frente a N{a.semillas}_ses)" if ref is not None and nm.endswith("_ses") else ""))
        resumen(nm, m)
    print(f"guardado {nombre} y {nombre}_ses en entrega/salidas  ({time.time() - t0:.0f}s)", flush=True)


def probs(df):
    """softmax del score dentro de cada snapshot."""
    return np.exp(lsm(df.sample_id.values, df.score.values))


def mezclar(a, b, w):
    """(1 − w)·softmax(a) + w·softmax(b) dentro del snapshot; a y b en formato común. Devuelve el formato común,
    ordenado por sample_id, row_index."""
    a = a.sort_values(K).reset_index(drop=True)
    b = a[K].merge(b, on=K, how="left")
    assert not b.score.isna().any(), "los dos modelos no tienen las mismas filas"
    return a[K].assign(score=(1 - w) * probs(a) + w * probs(b))


def mezcla(a):
    n = a.semillas
    nc, nn = f"C{n}_ses", f"N{n}_ses"
    oc, on_ = leer(nc), leer(nn)
    pc, pn = leer(nc, "preds"), leer(nn, "preds")
    assert all(x is not None for x in (oc, on_, pc, pn)), f"faltan {nc} / {nn}: entrenar antes C y N con --semillas {n}"
    # CV anidada del peso (informativa): en cada fold, el peso que mejor va en los otros 4
    H = pd.DataFrame({w: hits(mezclar(oc, on_, w)) for w in W_GRID})
    fold = labels().fold.reindex(H.index)
    el = [max(W_GRID, key=lambda w: H.loc[fold != f, w].mean()) for f in range(5)]
    anidada = sum(H.loc[fold == f, wf].sum() for f, wf in enumerate(el)) / len(H)
    print(f"CV anidada {anidada:.4f}; peso de N elegido por fold {el}; C solo {H[0.0].mean():.4f}, N solo {H[1.0].mean():.4f}")
    print("  " + " · ".join(f"w={w:.2f}: {H[w].mean():.4f}" for w in W_GRID))
    nombre = a.nombre or f"M{n}_{round(100 * a.peso)}"
    oof, pred = mezclar(oc, on_, a.peso), mezclar(pc, pn, a.peso)
    guardar(nombre, oof, pred)
    m = medidas(oof, on_)
    print(f"{nombre} = {1 - a.peso:.2f}·{nc} + {a.peso:.2f}·{nn}: CV {m['cv']:.4f} (frente a {nn})")
    resumen(nombre, m)


if __name__ == "__main__":
    args = parse()
    mezcla(args) if args.modelo == "mezcla" else entrenar(args)

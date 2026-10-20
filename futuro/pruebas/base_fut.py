"""
base_fut.py · Juanma (C) · Base común de los hilos de pruebas CON FUTURO (rompe la regla 1 a propósito).

- Datos y features de lgb_future (futuro/features_future.py: bloques base, logic, past, future; CV 0.901 con 3 semillas).
  Se calculan una vez y se guardan en futuro/pruebas/cache.parquet (lo comparten todos los hilos).
- Grupos de Adri (pruebas/base.py: alerta, contexto, exit_coord, deadline...) disponibles con add_groups().
- Salidas en futuro/pruebas/oof|preds/<nombre>.csv y registro en futuro/pruebas/resultados.jsonl
  (fuera de oof/ y de futuro/oof: no entran en ningún stacking salvo que se pida).

Uso:
    import sys; sys.path.insert(0, "futuro/pruebas")
    from base_fut import datos, PARAMS_FUT, lgb_cv, add_groups, guardar, registrar
    lab, te, fs = datos()                                  # fs: {"base","logic","past","future"} -> columnas
    F = fs["base"] + fs["logic"] + fs["past"] + fs["future"]
    oof, pred = lgb_cv(lab, F, seeds=(0,), test=te)
    guardar("f2_ref", lab, oof, te, pred); registrar("f2_ref", "f2_lgb", "...", n_features=len(F))
"""
import sys, time
from functools import partial
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "futuro" / "pruebas"
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "pruebas")]
import features_future as ff  # noqa: E402
import base  # noqa: E402

CACHE = OUT / "cache.parquet"
# parámetros de lgb_future (futuro/m_lgb_future.py), con 3 hilos: hay varios hilos de pruebas a la vez
PARAMS_FUT = dict(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, lambdarank_truncation_level=5,
                  reg_lambda=0.0, verbose=-1, n_jobs=3)
TYPES = base.TYPES
guardar = partial(base.guardar, out=OUT)
registrar = partial(base.registrar, out=OUT)
lgb_fit = base.lgb_fit


def datos():
    """(lab, te, fs). lab/te ordenados por sample_id, row_index; lab.fold de Data/folds.csv, te.fold = -1."""
    if CACHE.exists():
        d = pd.read_parquet(CACHE)
    else:
        t0 = time.time()
        d = ff.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(f".{time.time_ns()}.tmp")   # escritura atómica: otro hilo puede estar leyendo
        d.to_parquet(tmp); tmp.rename(CACHE)
        print(f"features de futuro calculadas en {time.time() - t0:.0f}s -> {CACHE}", flush=True)
    fs = ff.feature_sets(d)
    lab = d[d.split != "test"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True).assign(fold=-1)
    assert (lab.groupby("sample_id").target_index.sum() == 1).all()
    return lab, te, fs


def add_groups(df, groups):
    """Añade grupos de features de Adri (pruebas/base.py FEATURE_GROUPS) -> (df, columnas nuevas)."""
    new = [base.FEATURE_GROUPS[g](df) for g in groups]
    cols = [c for n in new for c in n.columns]
    return pd.concat([df] + new, axis=1), cols


def lgb_cv(d, F, params=None, seeds=(0,), test=None):
    """CV con Data/folds.csv y parámetros de lgb_future. Devuelve (oof, pred_test o None)."""
    return base.lgb_cv(d, F, {**PARAMS_FUT, **(params or {})}, seeds=seeds, test=test)

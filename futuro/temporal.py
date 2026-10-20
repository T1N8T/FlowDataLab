"""
temporal.py · Juanma (C) · RAMA "futuro" · Columnas de otros snapshots de la sesión para cualquier modelo.

!! ROMPE LA REGLA 1 A PROPÓSITO: solo para los modelos *_future, que guardan en futuro/. !!
Saca de futuro/features_future.py los bloques
- f_* (futuro): siguiente input al avión y su tipo, si encaja con la tarea, cuándo desaparece la tarea/avión...
- p_* (pasado) + blk_pos, blk_len, gap_prev, gap_next: edad en la lista, deltas con el snapshot anterior, racha.
No usan etiquetas. Los folds siguen siendo Data/folds.csv, así que la CV es comparable con la de main.

Uso:  from temporal import add_temporal, temporal_cols
      d = add_temporal(d)            # merge por (sample_id, row_index), mismo orden y nº de filas
"""
import sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "futuro"))
import features_future
_CACHE = None

EXTRA = ["blk_pos", "blk_len", "gap_prev", "gap_next"]


def temporal_cols(d):
    return [c for c in d.columns if c.startswith(("f_", "p_"))] + EXTRA


def add_temporal(d):
    global _CACHE
    if _CACHE is None:
        _CACHE = features_future.build()
    t = _CACHE
    cols = temporal_cols(t)
    t = t[["sample_id", "row_index"] + cols]
    t[cols] = t[cols].astype(np.float32)
    n, idx = len(d), d.index
    out = d.merge(t, on=["sample_id", "row_index"], how="left")
    assert len(out) == n and out[cols[0]].notna().any(), "merge de columnas temporales"
    out.index = idx
    print(f"+{len(cols)} columnas temporales (RAMA FUTURO)", flush=True)
    return out

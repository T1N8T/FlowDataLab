"""
mix.py · mezclas de OOF guardados (media ponderada de z-scores dentro del snapshot).
    python realista/lab/mix.py b045 sm_b045 xgb1             # pesos iguales; cada modelo = media de sus semillas
    python realista/lab/mix.py b045:1 sm_b045:0.5 noche:1    # 'noche' = realista/oof/real_adri_noche.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import lab as L  # noqa: E402

_LAB = None


def lab():
    global _LAB
    if _LAB is None:
        d = pd.read_pickle(L.OUT / "cache_features.pkl")
        _LAB = d[d.split != "test"][["sample_id", "row_index", "atco", "fold", "session", "target_index"]] \
            .reset_index(drop=True)
    return _LAB


def model_z(name, seeds=None):
    lb = lab()
    if name == "noche":
        return L.zscore(L.ref_noche(lb), lb.sample_id.values)
    fs = sorted((L.OUT / "oof").glob(f"{name}__s*.npy"))
    if seeds is not None:
        fs = [f for f in fs if int(f.stem.split("__s")[1]) in seeds]
    assert fs, name
    return np.mean([L.zscore(np.load(f), lb.sample_id.values) for f in fs], axis=0)


def mezcla(spec):
    parts = [(s.split(":")[0], float(s.split(":")[1]) if ":" in s else 1.0) for s in spec]
    return sum(w * model_z(n) for n, w in parts) / sum(w for _, w in parts)


if __name__ == "__main__":
    lb = lab()
    ref = model_z("noche")
    for s in sys.argv[1:]:
        print(L.linea(s, L.medir(lb, mezcla(s.split("+")), ref)))

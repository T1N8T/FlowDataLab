"""build_cache.py · calcula una vez las features realistas (features_real.build) -> models/real_lab/cache_features.pkl"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "real_lab"
sys.path[:0] = [str(ROOT / "realista"), str(ROOT / "futuro"), str(ROOT / "cv")]
import features_real as fr
t0 = time.time()
d = fr.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
f64 = [c for c in d.columns if d[c].dtype == "float64" and c != "t"]
d[f64] = d[f64].astype("float32")
OUT.mkdir(parents=True, exist_ok=True)
d.to_pickle(OUT / "cache_features.pkl")
fs = fr.feature_sets(d)
print({k: len(v) for k, v in fs.items()}, d.shape, f"{time.time()-t0:.0f}s", flush=True)

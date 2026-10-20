"""h2_mezcla.py · media de rangos (dentro del snapshot) de OOF de variantes h2. Uso: h2_mezcla.py nombre a b [c...]"""
import sys
import pandas as pd
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/pruebas")
from base import OUT, registrar

nombre, partes = sys.argv[1], sys.argv[2:]
dfs = [pd.read_csv(OUT / "oof" / f"{p}.csv") for p in partes]
m = dfs[0][["sample_id", "row_index"]].copy()
m["score"] = sum(df.groupby("sample_id").score.rank(pct=True) for df in dfs) / len(dfs)
registrar(nombre, hilo="h2_lgb", descripcion="media de rangos OOF: " + " + ".join(partes), oof_df=m)

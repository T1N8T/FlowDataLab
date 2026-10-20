"""
folds.py · común · Genera Data/folds.csv (columnas atco,fold) para GroupKFold de 5 folds por controlador.

- Cada controlador va entero a un fold, así que cada sesión también (una sesión = un controlador × un escenario).
- Reparto determinista y equilibrado por nº de muestras: controladores de mayor a menor, cada uno al fold con menos
  muestras hasta el momento. No depende de semillas: todos obtienen el mismo archivo.
Uso:  python src/folds.py
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "Data"
N_FOLDS = 5


def make_folds():
    d = pd.concat([pd.read_csv(DATA / f"{f}.csv", usecols=["sample_id", "session", "atco"])
                   for f in ("train", "validation")]).drop_duplicates("sample_id")
    assert d.groupby("session").atco.nunique().max() == 1, "una sesión con varios controladores"
    n = d.groupby("atco").size().sort_index().sort_values(ascending=False, kind="stable")
    load, fold = [0] * N_FOLDS, {}
    for atco, k in n.items():
        f = min(range(N_FOLDS), key=lambda i: (load[i], i))
        fold[atco] = f; load[f] += k
    out = pd.DataFrame({"atco": list(fold), "fold": list(fold.values())}).sort_values("atco")
    out.to_csv(DATA / "folds.csv", index=False)
    summary = d.assign(fold=d.atco.map(fold)).groupby("fold").agg(muestras=("sample_id", "size"),
                                                                  atcos=("atco", lambda s: sorted(s.unique())))
    print(summary.to_string())
    print("Escrito", DATA / "folds.csv")


if __name__ == "__main__":
    make_folds()

"""comun.py · rutas, datos, métrica top-3, formato de salidas y medidas robustas. Lo usan las dos líneas (futuro y realista).

Todo es relativo a la carpeta entrega/: datos en entrega/datos, salidas en entrega/salidas/{oof,preds,submissions}, cachés en
entrega/cache. Formato común de salidas: sample_id,row_index,score (score alto = más probable).
"""
from pathlib import Path
import numpy as np, pandas as pd

ENTREGA = Path(__file__).resolve().parents[1]
DATOS, SALIDAS, CACHE = ENTREGA / "datos", ENTREGA / "salidas", ENTREGA / "cache"
K = ["sample_id", "row_index"]
TIPOS = ['entry_conflict_resolution', 'exit_conflict_resolution', 'entry_coordination', 'exit_coordination',
         'assume_aircraft', 'transfer_aircraft', 'conflict_resolution', 'non_conformance_resolution', 'qos_improvement',
         'entry_conditions', 'exit_conditions', 'aircraft_request', 'zone_conflict', 'return_to_route']
CV023 = [2, 11, 10, 6, 17, 15]          # controladores de la CV más parecidos al 023 (privado) por variables sin etiqueta
_LABELS = None


def labels():
    """Una fila por snapshot etiquetado (train + validation): sample_id, target (row_index correcto), atco, session, scenario, fold."""
    global _LABELS
    if _LABELS is None:
        d = pd.concat([pd.read_csv(DATOS / f"{f}.csv", usecols=["sample_id", "row_index", "target_index", "atco", "session", "scenario"])
                       for f in ("train", "validation")])
        lab = d[d.target_index == 1].rename(columns={"row_index": "target"}).drop(columns="target_index")
        assert lab.sample_id.is_unique and lab.sample_id.nunique() == d.sample_id.nunique(), "≠ 1 etiqueta por snapshot"
        folds = pd.read_csv(DATOS / "folds.csv")
        lab["fold"] = lab.atco.map(dict(zip(folds.atco, folds.fold)))
        _LABELS = lab.set_index("sample_id")
    return _LABELS


def topk(df, k=3):
    """sample_id -> lista de los k row_index con mayor score (empates: row_index ascendente)."""
    x = df.sort_values(["sample_id", "score", "row_index"], ascending=[True, False, True])
    return x.groupby("sample_id", sort=False).row_index.apply(lambda r: list(r.iloc[:k]))


def hits(df, k=3):
    """Serie sample_id -> 1/0: ¿la tarea correcta está entre las k primeras? (solo snapshots etiquetados)."""
    lab = labels()
    p = topk(df, k)
    p = p[p.index.isin(lab.index)]
    return pd.Series([int(t in r) for t, r in zip(lab.target.loc[p.index], p)], index=p.index)


def to_submission(df):
    p = topk(df)
    return pd.DataFrame({"sample_id": p.index, "prediction": [" ".join(map(str, r)) for r in p]})


def guardar(nombre, oof=None, preds=None):
    """salidas/oof/<nombre>.csv, salidas/preds/<nombre>.csv y salidas/submissions/sub_<nombre>.csv."""
    for sub, df in (("oof", oof), ("preds", preds)):
        if df is not None:
            (SALIDAS / sub).mkdir(parents=True, exist_ok=True)
            df[K + ["score"]].to_csv(SALIDAS / sub / f"{nombre}.csv", index=False)
    if preds is not None:
        (SALIDAS / "submissions").mkdir(parents=True, exist_ok=True)
        to_submission(preds).to_csv(SALIDAS / "submissions" / f"sub_{nombre}.csv", index=False)


def leer(nombre, carpeta="oof"):
    p = SALIDAS / carpeta / f"{nombre}.csv"
    return pd.read_csv(p) if p.exists() else None


def lsm(sample_id, score):
    """log-softmax del score dentro de cada snapshot."""
    s = pd.Series(np.asarray(score, dtype="float64"))
    g = s.groupby(np.asarray(sample_id))
    sh = s - g.transform("max")
    return (sh - np.log(np.exp(sh).groupby(np.asarray(sample_id)).transform("sum"))).values


def medidas(oof, ref=None):
    """CV y medidas robustas de un OOF (y diferencia frente a una referencia): peor controlador, mediana por sesión,
    CV 023, por escenario, sesiones y controladores que suben/bajan."""
    lab = labels()
    h = hits(oof)
    L = lab.loc[h.index]
    m = {"cv": h.mean(), "peor_controlador": h.groupby(L.atco).mean().min(),
         "mediana_sesion": h.groupby(L.session).mean().median(), "cv_023": h[L.atco.isin(CV023)].mean()}
    for e, v in h.groupby(L.scenario).mean().items():
        m[f"escenario_{e}"] = v
    if ref is not None:
        r = hits(ref).reindex(h.index)
        ds, da = (h - r).groupby(L.session).mean(), (h - r).groupby(L.atco).mean()
        m.update(delta=(h - r).mean(), sesiones_suben=int((ds > 0).sum()), sesiones_bajan=int((ds < 0).sum()),
                 controladores_suben=int((da > 0).sum()), controladores_bajan=int((da < 0).sum()))
    return m

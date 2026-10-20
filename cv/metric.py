"""
metric.py · común · Top-3 hit rate y formato común de salidas (oof/, preds/, submissions/).

Entrada estándar: un DataFrame con columnas sample_id,row_index,score (score alto = más probable).
Empates dentro de un snapshot: se rompen por row_index ascendente (determinista).

Uso desde un modelo (desde la raíz del repo):
    import sys; sys.path.insert(0, "cv")
    from metric import top3, report, save_outputs
    report(oof_df)                               # CV global, por fold y por controlador
    save_outputs("lgb_rank", oof_df, preds_df)   # oof/lgb_rank.csv, preds/lgb_rank.csv, submissions/sub_lgb_rank.csv

Uso por línea de comandos:
    python cv/metric.py oof/lgb_rank.csv [oof/otro.csv ...]   # CV de uno o varios OOF
"""
import sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
K = 3
_LABELS = None


def labels():
    """Una fila por muestra etiquetada (train+validation): sample_id, target (row_index correcto), atco, fold."""
    global _LABELS
    if _LABELS is None:
        d = pd.concat([pd.read_csv(DATA / f"{f}.csv", usecols=["sample_id", "row_index", "target_index", "atco"])
                       for f in ("train", "validation")])
        lab = d[d.target_index == 1].rename(columns={"row_index": "target"})[["sample_id", "target", "atco"]]
        assert lab.sample_id.is_unique and lab.sample_id.nunique() == d.sample_id.nunique(), "≠ 1 etiqueta por muestra"
        folds = pd.read_csv(DATA / "folds.csv")
        lab["fold"] = lab.atco.map(dict(zip(folds.atco, folds.fold)))
        _LABELS = lab.reset_index(drop=True)
    return _LABELS


def topk(df, k=K):
    """sample_id -> lista de los k row_index con mayor score (el mejor primero)."""
    x = df.sort_values(["sample_id", "score", "row_index"], ascending=[True, False, True])
    return x.groupby("sample_id", sort=False).row_index.apply(lambda r: list(r.iloc[:k]))


def hits(df, k=K):
    """Serie sample_id -> 1/0 (acierto top-k) para las muestras etiquetadas presentes en df."""
    lab = labels().set_index("sample_id")
    pred = topk(df, k)
    pred = pred[pred.index.isin(lab.index)]
    return pd.Series([int(t in p) for t, p in zip(lab.target.loc[pred.index], pred)], index=pred.index)


def top3(df):
    """Top-3 hit rate (media por muestra, como el scorer oficial)."""
    return float(hits(df).mean())


def report(df, name="", verbose=True):
    """Imprime CV top-3 global, por fold y por controlador. Devuelve la CV global."""
    h = hits(df).rename("hit").reset_index().merge(labels(), on="sample_id")
    cv = h.hit.mean()
    by_fold = h.groupby("fold").hit.mean()
    by_atco = h.groupby("atco").hit.mean()
    print(f"{name:30s} CV top3 = {cv:.4f}  ({len(h)} muestras)  "
          f"min atco {by_atco.min():.3f}  media por atco {by_atco.mean():.3f}", flush=True)
    if verbose:
        print("  por fold: " + "  ".join(f"{f}:{v:.3f}" for f, v in by_fold.items()))
        print("  por atco: " + "  ".join(f"{a}:{v:.3f}" for a, v in by_atco.items()))
    return cv


def to_submission(df):
    """sample_id,prediction con 3 row_index separados por espacios, el mejor primero."""
    p = topk(df)
    return pd.DataFrame({"sample_id": p.index, "prediction": [" ".join(map(str, r)) for r in p]})


def _check(df, what):
    assert list(df.columns[:3]) == ["sample_id", "row_index", "score"], f"{what}: columnas sample_id,row_index,score"
    assert not df.duplicated(["sample_id", "row_index"]).any(), f"{what}: filas duplicadas"
    assert df.score.notna().all(), f"{what}: score con NaN"


def save_outputs(name, oof=None, preds=None, out=ROOT):
    """Guarda oof/<name>.csv, preds/<name>.csv y submissions/sub_<name>.csv en el formato común (dentro de out;
    futuro/ usa out=ROOT / "futuro" para que el stacking de main no coja esos modelos)."""
    if oof is not None:
        _check(oof, "oof")
        (out / "oof").mkdir(exist_ok=True, parents=True)
        oof[["sample_id", "row_index", "score"]].to_csv(out / "oof" / f"{name}.csv", index=False)
    if preds is not None:
        _check(preds, "preds")
        n_test = pd.read_csv(DATA / "test.csv", usecols=["sample_id"]).sample_id.nunique()
        assert preds.sample_id.nunique() == n_test, f"preds: {preds.sample_id.nunique()} muestras, test tiene {n_test}"
        for sub in ("preds", "submissions"):
            (out / sub).mkdir(exist_ok=True, parents=True)
        preds[["sample_id", "row_index", "score"]].to_csv(out / "preds" / f"{name}.csv", index=False)
        to_submission(preds).to_csv(out / "submissions" / f"sub_{name}.csv", index=False)
    print("Guardado:", name, "(oof)" if oof is not None else "", "(preds + sub)" if preds is not None else "")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        report(pd.read_csv(path), Path(path).stem)

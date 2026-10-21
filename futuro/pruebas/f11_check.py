"""f11_check.py · comprueba un envío (1863 snapshots, 3 row_index distintos y existentes) y lo compara en test con otro:
% de snapshots con el mismo top-3 y % con alguna entry_coordination en el top-3, por sesión (H1 de la sección 21b).
Uso: python futuro/pruebas/f11_check.py envio.csv [referencia.csv]"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
te = pd.read_csv(ROOT / "Data" / "test.csv", usecols=["sample_id", "row_index", "task_index", "session"])


def leer(p):
    s = pd.read_csv(p)
    assert len(s) == te.sample_id.nunique() and s.sample_id.is_unique and set(s.sample_id) == set(te.sample_id), p
    s["r"] = s.prediction.str.split().apply(lambda x: [int(v) for v in x])
    assert (s.r.apply(len) == 3).all() and (s.r.apply(lambda x: len(set(x))) == 3).all(), p
    ok = set(zip(te.sample_id, te.row_index))
    assert all((sid, r) in ok for sid, rr in zip(s.sample_id, s.r) for r in rr), p
    return s.set_index("sample_id").r


def entry_top3(r):
    ti = te.set_index(["sample_id", "row_index"]).task_index
    ses = te.drop_duplicates("sample_id").set_index("sample_id").session
    e = pd.Series([any(ti.loc[(sid, x)] == 2 for x in rr) for sid, rr in r.items()], index=r.index)
    return e.groupby(ses.reindex(r.index)).mean()


if __name__ == "__main__":
    a = leer(sys.argv[1])
    print(f"{Path(sys.argv[1]).name}: válido (1863 snapshots, 3 row_index distintos y existentes)")
    out = pd.DataFrame({"entry_coord en top-3": entry_top3(a)})
    if len(sys.argv) > 2:
        b = leer(sys.argv[2])
        ses = te.drop_duplicates("sample_id").set_index("sample_id").session
        out["mismo top-3 que ref"] = pd.Series([set(a[s]) == set(b[s]) for s in a.index], index=a.index).groupby(ses.reindex(a.index)).mean()
        out["ref: entry_coord en top-3"] = entry_top3(b)
    print(out.round(3).to_string())

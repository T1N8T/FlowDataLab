"""f11_gate.py · Pablo (A) · hilo f11 · segundo paso con un «experto en priors» y puerta por falta de huella (sin reentrenar).

El 76 % de los fallos de N están en snapshots cuya respuesta no deja huella (acierto 0.685 frente a 0.978), y el 023 tiene
más snapshots sin instrucción compatible inminente (023sc1: 45 % frente al 59 % de la CV). Ahí N decide con ruido del futuro.
Idea: el logit de sesión recibe, además de sus 9 features sobre N, la log-probabilidad de un modelo SIN futuro
(realista/real_todo: snapshot + pasado) multiplicada por indicadores de «no hay huella en el snapshot» (puertas), para que
solo pese donde N no tiene evidencia. Pesos por fold (los otros 4); para test, con todo el OOF.
Uso: python futuro/pruebas/f11_gate.py NOMBRE base experto [puertas]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import feat_session as FS, suavizado as S  # noqa: E402


def leer(m, carpeta):
    p = S.leer(m, carpeta)
    if p is None:
        q = ROOT / "realista" / carpeta / f"{m}.csv"
        p = pd.read_csv(q) if q.exists() else None
    return p


def puertas(d, base_score, cuales):
    """Indicadores por fila (constantes en el snapshot) de que N no tiene evidencia."""
    g = {}
    sid = d.sample_id.values
    mn = pd.Series(d.si_match.values).groupby(sid).transform("min").values
    end = pd.Series(d.f_task_end.values).groupby(sid).transform("min").values
    p = np.exp(S.lsm(d, base_score))
    pmax = pd.Series(p).groupby(sid).transform("max").values
    g["sinmatch30"] = ~(mn <= 30)
    g["sinmatch60"] = ~(mn <= 60)
    g["sinfin10"] = ~(end <= 10)
    g["duda"] = 1 - pmax
    return np.column_stack([g[c].astype(float) for c in cuales])


def X_de(d, base, exp, cuales):
    Xb = FS.session_feats(d, base).values
    le = S.lsm(d, exp)
    G = puertas(d, base, cuales)
    return np.hstack([Xb, le[:, None], le[:, None] * G])


def run(nombre, mb, me, cuales):
    lab, te, _ = datos_instr()
    k = ["sample_id", "row_index"]
    ob = lab[k].merge(leer(mb, "oof"), on=k, how="left").score.values
    oe = lab[k].merge(leer(me, "oof"), on=k, how="left").score.values
    assert not np.isnan(ob).any() and not np.isnan(oe).any()
    X = X_de(lab, ob, oe, cuales)
    out = np.full(len(lab), np.nan)
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        out[va] = X[va] @ FS.fit_logit(X[tr], lab.target_index.values[tr], lab.sample_id.values[tr])
    w = FS.fit_logit(X, lab.target_index.values, lab.sample_id.values)
    pb, pe = leer(mb, "preds"), leer(me, "preds")
    sp = None
    if pb is not None and pe is not None:
        tb = te[k].merge(pb, on=k, how="left").score.values
        tq = te[k].merge(pe, on=k, how="left").score.values
        sp = X_de(te, tb, tq, cuales) @ w
    print(nombre, "pesos:", dict(zip(FS.COLS + ["lsm_exp"] + [f"lsm_exp*{c}" for c in cuales], np.round(w, 2))))
    guardar(nombre, lab, out, te if sp is not None else None, sp)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else ["sinmatch30"])

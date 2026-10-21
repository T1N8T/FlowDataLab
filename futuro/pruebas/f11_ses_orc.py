"""f11_ses_orc.py · Pablo (A) · hilo f11 · el oráculo de coordinaciones SOLO en el segundo paso (prior en log con pesos
aprendidos), con el primer pase de N sin oráculo. En C el oráculo entra en el primer pase y puede vetar una coordinación
aunque la evidencia propia sea fuerte (006sc2); aquí su fuerza es un coeficiente del logit, ajustado por fold.
Para el fold f se usan las features del oráculo de la versión f (fuentes de folds != f) en entrenamiento y validación,
como en el primer pase de C; para test, la versión final. Uso: python futuro/pruebas/f11_ses_orc.py NOMBRE modelo_1er_pase"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import OUT, guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import feat_session as FS, suavizado as S, f11_lab  # noqa: E402
from metric import save_outputs  # noqa: E402


def orc_X(O, coord):
    """Prior del oráculo: log de las fracciones (suavizadas) en las filas de coordinación; 0 en las demás."""
    cols = []
    for c in ["or_ever", "or_near60", "or_near180", "or_cs_near60"]:
        v = np.log(O[c].fillna(0).values + 0.05)
        cols.append(np.where(coord, v, 0.0))
    cols.append(np.where(coord, O["or_dt_abs"].fillna(3600).clip(upper=3600).values / 600, 0.0))
    return np.column_stack(cols)


if __name__ == "__main__":
    nombre, modelo = sys.argv[1], sys.argv[2]
    lab, te, _ = datos_instr()
    X, ocols = f11_lab.orc_cache(lab, te, "coord")
    k = ["sample_id", "row_index"]
    o = lab[k].merge(S.leer(modelo, "oof"), on=k, how="left").score.values
    p = te[k].merge(S.leer(modelo, "preds"), on=k, how="left").score.values
    Xs, Xt = FS.session_feats(lab, o).values, FS.session_feats(te, p).values
    cl, ct = lab.task_index.isin([2, 3]).values, te.task_index.isin([2, 3]).values
    n = len(lab)
    out = np.full(n, np.nan)
    y, g = lab.target_index.values, lab.sample_id.values
    for f in range(5):
        O = pd.DataFrame({c: X[f"{c}__f{f}"].values[:n] for c in ocols})
        Z = np.hstack([Xs, orc_X(O, cl)])
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        w = FS.fit_logit(Z[tr], y[tr], g[tr])
        out[va] = Z[va] @ w
    Of = pd.DataFrame({c: X[f"{c}__full"].values for c in ocols})
    Zl, Zt = np.hstack([Xs, orc_X(Of.iloc[:n], cl)]), np.hstack([Xt, orc_X(Of.iloc[n:].reset_index(drop=True), ct)])
    w = FS.fit_logit(Zl, y, g)
    print("pesos:", dict(zip(FS.COLS + ["l_ever", "l_near60", "l_near180", "l_cs60", "dt"], np.round(w, 2))))
    sp = Zt @ w
    guardar(nombre, lab, out, te, sp)
    save_outputs(nombre, None, te[k].assign(score=sp), out=OUT)

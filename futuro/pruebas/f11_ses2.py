"""f11_ses2.py · Pablo (A) · hilo f11 · segundo paso de sesión con VARIOS primeros pases a la vez (sin reentrenar).

Las secciones 13c y 15 mezclaban modelos antes o después del segundo paso (media ponderada). Aquí el logit condicional
recibe las 9 features de sesión de cada primer pase (p. ej. N y N + oráculo) y aprende cuánto pesa cada uno. Pesos por
fold (los otros 4) como en feat_session.sesion; para test, con todo el OOF.
Uso: python futuro/pruebas/f11_ses2.py NOMBRE modelo1 modelo2 [...]   (OOF/preds de futuro/pruebas/ o futuro/)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import OUT, guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import feat_session as FS, suavizado as S  # noqa: E402


def sesion_multi(lab, oofs, te=None, preds=None):
    X = np.hstack([FS.session_feats(lab, o).values for o in oofs])
    out = np.full(len(lab), np.nan)
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        out[va] = X[va] @ FS.fit_logit(X[tr], lab.target_index.values[tr], lab.sample_id.values[tr])
    w = FS.fit_logit(X, lab.target_index.values, lab.sample_id.values)
    p = np.hstack([FS.session_feats(te, q).values for q in preds]) @ w if te is not None else None
    return out, p, w


if __name__ == "__main__":
    nombre, modelos = sys.argv[1], sys.argv[2:]
    lab, te, _ = datos_instr()
    k = ["sample_id", "row_index"]
    oofs = [lab[k].merge(S.leer(m, "oof"), on=k, how="left").score.values for m in modelos]
    pr = [S.leer(m, "preds") for m in modelos]
    has_te = all(p is not None for p in pr)
    preds = [te[k].merge(p, on=k, how="left").score.values for p in pr] if has_te else None
    so, sp, w = sesion_multi(lab, oofs, te if has_te else None, preds)
    print("pesos:", np.round(w, 2))
    guardar(nombre, lab, so, te if has_te else None, sp)

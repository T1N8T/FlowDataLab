"""f11_promedio.py · Pablo (A) · hilo f11 · media de los PRIMEROS pases de varias ejecuciones (p. ej. dos tandas de 10
semillas con --seed0 0 y 10) y segundo paso de sesión encima -> NOMBRE y NOMBRE_ses (oof, preds y envío).
Uso: python futuro/pruebas/f11_promedio.py NOMBRE run1 run2 [...]     (runs = nombres del 1.er pase, sin _ses)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import OUT, guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import feat_session, suavizado as S  # noqa: E402
from metric import save_outputs  # noqa: E402

if __name__ == "__main__":
    nombre, runs = sys.argv[1], sys.argv[2:]
    lab, te, _ = datos_instr()
    k = ["sample_id", "row_index"]
    oof = np.mean([lab[k].merge(S.leer(r, "oof"), on=k, how="left").score.values for r in runs], axis=0)
    pred = np.mean([te[k].merge(S.leer(r, "preds"), on=k, how="left").score.values for r in runs], axis=0)
    assert not np.isnan(oof).any() and not np.isnan(pred).any()
    so, sp, w = feat_session.sesion(lab, oof, te, pred)
    guardar(nombre, lab, oof, te, pred)
    guardar(nombre + "_ses", lab, so, te, sp)
    save_outputs(nombre + "_ses", None, te[k].assign(score=sp), out=OUT)
    print(nombre, "pesos", w)

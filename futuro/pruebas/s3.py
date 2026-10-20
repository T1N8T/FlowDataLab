"""
s3.py · Juanma (C) · Repite una prueba de futuro con las semillas 0, 1 y 2, como lgb_future (CV 0.9010), para
compararla en igualdad. Reaprovecha el OOF de la semilla 0 si ya existe (futuro/pruebas/oof/<prueba>.csv) y entrena
solo las semillas 1 y 2; la media de las tres es la misma que entrenar las tres desde cero. Siempre genera preds de test.

Salida: <prueba>_s3 en futuro/pruebas/oof|preds y una línea en futuro/pruebas/resultados.jsonl (sale en el panel).

Uso:  python futuro/pruebas/s3.py f2_adri [f1_otra ...]
Para añadir una prueba: una entrada en PRUEBAS con sus grupos de Adri extra, columnas a quitar y parámetros.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, datos, add_groups, lgb_cv, lgb_fit, PARAMS_FUT, guardar, registrar  # noqa: E402

# nombre -> (grupos de Adri a añadir, columnas a quitar, parámetros que cambian respecto a PARAMS_FUT)
PRUEBAS = {
    "f1_ref": ([], [], {}),
    "f2_adri": (["alerta", "contexto"], [], {}),
}
SEMILLAS = (0, 1, 2)


def main(nombre):
    grupos, quitar, params = PRUEBAS[nombre]
    t0 = time.time()
    lab, te, fs = datos()
    F = fs["base"] + fs["logic"] + fs["past"] + fs["future"]
    lab, extra = add_groups(lab, grupos)
    te, _ = add_groups(te, grupos)
    F = [c for c in F + extra if c not in quitar]

    previo = OUT / "oof" / f"{nombre}.csv"
    if previo.exists() and params.get("objective", "lambdarank") != "binary":
        s0 = pd.read_csv(previo)
        assert (s0.sample_id.values == lab.sample_id.values).all() and (s0.row_index.values == lab.row_index.values).all()
        oof12, _ = lgb_cv(lab, F, params, seeds=(1, 2))
        oof = (s0.score.values + 2 * oof12) / 3
        print(f"{nombre}: semilla 0 reaprovechada de {previo.name}", flush=True)
    else:
        oof, _ = lgb_cv(lab, F, params, seeds=SEMILLAS)
    p = {**PARAMS_FUT, **params}
    pred = np.mean([lgb_fit(lab, F, {**p, "random_state": s}).predict(te[F]) for s in SEMILLAS], axis=0)

    out = f"{nombre}_s3"
    guardar(out, lab, oof, te, pred)
    registrar(out, hilo="s3", descripcion=f"{nombre} con semillas 0,1,2 (comparable con lgb_future 0.9010)",
              segundos=time.time() - t0, n_features=len(F))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)

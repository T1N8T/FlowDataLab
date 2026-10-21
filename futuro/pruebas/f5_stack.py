"""f5_stack.py · stack de futuro + suavizado temporal con CV anidada (hilo f5_stack).
- stack: stacking/stack.py (media ponderada con CV anidada y meta-ranker) sobre los modelos pedidos, buscados en
  futuro/ y futuro/pruebas/ (como la sección 4 de panel_pruebas.ipynb). Guarda stackf_<tag>_avg / stackf_<tag>_meta.
- suav: suavizado `prob` (suavizado.suavizar_prob) con decay/k elegidos en 4 folds y medidos en el 5º
  (como la sección 7 de LGB_futuro.ipynb). Guarda suav_<modelo> con oof y preds; la CV anidada va en `extra`.
Uso: python futuro/pruebas/f5_stack.py stack TAG m1 m2 ...
     python futuro/pruebas/f5_stack.py suav m1 m2 ...
"""
import sys, itertools, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402
import suavizado as S  # noqa: E402  (añade stacking/ al path)
import stack  # noqa: E402
sys.path.insert(0, str(ROOT / "cv"))
from metric import save_outputs  # noqa: E402

GRID = list(itertools.product((0.5, 0.7, 0.85, 1.0), (1, 2, 3, 5, 10)))   # (decay, k)


def hacer_stack(tag, modelos):
    stack.DIRS, stack.OUT = [ROOT / "futuro", OUT], OUT
    guardado = {}
    stack.save_outputs = lambda name, oof, preds, out: guardado.update({name: (oof, preds)})
    t0, argv = time.time(), sys.argv
    sys.argv = ["stack.py"] + modelos
    try:
        stack.main()
    finally:
        sys.argv = argv
    for name, (oof, preds) in guardado.items():
        nombre = name.replace("stack", f"stackf_{tag}")
        save_outputs(nombre, oof, preds, out=OUT)
        registrar(nombre, hilo="f5_stack", descripcion="stacking de: " + ", ".join(modelos), oof_df=oof,
                  segundos=time.time() - t0, n_features=len(modelos))


def suavizar_anidado(modelos):
    lab, te, _ = datos()
    cols = ["sample_id", "row_index", "session", "block", "t", "task_name"]
    LS = lab[cols + ["target_index", "fold"]].assign(y=lab.target_index)
    TS = te[cols].copy()
    NB_L, NB_T = S.vecinos(LS, 10), S.vecinos(TS, 10)
    for modelo in modelos:
        t0 = time.time()
        L = LS.merge(S.leer(modelo, "oof"), on=["sample_id", "row_index"], how="left")
        assert L.score.notna().all(), f"{modelo}: OOF incompleto"
        base = np.exp(S.lsm(L, L.score.values))
        sc = {g: S.suavizar_prob(base, NB_L, *g) for g in GRID}
        n, tot0, tot1, elegidos = L.sample_id.nunique(), 0, 0, []
        for f in range(5):
            tr, va = (L.fold != f).values, (L.fold == f).values
            g = max(GRID, key=lambda g: S.hit3(L, sc[g], tr))
            m = L[va].sample_id.nunique()
            tot0, tot1 = tot0 + S.hit3(L, base, va) * m, tot1 + S.hit3(L, sc[g], va) * m
            elegidos.append(g)
        g_final = max(GRID, key=lambda g: S.hit3(L, sc[g]))
        print(f"{modelo}: CV anidada {tot0 / n:.4f} -> {tot1 / n:.4f} ({(tot1 - tot0) / n:+.4f}); "
              f"por fold {elegidos}; final {g_final}", flush=True)
        T = TS.merge(S.leer(modelo, "preds"), on=["sample_id", "row_index"], how="left")
        sp = S.suavizar_prob(np.exp(S.lsm(T, T.score.values)), NB_T, *g_final)
        nombre = f"suav_{modelo}"
        guardar(nombre, L, sc[g_final], T, sp)
        registrar(nombre, hilo="f5_stack", segundos=time.time() - t0,
                  descripcion=f"{modelo} suavizado prob decay={g_final[0]} k={g_final[1]} (CV anidada en extra)",
                  extra={"metodo": "prob", "params": list(g_final), "cv_sin_suavizar": round(tot0 / n, 4),
                         "cv_anidada": round(tot1 / n, 4), "params_por_fold": [list(g) for g in elegidos]})


def hacer_ses(modelos):
    """Segundo paso de sesión de Pablo (futuro/feat_session.py) sobre cualquier oof/preds -> <modelo>_ses."""
    sys.path.insert(0, str(ROOT / "futuro"))
    import feat_session
    lab, te, _ = datos()
    k = ["sample_id", "row_index"]
    for modelo in modelos:
        t0 = time.time()
        o = lab[k].merge(S.leer(modelo, "oof"), on=k, how="left").score.values
        p = te[k].merge(S.leer(modelo, "preds"), on=k, how="left").score.values
        assert not np.isnan(o).any() and not np.isnan(p).any(), modelo
        so, sp, w = feat_session.sesion(lab, o, te, p)
        guardar(f"{modelo}_ses", lab, so, te, sp)
        registrar(f"{modelo}_ses", hilo="f5_stack", segundos=time.time() - t0,
                  descripcion=f"{modelo} + segundo paso de sesión (feat_session.py)")


if __name__ == "__main__":
    modo, args = sys.argv[1], sys.argv[2:]
    if modo == "stack":
        hacer_stack(args[0], args[1:])
    elif modo == "suav":
        suavizar_anidado(args)
    elif modo == "ses":
        hacer_ses(args)

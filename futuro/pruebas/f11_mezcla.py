"""f11_mezcla.py · Pablo (A) · hilo f11 · mezcla ARITMÉTICA de probabilidades de dos modelos ya con segundo paso:
p = (1 − w) · p_A + w · p_B (softmax de cada score dentro del snapshot), w elegido en CV anidada (4 folds -> 5.º).
El segundo paso y los stacks combinan en log (producto de expertos): si el oráculo da ~0 a una tarea, la veta aunque N
esté seguro (006sc2: 18 de 21 entry_coordination acertadas por N y 0 por C). La mezcla aritmética no deja vetar a uno solo.
Uso: python futuro/pruebas/f11_mezcla.py NOMBRE modeloA modeloB      (A = principal, p. ej. C; B = N)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import OUT, guardar  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import suavizado as S  # noqa: E402
from metric import save_outputs  # noqa: E402

W = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def probs(d, s):
    return np.exp(S.lsm(d, s))


if __name__ == "__main__":
    nombre, ma, mb = sys.argv[1:4]
    lab, te, _ = datos_instr()
    k = ["sample_id", "row_index"]
    L = lab[k + ["target_index", "fold"]].assign(y=lab.target_index)
    pa = probs(L, L[k].merge(S.leer(ma, "oof"), on=k, how="left").score.values)
    pb = probs(L, L[k].merge(S.leer(mb, "oof"), on=k, how="left").score.values)
    sc = {w: (1 - w) * pa + w * pb for w in W}
    n, tot, el = L.sample_id.nunique(), 0.0, []
    for f in range(5):
        tr, va = (L.fold != f).values, (L.fold == f).values
        w = max(W, key=lambda w: S.hit3(L, sc[w], tr))
        tot += S.hit3(L, sc[w], va) * L[va].sample_id.nunique()
        el.append(w)
    wf = max(W, key=lambda w: S.hit3(L, sc[w]))
    print(f"{nombre}: CV anidada {tot / n:.4f} (A solo {S.hit3(L, pa):.4f}, B solo {S.hit3(L, pb):.4f}); "
          f"w por fold {el}; w final {wf}; CV con w final {S.hit3(L, sc[wf]):.4f}", flush=True)
    for w in W:
        print(f"  w={w:.2f}: {S.hit3(L, sc[w]):.4f}")
    T = te[k].copy()
    qa, qb = S.leer(ma, "preds"), S.leer(mb, "preds")
    sp = None
    if qa is not None and qb is not None:
        sp = (1 - wf) * probs(T, T.merge(qa, on=k, how="left").score.values) + wf * probs(T, T.merge(qb, on=k, how="left").score.values)
    guardar(nombre, lab, sc[wf], te if sp is not None else None, sp)
    if sp is not None:
        save_outputs(nombre, None, te[k].assign(score=sp), out=OUT)

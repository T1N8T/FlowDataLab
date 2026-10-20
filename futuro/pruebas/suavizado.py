"""
suavizado.py · Juanma (C) · Suavizado temporal de las predicciones (rompe la regla 1: usa snapshots vecinos).

Hay un snapshot cada ~3 s y el controlador suele seguir con la misma tarea varios seguidos. Se mezcla el score de
cada tarea con el de la MISMA tarea (session, task_name) en los k snapshots anteriores y siguientes de la misma racha
(block de features_future), con pesos que decaen con la distancia. Dos métodos:
- suavizar_prob (el de Pablo, 5b de LGB_futuro_pablo; el que usa main() porque gana en CV anidada):
    p'(t) = (p(t) + Σ_j decay^j · p(t±j)) / (1 + Σ_j decay^j · [la tarea está en ese vecino]),  p = softmax del score
- suavizar (log): s'(t) = lsm(t) + a · Σ_j decay^(j-1) · [lsm(t-j) + lsm(t+j)],  lsm = log-softmax del score
CV anidada (futuro/LGB_futuro_juanma.ipynb, sección 7): lgb_future 0.9010 -> 0.9042 (prob) / 0.9033 (log);
nn_future 0.8874 -> 0.8973 (prob) / 0.8776 (log). Sin fuga: los vecinos son de la misma sesión, así que del mismo controlador y fold, y
sus scores OOF tampoco vieron la etiqueta. Test: igual, con preds/ (no usa etiquetas).

Uso:  python futuro/pruebas/suavizado.py                    # barrido sobre lgb_future, guarda el mejor
      python futuro/pruebas/suavizado.py nn_future f1_best  # otros modelos (futuro/oof o futuro/pruebas/oof)
"""
import sys, itertools, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402
sys.path.insert(0, str(ROOT / "stacking"))
from stack import hit3  # noqa: E402


def leer(modelo, carpeta):
    for base in (ROOT / "futuro", OUT):
        p = base / carpeta / f"{modelo}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def vecinos(d, kmax):
    """Para cada fila: posición de la misma (session, task_name) j snapshots antes/después en la misma racha."""
    snap = d.drop_duplicates("sample_id")[["sample_id", "session", "block", "t"]].sort_values(["session", "t"])
    snap["pos"] = snap.groupby("block").cumcount()
    d = d.merge(snap[["sample_id", "pos"]], on="sample_id")
    key = d[["block", "task_name", "pos"]].reset_index().rename(columns={"index": "i"})
    out = {}
    for j in range(1, kmax + 1):
        for sgn, nom in ((-1, "prev"), (1, "next")):
            k2 = key.assign(pos=key.pos + sgn * j)
            m = k2.merge(key, on=["block", "task_name", "pos"], how="left", suffixes=("", "_v"))
            out[f"{nom}{j}"] = m.sort_values("i").i_v.values          # NaN si la tarea no está en ese vecino
    return out


def lsm(d, s):
    s = pd.Series(s, index=d.index)
    sh = s - s.groupby(d.sample_id).transform("max")
    return (sh - np.log(np.exp(sh).groupby(d.sample_id).transform("sum"))).values


def suavizar(base, nb, a, decay, k):
    s = base.copy()
    for j in range(1, k + 1):
        for nom in ("prev", "next"):
            idx = nb[f"{nom}{j}"]
            ok = ~np.isnan(idx)
            s[ok] += a * decay ** (j - 1) * base[idx[ok].astype(int)]
    return s


def suavizar_prob(p, nb, decay, k):
    """Variante de Pablo (LGB_futuro_pablo, 5b): media ponderada de PROBABILIDADES de la misma tarea en los ±k
    vecinos, normalizada por los vecinos presentes: (p(t) + Σ decay^j p(t±j)) / (1 + Σ decay^j [presente])."""
    num, den = p.copy(), np.ones_like(p)
    for j in range(1, k + 1):
        for nom in ("prev", "next"):
            idx = nb[f"{nom}{j}"]
            ok = ~np.isnan(idx)
            num[ok] += decay ** j * p[idx[ok].astype(int)]
            den[ok] += decay ** j
    return num / den


def main(modelos):
    lab, te, _ = datos()
    cols = ["sample_id", "row_index", "session", "block", "t", "task_name"]
    L, T = lab[cols + ["target_index"]].copy(), te[cols].copy()
    L["y"] = L.target_index
    nbL, nbT = vecinos(L, 10), vecinos(T, 10)
    for modelo in modelos:
        t0 = time.time()
        o, p = leer(modelo, "oof"), leer(modelo, "preds")
        L2 = L.merge(o, on=["sample_id", "row_index"], how="left")
        assert L2.score.notna().all(), f"{modelo}: OOF incompleto"
        base = np.exp(lsm(L2, L2.score.values))
        h0 = hit3(L2, base)
        res = sorted(((hit3(L2, suavizar_prob(base, nbL, dc, k)), dc, k)
                      for dc, k in itertools.product((0.5, 0.7, 0.85, 1.0), (1, 2, 3, 5, 10))), reverse=True)
        print(f"{modelo}: sin suavizar {h0:.4f}; mejores:", [f"{h:.4f} decay={dc} k={k}" for h, dc, k in res[:5]])
        h, dc, k = res[0]
        nombre = f"suav_{modelo}"
        so = suavizar_prob(base, nbL, dc, k)
        sp = None
        if p is not None:
            T2 = T.merge(p, on=["sample_id", "row_index"], how="left")
            sp = suavizar_prob(np.exp(lsm(T2, T2.score.values)), nbT, dc, k)
        guardar(nombre, L2, so, T2 if p is not None else None, sp)
        registrar(nombre, hilo="suavizado", segundos=time.time() - t0,
                  descripcion=f"{modelo} suavizado prob con la misma tarea en ±{k} snapshots (decay={dc}); "
                              f"parámetros elegidos sobre todo el OOF",
                  extra={"cv_sin_suavizar": round(h0, 4), "metodo": "prob", "decay": dc, "k": k})


if __name__ == "__main__":
    main(sys.argv[1:] or ["lgb_future"])

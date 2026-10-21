"""sinpista.py · línea "futuro" · ¿un experto sin futuro arregla los snapshots sin pista?
"Sin pista" = ningún candidato del snapshot tiene instrucción compatible en ≤ 30 s (si_match; se ve sin etiquetas, también en test):
41 % de la CV. Solo en esos snapshots se mezcla el log-softmax de la base (N o M30_30) con el de un modelo entrenado SIN futuro, que
aprendió prioridades (plazos, cercanía al borde, antigüedad): la línea limpia (`limpia`) o la realista (`real_adri_noche`).
Peso w con CV anidada (4 folds -> 5º). No entrena nada (~1 min).
Uso: python entrega/codigo/sinpista.py   (lo llama la sección 5b de futuro.ipynb)
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import hits, leer as _leer, lsm  # noqa: E402
from features_futuro import construir  # noqa: E402

K = ["sample_id", "row_index"]
CV023 = [2, 11, 10, 6, 17, 15]
W = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)


def leer(n):
    df = _leer(n)
    if df is None:
        raise FileNotFoundError(n)
    return df


def hit3(L, score, mask=None):
    """top-3 hit rate vectorizado (L ordenado por sample_id, con columna y)."""
    code = pd.factorize(L.sample_id)[0]
    tpos = np.flatnonzero(L.y.values)[code]
    s = np.asarray(score, float); ts = s[tpos]
    better = (s > ts) | ((s == ts) & (np.arange(len(s)) < tpos))
    hit = np.bincount(code, weights=better) < 3
    if mask is not None:
        keep = np.zeros(code.max() + 1, bool); keep[code[mask]] = True; hit = hit[keep]
    return float(hit.mean())


def main():
    lab, _, _ = construir()
    L = lab[K + ["fold", "atco", "target_index", "si_match"]].copy()
    L["y"] = L.target_index
    gate = L.groupby("sample_id").si_match.transform(lambda s: not (s <= 30).any()).values
    sinp = set(L.sample_id[gate])
    at = L.drop_duplicates("sample_id").set_index("sample_id").atco
    filas = []
    for base in ("N10_ses", "M30_30"):
        b = L[K].merge(leer(base), on=K).score.values
        lb = lsm(L.sample_id, np.log(np.clip(b, 1e-12, None)) if base == "M30_30" else b)   # M30 guarda probabilidades
        h0 = hits(L[K].assign(score=lb))
        for otro in ("limpia", "real_adri_noche"):
            lo = lsm(L.sample_id, L[K].merge(leer(otro), on=K).score.values)
            cands = {w: np.where(gate, (1 - w) * lb + w * lo, lb) for w in W}
            n, tot, el = L.sample_id.nunique(), 0, []
            for f in range(5):
                tr, va = (L.fold != f).values, (L.fold == f).values
                w = max(cands, key=lambda w: hit3(L, cands[w], tr))
                el.append(w)
                tot += hit3(L, cands[w], va) * L[va].sample_id.nunique()
            wb = max(W, key=lambda w: hit3(L, cands[w]))
            hb = hits(L[K].assign(score=cands[wb]))
            dc = (hb.groupby(at.reindex(hb.index)).mean() - h0.groupby(at.reindex(h0.index)).mean()) * 100
            filas.append(dict(base=base, experto=otro, cv_base=round(hit3(L, lb), 4), cv_anidada=round(tot / n, 4),
                              w_por_fold=el, acierto_sin_pista=f"{h0[h0.index.isin(sinp)].mean():.3f} -> {hb[hb.index.isin(sinp)].mean():.3f}",
                              controladores_suben_bajan=f"{(dc > 0).sum()} / {(dc < 0).sum()}", delta_cv023=round(dc.loc[CV023].mean(), 2)))
    T = pd.DataFrame(filas)
    print(f"snapshots sin pista: {len(sinp) / L.sample_id.nunique():.1%}")
    print(T.to_string(index=False))
    return T


if __name__ == "__main__":
    main()

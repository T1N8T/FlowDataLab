"""f13_sinpista.py · ¿un experto sin futuro arregla los snapshots sin pista? (hilo f13)
"Sin pista" = ningún candidato del snapshot tiene instrucción compatible en ≤ 30 s (si_match; se ve sin etiquetas, también en test):
41 % de la CV. Solo en esos snapshots se mezcla el log-softmax de la base (N o M30_30) con el de un modelo entrenado SIN futuro, que
aprendió prioridades (plazos, cercanía al borde, antigüedad): la línea limpia (lgb_adri_v2) o la realista (real_adri_noche_ses).
Peso w con CV anidada (4 folds -> 5º). No entrena nada (~1 min).
Uso: python futuro/pruebas/f13_sinpista.py
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cv"), str(ROOT / "futuro" / "pruebas")]
from metric import hits  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import suavizado as S  # noqa: E402

K = ["sample_id", "row_index"]
CV023 = [2, 11, 10, 6, 17, 15]
W = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)


def leer(n):
    for d in ("oof", "realista/oof", "futuro/oof", "futuro/pruebas/oof"):
        p = ROOT / d / f"{n}.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(n)


def main():
    lab, _, _ = datos_instr()
    L = lab[K + ["fold", "atco", "target_index", "si_match"]].copy()
    L["y"] = L.target_index
    gate = L.groupby("sample_id").si_match.transform(lambda s: not (s <= 30).any()).values
    sinp = set(L.sample_id[gate])
    at = L.drop_duplicates("sample_id").set_index("sample_id").atco
    filas = []
    for base in ("f10_instr_l63_s10_ses", "f11_M30_30"):
        b = L[K].merge(leer(base), on=K).score.values
        lb = S.lsm(L, np.log(np.clip(b, 1e-12, None)) if base == "f11_M30_30" else b)   # M30 guarda probabilidades
        h0 = hits(L[K].assign(score=lb))
        for otro in ("lgb_adri_v2", "real_adri_noche_ses"):
            lo = S.lsm(L, L[K].merge(leer(otro), on=K).score.values)
            cands = {w: np.where(gate, (1 - w) * lb + w * lo, lb) for w in W}
            n, tot, el = L.sample_id.nunique(), 0, []
            for f in range(5):
                tr, va = (L.fold != f).values, (L.fold == f).values
                w = max(cands, key=lambda w: S.hit3(L, cands[w], tr))
                el.append(w)
                tot += S.hit3(L, cands[w], va) * L[va].sample_id.nunique()
            wb = max(W, key=lambda w: S.hit3(L, cands[w]))
            hb = hits(L[K].assign(score=cands[wb]))
            dc = (hb.groupby(at.reindex(hb.index)).mean() - h0.groupby(at.reindex(h0.index)).mean()) * 100
            filas.append(dict(base=base, experto=otro, cv_base=round(S.hit3(L, lb), 4), cv_anidada=round(tot / n, 4),
                              w_por_fold=el, acierto_sin_pista=f"{h0[h0.index.isin(sinp)].mean():.3f} -> {hb[hb.index.isin(sinp)].mean():.3f}",
                              controladores_suben_bajan=f"{(dc > 0).sum()} / {(dc < 0).sum()}", delta_cv023=round(dc.loc[CV023].mean(), 2)))
    T = pd.DataFrame(filas)
    print(f"snapshots sin pista: {len(sinp) / L.sample_id.nunique():.1%}")
    print(T.to_string(index=False))
    return T


if __name__ == "__main__":
    main()

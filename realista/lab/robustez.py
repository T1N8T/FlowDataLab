"""
robustez.py · ¿aguanta una mezcla en un controlador nuevo como el 023? Para cada mezcla (media ponderada de z-scores en el
snapshot de modelos guardados en models/real_lab/oof; 'noche' = realista/oof/real_adri_noche.csv), frente a la referencia:
CV, CV 023, peor controlador, sesiones y controladores que suben / bajan, bootstrap de controladores (comparar.py),
privado simulado (f11_privado.py) y Δ en cada controlador parecido al 023 (*) y en el 020 (público).
Criterio de la sección 5c/5d: solo vale si no baja en los parecidos al 023 y el IC del bootstrap queda por encima de 0.

    python realista/lab/robustez.py noche noche:1+sty:0.25 noche:1+sty:0.25+trig:0.25
    python realista/lab/robustez.py --ref noche:1+sty:0.25 noche:1+sty:0.25+trig:0.5
"""
import io, sys, argparse, contextlib
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path(__file__).resolve().parent
sys.path[:0] = [str(EXP), str(EXP.parents[1] / "futuro" / "pruebas")]
import mix as MX  # noqa: E402
import comparar as C  # noqa: E402
import f11_privado as P  # noqa: E402
import metricas  # noqa: E402
from metric import hits  # noqa: E402

CV023 = [2, 11, 10, 6, 17, 15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="noche")
    ap.add_argument("mezclas", nargs="+")
    a = ap.parse_args()
    lb = MX.lab()
    DF = {s: lb[["sample_id", "row_index"]].assign(score=MX.mezcla(s.split("+"))) for s in [a.ref] + a.mezclas}
    C.leer = P.leer = lambda n: DF[n]
    i = metricas.info()
    r = hits(DF[a.ref])
    at = i.atco.reindex(r.index)
    print(f"referencia {a.ref}: CV {r.mean():.4f} · CV 023 {r[at.isin(CV023).values].mean():.4f}")
    for s in a.mezclas:
        h = hits(DF[s]).reindex(r.index)
        m = metricas.metricas(DF[s], DF[a.ref])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            C.comparar(a.ref, s)
        boot = [l for l in buf.getvalue().splitlines() if "bootstrap" in l][0].split(":", 1)[1].split("(")[0].strip()
        pm, plo, phi, pp = P.esperado(a.ref, s, [1, 2, 3])
        d = (h - r).groupby(at).mean() * 100
        d023 = d[d.index.isin(CV023)]
        print(f"\n{s}\n  CV {m['cv']:.4f} (Δ {m['d_cv'] * 100:+.2f}) · CV 023 {h[at.isin(CV023).values].mean():.4f} · peor "
              f"controlador {m['min_atco']:.3f} · sesiones {m['ses_up']}/{m['ses_down']} · controladores "
              f"{m['atco_up']}/{m['atco_down']}\n  bootstrap {boot}\n  privado simulado Δ {pm:+.2f} (p10 {plo:+.2f}), "
              f"P = {pp:.2f} · parecidos al 023: suben {int((d023 > 0).sum())}, bajan {int((d023 < 0).sum())} · 020 {d.get(20, np.nan):+.1f}")
        print("  " + "  ".join(f"{k:03d}{'*' if k in CV023 else ''}:{v:+.1f}" for k, v in d.items()))


if __name__ == "__main__":
    main()

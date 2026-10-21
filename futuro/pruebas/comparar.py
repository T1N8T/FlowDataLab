"""
comparar.py · Pablo (A) · RAMA FUTURO · Compara dos OOF con el criterio de la sección 14 de LGB_futuro.ipynb.

El privado es UN controlador nuevo. Para cada candidato: CV, peor controlador, mediana por sesión, CV por escenario,
sesiones y controladores que suben/bajan (futuro/metricas.py) y un bootstrap de la ganancia media sobre los 18
controladores (no incluye el ruido de las semillas, que mueve a todos a la vez: sección 14b).
Los OOF se buscan en futuro/oof/ y futuro/pruebas/oof/ (formato común sample_id,row_index,score).
Ojo: en los suav_* la CV que sale es la optimista (parámetros del suavizado elegidos con todo el OOF); la honesta es la
CV anidada que imprime suavizado.py / la celda de 9b (~0.1–0.2 menos).

Uso: python futuro/pruebas/comparar.py <referencia> <candidato> [<candidato2> ...]
     p. ej. python futuro/pruebas/comparar.py suav_lgb_future_clear10_ses suav_lgb_future_clear10_or_ses
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cv"), str(ROOT / "futuro")]
from metric import hits  # noqa: E402
import metricas  # noqa: E402


def leer(nombre):
    for base in (ROOT / "futuro" / "oof", ROOT / "futuro" / "pruebas" / "oof"):
        if (base / f"{nombre}.csv").exists():
            return pd.read_csv(base / f"{nombre}.csv")
    raise FileNotFoundError(nombre)


def comparar(ref, cand, n_boot=5000, seed=0):
    """Métricas de futuro/metricas.py (media, peor controlador, mediana por sesión, escenarios, sesiones y controladores
    que suben/bajan) + bootstrap sobre controladores de la ganancia media. Devuelve el dict de métricas."""
    r, c = leer(ref), leer(cand)
    m = metricas.metricas(c, r)
    print(metricas.resumen(cand, m, ref))
    i = metricas.info()
    a, b = hits(r), hits(c)
    at = i.atco.reindex(a.index)
    g = b.reindex(a.index).groupby(at).mean() - a.groupby(at).mean()
    rng = np.random.default_rng(seed)
    bs = np.array([g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)])
    print(f"  bootstrap de controladores: IC 90 % de la ganancia media [{np.percentile(bs, 5) * 100:+.2f}, "
          f"{np.percentile(bs, 95) * 100:+.2f}], P(media > 0) = {(bs > 0).mean():.2f}  (no incluye el ruido de semillas)")
    return m


if __name__ == "__main__":
    for c in sys.argv[2:]:
        comparar(sys.argv[1], c)

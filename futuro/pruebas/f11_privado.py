"""f11_privado.py · ganancia esperada en un controlador nuevo que vuela los 3 escenarios (como el 023, privado) y en
uno que vuela los escenarios 2 y 3 (como el 020, público): se combinan al azar sesiones de la CV de esos escenarios
(de controladores distintos), ponderadas por su número de snapshots, y se mide Δ = cand − ref. Igual que la sección 21c.
Uso: python futuro/pruebas/f11_privado.py ref cand [cand2 ...]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cv"), str(ROOT / "futuro"), str(ROOT / "futuro" / "pruebas")]
from metric import hits  # noqa: E402
import metricas  # noqa: E402
from comparar import leer  # noqa: E402


def esperado(ref, cand, escenarios, n=20000, seed=0):
    i = metricas.info()
    a, b = hits(leer(ref)), hits(leer(cand)).reindex(hits(leer(ref)).index)
    ses = i.session.reindex(a.index)
    t = pd.DataFrame(dict(d=b - a, n=1, sc=i.scenario.reindex(a.index), atco=i.atco.reindex(a.index), ses=ses))
    g = t.groupby("ses").agg(d=("d", "sum"), n=("n", "sum"), sc=("sc", "first"), atco=("atco", "first"))
    rng = np.random.default_rng(seed)
    pools = {s: g[g.sc == s] for s in escenarios}
    out = []
    for _ in range(n):
        pick = [pools[s].iloc[rng.integers(len(pools[s]))] for s in escenarios]
        if len({p.atco for p in pick}) < len(pick):
            continue
        out.append(sum(p.d for p in pick) / sum(p.n for p in pick))
    out = np.array(out) * 100
    return out.mean(), np.percentile(out, 10), np.percentile(out, 90), (out > 0).mean()


if __name__ == "__main__":
    ref = sys.argv[1]
    for c in sys.argv[2:]:
        for nom, esc in (("privado (023: escenarios 1+2+3)", [1, 2, 3]), ("público (020: escenarios 2+3)", [2, 3])):
            m, lo, hi, p = esperado(ref, c, esc)
            print(f"{c} frente a {ref} · {nom}: Δ esperado {m:+.2f} (p10 {lo:+.2f}, p90 {hi:+.2f}), P(> ref) = {p:.2f}")


def parejas(modelos, escenarios=(1, 2, 3), n=20000, seed=0):
    """E[máx(nota de la pareja)] en un controlador nuevo simulado (sesiones de la CV de los escenarios dados, de
    controladores distintos): Kaggle puntúa la mejor de las 2 entregas elegidas."""
    i = metricas.info()
    H = {m: hits(leer(m)) for m in modelos}
    idx = H[modelos[0]].index
    ses = i.session.reindex(idx)
    g = pd.DataFrame({m: H[m].reindex(idx) for m in modelos}).groupby(ses.values).agg(["sum", "size"])
    info = i.drop_duplicates("session").set_index("session")
    sc, at = info.scenario.reindex(g.index).values, info.atco.reindex(g.index).values
    rng = np.random.default_rng(seed)
    pools = {s: np.where(sc == s)[0] for s in escenarios}
    notas = []
    for _ in range(n):
        pick = [pools[s][rng.integers(len(pools[s]))] for s in escenarios]
        if len(set(at[pick])) < len(pick):
            continue
        tot = g.iloc[pick][(modelos[0], "size")].sum()
        notas.append([g.iloc[pick][(m, "sum")].sum() / tot for m in modelos])
    return pd.DataFrame(np.array(notas) * 100, columns=modelos)

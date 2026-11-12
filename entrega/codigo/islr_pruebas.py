"""islr_pruebas.py · El hilo de pruebas "a priori" de entrega/justificacion_islr.md, sobre los OOF ya calculados (no entrena).

Unidad de análisis: el CONTROLADOR (n = 18). Los snapshots de una sesión están correlacionados (ISLR §3.3.3, pp. 94–95).

P0 · Suelos de referencia (ISLR §2.2.3 y §4.4):
     azar (3/m por snapshot) y "prior por tipo": ordenar por el lift del tipo de tarea aprendido en los folds de
     entrenamiento (CV por controlador, datos/folds.csv). Es el clasificador generativo más simple posible (π_k).
P1 · Suelo de ruido: el mismo modelo con otras semillas u otra máquina. Ninguna mejora por debajo de esto cuenta.
P2 · Pares de modelos de la cadena (d = acierto por controlador de B − A):
     - permutación con cambio de signo, exacta para n ≤ 20 (ISLR §13.5, p. 574)
     - IC 95 % por bootstrap remuestreando controladores (ISLR §5.2)
     - Holm (FWER, §13.3.2) y Benjamini–Hochberg (FDR, §13.4) sobre la familia de pares
     - diferencia mínima detectable con potencia 0.8 y α = 0.05 (t pareado, n controladores)
     - P(B > A) en la media: bootstrap bayesiano (pesos Dirichlet(1,…,1) sobre los controladores)
     - P(B > A en UN controlador nuevo): predictiva t, d_nuevo ~ media + s·sqrt(1 + 1/n)·t_{n−1}
P3 · Regla de elección validada (dejar fuera un controlador): con los otros 17 se elige la candidata de mejor CV y se mide
     en el que queda fuera. Pérdida frente a la mejor a posteriori y frente a quedarse siempre con la referencia.

Uso:  python entrega/codigo/islr_pruebas.py
Salidas: salidas/notas/islr_P0_suelos.csv, islr_P1_ruido.csv, islr_P2_pares.csv, islr_P3_regla.csv
"""
import itertools
import numpy as np, pandas as pd
from scipy import stats
from comun import labels, hits, leer, SALIDAS, DATOS

RNG = np.random.default_rng(0)
B_BOOT = 10_000

# un modelo por paso de la cadena (README); los que no existan en salidas/oof se saltan
CADENA = ["limpia", "real_adri_noche", "futuro_base", "futuro_clear", "futuro_clear_ses", "A", "B", "N30_ses", "C30_ses",
          "M30_30"]
# mismo modelo, solo cambia el azar (semillas o máquina)
RUIDO = [("C10_ses", "C10_otra_maquina_ses"), ("N3_ses", "N10_ses"), ("N10_ses", "N30_ses"), ("C3_ses", "C10_ses"),
         ("C10_ses", "C30_ses")]
# candidatas finales para P3 (la primera es la referencia)
CANDIDATAS = ["N30_ses", "A", "B", "C30_ses", "M30_30"]


# ---------------------------------------------------------------------------------------------------------------- utilidades
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); acc = 0.0
    for r, i in enumerate(o):
        acc = max(acc, (m - r) * p[i]); adj[i] = min(1.0, acc)
    return adj


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); acc = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; acc = min(acc, p[i] * m / (r + 1)); adj[i] = acc
    return adj


def perm_signo(d, n_mc=200_000):
    """p-valor bilateral de la media de d con cambio de signo (exacto si n ≤ 20)."""
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    if n <= 20:
        s = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    else:
        s = RNG.choice([-1, 1], size=(n_mc, n))
    return float(np.mean(np.abs(s @ d / n) >= obs - 1e-12))


def boot_ic(d):
    d = np.asarray(d, float)
    m = d[RNG.integers(0, len(d), size=(B_BOOT, len(d)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def p_bayes_boot(d):
    w = RNG.dirichlet(np.ones(len(d)), size=B_BOOT)
    return float(np.mean(w @ np.asarray(d, float) > 0))


def p_predictiva(d):
    d = np.asarray(d, float); n = len(d); s = d.std(ddof=1)
    return float(stats.t.cdf(d.mean() / (s * np.sqrt(1 + 1 / n)), n - 1)) if s > 0 else float(d.mean() > 0)


def mde(d, alfa=0.05, potencia=0.8):
    n = len(d); s = np.std(d, ddof=1)
    return (stats.t.ppf(1 - alfa / 2, n - 1) + stats.t.ppf(potencia, n - 1)) * s / np.sqrt(n)


# ---------------------------------------------------------------------------------------------------------------------- P0
def p0_suelos(lab):
    d = pd.concat([pd.read_csv(DATOS / f"{f}.csv", usecols=["sample_id", "row_index", "target_index", "task_index"])
                   for f in ("train", "validation")])
    d = d[d.sample_id.isin(lab.index)]
    d["atco"] = d.sample_id.map(lab.atco); d["fold"] = d.sample_id.map(lab.fold)
    oof = []
    for f in sorted(lab.fold.unique()):
        tr = d[d.fold != f]
        lift = (tr[tr.target_index == 1].task_index.value_counts(normalize=True)
                / tr.task_index.value_counts(normalize=True)).fillna(0)
        te = d[d.fold == f].copy()
        te["score"] = np.log(te.task_index.map(lift).fillna(0) + 1e-9)
        oof.append(te[["sample_id", "row_index", "score"]])
    h = hits(pd.concat(oof))
    m = d.groupby("sample_id").size()
    azar = np.minimum(3 / m, 1.0).reindex(h.index)
    filas = []
    for nombre, x in (("azar (3/m)", azar), ("prior por tipo (lift, CV)", h)):
        pa = x.groupby(lab.atco.loc[x.index]).mean()
        filas.append(dict(modelo=nombre, cv=x.mean(), media_atco=pa.mean(), peor_atco=pa.min()))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------------------------------------------------- P1/P2
def fila_par(Pa, a, b, H):
    d = (Pa[b] - Pa[a]).values
    t, p_t = stats.ttest_rel(Pa[b], Pa[a])
    ic = boot_ic(d)
    return dict(A=a, B=b, dif_cv=H[b].mean() - H[a].mean(), dif_media_atco=d.mean(), sd_atco=d.std(ddof=1),
                ic95_boot_inf=ic[0], ic95_boot_sup=ic[1], atco_suben=int((d > 0).sum()), atco_bajan=int((d < 0).sum()),
                p_t=p_t, p_perm=perm_signo(d), mde_80=mde(d), P_media_mejor=p_bayes_boot(d),
                P_controlador_nuevo=p_predictiva(d))


# ---------------------------------------------------------------------------------------------------------------------- P3
def p3_regla(H, lab, cand):
    atco = lab.atco.loc[H.index]
    filas = []
    for c in sorted(atco.unique()):
        dentro, fuera = H[atco != c], H[atco == c]
        elegida = dentro[cand].mean().idxmax()
        acierto = fuera[cand].mean()
        filas.append(dict(atco=c, elegida=elegida, acierto_elegida=acierto[elegida], acierto_ref=acierto[cand[0]],
                          mejor_a_posteriori=acierto.idxmax(), acierto_mejor=acierto.max()))
    R = pd.DataFrame(filas)
    R["gana_a_ref"] = R.acierto_elegida - R.acierto_ref
    R["perdida_vs_mejor"] = R.acierto_mejor - R.acierto_elegida
    return R


# -------------------------------------------------------------------------------------------------------------------- main
def main():
    lab = labels()
    nombres = sorted({n for n in CADENA + CANDIDATAS + [x for p in RUIDO for x in p] if leer(n) is not None},
                     key=lambda n: (CADENA + CANDIDATAS + [x for p in RUIDO for x in p]).index(n))
    falta = set(CADENA + CANDIDATAS + [x for p in RUIDO for x in p]) - set(nombres)
    if falta:
        print("no están en salidas/oof (se saltan):", sorted(falta))
    H = pd.DataFrame({n: hits(leer(n)) for n in nombres}).reindex(lab.index)
    H = H.dropna()                                                  # solo snapshots con predicción de todos
    Pa = H.groupby(lab.atco.loc[H.index]).mean()
    notas = SALIDAS / "notas"; notas.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40)

    print("\n== P0 · suelos de referencia")
    P0 = p0_suelos(lab); P0.to_csv(notas / "islr_P0_suelos.csv", index=False); print(P0.to_string(index=False))

    print("\n== P1 · suelo de ruido (mismo modelo, solo cambia el azar)")
    P1 = pd.DataFrame([fila_par(Pa, a, b, H) for a, b in RUIDO if a in H and b in H])
    P1.to_csv(notas / "islr_P1_ruido.csv", index=False)
    print(P1[["A", "B", "dif_cv", "dif_media_atco", "sd_atco", "p_perm", "mde_80"]].to_string(index=False))

    print("\n== P2 · pares de la cadena (d = B − A por controlador)")
    cad = [n for n in CADENA if n in H]
    P2 = pd.DataFrame([fila_par(Pa, a, b, H) for a, b in itertools.combinations(cad, 2)])
    P2["p_perm_holm"] = holm(P2.p_perm); P2["p_perm_bh"] = bh(P2.p_perm)
    P2.to_csv(notas / "islr_P2_pares.csv", index=False)
    cols = ["A", "B", "dif_media_atco", "ic95_boot_inf", "ic95_boot_sup", "atco_suben", "atco_bajan", "p_perm",
            "p_perm_holm", "p_perm_bh", "mde_80", "P_media_mejor", "P_controlador_nuevo"]
    consec = P2[[cad.index(b) == cad.index(a) + 1 for a, b in zip(P2.A, P2.B)]]
    print("pasos consecutivos de la cadena:"); print(consec[cols].round(4).to_string(index=False))
    fin = P2[P2.A.isin(CANDIDATAS) & P2.B.isin(CANDIDATAS)]
    print("\nentre candidatas finales:"); print(fin[cols].round(4).to_string(index=False))

    print("\n== P3 · regla 'elegir por CV' validada dejando fuera cada controlador")
    cand = [c for c in CANDIDATAS if c in H]
    P3 = p3_regla(H, lab, cand); P3.to_csv(notas / "islr_P3_regla.csv", index=False)
    print(P3.elegida.value_counts().to_string())
    print(f"ganancia media frente a {cand[0]}: {P3.gana_a_ref.mean():+.4f} · pérdida media frente a la mejor a posteriori: "
          f"{P3.perdida_vs_mejor.mean():.4f} · veces que la elegida es la mejor: {(P3.elegida == P3.mejor_a_posteriori).sum()}"
          f"/{len(P3)}")


if __name__ == "__main__":
    main()

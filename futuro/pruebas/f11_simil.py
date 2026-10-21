"""
f11_simil.py · Juanma (C) · RAMA FUTURO · Estimar el privado (controlador 023) por similitud, sin etiquetas del 023.

1. Indicadores SIN etiquetas por controlador (CV: con OOF; test: con preds): % top-1 de coordinación y de cada tipo,
   confianza/entropía, % snapshots con un input en <= 10 s, rachas, nº de tareas, composición de la lista,
   desacuerdo entre modelos, ritmo de inputs.
2. Ponderación de los 18 controladores de la CV por un kernel gaussiano sobre indicadores estandarizados.
   Validación: leave-one-controller-out (LOCO) prediciendo el acierto y la diferencia entre modelos del controlador
   que se deja fuera; el ancho de banda y el conjunto de indicadores se eligen en LOCO anidado.
   Comprobación externa: misma receta para el 020 (sin el 020 en la referencia) frente al público conocido.
3. Estimación para el 023 de A, B, N y N+B (media de probabilidades) con intervalos (bootstrap de controladores
   + error LOCO). Autoevaluación con la posterior media calibrada (pseudo-etiquetas).
4. Comparación de las predicciones de test en las sesiones del 023 (y 020): dónde discrepan y qué tipos.

Salidas: futuro/pruebas/oof|preds/f11_simil_NB.csv (mezcla N+B) y registro en resultados.jsonl (hilo f11).
Uso: python futuro/pruebas/f11_simil.py
"""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402

MODELOS = dict(A="suav_lgb_future_clear10_ses", B="suav_lgb_future_clear10_orcoord_ses", N="f10_instr_l63_s10_ses")
PUBLICO = dict(A=0.947, B=0.934, N=0.951)
CV023 = [2, 11, 10, 6, 17, 15]
TIPOS = ["e_conf", "x_conf", "e_coord", "x_coord", "assume", "transfer", "conflict", "nonconf", "qos", "e_cond"]
rng = np.random.default_rng(0)


def tn(k):
    return TIPOS[k] if 0 <= k < len(TIPOS) else f"tipo{k}"


def leer(nombre, carpeta, d):
    x = pd.read_csv(OUT / carpeta / f"{nombre}.csv")
    x = d[["sample_id", "row_index"]].merge(x, on=["sample_id", "row_index"], how="left")
    assert x.score.notna().all(), nombre
    return x.score.values


def prob(d, s):
    """softmax por snapshot si son logits; si ya son probabilidades (suman 1), se dejan."""
    s = pd.Series(s, index=d.index)
    tot = s.groupby(d.sample_id).transform("sum")
    if s.min() >= 0 and np.allclose(tot, 1, atol=0.02):
        return (s / tot).values
    e = np.exp(s - s.groupby(d.sample_id).transform("max"))
    return (e / e.groupby(d.sample_id).transform("sum")).values


def rank_in(d, p):
    return pd.Series(-p, index=d.index).groupby(d.sample_id).rank(method="first").values


def por_snap(d, P, y=None):
    """Tabla por snapshot: sesión, atco, hit y tipo/tarea del top-1 de cada modelo, confianza, etc."""
    s = d.groupby("sample_id", sort=True)
    snap = s.agg(session=("session", "first"), atco=("atco", "first"), t=("t", "first"), block=("block", "first"),
                 blk_len=("blk_len", "first"), seq_len=("seq_len", "first"),
                 instr10=("f_next_input", lambda v: float(np.nanmin(v) <= 10) if np.isfinite(v).any() else 0.0),
                 an60=("a_n60", "max"))
    for k in range(10):
        snap[f"lst_{TIPOS[k]}"] = (d.task_index == k).groupby(d.sample_id).mean()
    R = {}
    for m, p in P.items():
        r = rank_in(d, p)
        R[m] = r
        top1 = d[r == 1].set_index("sample_id")
        snap[f"{m}_t1type"] = top1.task_index
        snap[f"{m}_t1task"] = top1.task_name
        snap[f"{m}_conf"] = pd.Series(p[r == 1], index=top1.index)
        snap[f"{m}_m3"] = pd.Series(p * (r <= 3), index=d.index).groupby(d.sample_id).sum()
        snap[f"{m}_ent"] = pd.Series(-p * np.log(np.clip(p, 1e-12, 1)), index=d.index).groupby(d.sample_id).sum()
        snap[f"{m}_top3"] = pd.Series(np.where(r <= 3, d.row_index, -1), index=d.index).groupby(d.sample_id) \
            .apply(lambda v: frozenset(v[v >= 0]))
        if y is not None:
            snap[f"{m}_hit"] = pd.Series(y * (r <= 3), index=d.index).groupby(d.sample_id).sum()
            snap[f"{m}_hit1"] = pd.Series(y * (r == 1), index=d.index).groupby(d.sample_id).sum()
    return snap, R


def indicadores(snap):
    """Indicadores sin etiquetas por controlador (media de sus sesiones ponderada por snapshots)."""
    g = snap.groupby("atco")
    X = pd.DataFrame(index=sorted(snap.atco.unique()))
    t1 = snap.N_t1type
    for k in range(9):
        X[f"t1_{TIPOS[k]}"] = (t1 == k).groupby(snap.atco).mean()
    X["t1_coord"] = t1.isin([2, 3]).groupby(snap.atco).mean()
    X["conf"] = g.N_conf.mean()
    X["m3"] = g.N_m3.mean()
    X["ent"] = g.N_ent.mean()
    X["instr10"] = g.instr10.mean()
    X["an60"] = g.an60.mean()
    X["seq_len"] = g.seq_len.mean()
    X["blk_len"] = g.blk_len.mean()
    # racha del top-1 de N: nº medio de snapshots consecutivos con la misma tarea en cabeza
    s2 = snap.sort_values(["session", "t"])
    cambio = (s2.N_t1task != s2.groupby("session").N_t1task.shift()) | (s2.t.diff().abs() > 4.5) | \
             (s2.session != s2.session.shift())
    run = cambio.groupby(s2.session).cumsum()
    X["run_t1"] = s2.groupby(["atco", s2.session, run]).size().groupby("atco").mean()
    for k in (2, 3, 4, 5, 6):
        X[f"lst_{TIPOS[k]}"] = g[f"lst_{TIPOS[k]}"].mean()
    X["dis_NB"] = (snap.N_top3 != snap.B_top3).groupby(snap.atco).mean()
    X["dis_NA"] = (snap.N_top3 != snap.A_top3).groupby(snap.atco).mean()
    X["dis1_NB"] = (snap.N_t1task != snap.B_t1task).groupby(snap.atco).mean()
    return X


SUBSETS = {
    "todos": None,
    "coord": ["t1_coord", "t1_e_coord", "t1_x_coord", "lst_e_coord", "lst_x_coord"],
    "pred": ["t1_coord", "t1_assume", "t1_transfer", "t1_conflict", "conf", "ent", "m3", "dis_NB", "dis_NA"],
    "comport": ["instr10", "an60", "blk_len", "run_t1", "seq_len", "t1_coord"],
    "desac": ["dis_NB", "dis_NA", "dis1_NB", "conf", "t1_coord"],
}
HS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]


def pesos(Xs, x0, h):
    d2 = ((Xs - x0) ** 2).mean(axis=1)      # distancia media por indicador (comparable entre subconjuntos)
    if not np.isfinite(h):
        return np.ones(len(Xs)) / len(Xs)
    w = np.exp(-d2 / (2 * h ** 2))
    return w / w.sum()


def estandar(X, cols, ref_idx):
    Z = X[cols] if cols else X
    mu, sd = Z.loc[ref_idx].mean(), Z.loc[ref_idx].std().replace(0, 1)
    return (Z - mu) / sd


def loco_err(X, Y, idx, cols, h):
    """Error LOCO (RMSE) prediciendo Y[c] (columnas: objetivos) para c en idx, con referencia idx - {c}."""
    err = []
    for c in idx:
        ref = [i for i in idx if i != c]
        Z = estandar(X, cols, ref)
        w = pesos(Z.loc[ref].values, Z.loc[c].values, h)
        err.append(Y.loc[c].values - w @ Y.loc[ref].values)
    return np.sqrt((np.array(err) ** 2).mean(axis=0))


def elegir(X, Y, idx):
    best = None
    for nom, cols in SUBSETS.items():
        for h in HS:
            e = loco_err(X, Y, idx, cols, h).mean()
            if best is None or e < best[0]:
                best = (e, nom, h)
    return best


def main():
    lab, te, _ = datos()
    P = {m: prob(lab, leer(n, "oof", lab)) for m, n in MODELOS.items()}
    Q = {m: prob(te, leer(n, "preds", te)) for m, n in MODELOS.items()}
    P["NB"], Q["NB"] = (P["N"] + P["B"]) / 2, (Q["N"] + Q["B"]) / 2
    guardar("f11_simil_NB", lab, P["NB"], te, Q["NB"])
    y = lab.target_index.values
    sl, RL = por_snap(lab, P, y)
    st, RT = por_snap(te, Q)
    ms = list(P)
    acc = pd.DataFrame({m: sl.groupby("atco")[f"{m}_hit"].mean() for m in ms})
    print("== Acierto real por controlador (CV OOF; A y B son suav_*, CV optimista) ==")
    print(acc.round(4).to_string())
    print("media", acc.mean().round(4).to_dict(), "| CV 023", acc.loc[CV023].mean().round(4).to_dict(),
          "| media por muestra", {m: round(sl[f"{m}_hit"].mean(), 4) for m in ms})
    XL, XT = indicadores(sl), indicadores(st)
    XT.index = [f"T{a}" for a in XT.index]
    X = pd.concat([XL, XT])
    print("\n== Indicadores sin etiquetas (CV con OOF; T20/T23 = test con preds) ==")
    with pd.option_context("display.width", 250):
        print(X.round(3).to_string())
    # ¿la confianza en test está inflada respecto a OOF? (modelo final entrenado con todo)
    print(f"\nconfianza media N: OOF {sl.N_conf.mean():.3f}  test {st.N_conf.mean():.3f}; "
          f"en 020: OOF(020sc1) {XL.loc[20, 'conf']:.3f}  test {XT.loc['T20', 'conf']:.3f}")

    # Objetivos: acierto de cada modelo y diferencias
    Y = acc.copy()
    for a, b in [("N", "B"), ("N", "A"), ("NB", "N"), ("NB", "B"), ("A", "B")]:
        Y[f"{a}-{b}"] = acc[a] - acc[b]
    idx = list(acc.index)
    dif = [c for c in Y.columns if "-" in c]
    print("\n== Validación LOCO del kernel (RMSE en puntos x100, objetivo = diferencias entre modelos) ==")
    tab = []
    for nom, cols in SUBSETS.items():
        for h in HS:
            e = loco_err(X, Y[dif], idx, cols, h)
            tab.append(dict(subset=nom, h=h, **{c: round(v * 100, 3) for c, v in zip(dif, e)},
                            media=round(e.mean() * 100, 3)))
    tab = pd.DataFrame(tab)
    print(tab.sort_values("media").head(12).to_string(index=False))
    base_err = tab[(tab.subset == "todos") & (tab.h == np.inf)].media.iloc[0]
    print(f"(h = inf es la media uniforme: {base_err:.3f})")
    # LOCO anidado: para cada c, elegir (subset, h) con LOCO interno sobre los otros 17 y predecir c
    nest, elegidos = [], []
    for c in idx:
        ref = [i for i in idx if i != c]
        e, nom, h = elegir(X, Y[dif], ref)
        Z = estandar(X, SUBSETS[nom], ref)
        w = pesos(Z.loc[ref].values, Z.loc[c].values, h)
        nest.append(Y.loc[c, dif].values - w @ Y.loc[ref, dif].values)
        elegidos.append((nom, h))
    nest = np.array(nest)
    unif = np.array([Y.loc[c, dif].values - Y.loc[[i for i in idx if i != c], dif].mean().values for c in idx])
    print("LOCO anidado RMSE (x100):", dict(zip(dif, (np.sqrt((nest ** 2).mean(0)) * 100).round(3))),
          "\nmedia uniforme RMSE (x100):", dict(zip(dif, (np.sqrt((unif ** 2).mean(0)) * 100).round(3))),
          "\nelecciones internas:", pd.Series([f"{a}/{b}" for a, b in elegidos]).value_counts().to_dict())
    # Signo: ¿acierta qué modelo gana en el controlador dejado fuera?
    for j, c in enumerate(dif):
        pr = Y[c].values - nest[:, j]
        print(f"  {c}: corr(pred, real) = {np.corrcoef(pr, Y[c].values)[0, 1]:+.2f}, signo acertado "
              f"{(np.sign(pr) == np.sign(Y[c].values)).mean():.2f}")

    # Regresión lineal sencilla de la diferencia sobre un indicador (LOCO), por si el kernel no sirve
    print("\n== LOCO de regresión univariante diferencia ~ indicador (RMSE x100; mejores) ==")
    reg = []
    for col in XL.columns:
        for c in dif:
            e = []
            for k in idx:
                ref = [i for i in idx if i != k]
                b1, b0 = np.polyfit(XL.loc[ref, col], Y.loc[ref, c], 1)
                e.append(Y.loc[k, c] - (b0 + b1 * XL.loc[k, col]))
            reg.append(dict(ind=col, obj=c, rmse=np.sqrt(np.mean(np.square(e))) * 100,
                            corr=np.corrcoef(XL[col], Y[c])[0, 1]))
    reg = pd.DataFrame(reg)
    uni = dict(zip(dif, np.sqrt((unif ** 2).mean(0)) * 100))
    reg["mejora_vs_unif"] = reg.rmse - reg.obj.map(uni)
    print(reg.sort_values("mejora_vs_unif").head(12).round(3).to_string(index=False))
    # anidado: en cada LOCO se elige el mejor indicador con LOCO interno sobre los otros 17
    def uni_fit(ref, c):
        best = None
        for col in XL.columns:
            e = []
            for k in ref:
                r2 = [i for i in ref if i != k]
                b1, b0 = np.polyfit(XL.loc[r2, col], Y.loc[r2, c], 1)
                e.append(Y.loc[k, c] - (b0 + b1 * XL.loc[k, col]))
            r = np.sqrt(np.mean(np.square(e)))
            if best is None or r < best[0]:
                best = (r, col)
        b1, b0 = np.polyfit(XL.loc[ref, best[1]], Y.loc[ref, c], 1)
        return best[1], b0, b1
    for c in ("N-B", "NB-B", "N-A"):
        e, cols_ = [], []
        for k in idx:
            ref = [i for i in idx if i != k]
            col, b0, b1 = uni_fit(ref, c)
            e.append(Y.loc[k, c] - (b0 + b1 * XL.loc[k, col])); cols_.append(col)
        col, b0, b1 = uni_fit(idx, c)
        print(f"  {c}: univariante ANIDADO RMSE {np.sqrt(np.mean(np.square(e))) * 100:.3f} (uniforme {uni[c]:.3f}); "
              f"elegidos {pd.Series(cols_).value_counts().to_dict()}; con todo: {col} -> T23 "
              f"{(b0 + b1 * XT.loc['T23', col]) * 100:+.2f}, T20 {(b0 + b1 * XT.loc['T20', col]) * 100:+.2f}")
    for col in ("t1_transfer", "seq_len", "t1_coord", "t1_e_coord", "conf"):
        for c in ("N-B", "NB-B", "N-A"):
            b1, b0 = np.polyfit(XL[col], Y[c], 1)
            print(f"    {c} ~ {col}: T23 {(b0 + b1 * XT.loc['T23', col]) * 100:+.2f}  T20 {(b0 + b1 * XT.loc['T20', col]) * 100:+.2f}"
                  f"  (corr {np.corrcoef(XL[col], Y[c])[0, 1]:+.2f})")

    # Estimación para T23 y comprobación con T20 (sin el 20 en la referencia)
    res = {}
    for tgt, ref in (("T23", idx), ("T20", [i for i in idx if i != 20])):
        e, nom, h = elegir(X, Y[dif], ref)
        Z = estandar(X, SUBSETS[nom], ref)
        w = pesos(Z.loc[ref].values, Z.loc[tgt].values, h)
        # para el acierto absoluto también se elige su propia configuración
        ea, noma, ha = elegir(X, Y[ms], ref)
        Za = estandar(X, SUBSETS[noma], ref)
        wa = pesos(Za.loc[ref].values, Za.loc[tgt].values, ha)
        est_d = w @ Y.loc[ref, dif].values
        est_a = wa @ Y.loc[ref, ms].values
        # bootstrap de controladores (con los pesos) + error LOCO
        B = []
        for _ in range(4000):
            ii = rng.integers(0, len(ref), len(ref))
            ww = w[ii] / w[ii].sum()
            B.append(ww @ Y.loc[ref, dif].values[ii])
        B = np.array(B)
        loco_rmse = loco_err(X, Y[dif], ref, SUBSETS[nom], h)
        ess = 1 / (w ** 2).sum()
        top = pd.Series(w, index=ref).sort_values(ascending=False).head(6).round(3).to_dict()
        print(f"\n== Estimación para {tgt} (diferencias: subset={nom}, h={h}; acierto: subset={noma}, h={ha}) ==")
        print(f"  pesos (ESS = {ess:.1f} controladores): {top}")
        print("  acierto esperado:", {m: round(v, 4) for m, v in zip(ms, est_a)})
        for j, c in enumerate(dif):
            lo, hi = np.percentile(B[:, j], [5, 95])
            print(f"  {c}: {est_d[j] * 100:+.2f} pts  (IC90 bootstrap [{lo * 100:+.2f}, {hi * 100:+.2f}], "
                  f"± error LOCO {loco_rmse[j] * 100:.2f};  P(>0) boot {(B[:, j] > 0).mean():.2f})")
        res[tgt] = dict(subset=nom, h=h, ess=ess, acc=dict(zip(ms, est_a)), dif=dict(zip(dif, est_d)),
                        ic=dict(zip(dif, np.percentile(B, [5, 95], axis=0).T.tolist())),
                        loco=dict(zip(dif, loco_rmse)), pesos=top)
    print(f"\n  público real 020: {PUBLICO}  ->  N-B {PUBLICO['N'] - PUBLICO['B']:+.3f}, "
          f"N-A {PUBLICO['N'] - PUBLICO['A']:+.3f}, A-B {PUBLICO['A'] - PUBLICO['B']:+.3f}")
    print(f"  CV real del 020 (020sc1): {acc.loc[20].round(4).to_dict()}")

    # Autoevaluación con pseudo-etiquetas: posterior = media de A, B, N calibrada (temperatura) en CV
    print("\n== Autoevaluación: acierto esperado de cada modelo bajo la posterior media calibrada ==")
    def post(d, PP, T):
        lp = np.log(np.clip((PP["A"] + PP["B"] + PP["N"]) / 3, 1e-12, 1)) / T
        e = np.exp(lp - pd.Series(lp, index=d.index).groupby(d.sample_id).transform("max").values)
        return e / pd.Series(e, index=d.index).groupby(d.sample_id).transform("sum").values
    # temperatura elegida por log-verosimilitud en CV (con todo el OOF: es un parámetro, se valida abajo por LOCO)
    best = None
    for T in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
        pp = post(lab, P, T)
        ll = np.log(np.clip(pp[y == 1], 1e-12, 1)).mean()
        if best is None or ll > best[0]:
            best = (ll, T)
    T = best[1]
    pl, pt = post(lab, P, T), post(te, Q, T)
    E_l = pd.DataFrame({m: pd.Series(pl * (RL[m] <= 3), index=lab.index).groupby(lab.atco).sum()
                        / lab.groupby("atco").sample_id.nunique() for m in ms})
    E_t = pd.DataFrame({m: pd.Series(pt * (RT[m] <= 3), index=te.index).groupby(te.atco).sum()
                        / te.groupby("atco").sample_id.nunique() for m in ms})
    print(f"  T = {T};  corr por controlador (esperado vs real):",
          {m: round(np.corrcoef(E_l[m], acc[m])[0, 1], 3) for m in ms})
    for c in ["N-B", "N-A", "NB-N", "NB-B"]:
        a, b = c.split("-")
        ed, rd = E_l[a] - E_l[b], acc[a] - acc[b]
        # calibración LOCO de la diferencia: real ≈ k * esperado
        pr = []
        for k in idx:
            ref = [i for i in idx if i != k]
            kk = (ed[ref] * rd[ref]).sum() / (ed[ref] ** 2).sum()
            pr.append(kk * ed[k])
        pr = np.array(pr)
        print(f"  {c}: corr(esperado, real) {np.corrcoef(ed, rd)[0, 1]:+.2f}; RMSE LOCO calibrado "
              f"{np.sqrt(np.mean((rd.values - pr) ** 2)) * 100:.2f} (uniforme {uni[c]:.2f}); "
              f"esperado en T20 {(E_t.loc[20, a] - E_t.loc[20, b]) * 100:+.2f}, T23 "
              f"{(E_t.loc[23, a] - E_t.loc[23, b]) * 100:+.2f}  (sesgo medio en CV esperado-real "
              f"{(ed - rd).mean() * 100:+.2f})")
    print("  acierto esperado en test:", E_t.round(4).to_dict("index"))
    print("  acierto esperado en CV (media controladores):", E_l.mean().round(4).to_dict(),
          " real:", acc.mean().round(4).to_dict())

    # Comparación de predicciones de test por sesión
    print("\n== Test: discrepancias por sesión ==")
    rows = []
    for ses, g in st.groupby("session"):
        r = dict(session=ses, n=len(g))
        for a, b in [("N", "B"), ("N", "A"), ("A", "B")]:
            r[f"top3≠ {a}{b}"] = round((g[f"{a}_top3"] != g[f"{b}_top3"]).mean(), 3)
            r[f"top1≠ {a}{b}"] = round((g[f"{a}_t1task"] != g[f"{b}_t1task"]).mean(), 3)
        for m in ("A", "B", "N"):
            r[f"coord1 {m}"] = round(g[f"{m}_t1type"].isin([2, 3]).mean(), 3)
        r["conf N"] = round(g.N_conf.mean(), 3)
        r["instr10"] = round(g.instr10.mean(), 3)
        rows.append(r)
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(rows).to_string(index=False))
    # tipos de las tareas que entran/salen del top-3 (N frente a B y A) en el 023 y en el 020
    for at in (23, 20):
        msk = (te.atco == at).values
        for a, b in [("N", "B"), ("N", "A")]:
            sol_a = msk & (RT[a] <= 3) & (RT[b] > 3)
            sol_b = msk & (RT[b] <= 3) & (RT[a] > 3)
            ca = te.task_index[sol_a].map(tn).value_counts()
            cb = te.task_index[sol_b].map(tn).value_counts()
            print(f"  atco {at:03d}: solo en top-3 de {a} (vs {b}): {ca.to_dict()}")
            print(f"             solo en top-3 de {b} (vs {a}): {cb.to_dict()}")
            # ¿cuánta probabilidad (posterior media) mueve cada cambio? = estimación del efecto
            print(f"             posterior media en esas filas: {a} {pt[sol_a].sum():.1f} vs {b} {pt[sol_b].sum():.1f}"
                  f" (sobre {msk.sum() and te[msk].sample_id.nunique()} snapshots)")
    # mismo análisis en la CV: en las filas en que N y B discrepan, ¿quién acierta? por tipo
    print("\n== CV: filas solo en top-3 de N vs solo de B, % que son la etiqueta, por tipo ==")
    for a, b in [("N", "B"), ("N", "A")]:
        sa = (RL[a] <= 3) & (RL[b] > 3)
        sb = (RL[b] <= 3) & (RL[a] > 3)
        t = pd.DataFrame({f"n_{a}": lab.task_index[sa].value_counts(), f"y_{a}": lab.target_index[sa].groupby(lab.task_index[sa]).sum(),
                          f"n_{b}": lab.task_index[sb].value_counts(), f"y_{b}": lab.target_index[sb].groupby(lab.task_index[sb]).sum()}).fillna(0)
        t.index = [tn(k) for k in t.index]
        t["neto"] = t[f"y_{a}"] - t[f"y_{b}"]
        print(f"-- {a} vs {b} --"); print(t.astype(int).to_string())

    # Nivel sesión (37 sesiones de CV): diferencia N-B y NB-B frente a indicadores de sesión; LOCO por controlador
    print("\n== Nivel sesión: diferencia ~ indicador, LOCO por controlador (RMSE x100) ==")
    def ses_ind(sn):
        g = sn.groupby("session")
        return pd.DataFrame(dict(atco=g.atco.first(), n=g.size(), t1_coord=g.N_t1type.apply(lambda v: v.isin([2, 3]).mean()),
                                 t1_transfer=g.N_t1type.apply(lambda v: (v == 5).mean()), seq_len=g.seq_len.mean(),
                                 conf=g.N_conf.mean(), instr10=g.instr10.mean(), dis_NB=(sn.N_top3 != sn.B_top3).groupby(sn.session).mean()))
    SL, ST = ses_ind(sl), ses_ind(st)
    for m in ms:
        SL[m] = sl.groupby("session")[f"{m}_hit"].mean()
    for c in ("N-B", "NB-B", "N-A"):
        a, b = c.split("-")
        yy = SL[a] - SL[b]
        w_ = SL.n.values
        e0 = []
        for at in SL.atco.unique():
            k = (SL.atco == at).values
            e0 += list(yy[k] - np.average(yy[~k], weights=w_[~k]))
        line = f"  {c}: media ponderada {np.average(yy, weights=w_) * 100:+.2f}; uniforme RMSE {np.sqrt(np.mean(np.square(e0))) * 100:.2f}"
        for col in ("t1_coord", "t1_transfer", "seq_len", "conf", "instr10", "dis_NB"):
            e = []
            for at in SL.atco.unique():
                k = (SL.atco == at).values
                b1, b0 = np.polyfit(SL[col][~k], yy[~k], 1, w=np.sqrt(w_[~k]))
                e += list(yy[k] - (b0 + b1 * SL[col][k]))
            b1, b0 = np.polyfit(SL[col], yy, 1, w=np.sqrt(w_))
            p23 = np.average(b0 + b1 * ST.loc[ST.atco == 23, col], weights=ST.loc[ST.atco == 23, "n"])
            line += (f"\n     ~{col}: corr {np.corrcoef(SL[col], yy)[0, 1]:+.2f}, RMSE {np.sqrt(np.mean(np.square(e))) * 100:.2f},"
                     f" pred 023 {p23 * 100:+.2f}")
        print(line)
    print(ST.round(3).to_string())

    registrar("f11_simil_NB", "f11", "media de probabilidades N (f10_instr_l63_s10_ses) + B (suav_lgb_future_clear10_orcoord_ses)",
              extra=dict(cv023=round(float(acc.loc[CV023, "NB"].mean()), 4),
                         est023_dif={k: round(float(v), 4) for k, v in res["T23"]["dif"].items()},
                         est023_acc={k: round(float(v), 4) for k, v in res["T23"]["acc"].items()}))
    json.dump({k: dict(subset=v["subset"], h=float(v["h"]), ess=float(v["ess"]),
                       acc={a: float(b) for a, b in v["acc"].items()}, dif={a: float(b) for a, b in v["dif"].items()},
                       ic={a: list(map(float, b)) for a, b in v["ic"].items()}, loco={a: float(b) for a, b in v["loco"].items()})
               for k, v in res.items()}, open(OUT / "f11_simil_est.json", "w"), indent=1)

if __name__ == "__main__":
    main()

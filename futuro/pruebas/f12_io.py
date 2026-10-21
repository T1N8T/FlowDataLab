"""
f12_io.py · Juanma (C) · RAMA FUTURO · Investigación operativa / inferencia bayesiana: asignación de episodios en la sesión.

Formulación
-----------
Sesión = conjunto de rachas r (snapshots contiguos, `block`); racha r = snapshots s_1..s_n ordenados por t.
Variables aleatorias:
    Y_s ∈ T(s)        tarea objetivo del snapshot s (T(s) = tareas abiertas de s)
    Z_{i,r} ∈ {0,1}   la tarea i se atiende (es objetivo en algún snapshot) en la racha r: Z_{i,r} = 1[∃ s∈r: Y_s = i]
Hechos observados en train (EDA): dentro de la racha el objetivo es constante a trozos y cada tarea forma UN solo
segmento (3645 de 3650); 1.17 tareas distintas por racha; cada tarea es objetivo en una sola racha (98.4 %).

Modelo conjunto (campo aleatorio de Markov sobre la sesión):
    P(Y) ∝ Π_r [ Π_s  e_s(Y_s) · Π_t A(Y_{t-1}, Y_t) ] · Π_i Ψ_i(Z_{i,·})
    e_s(i) = p_N(i|s)^α                          evidencia local = marginal de N (logit de sesión, OOF) templada
    A(k,k) = 1-ρ,  A(k,j) = ρ/(K_r-1)·φ_{i→r}     cadena semi-Markov: seguir con la tarea o cambiar (segmentos)
    Ψ_i(z) = exp(-λ · max(0, Σ_r z_r - 1))      restricción blanda "cada tarea a lo sumo un episodio en la sesión"
                                                  (λ = ∞: restricción dura Σ_r Z_{i,r} ≤ 1 del problema de asignación)
El problema de IO  max Σ log e  s.t. Σ_r x_{i,r} ≤ 1, Σ_i y_{i,s} = 1, y_{i,s} ≤ x_{i,r(s)}  es su MAP; aquí se usa la
relajación entrópica (energía libre de Bethe), que da MARGINALES en vez de una sola asignación:
  - Cada racha es un árbol (cadena): forward-backward exacto da q(Y_s = i) y x_{i,r} = P(Z_{i,r}=1) = 1 - Z_{-i}/Z
    (Z_{-i} = función de partición con e(i) = 0: K cadenas en paralelo).
  - Factor Ψ_i: mensajes de BP (loopy, 2-3 iteraciones con amortiguación). La racha r recibe, por tarea, el precio
        φ_{i→r} = Π_{r'≠r} (1 - x̂_{i,r'} (1 - e^{-λ}))      (x̂ = belief de cavidad: odds(x̂) = odds(x)/φ)
    que se paga UNA vez al entrar en el segmento de i (inicio de la racha o cambio). Es el "precio dual" de la
    capacidad de la tarea: si otra racha ya la reclama, entrar en ella aquí cuesta.
Salida: score = (1-w)·log p_N + w·log q(Y_s = i).
Top-3 bajo incertidumbre conjunta: para un snapshot, P(acierto | S) = Σ_{i∈S} q(Y_s=i) es lineal en las marginales,
así que el top-3 óptimo bajo CUALQUIER distribución conjunta es el top-3 de las marginales; la incertidumbre conjunta
entra solo a través de q (que ya es la marginal del modelo conjunto).
Estimación: e_s viene del OOF de N (sin etiquetas). Hiperparámetros (α, ρ, λ, w) por CV anidada 4->5 (para cada
fold externo se eligen con los otros 4 folds; ningún parámetro se ajusta con etiquetas del fold evaluado). Test: los
parámetros elegidos con los 5 folds y los preds de N. Sin fuga: la inferencia es no supervisada y por sesión
(una sesión está entera en un fold).

Uso: python futuro/pruebas/f12_io.py [base]      (base por defecto f10_instr_l63_s10_ses)
"""
import sys, time, json, itertools
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402
from suavizado import leer, lsm  # noqa: E402
sys.path.insert(0, str(ROOT / "stacking"))
from stack import hit3  # noqa: E402

CV023 = [2, 11, 10, 6, 17, 15]


def preparar(d):
    """Estructuras por racha: índices de filas (n_snap x K) con -1 si la tarea no está en el snapshot."""
    snap = d.drop_duplicates("sample_id")[["sample_id", "session", "block", "t"]].sort_values(["session", "block", "t"])
    pos = {s: k for k, s in enumerate(snap.sample_id)}
    rachas = []
    rows = d[["sample_id", "block", "session", "task_name"]].copy()
    rows["i"] = np.arange(len(d))
    rows["ps"] = rows.sample_id.map(pos)
    for (ses, blk), g in rows.groupby(["session", "block"], sort=False):
        tasks, tcode = np.unique(g.task_name.values, return_inverse=True)
        sids, scode = np.unique(g.ps.values, return_inverse=True)   # orden temporal
        M = np.full((len(sids), len(tasks)), -1, int)
        M[scode, tcode] = g.i.values
        rachas.append(dict(ses=ses, tasks=tasks, M=M))
    return rachas


def cadena(E, phi, rho):
    """Forward-backward en una racha. E: (n,K) evidencias (0 = ausente); phi: (K,) precio de entrada.
    Devuelve (q (n,K) marginales, x (K,) P(la tarea aparece en la racha))."""
    n, K = E.shape
    sw = rho / (K - 1) if K > 1 else 0.0
    st = 1 - rho if K > 1 else 1.0
    # forward normal
    a = np.empty((n, K)); c = np.empty(n)
    v = E[0] * phi
    c[0] = v.sum(); a[0] = v / c[0]
    for t in range(1, n):
        v = E[t] * (st * a[t - 1] + sw * phi * (1 - a[t - 1]))
        c[t] = v.sum(); a[t] = v / c[t]
    b = np.ones((n, K))
    for t in range(n - 2, -1, -1):
        u = E[t + 1] * b[t + 1]
        v = st * u + sw * ((phi * u).sum() - phi * u)
        b[t] = v / c[t + 1]
    q = a * b
    q /= q.sum(1, keepdims=True)
    logZ = np.log(c).sum()
    # K cadenas en paralelo, cada una sin la tarea i (fila i de A)
    if K == 1:
        return q, np.ones(1)
    mask = 1 - np.eye(K)
    v = mask * (E[0] * phi)[None, :]
    s = v.sum(1); s = np.where(s > 0, s, 1e-300); A = v / s[:, None]; lz = np.log(s)
    for t in range(1, n):
        v = mask * E[t][None, :] * (st * A + sw * phi[None, :] * (A.sum(1, keepdims=True) - A))
        s = v.sum(1); s = np.where(s > 0, s, 1e-300); A = v / s[:, None]; lz += np.log(s)
    x = 1 - np.exp(np.minimum(lz - logZ, 0))
    return q, np.clip(x, 1e-9, 1 - 1e-9)


def inferir(rachas, p, alpha, rho, lam, iters=3, damp=0.5):
    """Marginales q por fila (alineadas con p) con BP entre rachas de la misma sesión."""
    Es = []
    for R in rachas:
        M = R["M"]
        E = np.where(M >= 0, p[np.maximum(M, 0)], 0.0) ** alpha
        Es.append(E)
    phis = [np.ones(len(R["tasks"])) for R in rachas]
    q_out = np.zeros(len(p))
    n_it = 1 if lam == 0 else iters + 1
    for it in range(n_it):
        xs = []
        for R, E, phi in zip(rachas, Es, phis):
            q, x = cadena(E, phi, rho)
            xs.append(x)
            if it == n_it - 1:
                M = R["M"]; ok = M >= 0
                q_out[M[ok]] = q[ok]
        if it == n_it - 1:
            break
        # mensajes de cavidad hacia el factor de la tarea
        k = 1 - np.exp(-lam)
        tot = {}
        cav = []
        for R, x, phi in zip(rachas, xs, phis):
            odds = x / (1 - x) / phi
            xh = odds / (1 + odds)
            lt = np.log(np.maximum(1 - k * xh, 1e-12))
            cav.append(lt)
            for tn, l in zip(R["tasks"], lt):
                key = (R["ses"], tn)
                tot[key] = tot.get(key, 0.0) + l
        new = []
        for R, lt, phi in zip(rachas, cav, phis):
            T = np.array([tot[(R["ses"], tn)] for tn in R["tasks"]])
            ph = np.exp(T - lt)
            new.append(np.exp(damp * np.log(phi) + (1 - damp) * np.log(np.maximum(ph, 1e-12))))
        phis = new
    return q_out


def main(base="f10_instr_l63_s10_ses"):
    t0 = time.time()
    lab, te, _ = datos()
    cols = ["sample_id", "row_index", "session", "block", "t", "task_name", "fold", "atco"]
    L = lab[cols + ["target_index"]].copy(); L["y"] = L.target_index
    T = te[cols].copy()
    L = L.merge(leer(base, "oof"), on=["sample_id", "row_index"], how="left")
    T = T.merge(leer(base, "preds"), on=["sample_id", "row_index"], how="left")
    assert L.score.notna().all() and T.score.notna().all()
    lpL, lpT = lsm(L, L.score.values), lsm(T, T.score.values)
    pL, pT = np.exp(lpL), np.exp(lpT)
    rL, rT = preparar(L), preparar(T)
    print(f"preparado en {time.time() - t0:.0f}s; rachas lab {len(rL)}, test {len(rT)}", flush=True)
    fold = L.drop_duplicates("sample_id").set_index("sample_id").fold
    atco = L.drop_duplicates("sample_id").set_index("sample_id").atco

    def hits_snap(score):
        s = pd.Series(score, index=L.index)
        tpos = L.index[L.y == 1]
        ts = s.loc[tpos].values
        g = L.sample_id.values
        rank = pd.DataFrame(dict(g=g, b=s.values)).merge(
            pd.DataFrame(dict(g=L.sample_id.loc[tpos].values, ts=ts)), on="g")
        nb = (rank.b > rank.ts).groupby(rank.g).sum()
        return (nb < 3).astype(float)

    h0 = hits_snap(lpL)
    print(f"base {base}: CV {h0.mean():.4f}  CV023 {h0[atco.reindex(h0.index).isin(CV023).values].mean():.4f}", flush=True)

    grid_hmm = list(itertools.product((0.5, 0.75, 1.0), (0.02, 0.08, 0.2), (0.0, 1.0, 3.0, 30.0)))
    W = (0.3, 0.5, 0.7, 1.0)
    res = {}   # (alpha,rho,lam,w) -> hits por snapshot
    qs = {}
    for alpha, rho, lam in grid_hmm:
        t1 = time.time()
        q = inferir(rL, pL, alpha, rho, lam)
        lq = np.log(np.maximum(q, 1e-12))
        qs[(alpha, rho, lam)] = lq
        for w in W:
            res[(alpha, rho, lam, w)] = hits_snap((1 - w) * lpL + w * lq)
        best_w = max(W, key=lambda w: res[(alpha, rho, lam, w)].mean())
        print(f"a={alpha} rho={rho} lam={lam}: " + " ".join(f"w{w}={res[(alpha, rho, lam, w)].mean():.4f}" for w in W)
              + f"  ({time.time() - t1:.0f}s)", flush=True)

    # CV anidada 4->5
    fo = fold.reindex(h0.index).values
    nested = pd.Series(np.nan, index=h0.index)
    elegidos = {}
    for f in sorted(np.unique(fo)):
        tr = fo != f
        best = max(res, key=lambda k: res[k][tr].mean())
        elegidos[int(f)] = best
        nested[~tr] = res[best][~tr].values
    at = atco.reindex(h0.index).values
    m023 = np.isin(at, CV023)
    print("elegidos por fold:", elegidos)
    print(f"CV anidada {nested.mean():.4f} (base {h0.mean():.4f}); CV023 anidada {nested[m023].mean():.4f} "
          f"(base {h0[m023].mean():.4f})", flush=True)
    pa_n, pa_0 = nested.groupby(at).mean(), h0.groupby(at).mean()
    print("por controlador (anidada - base):", (pa_n - pa_0).round(4).to_dict())
    # sin restricción entre rachas (solo cadena) como ablación: mejor lam=0 anidado
    res0 = {k: v for k, v in res.items() if k[2] == 0.0}
    nest0 = pd.Series(np.nan, index=h0.index)
    for f in sorted(np.unique(fo)):
        tr = fo != f
        best = max(res0, key=lambda k: res0[k][tr].mean())
        nest0[~tr] = res0[best][~tr].values
    print(f"ablación solo cadena (lam=0): CV anidada {nest0.mean():.4f}; CV023 {nest0[m023].mean():.4f}", flush=True)

    best = max(res, key=lambda k: res[k].mean())
    alpha, rho, lam, w = best
    print("parámetros finales (todo el OOF):", best, f"CV optimista {res[best].mean():.4f}", flush=True)
    soof = (1 - w) * lpL + w * qs[(alpha, rho, lam)]
    qT = inferir(rT, pT, alpha, rho, lam)
    spred = (1 - w) * lpT + w * np.log(np.maximum(qT, 1e-12))
    nombre = "f12_io" if base == "f10_instr_l63_s10_ses" else f"f12_io_{base}"
    guardar(nombre, L, soof, T, spred)
    pa = nested.groupby(at).mean()
    registrar(nombre, hilo="f12", segundos=time.time() - t0,
              descripcion=f"IO/BP: cadena semi-Markov por racha + restricción blanda 1 episodio/tarea sobre {base}; "
                          f"params {best} (CV del OOF guardado = optimista; ver cv_anidada)",
              extra=dict(cv_anidada=round(float(nested.mean()), 4), cv023_anidada=round(float(nested[m023].mean()), 4),
                         cv_base=round(float(h0.mean()), 4), cv023_base=round(float(h0[m023].mean()), 4),
                         min_atco_anidada=round(float(pa.min()), 4),
                         cv_anidada_solo_cadena=round(float(nest0.mean()), 4),
                         elegidos={str(k): list(v) for k, v in elegidos.items()}, params=list(best)))


if __name__ == "__main__":
    main(*sys.argv[1:])

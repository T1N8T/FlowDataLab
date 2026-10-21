"""f11_trans.py · Juanma (C) · hilo f11 · HMM con transiciones APRENDIDAS entre tareas (rompe la regla 1: usa la sesión).

f6_ideas.hmm usaba "sigue con la misma tarea con prob. s o cambia uniforme" (sin mejora). Aquí la transición
P(B en el snapshot q | A en el snapshot p) entre snapshots consecutivos de la sesión es un logit condicional (softmax
sobre las tareas B de q) aprendido con las ETIQUETAS de los folds de entrenamiento (A = respuesta en p, B = en q):
    stay            B es la misma tarea que A
    samecs          B es otra tarea del mismo avión que A
    sw*A_res        cambio y el avión de A recibe entre p y q una instrucción COMPATIBLE con A (si_match de A en p <= dt)
    sw*A_any        cambio y el avión de A recibe alguna instrucción entre p y q
    sw*Bnew         cambio a una tarea que no estaba en p
    sw*Bnew*A_res
    sw*tipo(A)xtipo(B)   matriz 14x14 de cambios de tipo (L2)
con pesos distintos dentro de la racha (block, cada ~3 s) y entre rachas (hueco de 5-280 s).
Emisión = p^tau (p = softmax de N tras la sesión, f10_instr_l63_s3_ses). Forward-backward sobre la sesión entera:
    log T = lam_k * phi(A,B) normalizado por filas (lam_in dentro de la racha, lam_bt entre rachas)
    score = (1-w) log p + w log P(tarea | sesión)
(tau, lam_in, lam_bt, w) con CV anidada (4 folds -> 5º); los pesos de la transición del fold interno g se ajustan con
los folds que no son ni f ni g, y los del fold externo f con los 4 restantes. Test: pesos con todo lab.
Sin fuga: las etiquetas solo se usan para ajustar pesos en folds de entrenamiento; en el fold que se predice solo
entran los scores OOF de N (que no vieron su etiqueta) y features sin etiquetas.

Uso: python futuro/pruebas/f11_trans.py [MODELO_BASE]     (por defecto f10_instr_l63_s3_ses) -> f11_trans_<...>
"""
import sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import OUT, guardar, registrar  # noqa: E402
from f10_instr import datos_instr, medir  # noqa: E402
import suavizado as S  # noqa: E402

K = ["sample_id", "row_index"]
NT = 14
NF = 6                                  # features densas por tipo de paso
GRID = [(tau, li, lb, w) for tau, li, lb, w in itertools.product((0.35, 0.5, 0.75, 1.0), (0.5, 1.0), (0.0, 0.5, 1.0), (0.5, 1.0))]
GRID = [(1.0, 0.0, 0.0, 1.0)] + GRID    # "no hacer nada" (mismo orden que p)


def preparar(d):
    """Pares de snapshots consecutivos de cada sesión con sus matrices de features (Ka x Kb)."""
    code = pd.factorize(d.sample_id)[0]
    st = np.r_[0, np.flatnonzero(np.diff(code)) + 1]; en = np.r_[st[1:], len(d)]
    snap = d.iloc[st][["session", "block", "t"]].assign(c=np.arange(len(st))).sort_values(["session", "t"])
    tn = pd.factorize(d.session.astype(str) + "|" + d.task_name.astype(str))[0]
    cs = pd.factorize(d.session.astype(str) + "|" + d.cs.astype(str))[0]
    ty = d.task_index.fillna(0).astype(int).clip(0, NT - 1).values
    sim = d.si_match.values
    sinx = np.nanmin(d[[c for c in d.columns if c.startswith("si_next_")]].values, axis=1)
    y = d.y.values if "y" in d else None
    pairs, chains = [], []
    for ses, g in snap.groupby("session", sort=False):
        cc, bb, tt = g.c.values, g.block.values, g.t.values
        chains.append(cc)
        for i in range(1, len(cc)):
            P, Q = cc[i - 1], cc[i]
            ia, ib = np.arange(st[P], en[P]), np.arange(st[Q], en[Q])
            dt = tt[i] - tt[i - 1] + 1.5
            stay = tn[ia][:, None] == tn[ib][None, :]
            sw = ~stay
            samecs = (cs[ia][:, None] == cs[ib][None, :]) & sw
            ares = np.nan_to_num(sim[ia], nan=1e9) <= dt
            aany = np.nan_to_num(sinx[ia], nan=1e9) <= dt
            bnew = ~np.isin(tn[ib], tn[ia])
            F = np.stack([stay, samecs, sw & ares[:, None], sw & aany[:, None], sw & bnew[None, :],
                          sw & ares[:, None] & bnew[None, :]], -1).astype(np.int8)
            tp = (ty[ia][:, None] * NT + ty[ib][None, :]).astype(np.int16)
            tp[stay] = NT * NT                                  # casilla nula
            a = int(np.flatnonzero(y[ia])[0]) if y is not None else -1
            b = int(np.flatnonzero(y[ib])[0]) if y is not None else -1
            pairs.append(dict(P=P, Q=Q, k=int(bb[i] != bb[i - 1]), F=F, tp=tp, a=a, b=b))
    return dict(st=st, en=en, code=code, pairs=pairs, chains=chains, fold=d.fold.values[st],
                pidx={(p["P"], p["Q"]): j for j, p in enumerate(pairs)})


def fit_trans(H, folds, reg=2e-4):
    """Logit condicional de la transición con los pares de los folds dados. Devuelve pesos (2, NF) y (2, NT*NT+1)."""
    sel = [p for p in H["pairs"] if H["fold"][p["P"]] in folds and H["fold"][p["Q"]] in folds]
    Xs, Ts, Ys, Gs, Ks = [], [], [], [], []
    for j, p in enumerate(sel):
        Xs.append(p["F"][p["a"]]); Ts.append(p["tp"][p["a"]]); yv = np.zeros(p["F"].shape[1]); yv[p["b"]] = 1
        Ys.append(yv); Gs.append(np.full(len(yv), j)); Ks.append(np.full(len(yv), p["k"]))
    X = np.concatenate(Xs).astype(float); TP = np.concatenate(Ts).astype(int); Y = np.concatenate(Ys)
    G = np.concatenate(Gs); KK = np.concatenate(Ks); ng = len(sel)
    nd, nt = 2 * NF, 2 * (NT * NT + 1)
    TPk = TP + KK * (NT * NT + 1)
    Xk = np.zeros((len(X), nd)); Xk[KK == 0, :NF] = X[KK == 0]; Xk[KK == 1, NF:] = X[KK == 1]
    nullmask = np.zeros(nt, bool); nullmask[NT * NT] = nullmask[2 * NT * NT + 1] = True

    def nll(w):
        wd, wt = w[:nd], w[nd:].copy(); wt[nullmask] = 0
        s = Xk @ wd + wt[TPk]
        m = np.full(ng, -np.inf); np.maximum.at(m, G, s)
        e = np.exp(s - m[G]); z = np.bincount(G, weights=e, minlength=ng)
        lp = s - m[G] - np.log(z[G])
        r = np.exp(lp) - Y
        gd = Xk.T @ r / ng
        gt = np.bincount(TPk, weights=r, minlength=nt) / ng + 2 * reg * wt; gt[nullmask] = 0
        return -(lp * Y).sum() / ng + reg * (wt ** 2).sum(), np.r_[gd + 2e-5 * wd, gt]
    w = minimize(lambda w: (lambda f, g: (f + 1e-5 * (w[:nd] ** 2).sum(), g))(*nll(w)), np.zeros(nd + nt), jac=True,
                 method="L-BFGS-B", options=dict(maxiter=500)).x
    wt = w[nd:]; wt[nullmask] = 0
    return w[:nd].reshape(2, NF), wt.reshape(2, NT * NT + 1)


def trans_mats(H, W, lam, which):
    """log T por par (solo los pares en `which`), normalizado por filas."""
    wd, wt = W
    out = {}
    for j in which:
        p = H["pairs"][j]; k = p["k"]
        if lam[k] == 0:
            out[j] = None; continue
        phi = lam[k] * (p["F"] @ wd[k] + wt[k][p["tp"]])
        phi -= phi.max(1, keepdims=True)
        out[j] = np.exp(phi); out[j] /= out[j].sum(1, keepdims=True)
    return out


def fb(H, lp, chains, T, tau):
    """Forward-backward por cadena. lp = log p por fila. Devuelve log posterior por fila (NaN fuera de las cadenas)."""
    st, en = H["st"], H["en"]
    out = np.full(len(lp), np.nan)
    for cc in chains:
        n = len(cc)
        E = [np.exp(tau * (lp[st[c]:en[c]] - lp[st[c]:en[c]].max())) for c in cc]
        A = [None] * n; B = [None] * n
        A[0] = E[0] / E[0].sum()
        Ts = [T[H["pidx"][(cc[i - 1], cc[i])]] for i in range(1, n)]
        for i in range(1, n):
            t = Ts[i - 1]
            a = E[i] * (A[i - 1] @ t if t is not None else 1.0)
            A[i] = a / a.sum()
        B[n - 1] = np.ones(len(E[n - 1]))
        for i in range(n - 2, -1, -1):
            t = Ts[i]; v = E[i + 1] * B[i + 1]
            b = t @ v if t is not None else np.full(len(E[i]), v.sum())
            B[i] = b / b.sum()
        for i, c in enumerate(cc):
            q = A[i] * B[i]
            out[st[c]:en[c]] = np.log(q / q.sum() + 1e-300)
    return out


def posteriores(H, lp, folds, W, grid):
    """{config: score} para las filas de las sesiones de `folds`, con pesos W."""
    chains = [cc for cc in H["chains"] if H["fold"][cc[0]] in folds]
    which = [H["pidx"][(cc[i - 1], cc[i])] for cc in chains for i in range(1, len(cc))]
    res = {}
    for lam in sorted({(g[1], g[2]) for g in grid}):
        T = trans_mats(H, W, lam, which)
        for tau in sorted({g[0] for g in grid if (g[1], g[2]) == lam}):
            post = fb(H, lp, chains, T, tau)
            for g in grid:
                if (g[1], g[2], g[0]) == (lam[0], lam[1], tau):
                    res[g] = (1 - g[3]) * lp + g[3] * post
    return res


def main(base="f10_instr_l63_s3_ses"):
    t0 = time.time()
    lab, te, _ = datos_instr()
    L = lab.merge(pd.read_csv(OUT / "oof" / f"{base}.csv"), on=K, how="left").assign(y=lambda x: x.target_index)
    T_ = te.merge(pd.read_csv(OUT / "preds" / f"{base}.csv"), on=K, how="left").assign(fold=-1)
    assert L.score.notna().all() and T_.score.notna().all()
    lpL, lpT = S.lsm(L, L.score.values), S.lsm(T_, T_.score.values)
    HL, HT = preparar(L), preparar(T_)
    print(f"preparado: {len(HL['pairs'])} pares lab, {len(HT['pairs'])} test ({time.time() - t0:.0f}s)", flush=True)
    folds = sorted(L.fold.unique())
    W = {f: fit_trans(HL, [g for g in folds if g != f]) for f in folds}
    W["all"] = fit_trans(HL, folds)
    wd = W["all"][0]
    print("pesos (todo lab) dentro racha:", dict(zip(["stay", "samecs", "A_res", "A_any", "Bnew", "Bnew*A_res"], wd[0].round(2))),
          "\n               entre rachas:", dict(zip(["stay", "samecs", "A_res", "A_any", "Bnew", "Bnew*A_res"], wd[1].round(2))), flush=True)
    rowfold = L.fold.values
    # externo: posteriores del fold f con W[f]
    outer = {g: np.full(len(L), np.nan) for g in GRID}
    for f in folds:
        r = posteriores(HL, lpL, [f], W[f], GRID)
        m = rowfold == f
        for g in GRID:
            outer[g][m] = r[g][m]
    print(f"externo listo ({time.time() - t0:.0f}s)", flush=True)
    no_anid = sorted(((S.hit3(L, outer[g]), g) for g in GRID), reverse=True)
    print("sin anidar (optimista):", [f"{h:.4f} {g}" for h, g in no_anid[:6]], " nada:", f"{S.hit3(L, outer[GRID[0]]):.4f}", flush=True)
    # anidado
    nested, elegidos = np.full(len(L), np.nan), []
    for f in folds:
        inner = {g: np.full(len(L), np.nan) for g in GRID}
        for g2 in folds:
            if g2 == f:
                continue
            r = posteriores(HL, lpL, [g2], fit_trans(HL, [x for x in folds if x not in (f, g2)]), GRID)
            m = rowfold == g2
            for g in GRID:
                inner[g][m] = r[g][m]
        tr = rowfold != f
        best = max(GRID, key=lambda g: S.hit3(L, np.nan_to_num(inner[g], nan=-1e9), tr))
        elegidos.append(best)
        nested[rowfold == f] = outer[best][rowfold == f]
        print(f"  fold {f}: elegido {best}  ({time.time() - t0:.0f}s)", flush=True)
    gfin = no_anid[0][1]
    pT = posteriores(HT, lpT, [-1], W["all"], [gfin])[gfin]
    c0, c023_0, mn0 = medir("", L, lpL)
    c, c023, mn = medir("", L, nested)
    co, c023o, mno = medir("", L, outer[gfin])
    print(f"{base}: {c0:.4f} (CV 023 {c023_0:.4f}, peor {mn0:.3f}) -> anidada {c:.4f} (CV 023 {c023:.4f}, peor {mn:.3f}) "
          f"Δ {100 * (c - c0):+.2f} · final {gfin}: {co:.4f} (CV 023 {c023o:.4f}) ({time.time() - t0:.0f}s)", flush=True)
    tag = "" if base == "f10_instr_l63_s3_ses" else "_" + base
    for nm, o, p in ((f"f11_trans{tag}", outer[gfin], pT), (f"f11_trans{tag}_anid", nested, None)):
        guardar(nm, L, o, T_ if p is not None else None, p)
    ex = dict(cv_ref=round(c0, 4), cv_anidada=round(c, 4), cv_023_anidada=round(c023, 4), peor_anidada=round(mn, 4),
              cv_023=round(c023o, 4), params=list(gfin), params_por_fold=[list(x) for x in elegidos],
              pesos_in=wd[0].round(3).tolist(), pesos_bt=wd[1].round(3).tolist())
    registrar(f"f11_trans{tag}", hilo="f11", descripcion=f"HMM transiciones aprendidas sobre {base}, params {gfin} (todo OOF)",
              segundos=time.time() - t0, extra=ex)
    registrar(f"f11_trans{tag}_anid", hilo="f11", descripcion=f"HMM transiciones aprendidas sobre {base}, CV anidada",
              extra=ex)




# ----------------------------------------------------------------------------- modo feat
# El HMM sobre N_ses resta (N_ses ya mira la sesión). Variante: la log-posterior del HMM con transiciones aprendidas
# (emisión del 1.er pase o de N_ses) entra como una COLUMNA más del logit de sesión de feat_session (en vez de mezclar a
# mano), y el logit decide su peso. CV anidada estricta: en el fold interno g, transición y logit con folds ∉ {f, g}.
FGRID = [(tau, li, lb) for tau in (0.5, 1.0) for li, lb in ((1.0, 1.0), (1.0, 0.5), (0.5, 0.5), (1.0, 0.0))]


def feat_main(first="f10_instr_l63_s3", emis="first"):
    import feat_session as FS
    t0 = time.time()
    lab, te, _ = datos_instr()
    def cargar(d, nm, carp):
        v = d[K].merge(pd.read_csv(OUT / carp / f"{nm}.csv"), on=K, how="left").score.values
        assert not np.isnan(v).any(); return v
    L = lab.assign(y=lab.target_index); T_ = te.assign(fold=-1)
    s1L, s1T = cargar(L, first, "oof"), cargar(T_, first, "preds")
    if emis == "first":
        eL, eT = s1L, s1T
    else:
        eL, eT = cargar(L, first + "_ses", "oof"), cargar(T_, first + "_ses", "preds")
    lpL, lpT = S.lsm(L, eL), S.lsm(T_, eT)
    XL, XT = FS.session_feats(L, s1L).values, FS.session_feats(T_, s1T).values
    HL, HT = preparar(L), preparar(T_)
    folds = sorted(L.fold.unique()); rf = L.fold.values; yv = L.y.values; sid = L.sample_id.values
    grid = [(g[0], g[1], g[2], 1.0) for g in FGRID]
    Wc = {}
    def W(ex):
        k = tuple(sorted(ex))
        if k not in Wc:
            Wc[k] = fit_trans(HL, [x for x in folds if x not in ex])
        return Wc[k]
    def post_folds(evalf, ex):          # posteriores en las sesiones de evalf con transición sin los folds ex
        return posteriores(HL, lpL, [evalf], W(ex), grid)
    def logit_oof(Z, evalf, ex):
        tr = ~np.isin(rf, list(ex) + [evalf]); va = rf == evalf
        return Z[va] @ FS.fit_logit(Z[tr], yv[tr], sid[tr])
    # externo: para cada config, OOF con transición y logit sin el fold f
    outer = {g: np.full(len(L), np.nan) for g in grid}
    ref = np.full(len(L), np.nan)
    for f in folds:
        r = post_folds(f, [f]); va = rf == f
        for g in grid:
            Z = np.c_[XL, np.nan_to_num(r[g], nan=-50).clip(-50)]
            outer[g][va] = logit_oof(Z, f, [])
        ref[va] = logit_oof(XL, f, [])
    print(f"externo ({time.time() - t0:.0f}s): ref sesión {S.hit3(L, ref):.4f};",
          [f"{S.hit3(L, outer[g]):.4f} {g[:3]}" for g in grid], flush=True)
    nested, el = np.full(len(L), np.nan), []
    for f in folds:
        inner = {g: np.full(len(L), np.nan) for g in grid}
        for g2 in folds:
            if g2 == f:
                continue
            r = post_folds(g2, [f, g2]); va = rf == g2
            for g in grid:
                inner[g][va] = logit_oof(np.c_[XL, np.nan_to_num(r[g], nan=-50).clip(-50)], g2, [f])
        tr = rf != f
        best = max(grid, key=lambda g: S.hit3(L, np.nan_to_num(inner[g], nan=-1e9), tr))
        el.append(best); nested[rf == f] = outer[best][rf == f]
        print(f"  fold {f}: {best[:3]} ({time.time() - t0:.0f}s)", flush=True)
    gfin = max(grid, key=lambda g: S.hit3(L, outer[g]))
    # final: transición y logit con todo lab; config gfin
    rL = {}
    for f in folds:
        rr = post_folds(f, [f]); rL[f] = rr[gfin]
    zl = np.full(len(L), np.nan)
    for f in folds:
        zl[rf == f] = rL[f][rf == f]
    wl = FS.fit_logit(np.c_[XL, np.nan_to_num(zl, nan=-50).clip(-50)], yv, sid)
    pT = posteriores(HT, lpT, [-1], W([]), [gfin])[gfin]
    pred = np.c_[XT, np.nan_to_num(pT, nan=-50).clip(-50)] @ wl
    a, b, c = medir("", L, ref), medir("", L, nested), medir("", L, outer[gfin])
    print(f"feat emis={emis}: sesión ref {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) -> anidada {b[0]:.4f} "
          f"(CV 023 {b[1]:.4f}, peor {b[2]:.3f}) Δ {100 * (b[0] - a[0]):+.2f} · final {gfin[:3]} {c[0]:.4f} (CV 023 {c[1]:.4f}, "
          f"peor {c[2]:.3f}); peso de la columna HMM {wl[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    nm = f"f11_trans_feat_{emis}"
    guardar(nm, L, outer[gfin], T_, pred); guardar(nm + "_anid", L, nested)
    ex = dict(cv_ref_sesion=round(a[0], 4), cv_anidada=round(b[0], 4), cv_023_anidada=round(b[1], 4), peor_anidada=round(b[2], 4),
              cv_023=round(c[1], 4), params=list(gfin), params_por_fold=[list(x) for x in el], peso_hmm=round(float(wl[-1]), 3))
    registrar(nm, hilo="f11", descripcion=f"log-post HMM transiciones aprendidas (emisión {emis}) como columna del logit de sesión sobre {first}; config {gfin}",
              segundos=time.time() - t0, extra=ex)
    registrar(nm + "_anid", hilo="f11", descripcion=f"ídem, CV anidada", extra=ex)


if __name__ == "__main__":
    if sys.argv[1:2] == ["feat"]:
        feat_main(*sys.argv[2:])
    else:
        main(*sys.argv[1:])

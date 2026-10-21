"""
f12_kalman.py · Juanma (C) · RAMA FUTURO · Filtro de Kalman + suavizador RTS de la "atención" de cada tarea.

Formulación (modelo de espacio de estados lineal gaussiano, uno por cadena c = (session, task_name)):
  Sean t_1 < ... < t_L los instantes de los snapshots de la sesión en que aparece la tarea c,
  Δ_k = t_k - t_{k-1} (s) y g_k = 1 si t_{k-1} y t_k están en rachas (block) distintas (hueco).
  Estado latente   x_k = atención (en escala log) que el controlador presta a la tarea c en el snapshot k.
  Transición       x_k = x_{k-1} + δ·Δ_k + w_k,     w_k ~ N(0, q·Δ_k + q_gap·g_k)   (paseo aleatorio con deriva δ)
  Observación      y_k = x_k + v_k,                  v_k ~ N(0, r)
  con y_k = log-softmax (o logit de la prob.) del score del modelo N (f10_instr_l63_s10_ses, o su 1.er pase)
  de la fila de c en el snapshot k. w, v independientes entre sí y entre cadenas; prior difuso x_1 ~ N(y_1, ∞).
  La media posterior E[x_k | y_1..y_L] es invariante a escalar (q, q_gap, r) juntos, así que r = 1 y los
  parámetros libres son λ = q/r (por segundo), γ = q_gap/r, δ y la mezcla final
      score_k = (1 - w)·y_k + w·E[x_k | y_{1:L}]          (w = 1: posterior pura).
  Estimación de x: filtro de Kalman hacia delante (m_k|k, P_k|k) y suavizador de Rauch-Tung-Striebel hacia atrás,
  exactos (escalar), vectorizados sobre todas las cadenas.
  Estimación de (entrada, transformación, λ, γ, δ, w): rejilla, maximizando el top-3 en CV anidada 4->5
  (para el fold externo f se eligen con los otros 4 folds y se aplican a f); para test, con todo lab.
  Sin fuga: el suavizador solo usa scores OOF de la misma sesión (mismo controlador, mismo fold); las etiquetas
  solo entran en la elección de parámetros, siempre con folds de entrenamiento.
  Se compara también con el suavizado prob (suavizado.suavizar_prob) elegido con la misma CV anidada.

Uso: nohup .venv/bin/python futuro/pruebas/f12_kalman.py > futuro/pruebas/f12_kalman.log 2>&1 &
"""
import sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402
import suavizado as S  # noqa: E402

ATCO023 = [2, 11, 10, 6, 17, 15]
K = ["sample_id", "row_index"]
ENTRADAS = {"ses": "f10_instr_l63_s10_ses", "p1": "f10_instr_l63_s10"}
LAMS = [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
GAPS = [0.0, 1.0, 10.0, 1e6]
DRIFTS = [-0.01, 0.0, 0.01]
WS = [0.25, 0.5, 0.75, 1.0]
TRANS = ["lsm", "logit"]


def cadenas(d):
    """Matriz (n_cadenas, Lmax) con el índice de fila de cada observación, Δt y hueco, ordenada por t."""
    o = d[["session", "task_name", "t", "block"]].reset_index().rename(columns={"index": "i"})
    o = o.sort_values(["session", "task_name", "t"]).reset_index(drop=True)
    cid = o.groupby(["session", "task_name"], sort=False).ngroup().values
    pos = o.groupby(cid).cumcount().values
    n, L = cid.max() + 1, pos.max() + 1
    idx = np.full((n, L), -1)
    idx[cid, pos] = o.i.values
    dt = np.zeros((n, L)); gap = np.zeros((n, L))
    tt = o.t.values.astype(float); bb = o.block.values
    first = pos == 0
    dtv = np.r_[0, np.diff(tt)]; gv = np.r_[0, (bb[1:] != bb[:-1]).astype(float)]
    dtv[first] = 0; gv[first] = 0
    dt[cid, pos] = dtv; gap[cid, pos] = gv
    return idx, dt, gap


def rts(y, idx, dt, gap, lam, gam, drift):
    """Media posterior suavizada E[x_k | y_1:L] por fila (r = 1)."""
    valid = idx >= 0
    Y = np.where(valid, y[np.maximum(idx, 0)], 0.0)
    n, L = idx.shape
    mf = np.zeros((n, L)); Pf = np.zeros((n, L)); mp = np.zeros((n, L)); Pp = np.zeros((n, L))
    mf[:, 0] = Y[:, 0]; Pf[:, 0] = 1.0; mp[:, 0] = Y[:, 0]; Pp[:, 0] = np.inf
    for k in range(1, L):
        v = valid[:, k]
        mpk = mf[:, k - 1] + drift * dt[:, k]
        Ppk = Pf[:, k - 1] + lam * dt[:, k] + gam * gap[:, k]
        Kk = Ppk / (Ppk + 1.0)
        mp[:, k] = np.where(v, mpk, mf[:, k - 1]); Pp[:, k] = np.where(v, Ppk, Pf[:, k - 1])
        mf[:, k] = np.where(v, mpk + Kk * (Y[:, k] - mpk), mf[:, k - 1])
        Pf[:, k] = np.where(v, (1 - Kk) * Ppk, Pf[:, k - 1])
    ms = mf.copy()
    for k in range(L - 2, -1, -1):
        v = valid[:, k + 1]
        C = Pf[:, k] / Pp[:, k + 1]
        ms[:, k] = np.where(v, mf[:, k] + C * (ms[:, k + 1] - mp[:, k + 1]), mf[:, k])
    out = np.empty(len(y))
    out[idx[valid]] = ms[valid]
    return out


def obs(d, s, tr):
    l = S.lsm(d, s)
    if tr == "lsm":
        return l
    p = np.clip(np.exp(l), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def hits_snap(L, s):
    """Vector de aciertos por snapshot (orden de L._code)."""
    code, tpos = L._code.values, L._tpos.values
    ts = s[tpos]
    better = (s > ts) | ((s == ts) & (np.arange(len(s)) < tpos))
    return np.bincount(code, weights=better, minlength=code.max() + 1) < 3


def main():
    t0 = time.time()
    lab, te, _ = datos()
    cols = ["sample_id", "row_index", "session", "block", "t", "task_name"]
    L = lab[cols + ["target_index", "fold", "atco"]].copy(); L["y"] = L.target_index
    T = te[cols].copy()
    S.hit3(L, np.zeros(len(L)))        # crea _code/_tpos
    snap = L.drop_duplicates("sample_id")
    sf, sa = snap.fold.values, snap.atco.values
    m023 = np.isin(sa, ATCO023)
    folds = sorted(np.unique(sf))
    cL, cT = cadenas(L), cadenas(T)
    Y = {}
    for e, nm in ENTRADAS.items():
        o = L.merge(S.leer(nm, "oof"), on=K, how="left").score.values
        p = T.merge(S.leer(nm, "preds"), on=K, how="left").score.values
        assert not np.isnan(o).any() and not np.isnan(p).any()
        for tr in TRANS:
            Y[(e, tr)] = (obs(L, o, tr), obs(T, p, tr))
    ref = hits_snap(L, Y[("ses", "lsm")][0])
    print(f"N (ref) CV {ref.mean():.4f}  CV 023 {ref[m023].mean():.4f}", flush=True)

    confs, H = [], []
    for (e, tr), (yl, _) in Y.items():
        for lam, gam, dr in itertools.product(LAMS, GAPS, DRIFTS):
            xs = rts(yl, *cL, lam, gam, dr)
            for w in WS:
                confs.append(("kalman", e, tr, lam, gam, dr, w)); H.append(hits_snap(L, (1 - w) * yl + w * xs))
        print(f"  {e}/{tr} listo ({time.time() - t0:.0f}s)", flush=True)
    # suavizado prob de referencia, misma CV anidada
    nbL, nbT = S.vecinos(L, 10), S.vecinos(T, 10)
    for e in ENTRADAS:
        pb = np.exp(Y[(e, "lsm")][0])
        for dc, k in itertools.product((0.5, 0.7, 0.85, 1.0), (1, 2, 3, 5, 10)):
            confs.append(("prob", e, dc, k)); H.append(hits_snap(L, S.suavizar_prob(pb, nbL, dc, k)))
    # identidad (sin suavizar) de cada entrada
    for e in ENTRADAS:
        confs.append(("nada", e)); H.append(hits_snap(L, Y[(e, "lsm")][0]))
    H = np.array(H, dtype=np.float32)

    def anidada(sel):
        ids = np.flatnonzero(sel)
        out = np.zeros(H.shape[1], bool); eleg = []
        for f in folds:
            tr = sf != f
            b = ids[np.argmax(H[ids][:, tr].mean(1))]
            out[sf == f] = H[b, sf == f] > 0.5; eleg.append(confs[b])
        return out, eleg

    kind = np.array([c[0] for c in confs])
    res = {}
    for nom, sel in (("kalman", kind == "kalman"), ("kalman_ses", np.array([c[0] == "kalman" and c[1] == "ses" for c in confs])),
                     ("prob", kind == "prob"), ("kalman_o_nada", (kind == "kalman") | (kind == "nada"))):
        h, eleg = anidada(sel)
        pa = pd.Series(h).groupby(sa).mean()
        res[nom] = (h, eleg)
        print(f"{nom:14s} anidada {h.mean():.4f} (Δ vs N {100 * (h.mean() - ref.mean()):+.2f})  CV 023 {h[m023].mean():.4f}  "
              f"peor {pa.min():.3f}  elegidos {eleg}", flush=True)
    kid = np.flatnonzero(kind == "kalman")
    order = kid[np.argsort(-H[kid].mean(1))]
    print("mejores kalman (todo OOF, optimista):", [(confs[i][1:], round(float(H[i].mean()), 4)) for i in order[:8]], flush=True)
    # para cada λ, mejor resto: perfil
    for lam in LAMS:
        ii = [i for i in kid if confs[i][3] == lam]
        b = ii[int(np.argmax(H[ii].mean(1)))]
        print(f"  λ={lam:<6}: mejor {H[b].mean():.4f} {confs[b][1:]}", flush=True)

    # final: mejor kalman con todo lab -> oof y preds de test
    b = order[0]; _, e, tr, lam, gam, dr, w = confs[b]
    yl, yt = Y[(e, tr)]
    so = (1 - w) * yl + w * rts(yl, *cL, lam, gam, dr)
    sp = (1 - w) * yt + w * rts(yt, *cT, lam, gam, dr)
    guardar("f12_kalman", L, so, T, sp)
    hN, eN = res["kalman"]
    pa = pd.Series(hN).groupby(sa).mean()
    ex = dict(cv_ref=round(float(ref.mean()), 4), cv_anidada=round(float(hN.mean()), 4),
              cv_023_anidada=round(float(hN[m023].mean()), 4), peor_anidada=round(float(pa.min()), 4),
              cv_023=round(float(H[b][m023].mean()), 4), params=[str(x) for x in confs[b][1:]],
              params_por_fold=[[str(x) for x in c[1:]] for c in eN],
              cv_prob_anidada=round(float(res["prob"][0].mean()), 4))
    registrar("f12_kalman", hilo="f12", segundos=time.time() - t0,
              descripcion=f"Kalman/RTS por (session, task_name) sobre {ENTRADAS[e]} ({tr}), λ={lam}, γ={gam}, δ={dr}, "
                          f"w={w} (elegidos con todo el OOF; ver cv_anidada)", extra=ex)
    # OOF anidado como score: 1 en la elegida por fold no sirve para ranking; se guarda el score de la config elegida
    so_n = np.zeros(len(L))
    rowfold = L.fold.values
    for f, c in zip(folds, eN):
        _, e2, tr2, lam2, gam2, dr2, w2 = c
        y2 = Y[(e2, tr2)][0]
        s2 = (1 - w2) * y2 + w2 * rts(y2, *cL, lam2, gam2, dr2)
        so_n[rowfold == f] = s2[rowfold == f]
    guardar("f12_kalman_anid", L, so_n)
    registrar("f12_kalman_anid", hilo="f12", descripcion="Kalman/RTS, CV anidada 4->5 (sin preds)", extra=ex)
    print(f"fin ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

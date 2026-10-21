"""
f12_riesgos.py · Juanma (C) · RAMA FUTURO · Riesgos competitivos / proceso de llegada (logit condicional).

Formulación
-----------
Snapshot s (instante t) con lista R_s de tareas abiertas. Cada tarea i in R_s es un proceso de llegada de "atenciones"
del controlador, con intensidad (riesgo de causa específica, tiempo discreto, paso ~3 s)
        h_i(t) = h_0(t) · exp(beta' z_i(t)),
z_i(t) = covariables de la tarea en ese instante: tiempo en la lista (x_task_age), tiempos hasta sus plazos
(time_to_sector, k_timeToExitPoint, distance_to_zone, logic_deadline, time_to_MD), tiempo hasta la siguiente instrucción
compatible (si_match) y hasta la anterior, hasta que desaparece (f_task_end), si desaparece en la racha, tipo, alertas...
Supuestos: (1) riesgos competitivos independientes dado z (el controlador atiende a lo sumo una tarea por instante),
(2) riesgo proporcional con línea base h_0(t) común a todas las tareas del snapshot, (3) snapshots condicionalmente
independientes dado z (pseudo-verosimilitud: los snapshots de una racha están correlados; la sesión entra después).
Condicionando a que en el instante t "llega" exactamente una atención (lo que garantiza la etiqueta), h_0 se cancela:
        P(Y_s = i | z, una llegada) = h_i / sum_{j in R_s} h_j = softmax_i(beta' z)        (verosimilitud parcial de Cox)
z es una base no lineal: para cada variable continua u = log1p(x) (o asinh(x/60) si tiene signo), tramos lineales
(hinge (u - k)_+ en 5 cuantiles) + indicador de vacío (vacío = "no aplica", nunca 0); interacciones tipo × tiempo
(tramos de x_task_age, si_match, f_task_end, f_next_input, logic_deadline por grupo de tipo).
Estimación: máxima verosimilitud penalizada  max_beta  sum_s log P(Y_s | z) - (lam/2)||beta||^2  (L-BFGS, gradiente
exacto), con las columnas estandarizadas con el fold de entrenamiento. lam = 1e-3 por snapshot fijado a priori (no se
ajusta con etiquetas); se comprueba la sensibilidad con CV anidada 4->5 (sección "lam").
Uso del modelo: (i) solo, (ii) score adicional  s = lsm(N1) + w·lsm(CR)  antes del segundo paso de sesión
(feat_session.sesion), w por CV anidada 4->5; también (iii) después de la sesión: lsm(N) + w·lsm(CR).
Sin fuga: beta del fold f solo con folds != f; la estandarización y los cuantiles de los nudos no usan etiquetas.
Uso: OMP_NUM_THREADS=2 python futuro/pruebas/f12_riesgos.py
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from base_fut import OUT, ROOT, guardar, registrar, TYPES  # noqa: E402
from f10_instr import datos_instr  # noqa: E402
import feat_session  # noqa: E402

CV023 = [2, 11, 10, 6, 17, 15]
LOGT = ["x_task_age", "k_timeToExitPoint", "time_to_MD", "si_match", "si_prevmatch", "f_next_input", "f_task_end",
        "f_cs_end", "a_since_flight", "a_since_any", "p_task_start", "sx_task", "sc_task", "k_levelClearanceAgeS",
        "distance_to_zone", "f_task_end_rk", "si_match_rk", "logic_edf_rank_type", "l_rank_age_type", "f_next_input_rk",
        "p_since_any_rk", "l_n_type"]
SIGNED = ["time_to_sector", "logic_deadline", "logic_window_dist"]
BIN = ["f_next_match", "f_next_match_soon", "f_gone_next", "f_end_in_block", "logic_in_window", "conflict_ind",
       "HMI_conflict_safety_alert", "is_assumed", "within_sector", "k_aircraftNeedsToBeTransferred", "logic_feasible",
       "logic_offered", "logic_due_transfer", "logic_stca", "l_twin_AE", "a_is_last_cs", "p_new_in_block",
       "x_entry_coord_not_done", "k_exitConditionsAreMet", "multiple_conflicts_ind", "zone_conflict_ind",
       "monitoring_rule_activated", "qos_candidate"]
INTER = ["x_task_age", "si_match", "f_task_end", "f_next_input", "logic_deadline"]
GRUPOS = {"ecoord": [2], "xcoord": [3], "assume": [4], "transfer": [5], "exitcond": [10], "zone": [12], "rtr": [13],
          "conf": [0, 1, 6], "otro": [7, 8, 9, 11]}
SIMPLE = ["x_task_age", "time_to_sector", "k_timeToExitPoint", "distance_to_zone", "si_match", "f_end_in_block"]
LAM = 1e-3


def _u(d, c):
    x = d[c].values.astype(float)
    return np.arcsinh(x / 60.0) if c in SIGNED else np.log1p(np.maximum(x, 0))


def base(d, knots=None, simple=False):
    """Matriz Z (float32) y nombres. knots: dict var -> nudos (cuantiles de u; sin etiquetas)."""
    cols, names, kn = [], [], {} if knots is None else knots
    cont = [c for c in LOGT + SIGNED if (not simple or c in SIMPLE)]
    ubas = {}
    for c in cont:
        u = _u(d, c); na = np.isnan(u)
        if c not in kn:
            kn[c] = np.unique(np.nanquantile(u, [0.1, 0.3, 0.5, 0.7, 0.9]))
        u0 = np.where(na, 0.0, u)
        B = [u0] + [np.where(na, 0, np.maximum(u0 - k, 0)) for k in kn[c]]
        ubas[c] = B
        for j, b in enumerate(B):
            cols.append(b); names.append(f"{c}:h{j}")
        if na.any():
            cols.append(na.astype(float)); names.append(f"{c}:NA")
    for c in BIN:
        if simple and c not in SIMPLE:
            continue
        x = d[c].values.astype(float)
        cols.append(np.nan_to_num(x)); names.append(c)
        if np.isnan(x).any():
            cols.append(np.isnan(x).astype(float)); names.append(f"{c}:NA")
    if not simple:                       # f_next_act es categórica (tipo de la siguiente acción sobre el avión)
        for a in range(6):
            cols.append((d.f_next_act.values == a).astype(float)); names.append(f"f_next_act={a}")
    ti = d.task_index.values
    for g, tt in GRUPOS.items():
        m = np.isin(ti, tt).astype(float)
        cols.append(m); names.append(f"tipo={g}")
        if simple:
            continue
        for c in INTER:
            for j, b in enumerate(ubas[c]):
                cols.append(m * b); names.append(f"{g}*{c}:h{j}")
            na = d[c].isna().values
            if na.any():
                cols.append(m * na); names.append(f"{g}*{c}:NA")
        for c in ("f_next_match", "f_end_in_block", "is_assumed"):
            cols.append(m * np.nan_to_num(d[c].values.astype(float))); names.append(f"{g}*{c}")
    return np.column_stack(cols).astype(np.float32), names, kn


def _nll(w, X, y, g, ng, lam):
    s = (X @ w.astype(np.float32)).astype(float)
    m = np.full(ng, -np.inf); np.maximum.at(m, g, s)
    e = np.exp(s - m[g]); z = np.bincount(g, e, ng)
    lp = s - m[g] - np.log(z[g])
    r = (np.exp(lp) - y).astype(np.float32)
    return -(lp * y).sum() / ng + 0.5 * lam * w @ w, (X.T @ r).astype(float) / ng + lam * w


def fit(X, y, sid, lam=LAM):
    g = pd.factorize(sid)[0]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    r = minimize(_nll, np.zeros(X.shape[1]), args=(Xs, y.astype(float), g, g.max() + 1, lam), jac=True,
                 method="L-BFGS-B", options=dict(maxiter=600))
    return r.x / sd, mu, r            # coeficientes en la escala original de Z (score = Z @ b + cte)


def lsm(code, s):
    s = np.asarray(s, float)
    sh = s - pd.Series(s).groupby(code).transform("max").values
    return sh - np.log(pd.Series(np.exp(sh)).groupby(code).transform("sum").values)


def hits(code, y, s):
    """acierto top-3 por snapshot (empates: orden estable)."""
    o = np.lexsort((-np.asarray(s), code))
    c, yy = code[o], y[o]
    rk = np.arange(len(c)) - np.searchsorted(c, c)
    return np.bincount(c, (yy == 1) & (rk < 3), code.max() + 1)


class M:
    def __init__(self, lab):
        self.code = pd.factorize(lab.sample_id)[0]
        self.y = lab.target_index.values
        sn = lab.drop_duplicates("sample_id")
        self.fold, self.atco = sn.fold.values, sn.atco.values

    def __call__(self, s, mask=None):
        h = hits(self.code, self.y, s)
        m = np.ones(len(h), bool) if mask is None else mask
        return h[m].mean(), h[np.isin(self.atco, CV023)].mean(), pd.Series(h).groupby(self.atco).mean().min()


def leer(nombre, carpeta):
    for b in (ROOT / "futuro", OUT):
        p = b / carpeta / f"{nombre}.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(nombre)


def alinear(df, nombre, carpeta):
    x = df[["sample_id", "row_index"]].merge(leer(nombre, carpeta), on=["sample_id", "row_index"], how="left")
    assert x.score.notna().all(), nombre
    return x.score.values


def cv_cr(lab, te, simple, lam=LAM, log=print):
    """OOF del modelo de riesgos (beta por folds de entrenamiento) y pred de test (beta con todo)."""
    knots = {}
    Z, names, knots = base(pd.concat([lab, te], ignore_index=True), knots, simple)  # nudos: cuantiles sin etiquetas
    ZL, ZT = Z[:len(lab)], Z[len(lab):]
    y, sid = lab.target_index.values, lab.sample_id.values
    oof = np.zeros(len(lab))
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        b, mu, r = fit(ZL[tr], y[tr], sid[tr], lam)
        oof[va] = ZL[va] @ b
        log(f"  fold {f}: nll {r.fun:.4f} it {r.nit}")
    b, mu, r = fit(ZL, y, sid, lam)
    return oof, ZT @ b, b, names, ZL


def main():
    t0 = time.time()
    lab, te, _ = datos_instr()
    mz = M(lab)
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)
    res = {}
    N1o, N1p = alinear(lab, "f10_instr_l63_s10", "oof"), alinear(te, "f10_instr_l63_s10", "preds")
    No, Np = alinear(lab, "f10_instr_l63_s10_ses", "oof"), alinear(te, "f10_instr_l63_s10_ses", "preds")
    res["N1"], res["N"] = mz(N1o), mz(No)
    log(f"N1 {res['N1']}  N {res['N']}")
    cT = pd.factorize(te.sample_id)[0]

    # (i) modelo solo: simple (las covariables del enunciado, sin interacciones) y completo
    for nm, simple in (("simple", True), ("full", False)):
        oof, pred, b, names, ZL = cv_cr(lab, te, simple, log=log)
        res[f"cr_{nm}"] = mz(oof)
        log(f"CR {nm}: {len(names)} coef · CV {res[f'cr_{nm}']}")
        guardar(f"f12_riesgos_{nm}", lab, oof, te, pred)
        so, sp, _ = feat_session.sesion(lab, oof, te, pred)
        res[f"cr_{nm}_ses"] = mz(so)
        log(f"CR {nm} + sesión: {res[f'cr_{nm}_ses']}")
        guardar(f"f12_riesgos_{nm}_ses", lab, so, te, sp)
        pd.Series(b, index=names).to_csv(OUT / "logs" / f"f12_riesgos_{nm}_coef.csv")
        if nm == "simple":
            # interpretación: log-riesgo de cada variable en p10/p50/p90 (curva del efecto principal)
            sd = ZL.std(0)
            for c in SIMPLE:
                idx = [i for i, n in enumerate(names) if n.startswith(c + ":")]
                contrib = ZL[:, idx] @ b[idx]
                x = lab[c].values
                ok = ~np.isnan(x)
                q = np.nanquantile(x, [0.1, 0.5, 0.9])
                vals = [np.median(contrib[ok & (np.abs(x - qq) <= np.nanstd(x) * 0.05 + 1e-9)]) if
                        (ok & (np.abs(x - qq) <= np.nanstd(x) * 0.05 + 1e-9)).any() else np.nan for qq in q]
                log(f"  efecto {c}: x(p10,p50,p90)={np.round(q, 1)} log-riesgo={np.round(np.array(vals) - vals[1], 2)}"
                    + (f"  NA: {b[names.index(c + ':NA')]:.2f}" if c + ":NA" in names else ""))
            for n in names:
                if n.startswith("tipo=") or ":" not in n:
                    log(f"  coef {n}: {b[names.index(n)]:.2f}")
        if nm == "full":
            CRo, CRp = oof, pred

    # (ii) N1 + w·CR antes de la sesión, (iii) N + w·CR después; w por CV anidada 4->5
    lN1, lCR, lN = lsm(mz.code, N1o), lsm(mz.code, CRo), lsm(mz.code, No)
    lN1t, lCRt, lNt = lsm(cT, N1p), lsm(cT, CRp), lsm(cT, Np)
    WS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
    for modo in ("pre", "post"):
        H, outs = {}, {}
        for w in WS:
            if modo == "pre":
                so, _, _ = feat_session.sesion(lab, lN1 + w * lCR)
            else:
                so = lN + w * lCR
            H[w] = hits(mz.code, mz.y, so); outs[w] = so
            log(f"  {modo} w={w}: CV {H[w].mean():.4f} CV023 {H[w][np.isin(mz.atco, CV023)].mean():.4f}")
        nest = np.zeros(len(mz.fold)); wsel = {}
        for f in range(5):
            tr = mz.fold != f
            wb = max(WS, key=lambda w: (H[w][tr].mean(), -w))
            wsel[f] = wb; nest[mz.fold == f] = H[wb][mz.fold == f]
        wf = max(WS, key=lambda w: (H[w].mean(), -w))
        res[f"mix_{modo}_nested"] = (nest.mean(), nest[np.isin(mz.atco, CV023)].mean(),
                                     pd.Series(nest).groupby(mz.atco).mean().min())
        res[f"mix_{modo}_w"] = dict(folds=wsel, final=wf)
        log(f"{modo}: CV anidada {res[f'mix_{modo}_nested']} w por fold {wsel} w final {wf}")
        if modo == "pre":
            so, sp, _ = feat_session.sesion(lab, lN1 + wf * lCR, te, lN1t + wf * lCRt)
        else:
            so, sp = lN + wf * lCR, lNt + wf * lCRt
        nm = f"f12_riesgos_mix_{modo}"
        guardar(nm, lab, so, te, sp)
        registrar(nm, hilo="f12", descripcion=f"riesgos competitivos (logit condicional) + N, {modo}-sesión, w={wf}",
                  extra=dict(cv_anidada=round(res[f'mix_{modo}_nested'][0], 4), cv_023=round(mz(so)[1], 4), w=wf))
    for nm in ("full", "simple"):
        registrar(f"f12_riesgos_{nm}_ses", hilo="f12", descripcion=f"riesgos competitivos {nm} + sesión (solo)",
                  extra=dict(cv_023=round(res[f'cr_{nm}_ses'][1], 4)))
    json.dump({k: v for k, v in res.items()}, open(OUT / "logs" / "f12_riesgos_res.json", "w"), default=str, indent=1)
    log("fin")


if __name__ == "__main__":
    main()

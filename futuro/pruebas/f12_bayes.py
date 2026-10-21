"""f12_bayes.py · Juanma (C) · hilo f12 · Bayes jerárquico sobre la heterogeneidad de controladores (línea futuro).

Modelo (sobre los scores OOF de N = f10_instr_l63_s10_ses, que ya usa la sesión):
    Snapshot n del controlador c, tareas i = 1..m_n, tipo k(i) en {0..13}, score de N s_i.
    Y_n ~ Categorical(softmax(u_n)),   u_i = beta * s_i + gamma_{c, k(i)}
    gamma_c ~ N(mu, Sigma)  (efecto aleatorio de "estilo": preferencia del controlador por cada tipo de tarea),
    Sigma = diag(sigma_k^2), gamma_c independientes entre controladores, Y_n | gamma_c independientes.
Estimación (Bayes empírico, solo con controladores de entrenamiento del fold):
    1) beta, mu0: MLE conjunta agrupada (Newton) de u = beta*s + mu0_k.
    2) EM de máxima verosimilitud marginal con aproximación de Laplace (beta fija): E-step, para cada controlador c,
       posterior gamma_c | y_c ≈ N(m_c, S_c) con m_c = MAP bajo el prior N(mu, Sigma) (Newton) y S_c = H_c^{-1};
       M-step mu = media_c m_c, sigma_k^2 = media_c[(m_ck - mu_k)^2 + S_c,kk]. Hasta converger (<= 30 iter).
       (Un primer intento por método de momentos con MLE por controlador explotaba: tipos raros -> gamma no
       identificado, sd ~ 1e3.)
       (Invariancia de softmax: los gamma solo cuentan relativos dentro del snapshot; el prior lo regulariza.)
Predicción para un controlador NUEVO (en CV cada controlador del fold externo lo es; el 020 y el 023 en test):
    (a) media:          u = beta*s + mu
    (b) marginal MC:    P(i) = E_{gamma~N(mu,Sigma)}[softmax(beta*s + gamma)_i], M = 256 draws (mismos para todos)
    (c) transductivo:   gamma_c actualizado con los snapshots SIN etiqueta del propio controlador. Ojo: con un modelo
        discriminativo la verosimilitud de los datos sin etiqueta no depende de gamma (sum_i P(i) = 1), así que el EM
        clásico (E: r = P(i|gamma); M: MAP con r) tiene punto fijo gamma = mu: no aprende nada. Se usa EM
        "clasificador"/afilado: r_i ∝ P(i|gamma)^(1/T) (T -> 0: etiqueta dura top-1), M-step MAP con prior
        N(mu, kappa*Sigma); 10 iteraciones. (T, kappa) se eligen por CV anidada (4 folds internos -> 5 externos).
Sin fuga: beta, mu, Sigma y los hiperparámetros solo con etiquetas de folds de entrenamiento; (c) no usa etiquetas del
controlador que predice. Los s de N son OOF (el modelo N no vio la etiqueta del controlador).
Uso: python futuro/pruebas/f12_bayes.py   -> f12_bayes_{a,b,c} (oof+preds) en futuro/pruebas, registro hilo f12.
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402

BASE = "f10_instr_l63_s10_ses"
C023 = [2, 11, 10, 6, 17, 15]
K = 14
RNG = np.random.default_rng(0)
M_MC = 256


def leer(nombre, carpeta):
    for b in (OUT, ROOT / "futuro"):
        p = b / carpeta / f"{nombre}.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(nombre)


# ---------------------------------------------------------------- utilidades softmax por snapshot
def softmax_snap(u, code, nsnap):
    mx = np.full(nsnap, -np.inf); np.maximum.at(mx, code, u)
    e = np.exp(u - mx[code])
    z = np.bincount(code, e, nsnap)
    return e / z[code]


def newton(Z, off, r, code, nsnap, prior_mu, prior_prec, theta0, iters=30):
    """MAP de theta en u = off + Z theta, objetivo sum r*u - sum logsumexp - 0.5 (theta-mu)' P (theta-mu)."""
    th = theta0.copy(); P = Z.shape[1]
    for _ in range(iters):
        p = softmax_snap(off + Z @ th, code, nsnap)
        g = Z.T @ (r - p) - prior_prec * (th - prior_mu)
        A = Z.T @ (p[:, None] * Z)
        m = np.zeros((nsnap, P)); np.add.at(m, code, p[:, None] * Z)
        H = A - m.T @ m + np.diag(prior_prec + 1e-8)
        step = np.linalg.solve(H, g)
        th = th + step
        if np.abs(step).max() < 1e-6:
            break
    return th, H


class Ctrl:
    """Datos de un controlador (o conjunto) listos para Newton."""
    def __init__(self, d):
        self.code = pd.factorize(d.sample_id.values)[0]
        self.nsnap = self.code.max() + 1
        self.s = d.s.values
        self.k = d.k.values
        self.Zk = np.eye(K)[self.k]
        self.y = d.y.values.astype(float) if "y" in d else None


def fit_eb(dtr):
    """beta, mu, sigma2 (diag) con los controladores de dtr (tiene etiquetas)."""
    C = Ctrl(dtr)
    Z = np.column_stack([C.s, C.Zk])
    pm = np.zeros(K + 1); pp = np.r_[0.0, np.full(K, 1e-2)]
    th, _ = newton(Z, np.zeros(len(C.s)), C.y, C.code, C.nsnap, pm, pp, np.r_[1.0, np.zeros(K)])
    beta, mu0 = th[0], th[1:]
    # EM de Bayes empírico con aproximación de Laplace: E: posterior de gamma_c ~ N(m_c, S_c) con prior N(mu, tau2);
    # M: mu = media(m_c), tau2_k = media((m_ck - mu_k)^2 + S_ckk).
    grupos = [Ctrl(g) for _, g in dtr.groupby("atco")]
    mu, tau2 = mu0.copy(), np.full(K, 1.0)
    Gs = [mu0.copy() for _ in grupos]
    for it in range(30):
        G, V = [], []
        for j, c in enumerate(grupos):
            gh, H = newton(c.Zk, beta * c.s, c.y, c.code, c.nsnap, mu, 1 / tau2, Gs[j], iters=15)
            Gs[j] = gh; G.append(gh); V.append(np.diag(np.linalg.inv(H)))
        G, V = np.array(G), np.array(V)
        mu_n = G.mean(0)
        tau2_n = np.maximum(((G - mu_n) ** 2 + V).mean(0), 1e-4)
        conv = np.abs(mu_n - mu).max() < 1e-4 and np.abs(tau2_n - tau2).max() < 1e-4
        mu, tau2 = mu_n, tau2_n
        if conv:
            break
    # tipos que no aparecen en los datos: posterior = prior (tau2 sin información) -> sin varianza
    presentes = np.bincount(C.k, minlength=K) > 0
    tau2[~presentes] = 1e-4
    return dict(beta=beta, mu=mu, sigma2=tau2, G=G, V=V)


def pred_a(c, eb):
    return eb["beta"] * c.s + eb["mu"][c.k]


def pred_b(c, eb, draws):
    """log E[softmax]; draws: (M, K) estándar, se escala con mu, sigma."""
    gam = eb["mu"] + draws * np.sqrt(eb["sigma2"])
    acc = np.zeros(len(c.s))
    for g in gam:
        acc += softmax_snap(eb["beta"] * c.s + g[c.k], c.code, c.nsnap)
    return np.log(acc / len(gam) + 1e-300)


def pred_c(c, eb, T, kappa, iters=10):
    s2 = kappa * eb["sigma2"]
    prec = np.where(s2 > 0, 1 / np.maximum(s2, 1e-12), 1e6)
    gam = eb["mu"].copy(); off = eb["beta"] * c.s
    for _ in range(iters):
        u = off + gam[c.k]
        if T == 0:
            mx = np.full(c.nsnap, -np.inf); np.maximum.at(mx, c.code, u)
            r = (u == mx[c.code]).astype(float)
            r /= np.bincount(c.code, r, c.nsnap)[c.code]
        else:
            r = softmax_snap(u / T, c.code, c.nsnap)
        gam, _ = newton(c.Zk, off, r, c.code, c.nsnap, eb["mu"], prec, gam, iters=10)
    return off + gam[c.k]


def hits_by(d, score):
    d = d.assign(sc=score)
    tgt = d[d.y == 1][["sample_id", "sc", "row_index"]].rename(columns={"sc": "ts", "row_index": "tr"})
    m = d.merge(tgt, on="sample_id")
    better = (m.sc > m.ts) | ((m.sc == m.ts) & (m.row_index < m.tr))
    nb = better.groupby(m.sample_id).sum()
    h = (nb < 3).astype(float)
    at = d.drop_duplicates("sample_id").set_index("sample_id").atco.loc[h.index]
    return h, at


def resumen(d, score):
    h, at = hits_by(d, score)
    pa = h.groupby(at).mean()
    return dict(cv=round(h.mean(), 4), c023=round(h[at.isin(C023)].mean(), 4), peor=round(pa.min(), 4),
                por_atco={str(k): round(v, 4) for k, v in pa.items()})


GRID_C = [(T, kp) for T in (0, 0.5, 1.0) for kp in (0.5, 1.0, 2.0)]


def predecir_todo(dtr, dva, eb, draws):
    """Devuelve dict metodo -> score para las filas de dva (por controlador)."""
    out = {m: np.zeros(len(dva)) for m in ["a", "b"] + [f"c_{T}_{kp}" for T, kp in GRID_C]}
    pos = np.arange(len(dva))
    for a, idx in dva.groupby("atco").indices.items():
        c = Ctrl(dva.iloc[idx])
        out["a"][pos[idx]] = pred_a(c, eb)
        out["b"][pos[idx]] = pred_b(c, eb, draws)
        for T, kp in GRID_C:
            out[f"c_{T}_{kp}"][pos[idx]] = pred_c(c, eb, T, kp)
    return out


def main():
    t0 = time.time()
    lab, te, _ = datos()
    lab = lab[["sample_id", "row_index", "atco", "fold", "task_index", "target_index", "session"]].copy()
    te = te[["sample_id", "row_index", "atco", "task_index", "session"]].copy()
    oo = leer(BASE, "oof"); pr = leer(BASE, "preds")
    lab = lab.merge(oo.rename(columns={"score": "s"}), on=["sample_id", "row_index"], how="left")
    te = te.merge(pr.rename(columns={"score": "s"}), on=["sample_id", "row_index"], how="left")
    assert lab.s.notna().all() and te.s.notna().all()
    lab = lab.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    te = te.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    lab["k"] = lab.task_index.astype(int); te["k"] = te.task_index.astype(int)
    lab["y"] = lab.target_index.astype(int)
    draws = RNG.standard_normal((M_MC, K))
    metodos = ["a", "b"] + [f"c_{T}_{kp}" for T, kp in GRID_C]

    ref = resumen(lab, lab.s.values)
    print("N:", {k: v for k, v in ref.items() if k != "por_atco"}, flush=True)

    oof = {m: np.zeros(len(lab)) for m in metodos}
    elegido_c = {}
    eb_info = {}
    for f in range(5):
        tr = lab[lab.fold != f]; va_idx = np.flatnonzero(lab.fold.values == f)
        # --- CV interna para (T, kappa) de (c): 4 folds de entrenamiento
        inner = {m: [] for m in metodos if m.startswith("c_")}
        inner_h = {m: np.zeros(len(tr)) for m in inner}
        trr = tr.reset_index(drop=True)
        for g in sorted(trr.fold.unique()):
            ebi = fit_eb(trr[trr.fold != g])
            vi = np.flatnonzero(trr.fold.values == g)
            o = predecir_todo(None, trr.iloc[vi].reset_index(drop=True), ebi, draws)
            for m in inner:
                inner_h[m][vi] = o[m]
        sc_in = {m: resumen(trr, inner_h[m])["cv"] for m in inner}
        best = max(sc_in, key=sc_in.get); elegido_c[f] = best
        eb = fit_eb(tr)
        eb_info[f] = dict(beta=round(eb["beta"], 3), sd=np.round(np.sqrt(eb["sigma2"]), 3).tolist())
        o = predecir_todo(None, lab.iloc[va_idx].reset_index(drop=True), eb, draws)
        for m in metodos:
            oof[m][va_idx] = o[m]
        print(f"fold {f}: beta {eb['beta']:.3f} sd {np.round(np.sqrt(eb['sigma2']), 2)} | (c) interno -> {best} "
              f"{sc_in[best]:.4f} | {time.time() - t0:.0f}s", flush=True)
    res = {m: resumen(lab, oof[m]) for m in metodos}
    for m in metodos:
        print(f"{m:12s} cv {res[m]['cv']:.4f}  023 {res[m]['c023']:.4f}  peor {res[m]['peor']:.4f}", flush=True)
    # (c) con hiperparámetros elegidos en la CV interna de cada fold
    oof_c = np.zeros(len(lab))
    for f, m in elegido_c.items():
        idx = lab.fold.values == f; oof_c[idx] = oof[m][idx]
    rc = resumen(lab, oof_c)
    print(f"(c) anidada  cv {rc['cv']:.4f}  023 {rc['c023']:.4f}  peor {rc['peor']:.4f}  {elegido_c}", flush=True)

    # --- test: EB con los 18 controladores; (c) con los (T,kappa) más votados
    eb = fit_eb(lab)
    print("EB todo: beta", round(eb["beta"], 3), "mu", np.round(eb["mu"], 2), "sd", np.round(np.sqrt(eb["sigma2"]), 2))
    votos = pd.Series(list(elegido_c.values())).value_counts().index[0]
    T, kp = votos.split("_")[1:]; T, kp = float(T), float(kp)
    pt = {"a": np.zeros(len(te)), "b": np.zeros(len(te)), "c": np.zeros(len(te))}
    for a, idx in te.groupby("atco").indices.items():
        c = Ctrl(te.iloc[idx])
        pt["a"][idx] = pred_a(c, eb); pt["b"][idx] = pred_b(c, eb, draws); pt["c"][idx] = pred_c(c, eb, T, kp)
    finales = {"a": (oof["a"], res["a"]), "b": (oof["b"], res["b"]), "c": (oof_c, rc)}
    for m, (o, r) in finales.items():
        nombre = f"f12_bayes_{m}"
        guardar(nombre, lab, o, te, pt[m])
        registrar(nombre, "f12", f"Bayes jerárquico gamma_c~N(mu,Sigma) por tipo sobre {BASE}, método ({m})"
                  + (f", (T,kappa) por CV anidada, test {votos}" if m == "c" else ""), segundos=time.time() - t0,
                  extra=dict(cv_ref=ref["cv"], cv_023_ref=ref["c023"], cv_anidada=r["cv"], cv_023=r["c023"],
                             peor_anidada=r["peor"], eb_folds=eb_info,
                             elegido_c={str(k): v for k, v in elegido_c.items()} if m == "c" else None))
    with open(OUT / "f12_bayes_res.json", "w") as fh:
        json.dump(dict(ref=ref, res=res, c_anidada=rc, elegido_c=elegido_c, eb_test=dict(
            beta=eb["beta"], mu=eb["mu"].tolist(), sd=np.sqrt(eb["sigma2"]).tolist())), fh, indent=1)
    print(f"fin {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

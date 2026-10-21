"""
f12_calib.py · Juanma (C) · Calibración del acierto por snapshot y nota esperada en público (020) y privado (023).

FORMULACIÓN
-----------
Para cada modelo m y snapshot s (sesión j(s), controlador a(s), racha b(s)):
  H_{ms} ∈ {0,1}: la tarea etiquetada está en el top-3 de m.            H_{ms} ~ Bernoulli(π_{ms})
  x_{ms}: resumen de la lista de m en s (p1..p4, suma top-3 S3, logit S3, margen p3-p4, entropía, log nº tareas,
          tipo del top-1, log longitud de racha), con p = softmax del score de m dentro del snapshot.
  Calibración:  logit p_{ms} = g_m(x_{ms})  (logística, o isotónica sobre S3; se elige por log-loss en CV interna 4
          folds y se evalúa en el 5.º: CV anidada 4->5 con Data/folds.csv). Para test se ajusta con los 5 folds y se
          aplica a preds/ convertidas a probabilidad igual que el OOF.
  Heterogeneidad (lo que la calibración no explica): residuo medio de la sesión j,
          r_{mj} = mean_{s∈j}(H_{ms} - p_{ms}) = u_{m,a(j)} + v_{mj} + e_{mj}
          u_a ~ N(0, T_a) (controlador, vector en m),  v_j ~ N(0, T_s) (sesión dentro del controlador),
          e_j ~ N(0, S_j) (muestreo de Bernoullis correlacionadas dentro de cada racha: S_j por varianza de
          conglomerados = bootstrap por rachas en forma analítica, conjunta en m).
  Momentos: E[r_j r_j'] = T_a + T_s + S_j ;  E[r_j r_k'] = T_a  (j≠k, mismo controlador)  -> T_a, T_s (proy. PSD).
  Nota de un conjunto de test C (sesiones k, pesos w_k = n_k/n):
          Y_m = mean_s p_{ms} + u_m + Σ_k w_k v_{mk} + e_m ,   e ~ N(0, S_1/n) (S_1: covarianza por snapshot, de la CV)
  020 (público): el controlador 020 tiene la sesión 020sc1 en train -> u | r_{020sc1} ~ N(T_a G r, T_a - T_a G T_a),
          G = (T_a + T_s + S_{020sc1})^{-1}.   023 (privado): controlador nuevo -> u ~ N(0, T_a).
  P(mejor), E[max pareja] por Monte Carlo de la normal conjunta en m (las correlaciones entre modelos importan).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, registrar  # noqa: E402

MODELOS = {"lgb_future": ("lgb_future", 0.942), "A": ("suav_lgb_future_clear10_ses", 0.947),
           "OR": ("suav_lgb_future_clear10_or_ses", 0.940), "B": ("suav_lgb_future_clear10_orcoord_ses", 0.934),
           "N": ("f10_instr_l63_s10_ses", 0.951)}
PROB = {"A", "OR", "B"}           # sus scores ya son probabilidades


def leer(nombre, carpeta):
    for b in (ROOT / "futuro", OUT):
        p = b / carpeta / f"{nombre}.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(nombre)


def a_prob(d, s, es_prob):
    z = np.log(np.clip(s, 1e-12, None)) if es_prob else np.asarray(s, float)
    z = pd.Series(z, index=d.index)
    z = np.exp(z - z.groupby(d.sample_id).transform("max"))
    return (z / z.groupby(d.sample_id).transform("sum")).values


def resumen(d, p, y=None):
    """Una fila por snapshot: features x y (si hay etiquetas) H."""
    t = d[["sample_id", "task_index", "blk_len", "session", "atco", "block", "fold"]].assign(p=p)
    if y is not None:
        t["y"] = y
    t = t.sort_values(["sample_id", "p"], ascending=[True, False])
    t["r"] = t.groupby("sample_id").cumcount()
    g = t.groupby("sample_id")
    P = t[t.r < 4].pivot(index="sample_id", columns="r", values="p").fillna(0)
    out = pd.DataFrame({f"p{k+1}": P[k] for k in range(4)})
    out["s3"] = out.p1 + out.p2 + out.p3
    out["ls3"] = logit(out.s3.clip(1e-6, 1 - 1e-6))
    out["marg"] = np.log(out.p3 + 1e-9) - np.log(out.p4 + 1e-9)
    out["ent"] = g.p.apply(lambda v: -(v * np.log(v + 1e-12)).sum())
    out["lnt"] = np.log(g.size())
    top = t[t.r == 0].set_index("sample_id")
    for k in range(14):
        out[f"ty{k}"] = (top.task_index == k).astype(float)
    out["lblk"] = np.log(top.blk_len)
    for c in ("session", "atco", "block", "fold"):
        out[c] = top[c]
    if y is not None:
        out["H"] = (t[t.r < 3].groupby("sample_id").y.sum() > 0).astype(int)
    return out


FEAT = ["p1", "p2", "p3", "p4", "ls3", "marg", "ent", "lnt", "lblk"] + [f"ty{k}" for k in range(14)]
CANDS = [("lr", 0.1), ("lr", 1.0), ("lr_s3", 1.0), ("iso", None)]


def fit_pred(cand, Xtr, ytr, Xte):
    kind, C = cand
    if kind == "iso":
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(Xtr.s3, ytr)
        return m.predict(Xte.s3)
    cols = ["ls3"] if kind == "lr_s3" else FEAT
    mu, sd = Xtr[cols].mean(), Xtr[cols].std() + 1e-9
    m = LogisticRegression(C=C, max_iter=2000).fit((Xtr[cols] - mu) / sd, ytr)
    return m.predict_proba((Xte[cols] - mu) / sd)[:, 1]


def elegir(X):
    """Candidato con menor log-loss en CV por folds dentro de X."""
    ll = []
    for c in CANDS:
        pr = np.zeros(len(X))
        for f in X.fold.unique():
            tr, va = X.fold != f, X.fold == f
            pr[va.values] = fit_pred(c, X[tr], X.H[tr], X[va])
        ll.append(log_loss(X.H, np.clip(pr, 1e-6, 1 - 1e-6)))
    return CANDS[int(np.argmin(ll))]


def calibrar(X, Xt):
    """CV anidada 4->5: p OOF; y p de test con candidato elegido en los 5 folds."""
    p = np.zeros(len(X)); elegidos = []
    for f in sorted(X.fold.unique()):
        tr, va = X.fold != f, X.fold == f
        c = elegir(X[tr]); elegidos.append(c)
        p[va.values] = fit_pred(c, X[tr], X.H[tr], X[va])
    c = elegir(X)
    return p, fit_pred(c, X, X.H, Xt), elegidos, c


def psd(M):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


def cov_conglomerados(R, blocks):
    """Cov de la media de residuos R (n×m) con muestreo por rachas: Σ_b (R_b)(R_b)' / n², R_b = suma en la racha
    del residuo centrado (equivalente analítico del bootstrap por rachas)."""
    Rc = R - R.mean(0)
    Rb = pd.DataFrame(Rc).groupby(np.asarray(blocks)).sum().values
    nb = len(Rb)
    return Rb.T @ Rb / len(R) ** 2 * nb / max(nb - 1, 1)


def main():
    lab, te, _ = datos()
    ms = list(MODELOS)
    X, Xt, P, Pt, info = {}, {}, {}, {}, {}
    for k in ms:
        nom = MODELOS[k][0]
        o = lab[["sample_id", "row_index"]].merge(leer(nom, "oof"), how="left").score.values
        q = te[["sample_id", "row_index"]].merge(leer(nom, "preds"), how="left").score.values
        assert not np.isnan(o).any() and not np.isnan(q).any(), nom
        X[k] = resumen(lab, a_prob(lab, o, k in PROB), lab.target_index.values)
        Xt[k] = resumen(te, a_prob(te, q, k in PROB))
        P[k], Pt[k], el, c = calibrar(X[k], Xt[k])
        H = X[k].H.values
        info[k] = dict(cv=H.mean(), p_medio=P[k].mean(), brier=np.mean((H - P[k]) ** 2), brier_cte=H.var(),
                       ll=log_loss(H, np.clip(P[k], 1e-6, 1 - 1e-6)), ll_cte=log_loss(H, np.full(len(H), H.mean())),
                       elegidos=[str(e) for e in el], final=str(c))
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in info[k].items()}, flush=True)
    S0 = X[ms[0]]
    H = np.column_stack([X[k].H.values for k in ms]); Pm = np.column_stack([P[k] for k in ms])
    R = H - Pm
    ses = S0.session.values; atc = S0.atco.values; blk = S0.block.values
    # residuo por sesión, S_j y momentos
    sess = pd.unique(ses)
    r, S, n, a_of = {}, {}, {}, {}
    for j in sess:
        i = ses == j
        r[j] = R[i].mean(0); S[j] = cov_conglomerados(R[i], blk[i]); n[j] = i.sum(); a_of[j] = atc[i][0]
    print("sesgo medio (H - p) por modelo:", dict(zip(ms, R.mean(0).round(4))))
    tot = np.mean([np.outer(r[j], r[j]) - S[j] for j in sess], 0)
    pares = [(j, k) for j in sess for k in sess if j < k and a_of[j] == a_of[k]]
    Ta = psd(np.mean([(np.outer(r[j], r[k]) + np.outer(r[k], r[j])) / 2 for j, k in pares], 0))
    Ts = psd(tot - Ta)
    S1 = sum(S[j] * n[j] ** 2 for j in sess) / sum(n.values())          # cov por snapshot (con correlación de racha)
    print("sd controlador:", dict(zip(ms, np.sqrt(np.diag(Ta)).round(4))),
          "\nsd sesión:", dict(zip(ms, np.sqrt(np.diag(Ts)).round(4))),
          "\nsd muestreo (n=930):", dict(zip(ms, np.sqrt(np.diag(S1) / 930).round(4))), f"({len(pares)} pares)")
    # validación interna: cobertura por controlador (nota prevista p̄ ± z·sd, sin shrink)
    cov = []
    for a in np.unique(atc):
        i = atc == a; js = [j for j in sess if a_of[j] == a]; w = np.array([n[j] for j in js]) / i.sum()
        V = Ta + Ts * (w ** 2).sum() + S1 / i.sum()
        z = R[i].mean(0) / np.sqrt(np.diag(V)); cov.append(np.abs(z))
    cov = np.array(cov)
    print("cobertura interna 90% por controlador:", dict(zip(ms, (cov < 1.645).mean(0).round(3))),
          " 95%:", dict(zip(ms, (cov < 1.96).mean(0).round(3))))

    res = dict(info=info, sd_controlador=dict(zip(ms, np.sqrt(np.diag(Ta)))), sd_sesion=dict(zip(ms, np.sqrt(np.diag(Ts)))))
    rng = np.random.default_rng(0)
    pt = np.column_stack([Pt[k] for k in ms]); tses = Xt[ms[0]].session.values; tat = Xt[ms[0]].atco.values
    for a in (20, 23):
        i = tat == a; nt = i.sum()
        w = pd.Series(tses[i]).value_counts(normalize=True).values
        mu = pt[i].mean(0); V = Ts * (w ** 2).sum() + S1 / nt
        if a == 20:
            j1 = "020sc1"; G = np.linalg.inv(Ta + Ts + S[j1])
            mu = mu + Ta @ G @ r[j1]; V = V + Ta - Ta @ G @ Ta
            mu0 = pt[i].mean(0)
        else:
            V = V + Ta
        sd = np.sqrt(np.diag(V))
        print(f"\n== controlador 0{a} (n={nt}) ==")
        filas = {}
        for q, k in enumerate(ms):
            fila = dict(p_medio=float(pt[i, q].mean()), esperado=float(mu[q]), sd=float(sd[q]),
                        ic90=(float(mu[q] - 1.645 * sd[q]), float(mu[q] + 1.645 * sd[q])))
            if a == 20:
                obs = MODELOS[k][1]; fila.update(publico=obs, z=float((obs - mu[q]) / sd[q]),
                                                 dentro90=bool(abs(obs - mu[q]) < 1.645 * sd[q]))
            filas[k] = fila
            print(k, {x: (np.round(y, 4) if not isinstance(y, bool) else y) for x, y in fila.items()})
        sim = rng.multivariate_normal(mu, psd(V), 400000)
        if a == 20:
            obs = np.array([MODELOS[k][1] for k in ms])
            dmu = mu - mu.mean(); dobs = obs - obs.mean()
            print("orden previsto:", [ms[q] for q in np.argsort(-mu)], " orden real:", [ms[q] for q in np.argsort(-obs)])
            print("diferencias vs media (prev/real):", dict(zip(ms, zip(dmu.round(4), dobs.round(4)))))
            print("corr(prev, real) =", round(float(np.corrcoef(mu, obs)[0, 1]), 3),
                  " error medio (real - prev) =", round(float((obs - mu).mean()), 4))
            dif = obs - mu
            Vd = V  # test chi2 de las diferencias entre modelos (contrastes)
            C = np.eye(5)[1:] - np.eye(5)[:1]
            chi = float((C @ dif) @ np.linalg.pinv(C @ Vd @ C.T) @ (C @ dif))
            print("chi2 de contrastes (4 gl, p95=9.49):", round(chi, 2))
            res["publico_020"] = dict(filas=filas, chi2_contrastes=chi)
        else:
            idx = {k: ms.index(k) for k in ("A", "B", "N")}
            sub = sim[:, [idx["A"], idx["B"], idx["N"]]]
            best = np.bincount(sub.argmax(1), minlength=3) / len(sub)
            pb = dict(zip(["A", "B", "N"], best.round(3)))
            print("P(mejor entre A,B,N):", pb)
            pares_r = {}
            for x, y in (("N", "B"), ("N", "A"), ("A", "B")):
                mx = np.maximum(sim[:, idx[x]], sim[:, idx[y]])
                d = sim[:, idx[x]] - sim[:, idx[y]]
                pares_r[f"{x}+{y}"] = dict(E_max=float(mx.mean()), ic90=tuple(np.quantile(mx, [.05, .95]).round(4)),
                                           P_x_gana=float((d > 0).mean()), sd_dif=float(d.std()))
                print(f"{x}+{y}:", {u: (np.round(v, 4) if not isinstance(v, tuple) else v) for u, v in pares_r[f'{x}+{y}'].items()})
            res["privado_023"] = dict(filas=filas, P_mejor=pb, parejas=pares_r)
    (OUT / "f12_calib_res.json").write_text(json.dumps(res, indent=1, default=str))
    extra = dict(nota="no es un modelo: OOF = N; calibración y notas esperadas en f12_calib_res.json",
                 privado_023=res.get("privado_023", {}).get("P_mejor"))
    registrar("f12_calib", "f12", "calibración Bernoulli + efectos aleatorios controlador/sesión (ver JSON)",
              oof_df=leer(MODELOS["N"][0], "oof"), extra=extra)

if __name__ == "__main__":
    main()

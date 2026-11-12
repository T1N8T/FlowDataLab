"""islr_etapas.py · ¿Qué ideas del proyecto aportan información y cuáles son ruido de búsqueda? Regla de adopción explícita.

REGLA (se fija antes de mirar): un candidato se adopta frente al modelo actual solo si, con el controlador como unidad (n = 18),
  (a) la mejora media por controlador supera el MÁXIMO ESPERADO DEL RUIDO tras K pruebas:  Δ > σ_Δ · √(2 ln K)
      (σ_Δ = desviación de la diferencia de CV entre dos ejecuciones idénticas con distintas semillas), y
  (b) la prueba de permutación con cambio de signo (ISLR §13.5) es significativa con Bonferroni para K pruebas:
      p < 0.05 / K (ISLR §13.3.2; Bonferroni y no Holm porque las decisiones son secuenciales).

--historico  aplica la regla a la cadena que se hizo en el proyecto (OOF ya calculados, no entrena; ~1 min), con dos K:
             K = 620 (las configuraciones registradas de verdad: futuro/pruebas/resultados.jsonl, experiments/*.csv,
             realista/noche) y K = 9 (si solo se hubieran probado los pasos de la cadena, como en un plan por etapas).
--etapas     rehace el recorrido con un PROTOCOLO FIJO (LightGBM LambdaRank con parámetros por defecto razonables, sin
             ajustar, 3 semillas, CV por controlador), añadiendo las ideas por orden de prioridad:
               E1 información (bloques, en orden): logic → past → future → clear → instr → oraculo
               E2 estructura: segundo paso de sesión (sesion.py)
               E3 capacidad: 63 hojas + extra_trees (los parámetros de N)
               E4 conjunto: 10 semillas; mezcla en probabilidades con/sin oráculo (si el oráculo se adoptó)
               E5 comprobación híbrida (ISLR §6.1.2: la selección hacia delante no garantiza el mejor modelo): quitar cada
                  bloque adoptado y volver a probar cada bloque rechazado sobre el modelo final.
             Para cada bloque de información, además: (1) el mismo bloque con un modelo simple sin ajustar (logística
             condicional, λ fijo) — si también mejora, es información y no ajuste —, y (2) una nula por permutación (el
             bloque barajado entre snapshots, 3 veces). Al final reentrena con todo y escribe el envío.
             K = nº de pruebas previsto del plan (30). Todo se cachea en entrega/cache/etapas/: si se corta, continúa.

Uso:  python entrega/codigo/islr_etapas.py --historico
      python entrega/codigo/islr_etapas.py --etapas            # horas
Salidas: salidas/notas/islr_historico.csv, islr_etapas.csv, y (--etapas) salidas/submissions/sub_etapas_final.csv
"""
import argparse, hashlib, json, time
import numpy as np, pandas as pd
from comun import K, labels, hits, leer, guardar, lsm, SALIDAS, CACHE
from islr_pruebas import perm_signo

SIGMA_HIST = 0.0015         # ruido de semillas de la CV en el proyecto (±0.1–0.2 puntos con pocas semillas, README)
K_HIST, K_CADENA, K_PLAN = 620, 9, 30
CADENA = ["limpia", "real_adri_noche", "futuro_base", "futuro_clear", "futuro_clear_ses", "A", "B", "N30_ses",
          "C30_ses", "M30_30"]
QUE = {"real_adri_noche": "+ pasado (realista)", "futuro_base": "+ futuro", "futuro_clear": "+ clear",
       "futuro_clear_ses": "+ segundo paso de sesión (sobre clear)", "A": "+ 10 semillas + suavizado",
       "B": "+ oráculo de coordinaciones (sobre A)", "N30_ses": "+ instr + 63 hojas + extra_trees, 30 semillas",
       "C30_ses": "+ oráculo (sobre N)", "M30_30": "mezcla 0.7·C + 0.3·N"}


def por_atco(h):
    lab = labels()
    return h.groupby(lab.atco.loc[h.index]).mean()


def decidir(h_act, h_cand, sigma, k):
    """Aplica la regla. h_*: Series sample_id -> acierto (mismos snapshots). Devuelve dict con la decisión."""
    idx = h_act.index.intersection(h_cand.index)
    d = (por_atco(h_cand.loc[idx]) - por_atco(h_act.loc[idx])).values
    umbral = sigma * np.sqrt(2 * np.log(k))
    p = perm_signo(d)
    return dict(dif=d.mean(), umbral=umbral, p_perm=p, alfa=0.05 / k, suben=int((d > 0).sum()), bajan=int((d < 0).sum()),
                pasa_ruido=d.mean() > umbral, pasa_bonf=p < 0.05 / k, adopta=bool(d.mean() > umbral and p < 0.05 / k))


# ------------------------------------------------------------------------------------------------------------- histórico
def historico():
    priv = pd.read_csv(SALIDAS / "notas" / "metricas_con_kaggle.csv")
    H = {n: hits(leer(n)) for n in CADENA if leer(n) is not None}
    filas = []
    for k in (K_HIST, K_CADENA):
        actual = CADENA[0]
        for c in CADENA[1:]:
            if c not in H:
                continue
            r = decidir(H[actual], H[c], SIGMA_HIST, k)
            filas.append(dict(K=k, actual=actual, candidato=c, que=QUE.get(c, ""), **r))
            if r["adopta"]:
                actual = c
        filas.append(dict(K=k, actual=actual, candidato="== FINAL", que=""))
    R = pd.DataFrame(filas)
    R.to_csv(SALIDAS / "notas" / "islr_historico.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print(f"σ_Δ = {SIGMA_HIST * 100:.2f} puntos · umbral de ruido = σ·√(2 ln K): K=620 → "
          f"{SIGMA_HIST * np.sqrt(2 * np.log(K_HIST)) * 100:.2f} pts, K=9 → {SIGMA_HIST * np.sqrt(2 * np.log(K_CADENA)) * 100:.2f} pts\n")
    cols = ["actual", "candidato", "que", "dif", "umbral", "p_perm", "alfa", "suben", "bajan", "pasa_ruido", "pasa_bonf",
            "adopta"]
    for k in (K_HIST, K_CADENA):
        print(f"== K = {k}")
        print(R[R.K == k][cols].round(5).to_string(index=False), "\n")
    print("privado conocido de los envíos (para contrastar):")
    print(priv[["modelo", "cv", "privado"]].to_string(index=False))
    return R


# ----------------------------------------------------------------------------------------------------------------- etapas
PARAMS0 = dict(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
PARAMS_N = dict(num_leaves=63, min_child_samples=100, colsample_bytree=0.7, lambdarank_truncation_level=5,
                reg_lambda=0.0, extra_trees=True)
BLOQ_E1 = ["logic", "past", "future", "clear", "instr", "oraculo"]
DIR = CACHE / "etapas"
RAPIDO = False              # --rapido: solo fold 0, 30 árboles, caché aparte, sin test (comprobación de que corre)


class Ctx:
    def __init__(self):
        from features_futuro import construir
        from oraculo import oraculo
        self.lab, self.te, self.fs = construir()
        self.O = oraculo(self.lab, self.te)
        self.folds = sorted(self.lab.fold.unique())[:1] if RAPIDO else sorted(self.lab.fold.unique())
        DIR.mkdir(parents=True, exist_ok=True)

    def cols(self, bloques):
        return [c for b in ["base"] + [b for b in bloques if b != "oraculo"] for c in self.fs[b]]

    def tabla(self, bloques, f):
        """lab con el oráculo de la versión `f` (fold o "lab") si 'oraculo' está entre los bloques."""
        if "oraculo" not in bloques:
            return self.lab
        return pd.concat([self.lab, self.O[f].reset_index(drop=True)], axis=1)

    def F(self, bloques):
        return self.cols(bloques) + (list(self.O["lab"].columns) if "oraculo" in bloques else [])


def clave(**kw):
    return hashlib.md5(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cv_lgb(ctx, bloques, params, semillas, permutar=None, rep=0):
    """OOF (array alineado con ctx.lab) de LightGBM LambdaRank, media de semillas. Cacheado. `permutar`: bloque cuyas
    columnas se barajan (juntas) entre snapshots, en train y validación, para la nula por permutación."""
    import lightgbm as lgb
    kk = clave(m="lgb", b=sorted(bloques), p=params, s=semillas, perm=permutar, rep=rep)
    f_cache = DIR / f"{kk}.npy"
    if f_cache.exists():
        return np.load(f_cache)
    t0 = time.time()
    F = ctx.F(bloques)
    oof = np.zeros(len(ctx.lab))
    for f in ctx.folds:
        L = ctx.tabla(bloques, f)
        if permutar:
            pc = ctx.fs[permutar] if permutar != "oraculo" else list(ctx.O["lab"].columns)
            sid = L.sample_id.values
            u = np.unique(sid)
            perm = dict(zip(u, np.random.default_rng(100 + rep).permutation(u)))
            # cada snapshot recibe el bloque de otro snapshot (fila a fila por posición; si tiene menos filas, NaN)
            src = pd.DataFrame({"sample_id": [perm[s] for s in sid], "pos": L.groupby("sample_id").cumcount().values})
            blk = L[pc].assign(sample_id=sid, pos=L.groupby("sample_id").cumcount().values)
            L = L.copy()
            L[pc] = src.merge(blk, on=["sample_id", "pos"], how="left")[pc].values
        tr, va = (L.fold != f).values, (L.fold == f).values
        A = L[tr]
        g = A.groupby("sample_id", sort=False).size().values
        for s in semillas:
            m = lgb.LGBMRanker(**{**PARAMS0, **params, "random_state": s})
            m.fit(A[F], A.target_index, group=g, categorical_feature=["task_index"])
            oof[va] += m.predict(L.loc[va, F]) / len(semillas)
    np.save(f_cache, oof)
    print(f"      [{'+'.join(bloques) or 'base'}{' perm=' + permutar if permutar else ''}, {len(semillas)} sem.] "
          f"{time.time() - t0:.0f}s", flush=True)
    return oof


def cv_logit(ctx, bloques, lam=1e-3):
    """Modelo simple sin ajustar: logística condicional ridge con λ fijo (islr_escalera)."""
    from islr_escalera import Prep, logit_cond_fit
    kk = clave(m="logit", b=sorted(bloques), lam=lam)
    f_cache = DIR / f"{kk}.npy"
    if f_cache.exists():
        return np.load(f_cache)
    oof = np.zeros(len(ctx.lab))
    for f in ctx.folds:
        L = ctx.tabla(bloques, f)
        F = ctx.F(bloques)
        tr, va = (L.fold != f).values, (L.fold == f).values
        pp = Prep().fit(L[tr], F)
        w = logit_cond_fit(pp.transform(L[tr]), L.target_index.values[tr].astype(float), L.sample_id.values[tr], lam=lam)
        oof[va] = pp.transform(L[va]) @ w
    np.save(f_cache, oof)
    return oof


def H(ctx, oof):
    return hits(ctx.lab[K].assign(score=oof))


def con_sesion(ctx, oof):
    from sesion import sesion
    return sesion(ctx.lab, oof)[0]


def mezcla_cv(ctx, a, b):
    """Mezcla en probabilidades (1 − w)·a + w·b; w se elige por CV anidada (4 folds → 5.º) en 0, 0.1, …, 1."""
    sid = ctx.lab.sample_id.values
    pa, pb = np.exp(lsm(sid, a)), np.exp(lsm(sid, b))
    out = np.zeros(len(a))
    for f in ctx.folds:
        tr = (ctx.lab.fold != f).values
        best = max(np.linspace(0, 1, 11), key=lambda w: hits(ctx.lab.loc[tr, K].assign(score=(1 - w) * pa[tr] + w * pb[tr])).mean())
        out[~tr] = (1 - best) * pa[~tr] + best * pb[~tr]
    return out


def etapas():
    ctx = Ctx()
    filas, n_pruebas = [], [0]

    def prueba(etapa, desc, h_act, h_cand, extra=None):
        n_pruebas[0] += 1
        r = decidir(h_act, h_cand, sigma, K_PLAN)
        filas.append(dict(etapa=etapa, prueba=desc, cv_actual=h_act.mean(), cv_cand=h_cand.mean(), **r, **(extra or {})))
        print(f"  {etapa} · {desc}: Δ {r['dif'] * 100:+.2f} pts (umbral {r['umbral'] * 100:.2f}), p {r['p_perm']:.2g} "
              f"(α {r['alfa']:.2g}), {r['suben']}↑ {r['bajan']}↓ ⇒ {'ADOPTA' if r['adopta'] else 'no'}", flush=True)
        pd.DataFrame(filas).to_csv(SALIDAS / "notas" / "islr_etapas.csv", index=False)
        return r["adopta"]

    S3 = [0, 1, 2]
    # ruido: el modelo base con tres grupos de semillas distintos
    print("== E0 · ruido de semillas (base, 3 grupos de 3 semillas)", flush=True)
    cvs = [H(ctx, cv_lgb(ctx, [], {}, s)).mean() for s in ([0, 1, 2], [3, 4, 5], [6, 7, 8])]
    sigma = float(np.std(cvs, ddof=1) * np.sqrt(2))
    print(f"  CV base: {', '.join(f'{c:.4f}' for c in cvs)} ⇒ σ_Δ = {sigma * 100:.2f} pts, umbral (K={K_PLAN}) = "
          f"{sigma * np.sqrt(2 * np.log(K_PLAN)) * 100:.2f} pts", flush=True)

    # E1 · información
    print("== E1 · información (bloques en orden de prioridad)", flush=True)
    adoptados, rechazados = [], []
    act = cv_lgb(ctx, [], {}, S3)
    simple_base = H(ctx, cv_logit(ctx, []))
    for b in BLOQ_E1:
        cand = cv_lgb(ctx, adoptados + [b], {}, S3)
        nulas = [H(ctx, cv_lgb(ctx, adoptados + [b], {}, [0], permutar=b, rep=r)).mean() for r in range(3)]
        hs = H(ctx, cv_logit(ctx, adoptados + [b]))
        hs_act = H(ctx, cv_logit(ctx, adoptados))
        extra = dict(cv_nula_perm_max=max(nulas), cv_simple_act=hs_act.mean(), cv_simple_cand=hs.mean())
        if prueba("E1", f"+{b}", H(ctx, act), H(ctx, cand), extra):
            adoptados.append(b); act = cand
        else:
            rechazados.append(b)
    # E2 · estructura
    print("== E2 · estructura", flush=True)
    ses = con_sesion(ctx, act)
    usa_ses = prueba("E2", "+segundo paso de sesión", H(ctx, act), H(ctx, ses))
    fin = ses if usa_ses else act
    # E3 · capacidad
    print("== E3 · capacidad", flush=True)
    capN = cv_lgb(ctx, adoptados, PARAMS_N, S3)
    capN_f = con_sesion(ctx, capN) if usa_ses else capN
    usa_N = prueba("E3", "63 hojas + extra_trees", H(ctx, fin), H(ctx, capN_f))
    params = PARAMS_N if usa_N else {}
    if usa_N:
        fin = capN_f
    # E4 · conjunto
    print("== E4 · conjunto", flush=True)
    s10 = cv_lgb(ctx, adoptados, params, list(range(10)))
    s10_f = con_sesion(ctx, s10) if usa_ses else s10
    semillas = list(range(10)) if prueba("E4", "10 semillas", H(ctx, fin), H(ctx, s10_f)) else S3
    if semillas != S3:
        fin = s10_f
    mezcla_w = None
    if "oraculo" in adoptados:
        sin = cv_lgb(ctx, [b for b in adoptados if b != "oraculo"], params, semillas)
        sin = con_sesion(ctx, sin) if usa_ses else sin
        mz = mezcla_cv(ctx, fin, sin)
        if prueba("E4", "mezcla con/sin oráculo en probabilidades", H(ctx, fin), H(ctx, mz)):
            fin, mezcla_w = mz, True
    # E5 · comprobación híbrida
    print("== E5 · comprobación híbrida (quitar adoptados, reintentar rechazados)", flush=True)
    base_fin = cv_lgb(ctx, adoptados, params, S3)
    for b in list(adoptados):
        sin_b = cv_lgb(ctx, [x for x in adoptados if x != b], params, S3)
        prueba("E5", f"quitar {b} (Δ = con − sin)", H(ctx, sin_b), H(ctx, base_fin))
    for b in rechazados:
        con_b = cv_lgb(ctx, adoptados + [b], params, S3)
        prueba("E5", f"reintentar +{b} sobre el final", H(ctx, base_fin), H(ctx, con_b))

    R = pd.DataFrame(filas)
    R.to_csv(SALIDAS / "notas" / "islr_etapas.csv", index=False)
    desc = dict(bloques=adoptados, sesion=usa_ses, capacidad_N=usa_N, semillas=len(semillas), mezcla_oraculo=bool(mezcla_w))
    print(f"\n== MODELO FINAL: {desc} · CV {H(ctx, fin).mean():.4f} · pruebas hechas: {n_pruebas[0]} (K previsto {K_PLAN})")
    if not RAPIDO:
        final_test(ctx, adoptados, params, semillas, usa_ses, mezcla_w)


def final_test(ctx, bloques, params, semillas, usa_ses, mezcla):
    """Reentrena con todo (ISLR §6.5.2), aplica sesión/mezcla con pesos de todo el OOF y escribe el envío."""
    import lightgbm as lgb
    from sesion import sesion

    def pred(bl):
        F = ctx.F(bl)
        L = pd.concat([ctx.lab, ctx.O["lab"]], axis=1) if "oraculo" in bl else ctx.lab
        T = pd.concat([ctx.te, ctx.O["te"]], axis=1) if "oraculo" in bl else ctx.te
        g = L.groupby("sample_id", sort=False).size().values
        p = np.mean([lgb.LGBMRanker(**{**PARAMS0, **params, "random_state": s}).fit(
            L[F], L.target_index, group=g, categorical_feature=["task_index"]).predict(T[F]) for s in semillas], axis=0)
        oof = cv_lgb(ctx, bl, params, semillas)
        if usa_ses:
            oof, p, _ = sesion(ctx.lab, oof, ctx.te, p)
        return oof, p

    oof, p = pred(bloques)
    if mezcla:
        oof2, p2 = pred([b for b in bloques if b != "oraculo"])
        sid, sidt = ctx.lab.sample_id.values, ctx.te.sample_id.values
        pa, pb = np.exp(lsm(sid, oof)), np.exp(lsm(sid, oof2))
        w = max(np.linspace(0, 1, 11), key=lambda w: hits(ctx.lab[K].assign(score=(1 - w) * pa + w * pb)).mean())
        oof = (1 - w) * pa + w * pb
        p = (1 - w) * np.exp(lsm(sidt, p)) + w * np.exp(lsm(sidt, p2))
    guardar("etapas_final", ctx.lab[K].assign(score=oof), ctx.te[K].assign(score=p))
    print("envío: entrega/salidas/submissions/sub_etapas_final.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--historico", action="store_true")
    ap.add_argument("--etapas", action="store_true")
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    if a.rapido:
        RAPIDO, DIR = True, CACHE / "etapas_rapido"
        PARAMS0["n_estimators"] = 30
    if a.historico:
        historico()
    if a.etapas:
        etapas()

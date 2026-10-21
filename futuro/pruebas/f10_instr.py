"""f10_instr.py · bloque `instr` (si_*): qué TIPO de instrucción recibe el avión, cuándo, y si encaja con la tarea (hilo f10).
f_next_input / f_next_act solo miran la PRIMERA instrucción siguiente. Aquí, con la lista de inputs de cada avión en la sesión
(misma extracción que features_future.add_future: a_since_flight + a_last_*; sin etiquetas):
    si_next_<act>  s hasta la siguiente instrucción de cada tipo (assume, force_assume, nivel, lateral, transfer, otro)
    si_prev_<act>  s desde la última de cada tipo (hacia atrás: ¿ya se dio la que resuelve la tarea?)
    si_match       s hasta la siguiente instrucción COMPATIBLE con la tarea (tabla compat de features_future), aunque no
                   sea la primera; si_prevmatch: s desde la última compatible
    si_match_rk / si_match_rk_flight   rango de si_match en el snapshot / entre las tareas del mismo avión
    si_n60, si_n180  nº de instrucciones al avión en los próximos 60 / 180 s
Modos:
    python futuro/pruebas/f10_instr.py cv NOMBRE [semillas] [extra]  # 1.er pase (clear [+ instr]) + sesión, CV completa y "CV 023"
    python futuro/pruebas/f10_instr.py pseudo                        # pseudo-etiquetas transductivas (ver pseudo())
NOMBRE: f10_ref (clear), f10_instr (clear + instr), f10_instr_l63 (+ 63 hojas, extra_trees), f10_instr_or (+ oráculo coord).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import datos, lgb_cv, guardar, registrar, OUT, PARAMS_FUT, lgb_fit  # noqa: E402
import features_future as ff, feat_session  # noqa: E402
from metric import hits  # noqa: E402

ACT_N = ["assume", "force", "nivel", "lateral", "transfer", "otro"]
CV023 = [2, 11, 10, 6, 17, 15]          # controladores de la CV más parecidos al 023 (sección 16f)
CACHE = OUT / "cache_instr.parquet"
COMPAT = {4: [0, 1], 2: [0, 1], 5: [4], 12: [3], 13: [3], 10: [2], 3: [2, 3], 6: [2, 3], 1: [2, 3], 0: [0, 1, 2, 3],
          7: [2, 3], 8: [2, 3]}


def add_instr(d):
    ev = np.where(d.a_since_flight < 3600, np.round(d.t - d.a_since_flight), np.nan)
    k = d.assign(ev=ev).dropna(subset=["ev"]).sort_values("t").drop_duplicates(["session", "cs", "ev"])
    acts = k[["a_last_" + a for a in ff.ACTS]].values
    k = k.assign(act=np.where(acts.max(1) == 1, acts.argmax(1), len(ff.ACTS)))[["session", "cs", "ev", "act"]]
    n = len(d)
    nxt = np.full((n, len(ACT_N)), np.nan); prv = np.full((n, len(ACT_N)), np.nan)
    n60, n180 = np.zeros(n), np.zeros(n)
    t_all = d.t.values
    ev_g = {key: g for key, g in k.sort_values("ev").groupby(["session", "cs"])}
    for key, idx in d.groupby(["session", "cs"]).indices.items():
        if key not in ev_g:
            continue
        g = ev_g[key]; e, a = g.ev.values, g.act.values
        t = t_all[idx]
        lo = np.searchsorted(e, t + 1.5)
        n60[idx] = np.searchsorted(e, t + 60, side="right") - lo
        n180[idx] = np.searchsorted(e, t + 180, side="right") - lo
        for j in range(len(ACT_N)):
            ej = e[a == j]
            if not len(ej):
                continue
            p = np.searchsorted(ej, t + 1.5); ok = p < len(ej)
            nxt[idx[ok], j] = ej[p[ok]] - t[ok]
            q = np.searchsorted(ej, t + 1.5) - 1; ok = q >= 0
            prv[idx[ok], j] = t[ok] - ej[q[ok]]
    out = pd.DataFrame(index=d.index)
    for j, nm in enumerate(ACT_N):
        out["si_next_" + nm], out["si_prev_" + nm] = nxt[:, j], prv[:, j]
    ti = d.task_index.values
    match, pmatch = np.full(n, np.nan), np.full(n, np.nan)
    for typ, comp in COMPAT.items():
        m = ti == typ
        match[m] = np.nanmin(nxt[m][:, comp], axis=1) if m.any() else np.nan
        pmatch[m] = np.nanmin(prv[m][:, comp], axis=1) if m.any() else np.nan
    out["si_match"], out["si_prevmatch"], out["si_n60"], out["si_n180"] = match, pmatch, n60, n180
    out["si_match_rk"] = out.si_match.groupby(d.sample_id.values).rank(method="min")
    out["si_match_rk_flight"] = out.si_match.groupby([d.sample_id.values, d.cs.values]).rank(method="min")
    return out


def datos_instr():
    lab, te, fs = datos()
    if CACHE.exists():
        I = pd.read_parquet(CACHE)
    else:
        t0 = time.time()
        D = pd.concat([lab, te], ignore_index=True)
        I = add_instr(D)
        I.to_parquet(CACHE)
        print(f"bloque instr calculado en {time.time() - t0:.0f}s", flush=True)
    fs["instr"] = list(I.columns)
    lab = pd.concat([lab, I.iloc[:len(lab)].reset_index(drop=True)], axis=1)
    te = pd.concat([te, I.iloc[len(lab):].reset_index(drop=True)], axis=1)
    return lab, te, fs


def medir(nombre, lab, oof):
    h = hits(lab[["sample_id", "row_index"]].assign(score=oof))
    at = lab.drop_duplicates("sample_id").set_index("sample_id").atco.reindex(h.index)
    return h.mean(), h[at.isin(CV023)].mean(), h.groupby(at).mean().min()


VARIANTES = {"f10_ref": (["clear"], {}, False), "f10_instr": (["clear", "instr"], {}, False),
             "f10_instr_l63": (["clear", "instr"], dict(num_leaves=63, min_child_samples=100, extra_trees=True), False),
             "f10_instr_or": (["clear", "instr"], {}, True), "f10_ref_or": (["clear"], {}, True),
             "f10_instr_l63_or": (["clear", "instr"], dict(num_leaves=63, min_child_samples=100, extra_trees=True), True)}


def cv(nombre, seeds=3):
    t0 = time.time()
    lab, te, fs = datos_instr()
    bloques, p, orac = VARIANTES[nombre]
    F = [c for b in ["base", "logic", "past", "future"] + bloques for c in fs[b]]
    seeds = tuple(range(int(seeds)))
    P = {**PARAMS_FUT, **p, "n_jobs": 16}
    if orac:      # oráculo de coordinaciones de Pablo (feat_oracle vía m_lgb_future.oracle_data), calculado por fold
        import m_lgb_future as M
        O = M.oracle_data(dict(lab=lab, te=te), "coord")
        F = F + list(O["lab"].columns)
        oof = np.zeros(len(lab))
        for f in range(5):
            L = pd.concat([lab, O[f]], axis=1)
            tr, va = (L.fold != f).values, (L.fold == f).values
            oof[va] = np.mean([lgb_fit(L[tr], F, {**P, "random_state": s}).predict(L.loc[va, F]) for s in seeds], axis=0)
        L, T = pd.concat([lab, O["lab"]], axis=1), pd.concat([te, O["te"]], axis=1)
        pred = np.mean([lgb_fit(L, F, {**P, "random_state": s}).predict(T[F]) for s in seeds], axis=0)
    else:
        oof, pred = lgb_cv(lab, F, P, seeds=seeds, test=te)
    nm = f"{nombre}_s{len(seeds)}"
    guardar(nm, lab, oof, te, pred)
    so, sp, _ = feat_session.sesion(lab, oof, te, pred)
    guardar(nm + "_ses", lab, so, te, sp)
    a, b = medir(nm, lab, oof), medir(nm, lab, so)
    for n_, (c, c23, mn) in ((nm, a), (nm + "_ses", b)):
        registrar(n_, hilo="f10", descripcion=f"{'+'.join(bloques)} {p} oráculo={orac} seeds={list(seeds)}",
                  n_features=len(F), extra=dict(cv_023=round(c23, 4)))
    print(f"{nm}: 1.er pase {a[0]:.4f} (CV 023 {a[1]:.4f}) · + sesión {b[0]:.4f} (CV 023 {b[1]:.4f}, peor {b[2]:.3f})  "
          f"{len(F)} features ({time.time() - t0:.0f}s)", flush=True)


def pseudo(umbral=0.97, seeds=1):
    """Pseudo-etiquetas transductivas, evaluadas como se usarían: en cada fold, las rachas del fold de validación en que
    el 1.er pase + sesión está muy seguro (prob. top-1 >= umbral en TODOS sus snapshots) se añaden al entrenamiento con
    su top-1 como etiqueta, y se vuelve a predecir el fold. Así se mide si adaptarse al controlador nuevo ayuda."""
    import suavizado as S
    lab, te, fs = datos_instr()
    F = [c for b in ["base", "logic", "past", "future", "clear"] for c in fs[b]]
    P = {**PARAMS_FUT, "n_jobs": 10}
    seeds = tuple(range(int(seeds)))
    base = pd.read_csv(OUT / "oof" / "f10_ref_s3_ses.csv")
    L = lab.merge(base, on=["sample_id", "row_index"], how="left")
    p = np.exp(S.lsm(L, L.score.values))
    L["p"], L["rk"] = p, L.groupby("sample_id").score.rank(ascending=False, method="first")
    top = L[L.rk == 1][["sample_id", "block", "row_index", "p", "target_index"]]
    blk_min = top.groupby("block").p.min()
    seg = top[top.block.map(blk_min) >= umbral]
    print(f"umbral {umbral}: {seg.sample_id.nunique()} snapshots seguros ({seg.sample_id.nunique() / top.sample_id.nunique():.1%}); "
          f"su top-1 acierta {seg.target_index.mean():.3f}", flush=True)
    oof = np.zeros(len(lab))
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        ps = lab[va & lab.sample_id.isin(seg.sample_id).values].copy()
        pick = ps.sample_id.map(seg.set_index("sample_id").row_index)
        ps["target_index"] = (ps.row_index == pick).astype(int)
        A = pd.concat([lab[tr], ps], ignore_index=True)
        oof[va] = np.mean([lgb_fit(A, F, {**P, "random_state": s}).predict(lab.loc[va, F]) for s in seeds], axis=0)
    ref, _ = lgb_cv(lab, F, {**P}, seeds=seeds)
    so, _, _ = feat_session.sesion(lab, oof)
    sr, _, _ = feat_session.sesion(lab, ref)
    a, b = medir("", lab, so), medir("", lab, sr)
    print(f"pseudo ({len(seeds)} semilla): + sesión {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) frente a sin pseudo "
          f"{b[0]:.4f} (CV 023 {b[1]:.4f}, peor {b[2]:.3f})", flush=True)


if __name__ == "__main__":
    modo = sys.argv[1]
    if modo == "cv":
        cv(*sys.argv[2:])
    elif modo == "pseudo":
        pseudo(*[float(x) for x in sys.argv[2:3]])
    elif modo == "build":
        datos_instr()

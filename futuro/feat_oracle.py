"""
feat_oracle.py · Pablo (A) · RAMA FUTURO · "Oráculo" = prior colectivo (idea Pau #1 del EDA, sección 9).

Todas las sesiones de un escenario tienen el mismo tráfico (los aviones de test están al 100 % en train). Para cada
tarea se mira qué hicieron OTROS controladores en el mismo escenario y en el mismo momento (tiempo alineado):
    or_ever        fracción de las otras sesiones (que tienen la tarea) en que fue respuesta alguna vez
    or_near60/180  fracción en que fue respuesta a ±60 / ±180 s del mismo momento
    or_cs_near60   lo mismo para cualquier tarea del avión
    or_dt          segundos hasta el momento mediano en que la trabajan los demás (NaN si nadie); or_dt_abs
    or_*_rk        rangos dentro del snapshot
Variantes (feats(..., variant=)): "base" (lo de arriba), "B" + ventanas ±30/300/600 s, or_cs_ever y núcleo
exp(-|Δt|/120) a la respuesta más cercana; "C" + ventanas ±60/180 s en tiempo relativo a la primera aparición del
avión en la sesión (or_a*); "D" solo or_near180, or_dt, or_ever; "E" = B + C; "coord" = base, pero solo en
entry/exit_coordination (vacío en el resto de tipos).
USA ETIQUETAS de otras sesiones -> se calcula por fold con feats(d, mats, fold_out): solo cuentan sesiones etiquetadas
del mismo escenario, de folds != fold_out y de otro controlador que la fila. fold_out=None (modelo final y test):
todas las sesiones etiquetadas salvo las del propio controlador.
OJO: en main se descartó por la regla 4 (usa scenario y timestamp para alinear). Ver LGB_futuro.ipynb, sección 12.
"""
import numpy as np, pandas as pd

BIN = 10.0


def align(d):
    """Tiempo alineado entre sesiones del mismo escenario (no usa etiquetas)."""
    t0 = d.groupby("session").t.transform("min")
    first = d.groupby(["session", "cs"]).t.min().rename("ft").reset_index()
    first = first.merge(d.drop_duplicates("session")[["session", "scenario"]], on="session")
    first["rel"] = first.ft - first.session.map(d.groupby("session").t.min())
    first["ref"] = first.groupby(["scenario", "cs"]).rel.transform("median")
    corr = (first.rel - first.ref).groupby(first.session).median()
    return (d.t - t0 - d.session.map(corr)).values


def keys_near(names, tb_src, W):
    k = int(np.ceil(W / BIN))
    return pd.MultiIndex.from_arrays([np.repeat(names, 2 * k + 1), (tb_src[:, None] + np.arange(-k, k + 1)).ravel()])


def matrices(d, tau, extra=False):
    """Por escenario: indicadores fila x sesión-fuente (respuesta cerca / alguna vez / tarea presente).
    extra=True añade las matrices de las variantes B y C."""
    lab = d[d.split != "test"]
    is_ans = ((d.split != "test") & (d.target_index == 1)).values
    tau_a = (d.t - d.groupby(["session", "cs"]).t.transform("min")).values      # tiempo desde que aparece el avión
    ans = d.loc[is_ans, ["session", "scenario", "task_name", "cs"]].copy()
    ans["tau"], ans["tau_a"] = tau[is_ans], tau_a[is_ans]
    wins = [(60, "near60"), (180, "near180")] + ([(30, "near30"), (300, "near300"), (600, "near600")] if extra else [])
    out = {}
    for sc, g in d.groupby("scenario"):
        src = sorted(lab[lab.scenario == sc].session.unique())
        idx = g.index.values
        tb, tba = np.floor(tau[idx] / BIN).astype(int), np.floor(tau_a[idx] / BIN).astype(int)
        names = ["ever", "has", "cs60"] + [k for _, k in wins] + (["cs_ever", "a_near60", "a_near180"] if extra else [])
        M = {k: np.zeros((len(idx), len(src)), np.float32) for k in names}
        dt = np.full((len(idx), len(src)), np.nan, np.float32)
        mind = np.full((len(idx), len(src)), np.nan, np.float32) if extra else None
        rk = pd.MultiIndex.from_arrays([g.task_name.values, tb])
        for j, s in enumerate(src):
            a = ans[ans.session == s]
            ab = np.floor(a.tau.values / BIN).astype(int)
            has = set(lab.loc[lab.session == s, "task_name"].unique())
            M["has"][:, j] = g.task_name.isin(has).values
            M["ever"][:, j] = g.task_name.isin(set(a.task_name)).values
            for W, key in wins:
                M[key][:, j] = rk.isin(keys_near(a.task_name.values, ab, W))
            M["cs60"][:, j] = pd.MultiIndex.from_arrays([g.cs.values, tb]).isin(keys_near(a.cs.values, ab, 60))
            med = a.groupby("task_name").tau.median()
            dt[:, j] = g.task_name.map(med).values - tau[idx]
            if extra:
                M["cs_ever"][:, j] = g.cs.isin(set(a.cs)).values
                aba = np.floor(a.tau_a.values / BIN).astype(int)
                rka = pd.MultiIndex.from_arrays([g.task_name.values, tba])
                M["a_near60"][:, j] = rka.isin(keys_near(a.task_name.values, aba, 60))
                M["a_near180"][:, j] = rka.isin(keys_near(a.task_name.values, aba, 180))
                # distancia en tiempo a la respuesta más cercana de esa tarea en la fuente
                x = pd.DataFrame(dict(task_name=g.task_name.values, tau=tau[idx], i=np.arange(len(idx))))
                m2 = x.merge(a[["task_name", "tau"]], on="task_name", suffixes=("", "_s"))
                if len(m2):
                    m2["dd"] = (m2.tau - m2.tau_s).abs()
                    v = m2.groupby("i").dd.min()
                    mind[v.index.values, j] = v.values
        info = lab[lab.scenario == sc].drop_duplicates("session").set_index("session").loc[src]
        out[sc] = dict(idx=idx, M=M, dt=dt, mind=mind, fold=info.fold.values, atco=info.atco.values)
    return out


def feats(d, mats, fold_out, variant="base"):
    """Features or_* con las fuentes permitidas: folds != fold_out (None = todas) y atco distinto de la fila."""
    n = len(d)
    frac = {"or_ever": "ever", "or_near60": "near60", "or_near180": "near180"}
    if variant in ("B", "E"):
        frac.update(or_near30="near30", or_near300="near300", or_near600="near600")
    if variant in ("C", "E"):
        frac.update(or_a_near60="a_near60", or_a_near180="a_near180")
    F = {k: np.full(n, np.nan) for k in list(frac) + ["or_cs_near60", "or_dt", "or_cs_ever", "or_kernel"]}
    for sc, m in mats.items():
        idx = m["idx"]
        ok = np.ones((len(idx), len(m["fold"])), bool)
        if fold_out is not None:
            ok &= (m["fold"] != fold_out)[None, :]
        ok &= d.atco.values[idx][:, None] != m["atco"][None, :]
        has = (m["M"]["has"] * ok).sum(1)
        nsrc = ok.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            for col, key in frac.items():
                F[col][idx] = (m["M"][key] * ok).sum(1) / has
            F["or_cs_near60"][idx] = (m["M"]["cs60"] * ok).sum(1) / nsrc
            F["or_dt"][idx] = np.nanmedian(np.where(ok, m["dt"], np.nan), axis=1)
            if variant in ("B", "E"):
                F["or_cs_ever"][idx] = (m["M"]["cs_ever"] * ok).sum(1) / nsrc
                F["or_kernel"][idx] = np.nansum(np.where(ok, np.exp(-m["mind"] / 120.0), 0), axis=1) / has
    o = pd.DataFrame(F, index=d.index)
    if variant not in ("B", "E"):
        o = o.drop(columns=["or_cs_ever", "or_kernel"])
    rk = [c for c in o.columns if c not in ("or_dt",)]
    if variant == "D":
        return o[["or_near180", "or_dt", "or_ever"]]
    if variant == "coord":
        o.loc[~d.task_index.isin([2, 3]).values] = np.nan
    if variant in ("base", "coord"):
        rk = ["or_ever", "or_near60", "or_near180", "or_cs_near60"]
    for c in rk:
        o[c + "_rk"] = o[c].groupby(d.sample_id).rank(ascending=False, method="min")
    o["or_dt_abs"] = o.or_dt.abs()
    return o

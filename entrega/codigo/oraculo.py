"""oraculo.py · línea "futuro" · oráculo de coordinaciones (or_*): qué hicieron OTROS controladores en la misma situación.

Todas las sesiones de un escenario tienen el mismo tráfico. Para cada tarea se mira, en las otras sesiones etiquetadas del
mismo escenario y en el mismo momento (tiempo alineado entre sesiones), si esa tarea fue la respuesta:
    or_ever        fracción de las otras sesiones (que tienen la tarea) en que fue respuesta alguna vez
    or_near60/180  fracción en que fue respuesta a ±60 / ±180 s del mismo momento
    or_cs_near60   lo mismo para cualquier tarea del avión (fracción de las fuentes)
    or_dt          segundos hasta el momento mediano en que la trabajan los demás (NaN si nadie); or_dt_abs = |or_dt|
    or_*_rk        rangos dentro del snapshot (descendentes)
Variante "coord" (la entregada, modelo C): solo en entry_coordination y exit_coordination; vacío en el resto de tipos.

USA ETIQUETAS de otras sesiones, así que se calcula por fold: en el fold f de la CV solo cuentan sesiones etiquetadas de
folds != f y de otro controlador que la fila. Para el modelo final y test: todas las sesiones etiquetadas salvo las del
propio controlador. Reproduce futuro/feat_oracle.py (variante coord) + futuro/m_lgb_future.oracle_data.

Uso:
    from oraculo import oraculo
    O = oraculo(lab, te)     # {0..4: DataFrame alineado con lab (versión del fold), "lab": ..., "te": ...}, columnas COLS
Cacheado en entrega/cache/oraculo_coord.parquet (columnas <col>__f<fold> y <col>__full sobre las filas de lab y luego te).
"""
import time, warnings
import numpy as np, pandas as pd
from comun import CACHE

BIN = 10.0
CACHE_ORC = CACHE / "oraculo_coord.parquet"
COLS = sorted(["or_ever", "or_near60", "or_near180", "or_cs_near60", "or_dt", "or_ever_rk", "or_near60_rk",
               "or_near180_rk", "or_cs_near60_rk", "or_dt_abs"])          # orden en que entran al modelo C


def align(d):
    """Tiempo alineado entre sesiones del mismo escenario (no usa etiquetas): tiempo desde el inicio de la sesión menos el
    desfase mediano de la primera aparición de cada avión respecto a la mediana del escenario."""
    t0 = d.groupby("session").t.transform("min")
    first = d.groupby(["session", "cs"]).t.min().rename("ft").reset_index()
    first = first.merge(d.drop_duplicates("session")[["session", "scenario"]], on="session")
    first["rel"] = first.ft - first.session.map(d.groupby("session").t.min())
    first["ref"] = first.groupby(["scenario", "cs"]).rel.transform("median")
    corr = (first.rel - first.ref).groupby(first.session).median()
    return (d.t - t0 - d.session.map(corr)).values


def keys_near(names, tb_src, W):
    """Claves (nombre, casilla de 10 s) a ±W s de cada respuesta de la fuente."""
    k = int(np.ceil(W / BIN))
    return pd.MultiIndex.from_arrays([np.repeat(names, 2 * k + 1), (tb_src[:, None] + np.arange(-k, k + 1)).ravel()])


def matrices(d, tau):
    """Por escenario: indicadores fila x sesión-fuente (respuesta cerca / alguna vez / tarea presente) y desfase dt."""
    lab = d[d.split != "test"]
    is_ans = ((d.split != "test") & (d.target_index == 1)).values
    ans = d.loc[is_ans, ["session", "scenario", "task_name", "cs"]].copy()
    ans["tau"] = tau[is_ans]
    wins = [(60, "near60"), (180, "near180")]
    out = {}
    for sc, g in d.groupby("scenario"):
        src = sorted(lab[lab.scenario == sc].session.unique())
        idx = g.index.values
        tb = np.floor(tau[idx] / BIN).astype(int)
        M = {k: np.zeros((len(idx), len(src)), np.float32) for k in ["ever", "has", "cs60"] + [k for _, k in wins]}
        dt = np.full((len(idx), len(src)), np.nan, np.float32)
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
        info = lab[lab.scenario == sc].drop_duplicates("session").set_index("session").loc[src]
        out[sc] = dict(idx=idx, M=M, dt=dt, fold=info.fold.values, atco=info.atco.values)
    return out


def feats(d, mats, fold_out):
    """Features or_* (variante coord) con las fuentes permitidas: folds != fold_out (None = todas) y atco distinto."""
    n = len(d)
    frac = {"or_ever": "ever", "or_near60": "near60", "or_near180": "near180"}
    F = {k: np.full(n, np.nan) for k in list(frac) + ["or_cs_near60", "or_dt"]}
    for sc, m in mats.items():
        idx = m["idx"]
        ok = np.ones((len(idx), len(m["fold"])), bool)
        if fold_out is not None:
            ok &= (m["fold"] != fold_out)[None, :]
        ok &= d.atco.values[idx][:, None] != m["atco"][None, :]
        has = (m["M"]["has"] * ok).sum(1)
        nsrc = ok.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # filas sin fuentes -> NaN (sin aviso)
            for col, key in frac.items():
                F[col][idx] = (m["M"][key] * ok).sum(1) / has
            F["or_cs_near60"][idx] = (m["M"]["cs60"] * ok).sum(1) / nsrc
            F["or_dt"][idx] = np.nanmedian(np.where(ok, m["dt"], np.nan), axis=1)
    o = pd.DataFrame(F, index=d.index)
    o.loc[~d.task_index.isin([2, 3]).values] = np.nan            # solo entry / exit_coordination
    for c in ["or_ever", "or_near60", "or_near180", "or_cs_near60"]:
        o[c + "_rk"] = o[c].round(6).groupby(d.sample_id).rank(ascending=False, method="min")
    o["or_dt_abs"] = o.or_dt.abs()
    return o


def oraculo(lab, te, forzar=False):
    """{f: oráculo del fold f alineado con lab, "lab": versión final sobre lab, "te": versión final sobre te};
    cada DataFrame con las columnas COLS (ordenadas)."""
    n, N = len(lab), len(lab) + len(te)
    if CACHE_ORC.exists() and not forzar:
        X = pd.read_parquet(CACHE_ORC)
    else:
        t0 = time.time()
        D = pd.concat([lab, te], ignore_index=True)
        mats = matrices(D, align(D))
        parts = {}
        for f in sorted(lab.fold.unique()):
            o = feats(D, mats, f)
            parts.update({f"{c}__f{f}": np.concatenate([o[c].values[:n], np.full(N - n, np.nan)]) for c in COLS})
        o = feats(D, mats, None)
        parts.update({f"{c}__full": o[c].values for c in COLS})
        X = pd.DataFrame(parts)
        CACHE.mkdir(parents=True, exist_ok=True)
        X.to_parquet(CACHE_ORC)
        print(f"oráculo de coordinaciones calculado en {time.time() - t0:.0f}s -> {CACHE_ORC}", flush=True)
    assert len(X) == N, "la caché del oráculo no corresponde a estos datos (borrar entrega/cache/oraculo_coord.parquet)"
    folds = sorted({int(c.split("__f")[1]) for c in X.columns if "__f" in c and not c.endswith("__full")})
    O = {f: pd.DataFrame({c: X[f"{c}__f{f}"].values[:n] for c in COLS}) for f in folds}
    O["lab"] = pd.DataFrame({c: X[f"{c}__full"].values[:n] for c in COLS})
    O["te"] = pd.DataFrame({c: X[f"{c}__full"].values[n:] for c in COLS})
    return O


if __name__ == "__main__":
    from features_futuro import construir
    lab, te, _ = construir()
    O = oraculo(lab, te)
    print({k: v.shape for k, v in O.items()})

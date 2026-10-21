"""
claude_mejoras.py · RAMA "REALISTA" · Pruebas sobre la mejor configuración de la búsqueda nocturna (b045) para subir la CV
de real_adri_noche (0.6732). Mismas features, folds y métricas que noche_adri.py (reutiliza sus funciones).

Variantes (cada semilla se guarda en realista/claude/oof|pred; al relanzar se reutiliza):
    b045       referencia: lambdarank con los parámetros de b045
    b045_bin   mismo modelo con objetivo binary (diversidad para mezclar)
    b045_xe    mismo modelo con objetivo rank_xendcg
    b045_ctx   b045 + bloque ctx: diferencias de tiempos (tarea / cambio de estado frente a la última instrucción) y
               contexto del avión en el snapshot. Solo usa el snapshot y columnas que miran atrás.

Uso (desde la raíz):
    python realista/claude_mejoras.py cache                                   # features a la caché (una vez)
    python realista/claude_mejoras.py run --var b045_bin --seeds 3 --hilos 5
    python realista/claude_mejoras.py eval                                    # métricas y mezclas con real_adri_noche
"""
import argparse, sys, time, types
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.modules.setdefault("fcntl", types.SimpleNamespace(flock=lambda *a, **k: None, LOCK_EX=2))
ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import noche_adri as N  # noqa: E402
import metricas  # noqa: E402
from metric import hits, save_outputs  # noqa: E402

B045 = dict(n_estimators=756, lr=0.018642543216427748, leaves=37, min_child=25, subsample=0.9395578452213194,
            colsample=0.7720566531433419, reg_lambda=21.557215921190185, min_gain=0.0005006652310000156, max_bin=63,
            cat_smooth=1.5038235350480391, extra_trees=False, trunc=8)
VARS = dict(b045=("lambdarank", ()), b045_bin=("binary", ()), b045_xe=("rank_xendcg", ()),
            b045_ctx=("lambdarank", ("ctx",)), b045_trig=("lambdarank", ("trig",)),
            b045_bal=("lambdarank", (), "atco"))
OUT = REAL / "claude"


def fe_ctx(d):
    """Diferencias de tiempos y contexto del avión dentro del snapshot (sin otros snapshots salvo lo ya causal)."""
    o = pd.DataFrame(index=d.index)
    o["ctx_age_m_since"] = d.x_task_age - d.a_since_flight        # > 0: actuó sobre el avión después de aparecer la tarea
    o["ctx_age_m_any"] = d.x_task_age - d.a_since_any             # > 0: la tarea ya estaba en su última instrucción
    o["ctx_cz_m_since"] = d.cz_task - d.a_since_flight            # > 0: instrucción al avión después del cambio de estado
    o["ctx_cz_m_any"] = d.cz_task - d.a_since_any
    o["ctx_start_m_since"] = d.p_task_start - d.a_since_flight
    g = d.groupby(["sample_id", "cs"])
    o["ctx_cs_n"] = g.task_index.transform("size")
    o["ctx_cs_age_rk"] = g.x_task_age.rank(method="min")
    o["ctx_cs_cz_rel"] = d.cz_task - g.cz_task.transform("min")
    s = d.groupby("sample_id")
    o["ctx_age_m_since_rk"] = o.ctx_age_m_since.groupby(d.sample_id).rank(method="min")
    o["ctx_cz_m_since_rk"] = o.ctx_cz_m_since.groupby(d.sample_id).rank(method="min")
    o["ctx_tts_rk"] = s.time_to_sector.rank(method="min")
    o["ctx_tts_type_rk"] = d.groupby(["sample_id", "task_index"]).time_to_sector.rank(method="min")
    return o.astype("float32")


def fe_trig(d):
    """Qué disparó la racha actual: tiempos relativos al primer snapshot de la racha (pasado, así que causal).
    > 0 = ocurrió antes de que empezara la racha (segundos); < 0 = ocurrió durante la racha."""
    o = pd.DataFrame(index=d.index)
    tb = d.t - d.groupby("block").t.transform("min")              # segundos desde el inicio de la racha
    o["tg_task"] = d.p_task_start - tb                              # la tarea ya estaba al empezar la racha
    o["tg_age"] = d.x_task_age - tb
    o["tg_cz"] = d.cz_task - tb                                     # cambio de estado justo antes de la racha
    o["tg_act"] = d.a_since_flight - tb                             # < 0: instrucción al avión durante la racha
    o["tg_any"] = d.a_since_any - tb
    o["tg_gap"] = d.gap_prev.where(d.blk_pos == 0).groupby(d.block).transform("max")   # pausa antes de la racha
    for c in ("tg_task", "tg_age", "tg_cz", "tg_act"):
        o[c + "_rk"] = o[c].where(o[c] >= -3).groupby(d.sample_id).rank(method="min")   # el más reciente antes
    return o.astype("float32")


def setup(hilos):
    N.A = types.SimpleNamespace(prueba=False)
    N.OUT, N.THREADS = OUT, hilos
    for s in ("", "oof", "pred"):
        (OUT / s).mkdir(exist_ok=True, parents=True)
    lab, te, fs = N.datos()
    for nombre, fn in (("ctx", fe_ctx), ("trig", fe_trig)):
        a, b = fn(lab), fn(te)
        for c in a.columns:
            lab[c], te[c] = a[c].values, b[c].values
        fs[nombre] = list(a.columns)
    return lab, te, fs


def peso_atco(A_):
    """1/√(snapshots del controlador), media 1: los controladores pequeños pesan algo más al entrenar."""
    n = A_.drop_duplicates("sample_id").groupby("atco").size()
    w = 1.0 / np.sqrt(A_.atco.map(n).values)
    return w / w.mean()


def fit(A_, F, obj, seed, hilos, peso=None):
    p = N.lgb_params(B045)
    p.update(random_state=seed, verbose=-1, n_jobs=hilos)
    w = peso_atco(A_) if peso == "atco" else None
    if obj == "binary":
        p.pop("lambdarank_truncation_level")
        p["objective"] = "binary"
        return lgb.LGBMClassifier(**p).fit(A_[F], A_.target_index, sample_weight=w, categorical_feature=["task_index"])
    p["objective"] = obj
    if obj != "lambdarank":
        p.pop("lambdarank_truncation_level")
    return lgb.LGBMRanker(**p).fit(A_[F], A_.target_index, group=A_.groupby("sample_id", sort=False).size().values,
                                   sample_weight=w, categorical_feature=["task_index"])


def score(m, X):
    return m.predict(X, raw_score=True) if isinstance(m, lgb.LGBMClassifier) else m.predict(X)


def run(var, seeds, hilos):
    lab, te, fs = setup(hilos)
    obj, extra, *peso = VARS[var]
    peso = peso[0] if peso else None
    F = [c for b in N.FEATS + list(extra) for c in fs[b]]
    print(f"{var}: {obj}, {len(F)} features", flush=True)
    for s in range(seeds):
        po, pp = OUT / "oof" / f"{var}__s{s}.npy", OUT / "pred" / f"{var}__s{s}.npy"
        t0 = time.time()
        if not po.exists():
            oof = np.zeros(len(lab))
            for f in sorted(lab.fold.unique()):
                tr, va = (lab.fold != f).values, (lab.fold == f).values
                oof[va] = score(fit(lab[tr], F, obj, s, hilos, peso), lab.loc[va, F])
            np.save(po, oof)
        if not pp.exists():
            np.save(pp, score(fit(lab, F, obj, s, hilos, peso), te[F]))
        o = np.mean([np.load(OUT / "oof" / f"{var}__s{k}.npy") for k in range(s + 1)], axis=0)
        cv = hits(lab[["sample_id", "row_index"]].assign(score=o)).mean()
        print(f"  semilla {s}: {time.time() - t0:.0f}s · CV con {s + 1} semillas {cv:.4f}", flush=True)


# ----------------------------------------------------------------------------------------------------------- evaluación
def zs(df):
    g = df.groupby("sample_id").score
    return (df.score - g.transform("mean")) / (g.transform("std") + 1e-9)


def cargar(var, keys, kind):
    fs = sorted((OUT / kind).glob(f"{var}__s*.npy"))
    return (keys.assign(score=np.mean([np.load(p) for p in fs], axis=0)), len(fs)) if fs else (None, 0)


def evaluar():
    import pickle
    d = pd.read_pickle(REAL / "noche" / "cache_features.pkl")[["sample_id", "row_index", "split"]]
    kl = d[d.split != "test"][["sample_id", "row_index"]].reset_index(drop=True)
    kt = d[d.split == "test"][["sample_id", "row_index"]].reset_index(drop=True)
    best = kl.merge(pd.read_csv(REAL / "oof" / "real_adri_noche.csv"), on=["sample_id", "row_index"], how="left")
    best_te = kt.merge(pd.read_csv(REAL / "preds" / "real_adri_noche.csv"), on=["sample_id", "row_index"], how="left")
    info = metricas.info()
    cv023 = lambda o: (lambda h: h[info.atco.reindex(h.index).isin(N.CV023).values].mean())(hits(o))  # noqa: E731
    filas = []

    def add(nombre, o, ref, nsem=""):
        m = metricas.metricas(o, ref)
        filas.append(dict(modelo=nombre, semillas=nsem, cv=m["cv"], cv023=cv023(o), peor=m["min_atco"],
                          d_cv=m.get("d_cv"), ses=f"{m.get('ses_up', '')}/{m.get('ses_down', '')}",
                          atco=f"{m.get('atco_up', '')}/{m.get('atco_down', '')}"))

    add("real_adri_noche", best, None)
    oofs = {v: cargar(v, kl, "oof") for v in VARS}
    ref = oofs["b045"][0]
    for v, (o, n) in oofs.items():
        if o is None:
            continue
        add(v, o, ref if v != "b045" else best, n)
        if v != "b045":
            for w in (0.25, 0.5, 1.0):
                z = kl.assign(score=zs(best).values + w * zs(o).values)
                add(f"noche + {w} × {v}", z, best, n)
    r = pd.DataFrame(filas).round(4)
    print(r.to_string(index=False))
    r.to_csv(OUT / "resultados.csv", index=False)                  # lo lee LGB_realista.ipynb
    pickle.dump(dict(kl=kl, kt=kt, best_te=best_te), open(OUT / "keys.pkl", "wb"))
    return r


def guardar(nombre, partes):
    """partes: lista de (variante o 'noche', peso). Mezcla de z-scores -> realista/oof|preds|submissions/<nombre>."""
    import pickle
    k = pickle.load(open(OUT / "keys.pkl", "rb"))
    kl, kt = k["kl"], k["kt"]
    best = kl.merge(pd.read_csv(REAL / "oof" / "real_adri_noche.csv"), on=["sample_id", "row_index"], how="left")
    so, sp = np.zeros(len(kl)), np.zeros(len(kt))
    for v, w in partes:
        o = best if v == "noche" else cargar(v, kl, "oof")[0]
        p = k["best_te"] if v == "noche" else cargar(v, kt, "pred")[0]
        so += w * zs(o).values
        sp += w * zs(p).values
    o, p = kl.assign(score=so), kt.assign(score=sp)
    print(metricas.resumen(nombre, metricas.metricas(o, best), "real_adri_noche"))
    save_outputs(nombre, o, p, out=REAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "run", "eval", "guardar"])
    ap.add_argument("--var", default="b045")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hilos", type=int, default=5)
    ap.add_argument("--nombre", default="real_claude")
    ap.add_argument("--partes", default="noche:1", help="variante:peso,... (noche = real_adri_noche)")
    a = ap.parse_args()
    if a.cmd == "cache":
        setup(a.hilos)
    elif a.cmd == "run":
        run(a.var, a.seeds, a.hilos)
    elif a.cmd == "eval":
        evaluar()
    else:
        guardar(a.nombre, [(x.split(":")[0], float(x.split(":")[1])) for x in a.partes.split(",")])

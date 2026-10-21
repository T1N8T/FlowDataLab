"""f11_lab.py · Pablo (A) · hilo f11 · banco de pruebas del modelo N con todo el registro de la sección 14.

N = `f10_instr_l63_s10` (Juanma, sección 21): bloques base, logic, past, future, clear, instr; 63 hojas,
min_child 100, extra_trees; + segundo paso de sesión (feat_session.py). Aquí cada prueba cambia algo respecto a N:
bloques de features (incluidos los nuevos de f11_feats.py), parámetros de LightGBM, pesos, oráculo de coordinaciones.

Guarda futuro/pruebas/oof|preds/f11_<nombre>[_ses].csv, apunta en futuro/pruebas/resultados.jsonl (hilo f11) e imprime
las métricas de futuro/metricas.py frente a la referencia (por defecto f11_ref_s<semillas>_ses, misma semilla).

Uso (desde la raíz):
    python futuro/pruebas/f11_lab.py ref --seeds 3 --threads 8
    python futuro/pruebas/f11_lab.py l127 --params '{"num_leaves": 127}' --seeds 3
    python futuro/pruebas/f11_lab.py link --add link --seeds 3
    python futuro/pruebas/f11_lab.py orc --orc --seeds 3 --ref f11_orc_s3      # con oráculo de coordinaciones
    python futuro/pruebas/f11_lab.py final --seeds 10 --test                  # + preds de test y envío
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import OUT, PARAMS_FUT, guardar, registrar  # noqa: E402
from f10_instr import datos_instr, CV023  # noqa: E402
import feat_session, metricas  # noqa: E402
from metric import hits, save_outputs  # noqa: E402
import lightgbm as lgb  # noqa: E402

N_PARAMS = {**PARAMS_FUT, "num_leaves": 63, "min_child_samples": 100, "extra_trees": True}
N_FEATS = ["base", "logic", "past", "future", "clear", "instr"]


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0, help="primera semilla (para repartir semillas entre procesos)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--feats", default=",".join(N_FEATS))
    ap.add_argument("--add", default="", help="bloques nuevos de f11_feats.py, separados por comas")
    ap.add_argument("--drop", default="", help="columnas a quitar")
    ap.add_argument("--params", default="{}", help="JSON con cambios a los parámetros de N")
    ap.add_argument("--orc", action="store_true", help="oráculo de coordinaciones (feat_oracle, variante coord)")
    ap.add_argument("--orc-var", default="coord")
    ap.add_argument("--obj", default="lambdarank", help="lambdarank (N) o softmax (entropía cruzada listwise, f11)")
    ap.add_argument("--jitter", action="store_true", help="parámetros sacudidos por semilla (hojas, min_child, colsample, subsample)")
    ap.add_argument("--grado-orc", type=float, default=0.0,
                    help="relevancia 1 a las tareas que otros controladores trabajaron a ±60 s (or_near60 del oráculo "
                         "completo >= este umbral); la respuesta propia vale 2")
    ap.add_argument("--mono", default="", help="restricciones monótonas, p. ej. 'si_match:-1,si_match_rk:-1'")
    ap.add_argument("--ls", type=float, default=0.0, help="suavizado de etiquetas del objetivo softmax")
    ap.add_argument("--weight", default="", help="peso por snapshot: atco:p (1/n_atco^p), ses:p")
    ap.add_argument("--test", action="store_true", help="entrena con todo y guarda preds de test + envío")
    ap.add_argument("--ref", default="", help="referencia para las métricas (sin _ses); por defecto f11_ref_s<seeds>")
    ap.add_argument("--tag", default="")
    ap.add_argument("--medir", action="store_true", help="no entrena: solo mide los OOF ya guardados de <name>")
    return ap.parse_args(argv)


def orc_cache(lab, te, var):
    """Oráculo por fold (y para el modelo final), cacheado en parquet: es caro y no depende de la prueba.
    var "gate" (f11): el oráculo completo ("base") solo en las filas SIN huella propia (sin instrucción compatible
    <= 60 s, sin desaparecer <= 30 s y sin cambio de su estado <= 60 s); en las demás, vacío."""
    if var in ("exit", "entry"):       # oráculo de coordinaciones solo en exit_coordination (3) o entry_coordination (2)
        X, cols = orc_cache(lab, te, "coord")
        ti = np.concatenate([lab.task_index.values, te.task_index.values])
        X = X.copy()
        X.loc[ti != (3 if var == "exit" else 2), :] = np.nan
        return X, cols
    if var in ("Bcoord", "Ecoord"):    # variantes B/E del oráculo (más ventanas, núcleo, tiempo relativo) solo en coordinaciones
        X, cols = orc_cache(lab, te, var[0])
        ti = np.concatenate([lab.task_index.values, te.task_index.values])
        X = X.copy()
        X.loc[~np.isin(ti, [2, 3]), :] = np.nan
        return X, cols
    if var == "gate":
        X, cols = orc_cache(lab, te, "base")
        D = pd.concat([lab[["si_match", "f_task_end", "sx_task"]], te[["si_match", "f_task_end", "sx_task"]]],
                      ignore_index=True)
        traza = ((D.si_match <= 60) | (D.f_task_end <= 30) | (D.sx_task <= 60)).values
        X = X.copy()
        X.loc[traza, :] = np.nan
        return X, cols
    p = OUT / f"cache_orc_{var}.parquet"
    if p.exists():
        X = pd.read_parquet(p)
    else:
        import m_lgb_future as M
        O = M.oracle_data(dict(lab=lab, te=te), var)
        cols = list(O["lab"].columns)
        parts = {f"{c}__f{f}": O[f][c].values for f in range(5) for c in cols}
        full = pd.concat([O["lab"], O["te"]], ignore_index=True)
        parts.update({f"{c}__full": full[c].values for c in cols})
        n = len(lab) + len(te)
        X = pd.DataFrame({k: (np.concatenate([v, np.full(n - len(v), np.nan)]) if len(v) < n else v)
                          for k, v in parts.items()})
        X.to_parquet(p)
    cols = sorted({c.split("__")[0] for c in X.columns})
    return X, cols


def load(a):
    lab, te, fs = datos_instr()
    extra = [b for b in a.add.split(",") if b]
    if extra:
        import f11_feats
        D = pd.concat([lab, te], ignore_index=True)
        for b in extra:
            B = f11_feats.block(b, D)
            fs[b] = list(B.columns)
            lab = pd.concat([lab, B.iloc[:len(lab)].reset_index(drop=True)], axis=1)
            te = pd.concat([te, B.iloc[len(lab):].reset_index(drop=True)], axis=1)
    return lab, te, fs


def weights(a, A):
    if not a.weight:
        return None
    kind, p = a.weight.split(":")
    key = {"atco": "atco", "ses": "session"}[kind]
    n = A.groupby(key).sample_id.transform("nunique")
    w = 1.0 / n.values ** float(p)
    return w / w.mean()


LS = 0.0      # suavizado de etiquetas del objetivo softmax (--ls)


def softmax_obj(y, s, w, group):
    """Objetivo listwise: entropía cruzada de la softmax dentro de cada snapshot (una sola respuesta por snapshot).
    Gradiente p - y, hessiana p (1 - p). Da log-probabilidades calibradas, lo que usa el segundo paso de sesión.
    Con LS > 0, la etiqueta es (1 - LS) · y + LS / n_tareas (suavizado, como en la red de Pablo)."""
    st = np.concatenate([[0], np.cumsum(group)[:-1]])
    q = np.repeat(np.arange(len(group)), group)
    if LS:
        y = (1 - LS) * y + LS / np.asarray(group)[q]
    e = np.exp(s - np.maximum.reduceat(s, st)[q])
    p = e / np.add.reduceat(e, st)[q]
    g, h = p - y, np.maximum(p * (1 - p), 1e-6)
    return (g * w, h * w) if w is not None else (g, h)


def fit(a, A, F, P, seed, y=None):
    if a.jitter:      # bagging con parámetros sacudidos alrededor de N (cada semilla, otros valores)
        r = np.random.default_rng(1000 + seed)
        P = {**P, "num_leaves": int(r.choice([47, 63, 79])), "min_child_samples": int(r.choice([80, 100, 120])),
             "colsample_bytree": float(r.choice([0.6, 0.7, 0.8])), "subsample": float(r.choice([0.7, 0.8, 0.9]))}
    if a.obj == "softmax":
        P = {k: v for k, v in P.items() if k not in ("objective", "lambdarank_truncation_level")}
        P["objective"] = softmax_obj
    m = lgb.LGBMRanker(**{**P, "random_state": seed})
    m.fit(A[F], A.target_index if y is None else y, group=A.groupby("sample_id", sort=False).size().values,
          sample_weight=weights(a, A), categorical_feature=["task_index"])
    return m


def medir(a, nm, lab, n_feats=None, blocks=(), seeds=(), secs=None):
    """Métricas de la sección 14 (1.er pase y + sesión) frente a la referencia, leídas de los OOF guardados; las imprime
    y las apunta en resultados.jsonl."""
    ref = a.ref or f"f11_ref_s{a.seeds}"
    res = {}
    for suf in ("", "_ses"):
        df = pd.read_csv(OUT / "oof" / f"{nm}{suf}.csv")
        rp = OUT / "oof" / f"{ref}{suf}.csv"
        m = metricas.metricas(df, pd.read_csv(rp) if rp.exists() and nm != ref else None)
        h = hits(df)
        at = lab.drop_duplicates("sample_id").set_index("sample_id").atco.reindex(h.index)
        m["cv_023"] = h[at.isin(CV023)].mean()
        print(metricas.resumen(nm + suf, m, ref + suf) + f"\n  CV 023 {m['cv_023']:.4f}", flush=True)
        extra = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in m.items()
                 if k not in ("cv", "min_atco")}
        registrar(nm + suf, hilo="f11", descripcion=f"{a.tag} blocks={list(blocks)} params={a.params} orc={a.orc} "
                  f"weight={a.weight} drop={a.drop} obj={a.obj} ls={a.ls} mono={a.mono} grado_orc={a.grado_orc} jitter={a.jitter} seeds={list(seeds)} ref={ref}", n_features=n_feats, segundos=secs,
                  extra=extra)
        res[suf] = m
    return res


def run(a):
    global LS
    LS = a.ls
    t0 = time.time()
    lab, te, fs = load(a)
    blocks = [b for b in a.feats.split(",") if b] + [b for b in a.add.split(",") if b]
    drop = set(a.drop.split(","))
    F = [c for b in blocks for c in fs[b] if c not in drop]
    P = {**N_PARAMS, "n_jobs": a.threads, **json.loads(a.params)}
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    if a.orc:
        X, ocols = orc_cache(lab, te, a.orc_var)
        F = F + ocols
    if a.mono:
        mc = dict((k, int(v)) for k, v in (x.split(":") for x in a.mono.split(",")))
        P["monotone_constraints"] = [mc.get(c, 0) for c in F]
    if a.grado_orc:
        XB, _ = orc_cache(lab, te, "base")

        def ygr(version, mask):
            near = XB[f"or_near60__{version}"].values[:len(lab)][mask]
            t = lab.target_index.values[mask]
            return 2 * t + ((t == 0) & (np.nan_to_num(near) >= a.grado_orc)).astype(int)
    oof = np.zeros(len(lab))
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        L = lab
        if a.orc:
            L = pd.concat([lab, pd.DataFrame({c: X[f"{c}__f{f}"].values[:len(lab)] for c in ocols})], axis=1)
        yt = ygr(f"f{f}", tr) if a.grado_orc else None
        oof[va] = np.mean([fit(a, L[tr], F, P, s, yt).predict(L.loc[va, F]) for s in seeds], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    pred = None
    if a.test:
        L, T = lab, te
        if a.orc:
            L = pd.concat([lab, pd.DataFrame({c: X[f"{c}__full"].values[:len(lab)] for c in ocols})], axis=1)
            T = pd.concat([te, pd.DataFrame({c: X[f"{c}__full"].values[len(lab):] for c in ocols})], axis=1)
        yf = ygr("full", np.ones(len(lab), bool)) if a.grado_orc else None
        pred = np.mean([fit(a, L, F, P, s, yf).predict(T[F]) for s in seeds], axis=0)
    nm = f"f11_{a.name}_s{a.seeds}" + (f"o{a.seed0}" if a.seed0 else "")
    so, sp, w = feat_session.sesion(lab, oof, te if a.test else None, pred)
    guardar(nm, lab, oof, te if a.test else None, pred)
    guardar(nm + "_ses", lab, so, te if a.test else None, sp)
    if a.test:
        save_outputs(nm + "_ses", None, te[["sample_id", "row_index"]].assign(score=sp), out=OUT)
    res = medir(a, nm, lab, len(F), blocks, seeds, time.time() - t0)
    print(f"RESUMEN {nm}: 1.er pase {res['']['cv']:.4f} · + sesión {res['_ses']['cv']:.4f} "
          f"(mediana ses {res['_ses']['ses_median']:.4f}, peor {res['_ses']['min_atco']:.3f}, CV023 {res['_ses']['cv_023']:.4f})"
          f"  pesos {w}  {len(F)} feats  {time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    a = parse()
    if a.medir:
        medir(a, f"f11_{a.name}_s{a.seeds}" + (f"o{a.seed0}" if a.seed0 else ""), datos_instr()[0])
    else:
        run(a)

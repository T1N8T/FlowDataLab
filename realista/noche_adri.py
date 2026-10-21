"""
noche_adri.py · Adri (B) · RAMA "REALISTA" · Búsqueda larga (para dejarla corriendo de noche) sobre el LightGBM realista de
Pablo: snapshot actual + snapshots PASADOS, mismas features (realista/features_real.py), folds (Data/folds.csv) y métricas
(futuro/metricas.py). No mira al futuro ni usa etiquetas de otros controladores.

Etapas (cada una guarda lo que hace en realista/noche/; si se corta, al relanzar continúa donde iba):
  1. referencia   real_todo con los parámetros de siempre, 3 semillas                                   (~5 min)
  2. búsqueda     parámetros: parte de las mejores pruebas del Optuna de Pablo (realista/tune/t1.log) y las muta;
                  1 semilla por prueba, hasta --horas-busqueda                                          (~1.5 h)
  3. confirmar    las 6 mejores de la búsqueda (de Pablo y nuestras) con 3 semillas, frente a la referencia (~50 min)
  4. mezclas      media (z-score en el snapshot) de las k mejores configuraciones confirmadas; sin pesos ajustados
  5. extras       sobre la mejor configuración, 2 semillas: + hist (Pablo), + instr/atención (Juanma/Adri), peso por tramo
  6. final        lo que gane, con 10 semillas (o 4–6 por configuración si es una mezcla) + segundo paso causal de Pablo
                  -> realista/oof|preds|submissions/real_adri_noche(_ses).csv                         (~40–60 min)
Criterio para adoptar algo (el del equipo): sube la CV **y** suben más sesiones de las que bajan frente a la referencia.
Informe legible en realista/noche/INFORME.md; registro detallado en realista/noche/resultados.jsonl y log.txt.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe realista\\noche_adri.py                       # todo (~4 h)
    .venv\\Scripts\\python.exe realista\\noche_adri.py --horas-busqueda 1    # búsqueda más corta
    .venv\\Scripts\\python.exe realista\\noche_adri.py --prueba               # comprobación rápida del código (modelos diminutos)
"""
import argparse, json, math, os, random, sys, time, types, traceback
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.modules.setdefault("fcntl", types.SimpleNamespace(flock=lambda *a, **k: None, LOCK_EX=2))   # Windows
ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "realista"
sys.path[:0] = [str(REAL), str(ROOT / "futuro"), str(ROOT / "cv")]
import features_real as fr  # noqa: E402
import feat_session_causal  # noqa: E402
import metricas  # noqa: E402
from metric import save_outputs, hits  # noqa: E402

FEATS = ["base", "logic", "past", "causal", "geo"]           # los bloques de real_todo
CV023 = [2, 11, 10, 6, 17, 15]                               # controladores parecidos al 023 (Juanma, futuro 20f)
MIN_GAIN = 0.002
DEFAULT = dict(n_estimators=400, lr=0.05, leaves=31, min_child=50, subsample=0.8, colsample=0.7, reg_lambda=0.0,
               min_gain=0.0, max_bin=255, cat_smooth=10.0, extra_trees=False, trunc=5)      # = real_todo de Pablo
SPACE = dict(n_estimators=(150, 1600, "int"), lr=(0.008, 0.12, "float"), leaves=(7, 127, "int"),
             min_child=(5, 400, "int"), subsample=(0.4, 1.0, "lin"), colsample=(0.2, 1.0, "lin"),
             reg_lambda=(1e-3, 40, "float"), min_gain=(1e-4, 1.0, "float"), max_bin=([63, 127, 255], None, "cat"),
             cat_smooth=(1, 50, "float"), extra_trees=([False, True], None, "cat"), trunc=(3, 40, "int"))
ACTS = ["a_last_assume_control", "a_last_force_assume_control", "a_last_assign_level_clearance",
        "a_last_assign_lateral_clearance", "a_last_initiate_transfer_of_control"]
COMPAT = {4: [0, 1], 2: [0, 1], 5: [4], 12: [3], 13: [3], 10: [2], 3: [2, 3], 6: [2, 3], 1: [2, 3], 0: [0, 1, 2, 3],
          7: [2, 3], 8: [2, 3]}

A = None              # argumentos
OUT = None            # carpeta de trabajo
THREADS = os.cpu_count() or 8         # se puede cambiar con --hilos


# ----------------------------------------------------------------------------------------------------------- utilidades
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def informe(md):
    with open(OUT / "INFORME.md", "a", encoding="utf-8") as f:
        f.write(md.rstrip() + "\n\n")


def guardar_res(fila):
    with open(OUT / "resultados.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(fila, default=float) + "\n")


def leer_res():
    p = OUT / "resultados.jsonl"
    return [json.loads(x) for x in open(p, encoding="utf-8")] if p.exists() else []


def lgb_params(c):
    p = dict(objective="lambdarank", n_estimators=int(c["n_estimators"]), learning_rate=c["lr"], num_leaves=int(c["leaves"]),
             min_child_samples=int(c["min_child"]), subsample=c["subsample"], subsample_freq=1,
             colsample_bytree=c["colsample"], reg_lambda=c["reg_lambda"], min_split_gain=c["min_gain"],
             max_bin=int(c["max_bin"]), cat_smooth=c["cat_smooth"], extra_trees=bool(c["extra_trees"]),
             lambdarank_truncation_level=int(c["trunc"]))
    if A.prueba:
        p["n_estimators"] = 12
    return p


def clave(c):
    return json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in sorted(c.items())})


# ----------------------------------------------------------------------------------------------------------- datos
def fe_instr(d):
    """Instrucciones ya vistas (snapshot actual o anteriores): s desde la última de cada tipo y desde la última
    compatible con la tarea (mitad hacia atrás del bloque instr de Juanma)."""
    ev = np.where(d.a_since_flight < 3600, np.round(d.t - d.a_since_flight), np.nan)
    k = d.assign(ev=ev).dropna(subset=["ev"]).sort_values("t").drop_duplicates(["session", "cs", "ev"])
    acts = k[ACTS].values
    k = k.assign(act=np.where(acts.max(1) == 1, acts.argmax(1), len(ACTS)))[["session", "cs", "ev", "act", "t"]]
    n, t_all = len(d), d.t.values
    prv = np.full((n, len(ACTS) + 1), np.nan)
    ev_g = {key: g.sort_values("t") for key, g in k.groupby(["session", "cs"])}
    for key, idx in d.groupby(["session", "cs"]).indices.items():
        if key not in ev_g:
            continue
        g, t = ev_g[key], t_all[idx]
        for j in range(len(ACTS) + 1):
            gj = g[g.act == j]
            if not len(gj):
                continue
            q = np.searchsorted(gj.t.values, t, side="right") - 1
            ok = q >= 0
            prv[idx[ok], j] = t[ok] - np.maximum.accumulate(gj.ev.values)[q[ok]]
    out = pd.DataFrame(prv, index=d.index, columns=[f"ip_prev_{j}" for j in range(len(ACTS) + 1)])
    pm = np.full(n, np.nan)
    ti = d.task_index.values
    for typ, comp in COMPAT.items():
        m = ti == typ
        if m.any():
            with np.errstate(all="ignore"):
                pm[m] = np.nanmin(prv[m][:, comp], axis=1)
    out["ip_prevmatch"] = pm
    out["ip_prevmatch_rk"] = out.ip_prevmatch.groupby(d.sample_id.values).rank(method="min")
    out["ip_prevmatch_rk_ac"] = out.ip_prevmatch.groupby([d.sample_id.values, d.cs.values]).rank(method="min")
    return out.astype("float32")


def fe_atencion(d):
    k = ["session", "cs"]
    x = d.drop_duplicates(k + ["t"])[k + ["t", "a_since_flight"]].reset_index(drop=True)
    x["ev"] = np.where(x.a_since_flight < 3600, np.round(x.t - x.a_since_flight), np.nan)
    ev = x.dropna(subset=["ev"]).groupby(k + ["ev"]).t.min().rename("t_obs").reset_index()
    m = x[k + ["t"]].reset_index().merge(ev, on=k)
    vista = m.t_obs <= m.t
    m["n60"], m["n180"], m["nses"] = vista & (m.ev > m.t - 60), vista & (m.ev > m.t - 180), vista
    c = m.groupby("index")[["n60", "n180", "nses"]].sum()
    x[["at_n60", "at_n180", "at_nses"]] = c.reindex(x.index).fillna(0).values
    out = d[k + ["t"]].merge(x[k + ["t", "at_n60", "at_n180", "at_nses"]], on=k + ["t"], how="left")
    out.index = d.index
    return out[["at_n60", "at_n180", "at_nses"]].astype("float32")


def longitud_tramo(lab):
    y = lab[lab.target_index == 1].sort_values(["session", "t"])
    seg = ((y.task_name != y.groupby("session").task_name.shift()) | (y.blk_pos == 0)).cumsum()
    return lab.sample_id.map(dict(zip(y.sample_id, seg.map(seg.value_counts())))).values


def datos():
    cache = REAL / "noche" / "cache_features.pkl"            # compartida entre --prueba y la ejecución real
    t0 = time.time()
    if cache.exists():
        d = pd.read_pickle(cache)
        log(f"features leídas de la caché ({time.time() - t0:.0f}s)")
    else:
        log("calculando las features de Pablo (features_real.build, ~3-5 min)...")
        d = fr.build().sort_values(["sample_id", "row_index"]).reset_index(drop=True)
        f64 = [c for c in d.columns if d[c].dtype == "float64" and c != "t"]
        d[f64] = d[f64].astype("float32")
        cache.parent.mkdir(exist_ok=True, parents=True)
        d.to_pickle(cache)
        log(f"features calculadas y guardadas en la caché ({time.time() - t0:.0f}s)")
    fs = fr.feature_sets(d)
    lab = d[d.split != "test"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    if A.prueba:                                               # 2 sesiones por fold: solo para comprobar el código
        ses = lab.groupby("fold").session.unique().apply(lambda s: list(s[:2])).sum()
        lab = lab[lab.session.isin(ses)].reset_index(drop=True)
    for nombre, fn in (("instr", fe_instr), ("atencion", fe_atencion)):
        a, b = fn(lab), fn(te)
        for c in a.columns:
            lab[c], te[c] = a[c].values, b[c].values
        fs[nombre] = list(a.columns)
    lab["n_tramo"] = longitud_tramo(lab)
    log(f"lab {lab.sample_id.nunique()} snapshots, test {te.sample_id.nunique()}; bloques " +
        ", ".join(f"{k}={len(v)}" for k, v in fs.items()))
    return lab, te, fs


# ----------------------------------------------------------------------------------------------------------- modelo
class Modelo:
    def __init__(self, lab, te, fs):
        self.lab, self.te, self.fs = lab, te, fs
        self.atco = lab.drop_duplicates("sample_id").set_index("sample_id").atco
        (OUT / "oof").mkdir(exist_ok=True)
        (OUT / "pred").mkdir(exist_ok=True)

    def F(self, extra=()):
        return [c for b in FEATS + list(extra) for c in self.fs[b]]

    def _fit(self, A_, F, c, seed, w):
        m = lgb.LGBMRanker(**lgb_params(c), random_state=seed, verbose=-1, n_jobs=THREADS)
        return m.fit(A_[F], A_.target_index, group=A_.groupby("sample_id", sort=False).size().values,
                     sample_weight=w, categorical_feature=["task_index"])

    def oof_seed(self, nombre, c, seed, extra=(), peso=False):
        """OOF de una semilla (se guarda; si ya existe, se lee)."""
        p = OUT / "oof" / f"{nombre}__s{seed}.npy"
        if p.exists():
            return np.load(p)
        lab, F = self.lab, self.F(extra)
        w = 1.0 / np.sqrt(lab.n_tramo.values) if peso else None
        oof = np.zeros(len(lab), dtype="float64")
        for f in sorted(lab.fold.unique()):
            tr, va = (lab.fold != f).values, (lab.fold == f).values
            oof[va] = self._fit(lab[tr], F, c, seed, None if w is None else w[tr]).predict(lab.loc[va, F])
        np.save(p, oof)
        return oof

    def pred_seed(self, nombre, c, seed, extra=(), peso=False):
        p = OUT / "pred" / f"{nombre}__s{seed}.npy"
        if p.exists():
            return np.load(p)
        lab, F = self.lab, self.F(extra)
        w = 1.0 / np.sqrt(lab.n_tramo.values) if peso else None
        pr = self._fit(lab, F, c, seed, w).predict(self.te[F])
        np.save(p, pr)
        return pr

    def df(self, score):
        return self.lab[["sample_id", "row_index"]].assign(score=np.asarray(score))

    def medir(self, nombre, score, ref=None, extra=None):
        m = metricas.metricas(self.df(score), None if ref is None else self.df(ref))
        h = hits(self.df(score))
        m["cv023"] = h[self.atco.reindex(h.index).isin(CV023).values].mean()
        m["nombre"] = nombre
        if extra:
            m.update(extra)
        guardar_res(m)
        return m

    def zscore(self, score):
        s = pd.Series(np.asarray(score, dtype="float64"))
        g = s.groupby(self.lab.sample_id.values)
        return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values

    def zscore_te(self, score):
        s = pd.Series(np.asarray(score, dtype="float64"))
        g = s.groupby(self.te.sample_id.values)
        return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).values


def fila(m, ref_ok=True):
    r = f"| {m['nombre']} | {m['cv']:.4f} | {m['cv023']:.4f} | {m['min_atco']:.3f} | {m['ses_median']:.4f} |"
    if ref_ok and "d_cv" in m:
        r += f" {m['d_cv']:+.4f} | {m['ses_up']}/{m['ses_down']} | {m['atco_up']}/{m['atco_down']} |"
    else:
        r += " — | — | — |"
    return r


CAB = ("| modelo | CV | CV 023 | peor controlador | mediana por sesión | Δ CV vs ref | sesiones suben/bajan | "
       "controladores suben/bajan |\n|---|---|---|---|---|---|---|---|")


def pasa(m):
    return m.get("d_cv", -1) >= MIN_GAIN and m.get("ses_up", 0) > m.get("ses_down", 0)


# ----------------------------------------------------------------------------------------------------------- etapas
def pablo_optuna():
    """Pruebas completas del Optuna de Pablo (realista/tune/t1.log), solo lambdarank. Valor = CV con 1 semilla."""
    p = REAL / "tune" / "t1.log"
    if not p.exists():
        return []
    import collections
    pr, val = collections.defaultdict(dict), {}
    for line in open(p, encoding="utf-8"):
        o = json.loads(line)
        if o.get("op_code") == 5:
            dd = json.loads(o["distribution"])
            v = o["param_value_internal"]
            if dd["name"] == "CategoricalDistribution":
                v = dd["attributes"]["choices"][int(v)]
            elif dd["name"] == "IntDistribution":
                v = int(round(v))
            pr[o["trial_id"]][o["param_name"]] = v
        if o.get("op_code") == 6 and o.get("state") == 1:
            val[o["trial_id"]] = o["values"][0]
    out = []
    for t, v in val.items():
        c = pr[t]
        if c.get("objective") != "lambdarank":
            continue
        c = {k: c[k] for k in DEFAULT}
        out.append(dict(nombre=f"pablo_t{t}", cfg=c, cv1=v, origen="Optuna de Pablo"))
    return out


def mutar(c, rng):
    c = dict(c)
    for k in rng.sample(sorted(SPACE), k=rng.randint(2, 4)):
        lo, hi, kind = SPACE[k]
        if kind == "cat":
            c[k] = rng.choice(lo)
        elif kind == "lin":
            c[k] = float(min(hi, max(lo, c[k] + rng.gauss(0, 0.08))))
        else:
            v = c[k] * math.exp(rng.gauss(0, 0.35))
            v = min(hi, max(lo, v))
            c[k] = int(round(v)) if kind == "int" else float(v)
    return c


def etapa_busqueda(M, ref):
    log("== ETAPA 2 · búsqueda de parámetros (1 semilla) ==")
    hechos = [r for r in leer_res() if r.get("etapa") == "busqueda"]
    pool = pablo_optuna() + [dict(nombre=r["nombre"], cfg=r["cfg"], cv1=r["cv"], origen="búsqueda") for r in hechos]
    vistos = {clave(p["cfg"]) for p in pool} | {clave(DEFAULT)}
    rng = random.Random(len(hechos) + 7)
    fin = OUT / "fin_busqueda.txt"                # la hora límite se fija la primera vez (si se relanza, no se alarga)
    if not fin.exists():
        fin.write_text(str(time.time() + A.horas_busqueda * 3600))
    t_fin = float(fin.read_text())
    n, intentos = len(hechos), 0
    while time.time() < t_fin and n < A.max_pruebas and intentos < 10 * A.max_pruebas:
        intentos += 1
        top = sorted(pool, key=lambda p: -p["cv1"])[:5]
        if not top:
            top = [dict(cfg=DEFAULT, cv1=0)]
        padre = rng.choices(top, weights=[5, 4, 3, 2, 1][:len(top)])[0]
        c = mutar(padre["cfg"], rng)
        if clave(c) in vistos:
            continue
        vistos.add(clave(c))
        nombre = f"b{n:03d}"
        t0 = time.time()
        try:
            oof = M.oof_seed(nombre, c, 0)
        except Exception:
            log(f"{nombre} falló:\n{traceback.format_exc()}")
            continue
        m = M.medir(nombre, oof, ref=None, extra=dict(etapa="busqueda", cfg=c, padre=padre.get("nombre", "default"),
                                                      secs=int(time.time() - t0)))
        pool.append(dict(nombre=nombre, cfg=c, cv1=m["cv"], origen="búsqueda"))
        n += 1
        mejor = max(pool, key=lambda p: p["cv1"])
        log(f"{nombre}: CV {m['cv']:.4f} (CV 023 {m['cv023']:.4f}, peor {m['min_atco']:.3f}) en {time.time() - t0:.0f}s"
            f" | mejor hasta ahora {mejor['nombre']} {mejor['cv1']:.4f}")
    top = sorted(pool, key=lambda p: -p["cv1"])[:10]
    informe("## 2 · Búsqueda de parámetros (1 semilla)\n\n"
            f"{n} pruebas propias (mutaciones de las mejores) + {len(pablo_optuna())} del Optuna de Pablo. Las 10 mejores:\n\n"
            "| prueba | origen | CV (1 semilla) | parámetros |\n|---|---|---|---|\n" +
            "\n".join(f"| {p['nombre']} | {p['origen']} | {p['cv1']:.4f} | `{clave(p['cfg'])}` |" for p in top))
    return pool


def etapa_confirmar(M, pool, ref_oof, k=6):
    log("== ETAPA 3 · confirmar las mejores con 3 semillas ==")
    cand = sorted(pool, key=lambda p: -p["cv1"])[:k]
    conf = []
    for p in cand:
        t0 = time.time()
        oofs = [M.oof_seed(p["nombre"], p["cfg"], s) for s in range(3)]
        m = M.medir(p["nombre"] + "_3s", np.mean(oofs, axis=0), ref=ref_oof,
                    extra=dict(etapa="confirmar", cfg=p["cfg"], cv1=p["cv1"]))
        conf.append(dict(nombre=p["nombre"], cfg=p["cfg"], oof=np.mean(oofs, axis=0), m=m))
        log(f"{p['nombre']}: 1 semilla {p['cv1']:.4f} -> 3 semillas {m['cv']:.4f} (Δ vs ref {m['d_cv']:+.4f}, "
            f"sesiones {m['ses_up']}/{m['ses_down']}, CV 023 {m['cv023']:.4f}) en {time.time() - t0:.0f}s")
    conf.sort(key=lambda x: -x["m"]["cv"])
    informe("## 3 · Confirmación con 3 semillas (frente a `real_todo` con 3 semillas)\n\n" + CAB + "\n" +
            "\n".join(fila(x["m"]) for x in conf))
    return conf


def etapa_mezclas(M, conf, ref_oof, ref_m):
    log("== ETAPA 4 · mezclas de las mejores configuraciones ==")
    opciones = [dict(tipo="ref", nombres=[], m=ref_m, oof=ref_oof)]
    for x in conf:
        opciones.append(dict(tipo="sola", nombres=[x["nombre"]], m=x["m"], oof=x["oof"]))
    for kk in range(2, min(5, len(conf)) + 1):
        z = np.mean([M.zscore(x["oof"]) for x in conf[:kk]], axis=0)
        m = M.medir(f"mezcla_top{kk}", z, ref=ref_oof, extra=dict(etapa="mezcla", miembros=[x["nombre"] for x in conf[:kk]]))
        opciones.append(dict(tipo="mezcla", nombres=[x["nombre"] for x in conf[:kk]], m=m, oof=z))
        log(f"mezcla top-{kk}: CV {m['cv']:.4f} (Δ {m['d_cv']:+.4f}, sesiones {m['ses_up']}/{m['ses_down']})")
    validas = [o for o in opciones if o["tipo"] == "ref" or pasa(o["m"])]
    elegida = max(validas, key=lambda o: o["m"]["cv"])
    informe("## 4 · Mezclas (media de z-scores en el snapshot, pesos iguales)\n\n" + CAB + "\n" +
            "\n".join(fila(o["m"]) for o in opciones if o["tipo"] == "mezcla") +
            f"\n\n**Elegida para la etapa 6:** `{elegida['m']['nombre']}` ({elegida['tipo']}, CV {elegida['m']['cv']:.4f}). "
            "Criterio: la de mayor CV entre las que suben la CV ≥ 0.002 y suben más sesiones de las que bajan "
            "frente a la referencia (si ninguna, la referencia).")
    log(f"elegida: {elegida['m']['nombre']}")
    return elegida


def etapa_extras(M, conf_cfgs, base_nombre, base_cfg):
    """Sobre la mejor configuración (2 semillas): + hist, + instr/atención, peso por tramo. Devuelve el extra ganador."""
    log("== ETAPA 5 · extras sobre la mejor configuración (2 semillas) ==")
    base = np.mean([M.oof_seed(base_nombre, base_cfg, s) for s in range(2)], axis=0)
    mb = M.medir(base_nombre + "_2s", base, extra=dict(etapa="extras"))
    variantes = [("hist", ["hist"], False), ("instr_atencion", ["instr", "atencion"], False), ("peso_tramo", [], True)]
    filas, gan = [fila(mb, False)], None
    for nom, extra, peso in variantes:
        if any(b not in M.fs for b in extra):
            continue
        n = f"{base_nombre}+{nom}"
        oof = np.mean([M.oof_seed(n, base_cfg, s, extra=extra, peso=peso) for s in range(2)], axis=0)
        m = M.medir(n + "_2s", oof, ref=base, extra=dict(etapa="extras"))
        filas.append(fila(m))
        log(f"{n}: CV {m['cv']:.4f} (Δ {m['d_cv']:+.4f}, sesiones {m['ses_up']}/{m['ses_down']})")
        if pasa(m) and (gan is None or m["cv"] > gan[2]):
            gan = (extra, peso, m["cv"], nom)
    informe("## 5 · Extras sobre la mejor configuración (2 semillas, frente a ella misma)\n\n" + CAB + "\n" +
            "\n".join(filas) + f"\n\n**Extra que se queda:** {gan[3] if gan else 'ninguno'}.")
    return (gan[0], gan[1]) if gan else ((), False)


def etapa_final(M, elegida, cfgs, extra, peso, ref_oof):
    log("== ETAPA 6 · versión final ==")
    nombres = elegida["nombres"] or ["ref"]
    semillas = A.semillas_final if len(nombres) == 1 else max(4, math.ceil(A.semillas_final * 1.2 / len(nombres)))
    if A.prueba:
        semillas = 2
    suf = ("+" + "+".join(extra) if extra else "") + ("+peso" if peso else "")
    oofs, preds = [], []
    for nb in nombres:
        c = cfgs[nb]
        t0 = time.time()
        o = np.mean([M.oof_seed(nb + suf, c, s, extra=extra, peso=peso) for s in range(semillas)], axis=0)
        p = np.mean([M.pred_seed(nb + suf, c, s, extra=extra, peso=peso) for s in range(semillas)], axis=0)
        oofs.append(o if len(nombres) == 1 else M.zscore(o))
        preds.append(p if len(nombres) == 1 else M.zscore_te(p))
        log(f"final {nb}{suf}: {semillas} semillas en {time.time() - t0:.0f}s")
    oof, pred = np.mean(oofs, axis=0), np.mean(preds, axis=0)
    nombre = "real_adri_noche" + ("_prueba" if A.prueba else "")
    m1 = M.medir(nombre, oof, ref=ref_oof, extra=dict(etapa="final", miembros=nombres, extra=list(extra), peso=peso,
                                                      semillas=semillas))
    so, sp, pesos = feat_session_causal.sesion(M.lab, oof, M.te, pred)
    m2 = M.medir(nombre + "_ses", so, ref=ref_oof, extra=dict(etapa="final"))
    destino = OUT if A.prueba else REAL
    save_outputs(nombre, M.df(oof), M.te[["sample_id", "row_index"]].assign(score=pred), out=destino)
    save_outputs(nombre + "_ses", M.df(so), M.te[["sample_id", "row_index"]].assign(score=sp), out=destino)
    sub = pd.read_csv(destino / "submissions" / f"sub_{nombre}_ses.csv")
    ok = len(sub) == M.te.sample_id.nunique() and sub.sample_id.is_unique and sub.prediction.str.split().str.len().eq(3).all()
    log(f"FINAL {nombre}: CV {m1['cv']:.4f} · + segundo paso {m2['cv']:.4f} · submission válida: {ok}")
    informe("## 6 · Versión final\n\n"
            f"Miembros: {', '.join(f'`{x}`' for x in nombres)} · extra: {list(extra) or '—'} · peso por tramo: {peso} · "
            f"{semillas} semillas por miembro · segundo paso causal de Pablo (pesos {pesos}).\n\n" + CAB + "\n" +
            fila(m1) + "\n" + fila(m2) + "\n\n"
            f"**Para Kaggle:** `realista/submissions/sub_{nombre}_ses.csv` (o `sub_{nombre}.csv` si el segundo paso no "
            f"mejora). Submission comprobada: {ok}.")
    return m1, m2


# ----------------------------------------------------------------------------------------------------------- main
def main():
    global A, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas-busqueda", type=float, default=1.5)
    ap.add_argument("--max-pruebas", type=int, default=80)
    ap.add_argument("--semillas-final", type=int, default=10)
    ap.add_argument("--hilos", type=int, default=0, help="hilos de LightGBM (0 = todos los del ordenador)")
    ap.add_argument("--prueba", action="store_true", help="comprobación del código: pocos datos, modelos diminutos")
    A = ap.parse_args()
    global THREADS
    THREADS = A.hilos or THREADS
    if A.prueba:
        A.horas_busqueda, A.max_pruebas = 0.05, 3
    OUT = REAL / ("noche_prueba" if A.prueba else "noche")
    OUT.mkdir(exist_ok=True, parents=True)
    t_ini = time.time()
    if (OUT / "INFORME.md").exists():
        informe("---\n*(relanzado el " + time.strftime("%d/%m %H:%M") + ": lo ya calculado se reutiliza)*")
    else:
        informe(f"# Búsqueda nocturna del modelo realista · {time.strftime('%d/%m %H:%M')}\n\n"
                "Base: `real_todo` de Pablo (snapshot + pasado; bloques base, logic, past, causal, geo). Folds de "
                "`Data/folds.csv`, métricas de `futuro/metricas.py`. *CV 023* = controladores parecidos al 023 "
                "(Juanma). Se relanza igual si se corta: las semillas ya calculadas se leen de `realista/noche/oof|pred`.")
    log(f"inicio · {THREADS} hilos · búsqueda {A.horas_busqueda} h")
    lab, te, fs = datos()
    M = Modelo(lab, te, fs)

    log("== ETAPA 1 · referencia: real_todo con 3 semillas ==")
    ref_oof = np.mean([M.oof_seed("ref", DEFAULT, s) for s in range(3)], axis=0)
    ref_m = M.medir("ref_3s", ref_oof, extra=dict(etapa="referencia"))
    log(f"referencia: CV {ref_m['cv']:.4f} · CV 023 {ref_m['cv023']:.4f} · peor {ref_m['min_atco']:.3f}")
    if not any(r.get("etapa") == "referencia" for r in leer_res()[:-1]):
        informe("## 1 · Referencia\n\n" + CAB + "\n" + fila(ref_m, False) +
                "\n\n(Pablo dio 0.6703 con estos mismos parámetros y 3 semillas.)")

    pool = etapa_busqueda(M, ref_oof)
    conf = etapa_confirmar(M, pool, ref_oof)
    cfgs = {x["nombre"]: x["cfg"] for x in conf}
    cfgs["ref"] = DEFAULT
    elegida = etapa_mezclas(M, conf, ref_oof, ref_m)
    mejor = conf[0] if conf and pasa(conf[0]["m"]) else dict(nombre="ref", cfg=DEFAULT)
    extra, peso = etapa_extras(M, cfgs, mejor["nombre"], mejor["cfg"])
    m1, m2 = etapa_final(M, elegida, cfgs, extra, peso, ref_oof)
    informe(f"## Resumen\n\nReferencia `real_todo` (3 semillas): **{ref_m['cv']:.4f}** → versión final: **{m1['cv']:.4f}**, "
            f"**{m2['cv']:.4f}** con el segundo paso. Tiempo total: {(time.time() - t_ini) / 3600:.1f} h.")
    log(f"TERMINADO en {(time.time() - t_ini) / 3600:.1f} h. Informe: {OUT / 'INFORME.md'}")


if __name__ == "__main__":
    main()

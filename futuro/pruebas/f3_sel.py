"""f3_sel.py · hilo f3 · selección de variables con futuro (bloques, importancias, validación adversaria).
Uso: python futuro/pruebas/f3_sel.py <etapa>   etapas: ref, abl, imp, adv, best
Resumen de cada CV en futuro/pruebas/notas/f3_cv.csv; log en notas/f3_log.txt (con nohup/tee).
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import datos, lgb_cv, lgb_fit, guardar, registrar, PARAMS_FUT  # noqa: E402

OUT = Path(__file__).resolve().parent
NOTAS = OUT / "notas"; NOTAS.mkdir(exist_ok=True)
CVF = NOTAS / "f3_cv.csv"
lab, te, fs = datos()
F = fs["base"] + fs["logic"] + fs["past"] + fs["future"]


def metr(oof):
    d = lab[["sample_id", "atco", "fold", "target_index"]].assign(s=oof)
    d["r"] = d.groupby("sample_id").s.rank(ascending=False, method="first")
    t = d[d.target_index == 1]
    h = (t.r <= 3).astype(float)
    return h.mean(), h.groupby(t.atco).mean().min(), h.groupby(t.fold).mean().std()


def run(nombre, cols, desc, seeds=(0,), test=False):
    t0 = time.time()
    oof, pred = lgb_cv(lab, cols, seeds=seeds, test=te if test else None)
    guardar(nombre, lab, oof, te if test else None, pred)
    registrar(nombre, hilo="f3_sel", descripcion=desc, n_features=len(cols), segundos=time.time() - t0)
    cv, mn, sd = metr(oof)
    row = pd.DataFrame([dict(nombre=nombre, cv=round(cv, 4), min_atco=round(mn, 4), std_folds=round(sd, 4),
                             n_features=len(cols), seeds=len(seeds), desc=desc)])
    row.to_csv(CVF, mode="a", header=not CVF.exists(), index=False)
    print(f"{nombre:28s} cv {cv:.4f} min {mn:.4f} std {sd:.4f} n {len(cols)} ({time.time()-t0:.0f}s)", flush=True)
    return oof


def etapa_ref():
    # f3_ref con bucle propio (= lgb_cv semilla 0) para guardar gain por fold y permutación sobre OOF
    oof = np.zeros(len(lab)); gains = []; perm = {c: [] for c in F}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        m = lgb_fit(lab[tr], F, {**PARAMS_FUT, "random_state": 0})
        gains.append(m.booster_.feature_importance("gain"))
        oof[va] = m.predict(lab.loc[va, F])
        # permutación barata: 20% de los snapshots del fold
        sids = lab.loc[va, "sample_id"].unique()
        sub = va & lab.sample_id.isin(rng.choice(sids, len(sids) // 5, replace=False)).values
        V = lab.loc[sub, F].copy(); vmeta = lab.loc[sub, ["sample_id", "target_index"]]
        p = oof[sub]
        def h3(s):
            r = vmeta.assign(s=s).groupby("sample_id").s.rank(ascending=False, method="first")
            return float((r[vmeta.target_index.values == 1] <= 3).mean())
        base_h = h3(p)
        for c in F:
            old = V[c].values.copy()
            V[c] = rng.permutation(old)
            perm[c].append(base_h - h3(m.predict(V)))
            V[c] = old
        print(f"fold {f} hecho ({time.time()-t0:.0f}s)", flush=True)
    guardar("f3_ref", lab, oof)
    registrar("f3_ref", hilo="f3_sel", descripcion="todos los bloques (125), 1 semilla", n_features=len(F),
              segundos=time.time() - t0)
    cv, mn, sd = metr(oof)
    pd.DataFrame([dict(nombre="f3_ref", cv=round(cv, 4), min_atco=round(mn, 4), std_folds=round(sd, 4),
                       n_features=len(F), seeds=1, desc="todos")]).to_csv(CVF, mode="a", header=not CVF.exists(),
                                                                          index=False)
    print(f"f3_ref cv {cv:.4f} min {mn:.4f} std {sd:.4f}", flush=True)
    g = np.mean(gains, axis=0)
    blk = {c: k for k, v in fs.items() for c in v}
    imp = pd.DataFrame(dict(feature=F, bloque=[blk[c] for c in F], gain=g / g.sum(),
                            perm_drop=[np.mean(perm[c]) for c in F], perm_std=[np.std(perm[c]) for c in F]))
    imp["rank_gain"] = imp.gain.rank(ascending=False); imp["rank_perm"] = imp.perm_drop.rank(ascending=False)
    imp["rank_mix"] = ((imp.rank_gain + imp.rank_perm) / 2).rank()
    imp.sort_values("rank_mix").to_csv(NOTAS / "f3_importancias.csv", index=False)


def etapa_abl():
    B, L, P, Fu = fs["base"], fs["logic"], fs["past"], fs["future"]
    run("f3_sin_logic", B + P + Fu, "sin bloque logic")
    run("f3_sin_past", B + L + Fu, "sin bloque past")
    run("f3_sin_future", B + L + P, "sin bloque future")
    run("f3_solo_fp", ["task_index"] + P + Fu, "solo future+past+task_index")
    for pre in ("a_", "x_", "k_", "l_"):
        run(f"f3_sin_{pre[0]}", [c for c in F if not c.startswith(pre)], f"sin base {pre}*")
    resto = [c for c in B if not c.startswith(("a_", "x_", "k_", "l_")) and c != "task_index"]
    run("f3_sin_resto", [c for c in F if c not in resto], "sin base resto (columnas originales salvo task_index)")


def etapa_imp():
    imp = pd.read_csv(NOTAS / "f3_importancias.csv").sort_values("rank_mix")
    for q in (0.3, 0.5):
        keep = imp.feature.iloc[: int(round(len(imp) * (1 - q)))].tolist()
        if "task_index" not in keep: keep.append("task_index")
        run(f"f3_imp{int(q*100)}", [c for c in F if c in keep], f"quita el {int(q*100)}% menos importante (gain+perm)")


def etapa_adv():
    d = pd.concat([lab.sample(frac=0.35, random_state=0).assign(y=0), te.assign(y=1)], ignore_index=True)
    X = d[F]; y = d.y.values; oof = np.zeros(len(d)); gains = []
    for tr, va in GroupKFold(4).split(X, y, d.sample_id):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                               colsample_bytree=0.7, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=3)
        m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        gains.append(m.booster_.feature_importance("gain"))
    auc = roc_auc_score(y, oof)
    uni = {c: abs(roc_auc_score(y, d[c].fillna(-1e9)) - 0.5) + 0.5 for c in F}
    g = np.mean(gains, axis=0)
    adv = pd.DataFrame(dict(feature=F, gain=g / g.sum(), auc_univariada=[uni[c] for c in F])).sort_values(
        "gain", ascending=False)
    adv.to_csv(NOTAS / "f3_adversaria.csv", index=False)
    print(f"AUC adversaria {auc:.4f}\n{adv.head(15).to_string()}", flush=True)
    json.dump(dict(auc=auc), open(NOTAS / "f3_adv_auc.json", "w"))
    top = [c for c in adv.feature if c != "task_index"]
    for k in (5, 10):
        run(f"f3_adv{k}", [c for c in F if c not in top[:k]], f"sin las {k} más delatoras: {top[:k]}")


def etapa_best(nombre_cfg):
    cfg = json.load(open(NOTAS / "f3_best_cols.json"))
    run("f3_best", cfg["cols"], f"mejor selección ({nombre_cfg}), 3 semillas", seeds=(0, 1, 2), test=True)


def etapa_sin_a():
    """sin a_* (mejor ablación con 1 semilla) con 3 semillas y preds de test."""
    run("f3_best_sin_a", [c for c in F if not c.startswith("a_")], "sin base a_*, 3 semillas", seeds=(0, 1, 2),
        test=True)


if __name__ == "__main__":
    e = sys.argv[1]
    {"ref": etapa_ref, "abl": etapa_abl, "sin_a": etapa_sin_a, "imp": etapa_imp, "adv": etapa_adv}.get(e, lambda: etapa_best(e))()


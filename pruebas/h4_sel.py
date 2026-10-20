"""h4_sel · selección de variables y deriva entre controladores (solo snapshot, regla 1).

Etapas:  python pruebas/h4_sel.py imp | abl | adv | best
"""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/pruebas")
from base import lab_test, build, PARAMS, lgb_cv, lgb_fit, guardar, registrar, OUT

NOTAS = OUT / "notas"; NOTAS.mkdir(exist_ok=True)
HILO = "h4_sel"
lab, test = lab_test()
d, F = build(lab)
dt, _ = build(test)
ORIG = [c for c in F if not any(c.startswith(p) for p in ("a_", "x_", "k_", "l_", "al_", "cx_"))]


def top3(d, score):
    r = pd.Series(score, index=d.index).groupby(d.sample_id).rank(ascending=False, method="first")
    return float(((r <= 3) & (d.target_index == 1)).groupby(d.sample_id).max().mean())


def run(nombre, Fs, desc, seeds=(0,), with_test=False):
    t = time.time()
    oof, pred = lgb_cv(d, Fs, seeds=seeds, test=dt if with_test else None)
    guardar(nombre, d, oof, dt, pred)
    return registrar(nombre, hilo=HILO, descripcion=desc, segundos=time.time() - t, n_features=len(Fs))


def etapa_imp():
    t = time.time()
    oof = np.zeros(len(d)); gain = pd.DataFrame(index=F); perm = {c: [] for c in F}
    rng = np.random.default_rng(0)
    for f in range(5):
        tr, va = (d.fold != f).values, (d.fold == f).values
        m = lgb_fit(d[tr], F, {**PARAMS, "random_state": 0})
        gain[f] = m.booster_.feature_importance("gain")
        V = d[va].reset_index(drop=True)
        # submuestra de snapshots para la permutación (barata)
        sids = V.sample_id.drop_duplicates().sample(frac=0.5, random_state=f)
        V = V[V.sample_id.isin(sids)].reset_index(drop=True)
        X = V[F].copy()
        base = top3(V, m.predict(X))
        oof[va] = m.predict(d.loc[va, F])
        for c in F:
            Xp = X.copy(); Xp[c] = rng.permutation(Xp[c].values)
            if c == "task_index":
                Xp[c] = Xp[c].astype(X[c].dtype)
            perm[c].append(base - top3(V, m.predict(Xp)))
        print(f"fold {f} hecho {time.time()-t:.0f}s", flush=True)
    guardar("h4_ref", d, oof)
    registrar("h4_ref", hilo=HILO, descripcion="lgb_adri 99 features, 1 semilla (referencia)",
              segundos=time.time() - t, n_features=len(F))
    imp = pd.DataFrame({"gain_medio": gain.mean(1), "gain_frac": gain.mean(1) / gain.mean(1).sum(),
                        "perm_drop": {c: np.mean(v) for c, v in perm.items()},
                        "perm_std": {c: np.std(v) for c, v in perm.items()}})
    imp["rank_gain"] = imp.gain_medio.rank(ascending=False)
    imp["rank_perm"] = imp.perm_drop.rank(ascending=False)
    imp["rank_medio"] = (imp.rank_gain + imp.rank_perm) / 2
    imp.sort_values("rank_medio").to_csv(NOTAS / "h4_importancias.csv", index_label="feature")
    print(imp.sort_values("rank_medio").head(25)); print(imp.sort_values("rank_medio").tail(25))


def etapa_abl():
    imp = pd.read_csv(NOTAS / "h4_importancias.csv", index_col=0).sort_values("rank_medio")
    grupos = {"a": [c for c in F if c.startswith("a_")], "x": [c for c in F if c.startswith("x_")],
              "k": [c for c in F if c.startswith("k_")], "l": [c for c in F if c.startswith("l_")],
              "orig": [c for c in ORIG if c != "task_index"],
              "al_cx": [c for c in F if c.startswith(("al_", "cx_"))]}
    for g, q in grupos.items():
        run(f"h4_sin_{g}", [c for c in F if c not in q], f"sin grupo {g} ({len(q)} cols)")
    for p in (20, 40):
        n = int(round(len(F) * p / 100))
        q = [c for c in imp.index[-n:] if c != "task_index"]
        run(f"h4_sin_low{p}", [c for c in F if c not in q], f"sin el {p}% menos importante (rank medio gain+perm)")
    q = imp.index[imp.perm_drop <= 0].tolist()
    q = [c for c in q if c != "task_index"]
    run("h4_sin_permneg", [c for c in F if c not in q], f"sin las {len(q)} con perm_drop<=0")
    json.dump(q, open(NOTAS / "h4_permneg.json", "w"))


def adv_auc(X, y, groups=None):
    """AUC OOF de un clasificador LightGBM (3 folds por grupo si se dan)."""
    p = np.zeros(len(y))
    clf = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, colsample_bytree=0.7, subsample=0.5,
               subsample_freq=1, n_jobs=2, verbose=-1)
    folds = (groups % 3) if groups is not None else np.arange(len(y)) % 3
    gains = []
    for k in range(3):
        tr, va = folds != k, folds == k
        m = lgb.LGBMClassifier(**clf).fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
        gains.append(m.booster_.feature_importance("gain"))
    g = pd.Series(np.mean(gains, 0), index=X.columns)
    return roc_auc_score(y, p), (g / g.sum()).sort_values(ascending=False)


def etapa_adv():
    Fa = [c for c in F if c != "task_index"] + ["task_index"]
    # test vs lab: snapshots enteros como grupo, submuestra de filas
    A = pd.concat([d[Fa + ["sample_id", "atco"]].assign(y=0), dt[Fa + ["sample_id", "atco"]].assign(y=1)],
                  ignore_index=True)
    A = A.sample(frac=0.35, random_state=0).reset_index(drop=True)
    res = {}
    auc, g = adv_auc(A[Fa], A.y.values, groups=A.sample_id.values)
    res["test_vs_lab"] = (auc, g)
    print(f"test vs lab AUC {auc:.3f}\n{g.head(12)}", flush=True)
    # test 023 vs lab, y 020 vs resto, y algunos controladores vs resto (solo lab)
    t23 = A[(A.y == 0) | (A.atco == 23)]
    auc, g = adv_auc(t23[Fa].reset_index(drop=True), t23.y.values, groups=t23.sample_id.values)
    res["023_vs_lab"] = (auc, g); print(f"023 vs lab AUC {auc:.3f}\n{g.head(12)}", flush=True)
    L = A[A.y == 0].reset_index(drop=True)
    for a in (20, 5, 15, 18, 1):
        auc, g = adv_auc(L[Fa], (L.atco == a).astype(int).values, groups=L.sample_id.values)
        res[f"atco{a}_vs_resto"] = (auc, g); print(f"atco {a} vs resto AUC {auc:.3f}\n{g.head(8)}", flush=True)
    tab = pd.DataFrame({k: v[1] for k, v in res.items()})
    tab["media"] = tab.mean(1)
    tab = tab.sort_values("media", ascending=False)
    tab.to_csv(NOTAS / "h4_adversaria.csv", index_label="feature")
    json.dump({k: round(v[0], 4) for k, v in res.items()}, open(NOTAS / "h4_adv_auc.json", "w"), indent=1)
    print(tab.head(15))


def etapa_adv_abl():
    tab = pd.read_csv(NOTAS / "h4_adversaria.csv", index_col=0)
    imp = pd.read_csv(NOTAS / "h4_importancias.csv", index_col=0)
    # ranking de deriva: test_vs_lab y 023_vs_lab pesan más
    s = (tab["test_vs_lab"] + tab["023_vs_lab"]) / 2 + 0.5 * tab.filter(like="atco").mean(1)
    s = s.drop("task_index").sort_values(ascending=False)
    print(pd.concat([s.rename("deriva"), imp[["rank_medio", "perm_drop"]]], axis=1).loc[s.index[:15]])
    for n in (5, 10):
        q = s.index[:n].tolist()
        run(f"h4_sin_deriva{n}", [c for c in F if c not in q], f"sin las {n} más delatoras (adversaria): {q}")
        # sustituir por rango dentro del snapshot
        for df in (d, dt):
            for c in q:
                df[f"rk_{c}"] = df.groupby("sample_id")[c].rank(method="average", pct=True)
        Fr = [c for c in F if c not in q] + [f"rk_{c}" for c in q]
        run(f"h4_rank_deriva{n}", Fr, f"las {n} más delatoras sustituidas por su rango pct en el snapshot")
    json.dump(s.index[:10].tolist(), open(NOTAS / "h4_deriva10.json", "w"))




# ---------------------------------------------------------------- versión reducida (máquina cargada, poco tiempo)
def _imp():
    return pd.read_csv(NOTAS / "h4_importancias.csv", index_col=0).sort_values("rank_medio")


def quitar_low(p):
    imp = _imp(); n = int(round(len(F) * p / 100))
    return [c for c in imp.index[-n:] if c != "task_index"]


def quitar_deriva(n):
    tab = pd.read_csv(NOTAS / "h4_adversaria.csv", index_col=0)
    s = (tab["test_vs_lab"] + tab["023_vs_lab"]) / 2 + 0.5 * tab.filter(like="atco").mean(1)
    return s.drop("task_index").sort_values(ascending=False).index[:n].tolist()


def etapa_abl2():
    for p in (20, 40):
        q = quitar_low(p)
        run(f"h4_sin_low{p}", [c for c in F if c not in q], f"sin el {p}% menos importante (rank medio gain+perm)")
    q = quitar_deriva(6)
    run("h4_sin_deriva", [c for c in F if c not in q], f"sin las 6 más delatoras (adversaria): {q}")


def etapa_best():
    q = json.loads(sys.argv[2])
    run("h4_sel_best", [c for c in F if c not in q], f"selección h4: quita {len(q)} cols", seeds=(0, 1),
        with_test=True)


ETAPAS = {"imp": etapa_imp, "abl": etapa_abl, "adv": etapa_adv, "advabl": etapa_adv_abl, "abl2": etapa_abl2, "best": etapa_best}
if __name__ == "__main__" and sys.argv[1] in ETAPAS:
    ETAPAS[sys.argv[1]]()


def etapa_uno():
    """python h4_sel.py uno low20|deriva  (n_jobs=1 para correr dos a la vez sin pasar de 2 hilos)"""
    import base; base.PARAMS["n_jobs"] = 1
    if sys.argv[2] == "low20":
        q = quitar_low(20); run("h4_sin_low20", [c for c in F if c not in q], f"sin el 20% menos importante: {q}")
    else:
        q = quitar_deriva(6); run("h4_sin_deriva", [c for c in F if c not in q],
                                  f"sin las 6 más delatoras (adversaria, constantes por snapshot): {q}")


if __name__ == "__main__" and sys.argv[1] == "uno":
    etapa_uno()


if __name__ == "__main__" and sys.argv[1] == "lista":
    # python h4_sel.py lista <nombre> '<json cols a quitar>' '<descripción>'
    import base; base.PARAMS["n_jobs"] = 1
    q = json.loads(sys.argv[3]); run(sys.argv[2], [c for c in F if c not in q], sys.argv[4] + f": {q}")

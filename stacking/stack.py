"""
stack.py · Juanma (C) · Combina los modelos de oof/ y preds/ (formato común sample_id,row_index,score).

(a) avg  : media ponderada de puntuaciones normalizadas dentro del snapshot (log-softmax o rank). Pesos por búsqueda
           coordinada maximizando top-3. La CV que se imprime es ANIDADA: pesos ajustados en 4 folds, medidos en el 5º.
(b) meta : LightGBM LambdaRank pequeño (nivel 2) sobre las salidas OOF de todos los modelos (rank, z-score y
           log-softmax dentro del snapshot) + task_index, l_twin_AE, seq_len. CV con los mismos folds (Data/folds.csv).
           Solo se entrena con OOF; para test se aplica a preds/.

Solo entran los modelos que tienen oof/<m>.csv Y preds/<m>.csv (salvo --oof-only, que vale para medir CV).
Se ignoran los propios stack_*.

Uso:  python stacking/stack.py                       # todos los modelos disponibles
      python stacking/stack.py lgb_rank nn_base     # solo esos
      python stacking/stack.py --oof-only           # incluye modelos sin preds (solo CV, no guarda)
Salidas: oof|preds/stack_avg.csv, stack_meta.csv y submissions/sub_stack_avg.csv, sub_stack_meta.csv
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cv"))
from metric import report, save_outputs, labels

DATA = ROOT / "Data"
CTX = ["task_index", "l_twin_AE", "seq_len"]


# ----------------------------------------------------------------------------- carga
def available(names, oof_only):
    oofs = {p.stem for p in (ROOT / "oof").glob("*.csv") if not p.stem.startswith("stack")}
    preds = {p.stem for p in (ROOT / "preds").glob("*.csv")} if (ROOT / "preds").exists() else set()
    ok = sorted(oofs if oof_only else oofs & preds)
    if names:
        miss = set(names) - set(ok)
        assert not miss, f"sin oof{'' if oof_only else '+preds'}: {miss}"
        ok = names
    skipped = sorted(oofs - set(ok))
    if skipped:
        print("Sin usar (falta preds o no pedido):", skipped)
    assert ok, "no hay modelos en oof/"
    return ok


def base_rows(split):
    files = ["train", "validation"] if split == "lab" else ["test"]
    d = pd.concat([pd.read_csv(DATA / f"{f}.csv", usecols=["sample_id", "row_index", "atco"] + CTX) for f in files])
    if split == "lab":
        folds = pd.read_csv(DATA / "folds.csv")
        d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold)))
        t = labels().set_index("sample_id").target
        d["y"] = (d.row_index == d.sample_id.map(t)).astype(int)
    return d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def attach(d, models, folder):
    for m in models:
        s = pd.read_csv(ROOT / folder / f"{m}.csv").rename(columns={"score": m})
        n0 = len(d)
        d = d.merge(s, on=["sample_id", "row_index"], how="left")
        assert len(d) == n0
        cover = d.groupby("sample_id")[m].apply(lambda x: x.notna().any()).mean()
        if cover < 1:
            print(f"  AVISO {folder}/{m}: cubre {cover:.1%} de los snapshots; el resto se rellena con el mínimo")
        # fila sin score -> peor que todas las del snapshot
        d[m] = d[m].fillna(d.groupby("sample_id")[m].transform("min") - 1).fillna(0)
    return d


# ----------------------------------------------------------------------------- normalizaciones dentro del snapshot
def norm(d, m):
    g = d.groupby("sample_id")[m]
    s = d[m]
    z = (s - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)
    sh = s - g.transform("max")
    # escala del log-softmax: se estandariza para que modelos con escalas distintas (logits vs. probabilidades) pesen igual
    zs = sh / g.transform("std").replace(0, 1).fillna(1)
    lsm = zs - np.log(np.exp(zs).groupby(d.sample_id).transform("sum"))
    rk = g.rank(ascending=False, method="first")
    return {"z": z.values, "lsm": lsm.values, "rk": rk.values, "rpct": (1 - (rk - 1) / g.transform("size")).values}


def hit3(d, score, mask=None):
    """Top-3 rápido (vectorizado): nº de filas del snapshot que ganan a la correcta < 3. Empates: gana la fila anterior.
    d debe estar ordenado por sample_id y tener exactamente un y=1 por snapshot."""
    if "_code" not in d:
        d["_code"] = pd.factorize(d.sample_id)[0]
        d["_tpos"] = np.flatnonzero(d.y.values)[d._code.values]
    code, tpos = d._code.values, d._tpos.values
    s = np.asarray(score, float)
    ts = s[tpos]
    better = (s > ts) | ((s == ts) & (np.arange(len(s)) < tpos))
    n_better = np.bincount(code, weights=better, minlength=code.max() + 1)
    hit = n_better < 3
    if mask is not None:
        keep = np.zeros(code.max() + 1, bool); keep[code[mask]] = True
        hit = hit[keep]
    return float(hit.mean())


# ----------------------------------------------------------------------------- (a) media ponderada
def fit_weights(d, X, mask, grid=(0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3), rounds=3):
    w = np.ones(X.shape[1])
    best = hit3(d, X @ w, mask)
    for _ in range(rounds):
        changed = False
        for j in range(X.shape[1]):
            for v in grid:
                w2 = w.copy(); w2[j] = v
                if w2.sum() == 0:
                    continue
                h = hit3(d, X @ w2, mask)
                if h > best + 1e-6:
                    best, w, changed = h, w2, True
        if not changed:
            break
    return w / w.sum(), best


def stack_avg(lab, te, models, kind):
    X = np.column_stack([lab[f"{kind}_{m}"] for m in models])
    Xt = np.column_stack([te[f"{kind}_{m}"] for m in models]) if te is not None else None
    oof = np.zeros(len(lab))
    for f in range(5):
        w, _ = fit_weights(lab, X, (lab.fold != f).values)
        va = (lab.fold == f).values
        oof[va] = X[va] @ w
    w, _ = fit_weights(lab, X, None)
    print(f"  pesos finales ({kind}): " + "  ".join(f"{m}={v:.2f}" for m, v in zip(models, w)))
    return oof, (Xt @ w if Xt is not None else None)


# ----------------------------------------------------------------------------- (b) meta-modelo
def meta_model(seed=0):
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.03, num_leaves=7,
                          min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=5.0, lambdarank_truncation_level=5, random_state=seed, verbose=-1, n_jobs=6)


def meta_fit(A, F, seed=0):
    return meta_model(seed).fit(A[F], A.y, group=A.groupby("sample_id", sort=False).size().values,
                                categorical_feature=["task_index"])


def stack_meta(lab, te, F, seeds=3):
    oof = np.zeros(len(lab))
    for f in range(5):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        oof[va] = np.mean([meta_fit(lab[tr], F, s).predict(lab.loc[va, F]) for s in range(seeds)], axis=0)
    pt = np.mean([meta_fit(lab, F, s).predict(te[F]) for s in range(seeds)], axis=0) if te is not None else None
    return oof, pt


# ----------------------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    oof_only = "--oof-only" in sys.argv
    models = available(args, oof_only)
    print("Modelos:", models)

    lab = attach(base_rows("lab"), models, "oof")
    te = None if oof_only else attach(base_rows("test"), models, "preds")
    for m in models:
        for df in [lab] + ([te] if te is not None else []):
            for k, v in norm(df, m).items():
                df[f"{k}_{m}"] = v

    print("\nCV de cada modelo (sobre las filas etiquetadas):")
    for m in models:
        print(f"  {m:28s} {hit3(lab, lab[m].values):.4f}")
    # correlación de aciertos entre modelos (cuanto más baja, más aporta combinarlos)
    if len(models) > 1:
        H = pd.DataFrame({m: lab.assign(r=lab[f"rk_{m}"]).query("y == 1").set_index("sample_id").r.le(3).astype(int)
                          for m in models})
        print("\nCorrelación de aciertos top-3:\n" + H.corr().round(2).to_string())

    out = lambda df, s: df[["sample_id", "row_index"]].assign(score=s)
    results = {}
    print()
    for kind in ("lsm", "rpct"):
        o, p = stack_avg(lab, te, models, kind)
        results[f"stack_avg_{kind}"] = (o, p)
        print(f"  stack_avg ({kind}) CV anidada top3 = {hit3(lab, o):.4f}")
    F = [f"{k}_{m}" for m in models for k in ("z", "lsm", "rk")] + CTX
    o, p = stack_meta(lab, te, F)
    results["stack_meta"] = (o, p)
    print()
    for name, (o, p) in results.items():
        report(out(lab, o), name, verbose=False)

    if oof_only:
        print("\n--oof-only: no se guardan salidas")
        return
    best_avg = max(("stack_avg_lsm", "stack_avg_rpct"), key=lambda k: hit3(lab, results[k][0]))
    for name, key in (("stack_avg", best_avg), ("stack_meta", "stack_meta")):
        o, p = results[key]
        save_outputs(name, out(lab, o), out(te, p))
    print(f"stack_avg = {best_avg}")


if __name__ == "__main__":
    main()

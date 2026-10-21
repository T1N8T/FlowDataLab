"""f7_tramo.py · modelo por tramo (hilo f7). La unidad es (racha, tarea), no (snapshot, tarea).
El 74 % de los fallos están en rachas donde el 1.er pase no acierta en ningún snapshot: el error es de racha entera.
1. Para cada (block, task_name): agregados del 1.er pase (OOF) en la racha (media/máx/mín del log-softmax, rango medio,
   fracción de snapshots en top-1/top-3, presencia) + media/mín de las features future y clear + task_index, longitud.
2. LightGBM LambdaRank por racha (grupo = block), etiqueta = nº de snapshots de la racha en que la tarea es la respuesta
   (graduada, máx 3). Mismos folds (Data/folds.csv): todas las filas de una sesión van al mismo fold, sin fuga.
3. Score por fila = lsm del 1.er pase + w · log-softmax del score de racha (w con CV anidada); luego el segundo paso
   de sesión (feat_session) y el suavizado se aplican como siempre (f5_stack.py ses / suav).
Uso: python futuro/pruebas/f7_tramo.py MODELO [...]   -> tramo_<modelo>
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import datos, guardar, registrar  # noqa: E402
import suavizado as S  # noqa: E402

K = ["sample_id", "row_index"]
P2 = dict(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, lambdarank_truncation_level=5, verbose=-1, n_jobs=16,
          label_gain=[0, 1, 3, 7])


def agregados(d, score, cols):
    d = d[["sample_id", "block", "task_name", "task_index"] + cols].copy()
    d["lsm"] = S.lsm(d, score)
    d["rk"] = d.groupby("sample_id").lsm.rank(ascending=False, method="first")
    d["top1"], d["top3"] = (d.rk <= 1).astype(float), (d.rk <= 3).astype(float)
    d["nsnap"] = d.groupby("block").sample_id.transform("nunique")
    g = d.groupby(["block", "task_name"], sort=False)
    a = g.agg(task_index=("task_index", "first"), nsnap=("nsnap", "first"), npres=("sample_id", "size"),
              lsm_mean=("lsm", "mean"), lsm_max=("lsm", "max"), lsm_min=("lsm", "min"), rk_mean=("rk", "mean"),
              top1=("top1", "mean"), top3=("top3", "mean"))
    a["pres"] = a.npres / a.nsnap
    a = a.join(g[cols].mean().add_suffix("_m")).join(g[cols].min().add_suffix("_min")).reset_index()
    return a


def correr(modelo):
    t0 = time.time()
    lab, te, fs = datos()
    cols = fs["future"] + fs["clear"]
    so = lab[K].merge(S.leer(modelo, "oof"), on=K, how="left").score.values
    sp = te[K].merge(S.leer(modelo, "preds"), on=K, how="left").score.values
    A, AT = agregados(lab, so, cols), agregados(te, sp, cols)
    y = lab.assign(_y=lab.target_index).groupby(["block", "task_name"], sort=False)._y.sum().clip(upper=3)
    A = A.merge(y.rename("y").reset_index(), on=["block", "task_name"])
    A["fold"] = A.block.map(lab.drop_duplicates("block").set_index("block").fold)
    A = A.sort_values(["block", "task_name"]).reset_index(drop=True)
    AT = AT.sort_values(["block", "task_name"]).reset_index(drop=True)
    F = [c for c in A.columns if c not in ("block", "task_name", "y", "fold")]

    def fit(D, s):
        return lgb.LGBMRanker(**{**P2, "random_state": s}).fit(
            D[F], D.y, group=D.groupby("block", sort=False).size().values, categorical_feature=["task_index"])
    A["bs"] = 0.0
    for f in range(5):
        tr, va = (A.fold != f).values, (A.fold == f).values
        A.loc[va, "bs"] = np.mean([fit(A[tr], s).predict(A.loc[va, F]) for s in range(3)], axis=0)
    AT["bs"] = np.mean([fit(A, s).predict(AT[F]) for s in range(3)], axis=0)
    # de racha a fila: log-softmax del score de racha dentro de la racha
    for D in (A, AT):
        sh = D.bs - D.groupby("block").bs.transform("max")
        D["bl"] = sh - np.log(np.exp(sh).groupby(D.block).transform("sum"))
    L = lab[K + ["block", "task_name", "fold"]].assign(y=lab.target_index)
    L = L.merge(A[["block", "task_name", "bl"]], on=["block", "task_name"], how="left")
    T = te[K + ["block", "task_name"]].merge(AT[["block", "task_name", "bl"]], on=["block", "task_name"], how="left")
    l0, t0s = S.lsm(L, so), S.lsm(T, sp)
    cands = {w: l0 + w * L.bl.values for w in (0, 0.25, 0.5, 1, 2, 4)}
    n, tot, el = L.sample_id.nunique(), 0, []
    for f in range(5):
        tr, va = (L.fold != f).values, (L.fold == f).values
        w = max(cands, key=lambda w: S.hit3(L, cands[w], tr))
        tot += S.hit3(L, cands[w], va) * L[va].sample_id.nunique(); el.append(w)
    w = max(cands, key=lambda w: S.hit3(L, cands[w]))
    h_racha = S.hit3(L, L.bl.values)
    print(f"tramo_{modelo}: 1.er pase {S.hit3(L, l0):.4f}, solo racha {h_racha:.4f}, combinado CV anidada {tot / n:.4f}; "
          f"w por fold {el}, final {w} ({time.time() - t0:.0f}s)", flush=True)
    nombre = f"tramo_{modelo}"
    guardar(nombre, lab, cands[w], te, t0s + w * T.bl.values)
    registrar(nombre, hilo="f7", descripcion=f"modelo por racha sobre {modelo}, w={w}",
              extra=dict(cv_anidada=round(tot / n, 4), cv_solo_racha=round(h_racha, 4), params=[w], params_por_fold=el))


if __name__ == "__main__":
    for m in sys.argv[1:]:
        correr(m)

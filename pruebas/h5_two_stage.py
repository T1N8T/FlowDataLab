"""
h5_two_stage.py · hilo h5_2etapas · Modelo de dos etapas avión -> tarea (solo snapshot, regla 1).

Etapa 1 (avión): una fila por (sample_id, cs) con agregados min/max/mean de las features de sus tareas,
  multi-hot de tipos, nº de tareas y rangos del avión dentro del snapshot. LambdaRank con group = sample_id.
  -> log P(avión) = log-softmax dentro del snapshot.
Etapa 2 (tarea | avión): LambdaRank sobre filas con group = (sample_id, cs), entrenado solo con aviones que
  contienen la correcta (y con >1 tarea). -> log P(tarea | avión) = log-softmax dentro del avión.
Score = log P(avión) + log P(tarea | avión)   -> h5_two_stage
Variante: log P(avión) (OOF anidado por folds) como feature extra del LightGBM de filas -> h5_stage1_feat

Uso: .venv/bin/python pruebas/h5_two_stage.py [--test]
"""
import sys, time, argparse
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/pruebas")
from base import lab_test, build, PARAMS, lgb_cv, guardar, registrar, TYPES, RANK_COLS  # noqa: E402
from metric import hits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true", help="preds de test de la mejor variante")
args = ap.parse_args()
T0 = time.time()

lab, test = lab_test()
d, F = build(lab)
dt, _ = build(test)
d = d.assign(abs_tts=d.time_to_sector.abs())
dt = dt.assign(abs_tts=dt.time_to_sector.abs())
NUM = [c for c in F if c != "task_index"]
print(f"filas {len(d)}  features {len(F)}", flush=True)


# ----------------------------------------------------------------------------- etapa 1: tabla de aviones
def aircraft_table(x):
    """Una fila por (sample_id, cs), ordenada por sample_id, cs. Solo usa filas del mismo snapshot."""
    g = x.groupby(["sample_id", "cs"], sort=True)
    agg = g[NUM + ["abs_tts"]].agg(["min", "max", "mean"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    oh = pd.get_dummies(x.task_index).reindex(columns=range(len(TYPES)), fill_value=0).astype(int)
    oh.columns = [f"has_{t}" for t in TYPES]
    oh = oh.groupby([x.sample_id, x.cs]).max()
    a = pd.concat([agg, oh], axis=1)
    a["ac_n_tasks"] = g.size()
    a["ac_n_types"] = oh.sum(axis=1)
    a = a.reset_index()
    gs = a.groupby("sample_id")
    for c in ["a_since_flight_min", "x_task_age_max", "x_task_age_min", "x_state_age_min", "abs_tts_min",
              "time_to_MD_min", "k_timeToExitPoint_min", "distance_to_zone_min", "a_since_any_min"]:
        a[f"ra_{c}"] = gs[c].rank(method="min")
        a[f"da_{c}"] = a[c] - gs[c].transform("min")
    a["sn_n_aircraft"] = gs.cs.transform("size")
    if "target_index" in x:
        a["y"] = g.target_index.max().values
    return a


A = aircraft_table(d)
At = aircraft_table(dt)
A["fold"] = A.sample_id.map(d.groupby("sample_id").fold.first())
FA = [c for c in A.columns if c not in ("sample_id", "cs", "y", "fold")]
print(f"aviones {len(A)}  features avión {len(FA)}  ({time.time()-T0:.0f}s)", flush=True)

P1 = {**PARAMS, "n_estimators": 400}


def fit_rank(X, F_, y, groups, params, seed=0):
    m = lgb.LGBMRanker(**{**params, "random_state": seed})
    m.fit(X[F_], y, group=groups)
    return m


def gsizes(keys):
    """Tamaños de grupo para claves contiguas."""
    return keys.groupby((keys != keys.shift()).cumsum(), sort=False).size().values


def log_softmax(s, keys):
    s = pd.Series(np.asarray(s, float))
    k = [np.asarray(v) for v in keys] if isinstance(keys, list) else np.asarray(keys)
    mx = s.groupby(k).transform("max")
    z = np.exp(s - mx)
    return (s - mx - np.log(z.groupby(k).transform("sum"))).values


def stage1(tr, ap_):
    m = fit_rank(tr, FA, tr.y, gsizes(tr.sample_id), P1)
    return log_softmax(m.predict(ap_[FA]), ap_.sample_id.values)


# ----------------------------------------------------------------------------- etapa 2: tarea | avión
d2 = d.sort_values(["sample_id", "cs", "row_index"])
dt2 = dt.sort_values(["sample_id", "cs", "row_index"])


def add_in_ac(x):
    g = x.groupby(["sample_id", "cs"])
    x = x.copy()
    x["in_n_tasks"] = g.task_index.transform("size")
    for c in ["x_task_age", "x_state_age", "abs_tts", "time_to_MD", "k_timeToExitPoint"]:
        x[f"in_r_{c}"] = g[c].rank(method="min")
    oh = pd.get_dummies(x.task_index).reindex(columns=range(len(TYPES)), fill_value=0).astype(int)
    for i, t in enumerate(TYPES):
        x[f"in_has_{t}"] = oh[i].groupby([x.sample_id, x.cs]).transform("max")
    return x


d2 = add_in_ac(d2)
dt2 = add_in_ac(dt2)
F2 = F + [c for c in d2.columns if c.startswith("in_")]
P2 = {**PARAMS, "n_estimators": 300}


def stage2(tr, ap_):
    ac_y = tr.groupby(["sample_id", "cs"]).target_index.transform("max")
    sub = tr[(ac_y == 1) & (tr.in_n_tasks > 1)]
    m = lgb.LGBMRanker(**{**P2, "random_state": 0})
    key = sub.sample_id.astype(str) + "_" + sub.cs
    m.fit(sub[F2], sub.target_index, group=gsizes(key), categorical_feature=["task_index"])
    return log_softmax(m.predict(ap_[F2]), [ap_.sample_id.values, ap_.cs.values])


def top3_rate(df, score_col, key="sample_id", y="y"):
    x = df.sort_values([key, score_col], ascending=[True, False])
    x = x.assign(r=x.groupby(key).cumcount())
    return float(x[x[y] == 1].r.lt(3).mean())


# ----------------------------------------------------------------------------- CV de las dos etapas
A["lp1"] = np.nan
d2["lp2"] = np.nan
for f in range(5):
    t = time.time()
    tra, vaa = A[A.fold != f], A[A.fold == f]
    A.loc[vaa.index, "lp1"] = stage1(tra, vaa)
    tr2, va2 = d2[d2.fold != f], d2[d2.fold == f]
    d2.loc[va2.index, "lp2"] = stage2(tr2, va2)
    print(f"fold {f}  ({time.time()-t:.0f}s)", flush=True)

st1_top3 = top3_rate(A, "lp1")
st1_top1 = float(A.sort_values(["sample_id", "lp1"], ascending=[True, False]).groupby("sample_id").head(1).y.mean())
d2 = d2.merge(A[["sample_id", "cs", "lp1"]], on=["sample_id", "cs"], how="left").set_index(d2.index)
# acierto de la etapa 2 (top-1 dentro del avión correcto, solo aviones con >1 tarea)
cor = d2[d2.groupby(["sample_id", "cs"]).target_index.transform("max") == 1]
cor = cor[cor.in_n_tasks > 1]
st2_top1 = float(cor.sort_values(["sample_id", "lp2"], ascending=[True, False])
                 .groupby("sample_id").head(1).target_index.mean())
print(f"etapa 1: top1 avión {st1_top1:.4f}  top3 avión {st1_top3:.4f}  | etapa 2: top1 en avión (>1 tarea) "
      f"{st2_top1:.4f}", flush=True)

# peso de la etapa 2 (un escalar; w=1 es el producto de probabilidades)
res = {}
for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
    sc = (d2.lp1 + w * d2.lp2).sort_index()
    res[w] = hits(d[["sample_id", "row_index"]].assign(score=sc.values)).mean()
    print(f"  w={w}: CV {res[w]:.4f}", flush=True)
W = 1.0
oof_2s = (d2.lp1 + W * d2.lp2).sort_index().values
extra = dict(stage1_top1_avion=round(st1_top1, 4), stage1_top3_avion=round(st1_top3, 4),
             stage2_top1_en_avion=round(st2_top1, 4), cv_por_w={str(k): round(float(v), 4) for k, v in res.items()})
pred = None
if args.test:
    lpt = stage1(A, At)
    dt2 = dt2.assign(lp2=stage2(d2, dt2))
    dt2 = dt2.merge(At[["sample_id", "cs"]].assign(lp1=lpt), on=["sample_id", "cs"], how="left").set_index(dt2.index)
    pred = (dt2.lp1 + W * dt2.lp2).sort_index().values
guardar("h5_two_stage", d, oof_2s, dt, pred)
registrar("h5_two_stage", hilo="h5_2etapas",
          descripcion=f"2 etapas: LambdaRank aviones (top3 avión {st1_top3:.3f}) + LambdaRank tarea|avión; "
                      f"score=logP1+logP2", segundos=time.time() - T0, n_features=len(FA) + len(F2), extra=extra)

# ----------------------------------------------------------------------------- variante: log P(avión) como feature
# OOF anidado: para el fold externo f, las filas de entrenamiento llevan lp1 de una CV interna sobre los otros
# folds; las de validación, lp1 de la etapa 1 entrenada con todos los folds de entrenamiento (= A.lp1).
t1 = time.time()
oof_v = np.zeros(len(d))
FV = F + ["s1_lp1", "s1_rank"]


def attach(x, a_lp):
    m = x[["sample_id", "cs"]].merge(a_lp, on=["sample_id", "cs"], how="left")
    x = x.copy()
    x["s1_lp1"] = m.lp1.values
    x["s1_rank"] = x.groupby("sample_id").s1_lp1.rank(method="dense", ascending=False)
    return x


for f in range(5):
    inner = A[A.fold != f].copy()
    inner["lp1"] = np.nan
    for g_ in sorted(inner.fold.unique()):
        tri, vai = inner[inner.fold != g_], inner[inner.fold == g_]
        inner.loc[vai.index, "lp1"] = stage1(tri, vai)
    tr = attach(d[d.fold != f], inner[["sample_id", "cs", "lp1"]])
    va = attach(d[d.fold == f], A.loc[A.fold == f, ["sample_id", "cs", "lp1"]])
    m = lgb.LGBMRanker(**{**PARAMS, "random_state": 0})
    m.fit(tr[FV], tr.target_index, group=tr.groupby("sample_id", sort=False).size().values,
          categorical_feature=["task_index"])
    oof_v[(d.fold == f).values] = m.predict(va[FV])
    print(f"variante fold {f} ({time.time()-t1:.0f}s)", flush=True)
guardar("h5_stage1_feat", d, oof_v)
registrar("h5_stage1_feat", hilo="h5_2etapas",
          descripcion="LightGBM de filas lgb_adri + log P(avión) de la etapa 1 (OOF anidado) y su rango",
          segundos=time.time() - t1, n_features=len(FV))

# ----------------------------------------------------------------------------- correlación con lgb_adri
ref = pd.read_csv("/home/jm/proyectos/FlowDataLab/oof/lgb_adri.csv")
h_ref = hits(ref)
for n, o in [("h5_two_stage", oof_2s), ("h5_stage1_feat", oof_v)]:
    h = hits(d[["sample_id", "row_index"]].assign(score=o)).reindex(h_ref.index)
    rk = d[["sample_id", "row_index"]].assign(s=o).merge(ref, on=["sample_id", "row_index"])
    print(f"{n}: corr aciertos con lgb_adri {np.corrcoef(h, h_ref)[0,1]:.3f}  "
          f"(ambos {((h==1)&(h_ref==1)).mean():.3f}, solo él {((h==1)&(h_ref==0)).mean():.3f}, "
          f"solo adri {((h==0)&(h_ref==1)).mean():.3f})  spearman score {rk.s.corr(rk.score, method='spearman'):.3f}",
          flush=True)

print(f"total {time.time()-T0:.0f}s")

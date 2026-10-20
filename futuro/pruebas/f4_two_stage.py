"""
f4_two_stage.py · hilo f4_2etapas · Dos etapas avión -> tarea CON FUTURO (rompe la regla 1 a propósito).

Etapa 1 (avión): una fila por (sample_id, cs) con min/max/mean de las features de sus tareas (bloques base, logic,
  past, future de base_fut), multi-hot de tipos, nº de tareas, rangos/diferencias del avión en el snapshot.
  LambdaRank group=sample_id -> log P(avión) = log-softmax dentro del snapshot.
Etapa 2 (tarea | avión): LambdaRank sobre filas, group=(sample_id, cs), solo aviones con la correcta y >1 tarea.
Score = log P(avión) + w·log P(tarea | avión)                                    -> f4_two_stage
Variante: log P(avión) (OOF anidado, 4 sub-folds internos baratos) como feature del LightGBM de filas con
  todos los bloques                                                               -> f4_stage1_feat
Uso: .venv/bin/python futuro/pruebas/f4_two_stage.py   (log en futuro/pruebas/notas/f4_log.txt)
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path[:0] = ["/home/jm/proyectos/FlowDataLab/futuro/pruebas", "/home/jm/proyectos/FlowDataLab/pruebas"]
from base_fut import datos, PARAMS_FUT, lgb_fit, guardar, registrar, TYPES  # noqa: E402
from metric import hits  # noqa: E402

T0 = time.time()
lab, te, fs = datos()
F = fs["base"] + fs["logic"] + fs["past"] + fs["future"]
for x in (lab, te):
    x["abs_tts"] = x.time_to_sector.abs()
NUM = [c for c in F if c != "task_index"] + ["abs_tts"]
print(f"filas {len(lab)}  features {len(F)}", flush=True)


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- etapa 1
KEY = ["a_since_flight_min", "x_task_age_max", "x_task_age_min", "x_state_age_min", "abs_tts_min", "time_to_MD_min",
       "k_timeToExitPoint_min", "a_since_any_min", "f_next_input_min", "f_next_input2_min", "f_next_act_min",
       "f_task_end_min", "f_cs_end_min", "p_task_start_max", "f_next_input_rk_min", "logic_edf_rank_min"]


def aircraft_table(x):
    g = x.groupby(["sample_id", "cs"], sort=True)
    agg = g[NUM].agg(["min", "max", "mean"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    oh = pd.get_dummies(x.task_index).reindex(columns=range(len(TYPES)), fill_value=0).astype(np.int8)
    oh.columns = [f"has_{t}" for t in TYPES]
    oh = oh.groupby([x.sample_id, x.cs]).max()
    a = pd.concat([agg, oh], axis=1)
    a["ac_n_tasks"] = g.size()
    a["ac_n_types"] = oh.sum(axis=1)
    a = a.reset_index()
    gs = a.groupby("sample_id")
    for c in [k for k in KEY if k in a.columns]:
        a[f"ra_{c}"] = gs[c].rank(method="min")
        a[f"da_{c}"] = a[c] - gs[c].transform("min")
    a["sn_n_aircraft"] = gs.cs.transform("size")
    if "target_index" in x:
        a["y"] = g.target_index.max().values
    return a


A = aircraft_table(lab)
At = aircraft_table(te)
A["fold"] = A.sample_id.map(lab.groupby("sample_id").fold.first())
FA = [c for c in A.columns if c not in ("sample_id", "cs", "y", "fold")]
log(f"aviones {len(A)}  features avión {len(FA)}")

P1 = {**PARAMS_FUT, "n_estimators": 400, "colsample_bytree": 0.5}
P1_inner = {**PARAMS_FUT, "n_estimators": 150, "learning_rate": 0.1, "colsample_bytree": 0.5}
P2 = {**PARAMS_FUT, "n_estimators": 300}


def gsizes(keys):
    keys = pd.Series(np.asarray(keys))
    return keys.groupby((keys != keys.shift()).cumsum(), sort=False).size().values


def log_softmax(s, keys):
    s = pd.Series(np.asarray(s, float))
    k = [np.asarray(v) for v in keys] if isinstance(keys, list) else np.asarray(keys)
    mx = s.groupby(k).transform("max")
    z = np.exp(s - mx)
    return (s - mx - np.log(z.groupby(k).transform("sum"))).values


def stage1(tr, ap_, params=P1):
    m = lgb.LGBMRanker(**{**params, "random_state": 0})
    m.fit(tr[FA], tr.y, group=gsizes(tr.sample_id))
    return log_softmax(m.predict(ap_[FA]), ap_.sample_id.values)


# ----------------------------------------------------------------------------- etapa 2
def add_in_ac(x):
    x = x.sort_values(["sample_id", "cs", "row_index"]).copy()
    g = x.groupby(["sample_id", "cs"])
    x["in_n_tasks"] = g.task_index.transform("size")
    for c in ["x_task_age", "x_state_age", "abs_tts", "time_to_MD", "k_timeToExitPoint", "f_task_end",
              "p_task_start", "logic_edf_rank"]:
        x[f"in_r_{c}"] = g[c].rank(method="min")
    oh = pd.get_dummies(x.task_index).reindex(columns=range(len(TYPES)), fill_value=0).astype(np.int8)
    for i, t in enumerate(TYPES):
        x[f"in_has_{t}"] = oh[i].groupby([x.sample_id, x.cs]).transform("max")
    return x


d2 = add_in_ac(lab)
dt2 = add_in_ac(te)
F2 = F + [c for c in d2.columns if c.startswith("in_")]


def stage2(tr, ap_):
    ac_y = tr.groupby(["sample_id", "cs"]).target_index.transform("max")
    sub = tr[(ac_y == 1) & (tr.in_n_tasks > 1)]
    m = lgb.LGBMRanker(**{**P2, "random_state": 0})
    m.fit(sub[F2], sub.target_index, group=gsizes(sub.sample_id.astype(str) + "_" + sub.cs),
          categorical_feature=["task_index"])
    return log_softmax(m.predict(ap_[F2]), [ap_.sample_id.values, ap_.cs.values])


# ----------------------------------------------------------------------------- CV dos etapas
A["lp1"] = np.nan
d2["lp2"] = np.nan
for f in range(5):
    A.loc[A.fold == f, "lp1"] = stage1(A[A.fold != f], A[A.fold == f])
    d2.loc[d2.fold == f, "lp2"] = stage2(d2[d2.fold != f], d2[d2.fold == f])
    log(f"dos etapas fold {f}")

As = A.sort_values(["sample_id", "lp1"], ascending=[True, False])
As["r"] = As.groupby("sample_id").cumcount()
st1_top1 = float(As[As.y == 1].r.eq(0).mean())
st1_top3 = float(As[As.y == 1].r.lt(3).mean())
d2 = d2.merge(A[["sample_id", "cs", "lp1"]], on=["sample_id", "cs"], how="left").set_index(d2.index)
cor = d2[(d2.groupby(["sample_id", "cs"]).target_index.transform("max") == 1) & (d2.in_n_tasks > 1)]
st2_top1 = float(cor.sort_values(["sample_id", "lp2"], ascending=[True, False])
                 .groupby("sample_id").head(1).target_index.mean())
log(f"etapa 1: top1 avión {st1_top1:.4f} top3 avión {st1_top3:.4f} | etapa 2 top1 en avión {st2_top1:.4f}")

res = {}
for w in [0.5, 1.0, 1.5, 2.0]:
    sc = (d2.lp1 + w * d2.lp2).sort_index()
    res[w] = hits(lab[["sample_id", "row_index"]].assign(score=sc.values)).mean()
    log(f"  w={w}: CV {res[w]:.4f}")
W = 1.0
oof_2s = (d2.lp1 + W * d2.lp2).sort_index().values

lpt = stage1(A, At)
dt2 = dt2.assign(lp2=stage2(d2, dt2))
dt2 = dt2.merge(At[["sample_id", "cs"]].assign(lp1=lpt), on=["sample_id", "cs"], how="left").set_index(dt2.index)
pred_2s = (dt2.lp1 + W * dt2.lp2).sort_index().values
guardar("f4_two_stage", lab, oof_2s, te, pred_2s)
extra = dict(etapa1_top1=round(st1_top1, 4), etapa1_top3=round(st1_top3, 4), etapa2_top1_en_avion=round(st2_top1, 4),
             cv_por_w={str(k): round(float(v), 4) for k, v in res.items()})
registrar("f4_two_stage", hilo="f4_2etapas",
          descripcion=f"2 etapas con futuro: LambdaRank aviones (top1 {st1_top1:.3f}, top3 {st1_top3:.3f}) + "
                      f"LambdaRank tarea|avión; score=logP1+logP2", segundos=time.time() - T0,
          n_features=len(FA) + len(F2), extra=extra)
A[["sample_id", "cs", "fold", "y", "lp1"]].to_parquet("/home/jm/proyectos/FlowDataLab/futuro/pruebas/f4_lp1.parquet")

# ----------------------------------------------------------------------------- variante: log P(avión) como feature
t1 = time.time()
FV = F + ["s1_lp1", "s1_rank"]


def attach(x, a_lp):
    m = x[["sample_id", "cs"]].merge(a_lp, on=["sample_id", "cs"], how="left")
    x = x.copy()
    x["s1_lp1"] = m.lp1.values
    x["s1_rank"] = x.groupby("sample_id").s1_lp1.rank(method="dense", ascending=False)
    return x


# para que train y validación usen lp1 del mismo tipo de modelo, la etapa 1 de la variante usa P1_inner en ambos
oof_v = np.zeros(len(lab))
for f in range(5):
    inner = A[A.fold != f].copy()
    inner["lp1"] = np.nan
    for g_ in sorted(inner.fold.unique()):
        inner.loc[inner.fold == g_, "lp1"] = stage1(inner[inner.fold != g_], inner[inner.fold == g_], P1_inner)
    va_a = A.loc[A.fold == f, ["sample_id", "cs"]].assign(lp1=stage1(A[A.fold != f], A[A.fold == f], P1_inner))
    tr = attach(lab[lab.fold != f], inner[["sample_id", "cs", "lp1"]])
    va = attach(lab[lab.fold == f], va_a)
    oof_v[(lab.fold == f).values] = lgb_fit(tr, FV, {**PARAMS_FUT, "random_state": 0}).predict(va[FV])
    log(f"variante fold {f}")
# test: train con lp1 OOF (P1_inner, 5 folds), test con etapa 1 entrenada con todo
a_oof = A[["sample_id", "cs"]].assign(lp1=np.nan)
for f in range(5):
    a_oof.loc[A.fold == f, "lp1"] = stage1(A[A.fold != f], A[A.fold == f], P1_inner)
tr = attach(lab, a_oof)
tt = attach(te, At[["sample_id", "cs"]].assign(lp1=stage1(A, At, P1_inner)))
pred_v = lgb_fit(tr, FV, {**PARAMS_FUT, "random_state": 0}).predict(tt[FV])
guardar("f4_stage1_feat", lab, oof_v, te, pred_v)
registrar("f4_stage1_feat", hilo="f4_2etapas",
          descripcion="LightGBM de filas (todos los bloques, 1 semilla) + log P(avión) de la etapa 1 (OOF anidado, "
                      "4 sub-folds, 150 árboles) y su rango", segundos=time.time() - t1, n_features=len(FV))

# ----------------------------------------------------------------------------- correlación con lgb_future
ref = pd.read_csv("/home/jm/proyectos/FlowDataLab/futuro/oof/lgb_future.csv")
h_ref = hits(ref)
for n, o in [("f4_two_stage", oof_2s), ("f4_stage1_feat", oof_v)]:
    h = hits(lab[["sample_id", "row_index"]].assign(score=o)).reindex(h_ref.index)
    rk = lab[["sample_id", "row_index"]].assign(s=o).merge(ref, on=["sample_id", "row_index"])
    log(f"{n}: corr aciertos {np.corrcoef(h, h_ref)[0,1]:.3f} ambos {((h==1)&(h_ref==1)).mean():.3f} "
        f"solo él {((h==1)&(h_ref==0)).mean():.3f} solo ref {((h==0)&(h_ref==1)).mean():.3f} "
        f"oráculo {((h==1)|(h_ref==1)).mean():.4f} spearman {rk.s.corr(rk.score, method='spearman'):.3f}")
    m = rk.assign(r1=rk.groupby("sample_id").s.rank(pct=True), r2=rk.groupby("sample_id").score.rank(pct=True))
    for a in (0.2, 0.3, 0.5):
        log(f"   mezcla rangos {a}·{n} + {1-a}·lgb_future: "
            f"{hits(m[['sample_id','row_index']].assign(score=a*m.r1+(1-a)*m.r2)).mean():.4f}")
log("fin")

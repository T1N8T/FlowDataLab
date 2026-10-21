"""f11_estilo.py · ESTILO del controlador por sesión, sin etiquetas (hilo f11).

Idea: el 023 (privado) coordina poco; si el modelo sabe "cómo trabaja" el controlador de la sesión puede ajustar la
prioridad por tipo. Todo se calcula con las filas de la sesión (snapshots pasados y futuros, permitido en la línea futuro)
y NUNCA con target_index, así que vale igual en test.

Bloques (columnas iguales en toda la sesión = st_*, interacciones con la fila = sti_*):
  st_snap_min      snapshots por minuto            st_blk_min    rachas (block) por minuto
  st_ins_min       instrucciones por minuto (eventos únicos avión+instante, misma extracción que f10_instr)
  st_lat_med       latencia mediana entre la 1.ª aparición de una tarea y la 1.ª instrucción compatible (si_match)
  st_noins         fracción de tareas que desaparecen sin instrucción compatible mientras estaban en la lista (+30 s)
  st_life_med      vida mediana de una tarea en la lista (último t - primer t)
  st_len_mean      nº medio de tareas por snapshot      st_coord_rows  fracción de filas de coordinación (tipos 2, 3)
  st_assumed       fracción de filas con is_assumed
  modelo (score del 1.er pase N, OOF/preds, no etiquetas):
  st_top1_coord    fracción de snapshots de la sesión cuyo top-1 es coordinación
  st_top1_<grupo>  idem para conflicto (0,1,6), assume/transfer (4,5)
interacciones:
  sti_coord        st_top1_coord si la fila es coordinación, si no NaN
  sti_type_share   fracción de snapshots de la sesión cuyo top-1 es del MISMO tipo que la fila (prior de tipo por sesión)
  sti_type_pmean   probabilidad media (softmax del 1.er pase) que la sesión da a ese tipo
  sti_match_rel    si_match / st_lat_med     sti_age_rel  x_task_age / st_life_med
  sti_rows_type    fracción de filas de la sesión de ese tipo
Fuga: las st_top1_* usan el OOF del 1.er pase (f10_instr_l63_s3), que para cada fila no vio su fold; es el mismo nivel
de apilado que feat_session (despreciable a nivel de sesión). Sin etiquetas en nada más.

Uso: python futuro/pruebas/f11_estilo.py VARIANTE [semillas]
  VARIANTES: sty (st_* sin modelo), int (sti_* sin modelo), mod (st_top1_* + sti de modelo), all (todo)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import lgb_cv, guardar, registrar, OUT, PARAMS_FUT  # noqa: E402
from f10_instr import datos_instr, medir, add_instr  # noqa: E402
import features_future as ff, feat_session  # noqa: E402

COORD, CONFL, AT = [2, 3], [0, 1, 6], [4, 5]
P_N = dict(num_leaves=63, min_child_samples=100, extra_trees=True)


def estilo(D, score):
    """D: lab+te concatenados (con bloque instr). score: 1.er pase alineado con D. Devuelve DataFrame alineado."""
    g = D.groupby("session")
    snaps = D.drop_duplicates("sample_id")
    dur = (snaps.groupby("session").t.max() - snaps.groupby("session").t.min()).clip(lower=60) / 60
    S = pd.DataFrame(index=dur.index)
    S["st_snap_min"] = snaps.groupby("session").size() / dur
    S["st_blk_min"] = snaps.groupby("session").block.nunique() / dur
    ev = np.where(D.a_since_flight < 3600, np.round(D.t - D.a_since_flight), np.nan)
    k = D.assign(ev=ev).dropna(subset=["ev"]).drop_duplicates(["session", "cs", "ev"])
    S["st_ins_min"] = k.groupby("session").size().reindex(S.index).fillna(0) / dur
    # por tarea (sesión, task_name): primera y última aparición
    o = D.sort_values("t")
    first = o.drop_duplicates(["session", "task_name"], keep="first")
    last = o.drop_duplicates(["session", "task_name"], keep="last")
    first = first.set_index(["session", "task_name"]); last = last.set_index(["session", "task_name"])
    life = last.t - first.t.reindex(last.index)
    lat = first.si_match                    # s desde la 1.ª aparición hasta la 1.ª instrucción compatible
    S["st_lat_med"] = lat.groupby(level=0).median()
    S["st_noins"] = (~(lat <= life.reindex(lat.index) + 30)).groupby(level=0).mean()
    S["st_life_med"] = life.groupby(level=0).median()
    S["st_len_mean"] = snaps.groupby("session").seq_len.mean()
    S["st_coord_rows"] = D.task_index.isin(COORD).groupby(D.session).mean()
    S["st_assumed"] = (D.is_assumed == 1).groupby(D.session).mean()
    # modelo: top-1 y probabilidades por tipo
    x = D[["sample_id", "session", "task_index"]].assign(s=np.asarray(score))
    x["s"] -= x.groupby("sample_id").s.transform("max")
    x["p"] = np.exp(x.s); x["p"] /= x.groupby("sample_id").p.transform("sum")
    top = x.sort_values(["sample_id", "s"], ascending=[True, False]).drop_duplicates("sample_id")
    M = pd.DataFrame(index=S.index)
    for nm, ty in (("coord", COORD), ("confl", CONFL), ("at", AT)):
        M["st_top1_" + nm] = top.task_index.isin(ty).groupby(top.session).mean()
    tshare = pd.crosstab(top.session, top.task_index, normalize="index")
    pmean = x.groupby(["session", "task_index"]).p.sum() / x.groupby("session").sample_id.nunique().reindex(
        x.groupby(["session", "task_index"]).p.sum().index.get_level_values(0)).values
    rshare = pd.crosstab(D.session, D.task_index, normalize="index")
    out = pd.DataFrame(index=D.index)
    for c in S.columns:
        out[c] = D.session.map(S[c]).values
    for c in M.columns:
        out[c] = D.session.map(M[c]).values
    key = pd.MultiIndex.from_arrays([D.session, D.task_index])
    out["sti_coord"] = np.where(D.task_index.isin(COORD), out.st_top1_coord, np.nan)
    out["sti_type_share"] = tshare.stack().reindex(key).fillna(0).values
    out["sti_type_pmean"] = pmean.reindex(key).fillna(0).values
    out["sti_match_rel"] = D.si_match.values / out.st_lat_med.clip(lower=1).values
    out["sti_age_rel"] = D.x_task_age.values / out.st_life_med.clip(lower=1).values
    out["sti_rows_type"] = rshare.stack().reindex(key).fillna(0).values
    return out


GRUPOS = {
    "sty": ["st_snap_min", "st_blk_min", "st_ins_min", "st_lat_med", "st_noins", "st_life_med", "st_len_mean",
            "st_coord_rows", "st_assumed"],
    "int": ["sti_match_rel", "sti_age_rel", "sti_rows_type"],
    "mod": ["st_top1_coord", "st_top1_confl", "st_top1_at", "sti_coord", "sti_type_share", "sti_type_pmean"],
}
GRUPOS["all"] = GRUPOS["sty"] + GRUPOS["int"] + GRUPOS["mod"]
GRUPOS["intmod"] = GRUPOS["int"] + GRUPOS["mod"]
GRUPOS["modi"] = ["sti_coord", "sti_type_share", "sti_type_pmean"]     # solo interacciones del modelo


def run(var, seeds=3, nj=4):
    t0 = time.time()
    lab, te, fs = datos_instr()
    o = pd.read_csv(OUT / "oof" / "f10_instr_l63_s3.csv"); p = pd.read_csv(OUT / "preds" / "f10_instr_l63_s3.csv")
    so = lab[["sample_id", "row_index"]].merge(o, how="left").score.values
    sp = te[["sample_id", "row_index"]].merge(p, how="left").score.values
    assert not np.isnan(so).any() and not np.isnan(sp).any()
    D = pd.concat([lab, te], ignore_index=True)
    E = estilo(D, np.r_[so, sp])
    lab = pd.concat([lab, E.iloc[:len(lab)].reset_index(drop=True)], axis=1)
    te = pd.concat([te, E.iloc[len(lab):].reset_index(drop=True)], axis=1)
    if var == "show":
        print(lab.groupby("atco")[GRUPOS["sty"] + GRUPOS["mod"][:3]].mean().round(3).to_string())
        print(te.groupby("atco")[GRUPOS["sty"] + GRUPOS["mod"][:3]].mean().round(3).to_string())
        return
    F = [c for b in ["base", "logic", "past", "future", "clear", "instr"] for c in fs[b]] + GRUPOS[var]
    seeds = tuple(range(int(seeds)))
    oof, pred = lgb_cv(lab, F, {**PARAMS_FUT, **P_N, "n_jobs": nj}, seeds=seeds, test=te)
    nm = f"f11_estilo_{var}_s{len(seeds)}"
    guardar(nm, lab, oof, te, pred)
    s_o, s_p, _ = feat_session.sesion(lab, oof, te, pred)
    guardar(nm + "_ses", lab, s_o, te, s_p)
    a, b = medir(nm, lab, oof), medir(nm, lab, s_o)
    for n_, (c, c23, mn) in ((nm, a), (nm + "_ses", b)):
        registrar(n_, hilo="f11", descripcion=f"N + estilo por sesión [{var}] {GRUPOS[var]} seeds={list(seeds)}",
                  n_features=len(F), extra=dict(cv_023=round(c23, 4)))
    print(f"{nm}: 1.er pase {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) · + sesión {b[0]:.4f} "
          f"(CV 023 {b[1]:.4f}, peor {b[2]:.3f})  {len(F)} features ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__" and sys.argv[1:2] != ["post"]:
    run(*sys.argv[1:2], *[int(x) for x in sys.argv[2:4]])


def post(base="f10_instr_l63_s3"):
    """Variante barata: el logit de sesión de feat_session con columnas extra de estilo (interacciones con el tipo),
    pesos ajustados solo con los folds de entrenamiento. Se compara con el logit de sesión normal."""
    lab, te, fs = datos_instr()
    o = pd.read_csv(OUT / "oof" / f"{base}.csv"); p = pd.read_csv(OUT / "preds" / f"{base}.csv")
    so = lab[["sample_id", "row_index"]].merge(o, how="left").score.values
    sp = te[["sample_id", "row_index"]].merge(p, how="left").score.values
    D = pd.concat([lab, te], ignore_index=True)
    E = estilo(D, np.r_[so, sp])
    ty = D.task_index.values
    coord = np.isin(ty, COORD).astype(float)
    sty = (E[GRUPOS["sty"]] - E[GRUPOS["sty"]].iloc[:len(lab)].mean()) / E[GRUPOS["sty"]].iloc[:len(lab)].std()
    extras = {
        "coordxtop1": coord * (E.st_top1_coord.values - 0.1),
        "logshare": np.log(E.sti_type_share.values + 0.01),
        "logpmean": np.log(E.sti_type_pmean.values + 1e-3),
        "coordxsty": np.column_stack([coord * sty[c].values for c in sty.columns]),
        "conflxtop1": np.isin(ty, CONFL) * (E.st_top1_confl.values - 0.04),
    }
    combos = {"coordxtop1": ["coordxtop1"], "logshare": ["logshare"], "logpmean": ["logpmean"],
              "coordxsty": ["coordxsty"], "top1s": ["coordxtop1", "conflxtop1"],
              "todo": ["coordxtop1", "conflxtop1", "logshare", "coordxsty"]}
    X0 = np.r_[feat_session.session_feats(lab, so).values, feat_session.session_feats(te, sp).values]
    y, sid, fold = lab.target_index.values, lab.sample_id.values, lab.fold.values
    n = len(lab)
    def evalua(X, nombre, guarda=False):
        out = np.full(n, np.nan)
        for f in range(5):
            tr, va = fold != f, fold == f
            out[va] = X[:n][va] @ feat_session.fit_logit(X[:n][tr], y[tr], sid[tr])
        w = feat_session.fit_logit(X[:n], y, sid)
        a = medir(nombre, lab, out)
        print(f"post {nombre}: {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f})  pesos extra {np.round(w[X0.shape[1]:], 3)}",
              flush=True)
        if guarda:
            guardar(nombre, lab, out, te, X[n:] @ w)
            registrar(nombre, hilo="f11", descripcion=f"logit de sesión de {base} + columnas de estilo",
                      extra=dict(cv_023=round(a[1], 4)))
        return a
    evalua(X0, "ref")
    for k, cs in combos.items():
        X = np.column_stack([X0] + [extras[c] for c in cs])
        evalua(X, f"f11_estilo_post_{k}", guarda=(k in ("coordxtop1", "todo")))


if __name__ == "__main__" and sys.argv[1:2] == ["post"]:
    post(*sys.argv[2:3])

"""f11_instrx.py · bloque instr AMPLIADO (si2_*) sobre N (f10_instr_l63) · hilo f11.
Misma lista de instrucciones por avión que f10_instr.add_instr (a_since_flight + a_last_*; sin etiquetas). Añade:
    secuencia     si2_seq1..3 tipo de las 3 siguientes instrucciones al avión; si2_t2/t3 s hasta la 2.ª/3.ª;
                  si2_nmatch3 cuántas de esas 3 son compatibles con la tarea
    acoplamiento  si2_match_end = si_match - f_task_end (¿la tarea desaparece justo tras la instrucción compatible?),
                  si2_gone_match (indicador -3 <= si2_match_end <= 10);
                  si2_res / si2_res_gap: última instrucción compatible antes del fin de la tarea (+6 s), s desde t y
                  s entre ella y el fin de la tarea; si2_res_rk rango de |si2_res| en el snapshot
    orden         si2_nbefore_any: nº de instrucciones a CUALQUIER avión de la sesión antes de la compatible de éste;
                  si2_first_any: la siguiente instrucción de la sesión es para este avión;
                  si2_match_rk_cs: rango denso por avión de si_match; si2_ntie_flight: tareas del avión con el mismo si_match
    racha         si2_nmatch_blk / si2_ninstr_blk: compatibles / todas al avión durante la racha (block, +5 s);
                  si2_nmatch60 / 180: compatibles en los próximos 60 / 180 s
Uso:  python futuro/pruebas/f11_instrx.py cv [variante] [semillas]
    variante: full (instr + instr2), seq, coup, ord, blk (instr + solo ese subgrupo)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import datos, lgb_cv, guardar, registrar, OUT, PARAMS_FUT  # noqa: E402
import features_future as ff, feat_session  # noqa: E402
from f10_instr import datos_instr, medir, COMPAT, ACT_N  # noqa: E402

CACHE2 = OUT / "cache_instr2.parquet"
CMASK = np.zeros((32, len(ACT_N)), bool)
for _ty, _c in COMPAT.items():
    CMASK[_ty, _c] = True
GRUPOS = dict(seq=["si2_seq1", "si2_seq2", "si2_seq3", "si2_t2", "si2_t3", "si2_nmatch3"],
              coup=["si2_match_end", "si2_gone_match", "si2_res", "si2_res_gap", "si2_res_rk"],
              ord=["si2_nbefore_any", "si2_first_any", "si2_match_rk_cs", "si2_ntie_flight"],
              blk=["si2_nmatch_blk", "si2_ninstr_blk", "si2_nmatch60", "si2_nmatch180"])


def add_instr2(d):
    """d: lab+te concatenados con el bloque instr (si_match) y f_task_end. Devuelve DataFrame alineado con d."""
    ev = np.where(d.a_since_flight < 3600, np.round(d.t - d.a_since_flight), np.nan)
    k = d.assign(ev=ev).dropna(subset=["ev"]).sort_values("t").drop_duplicates(["session", "cs", "ev"])
    acts = k[["a_last_" + a for a in ff.ACTS]].values
    k = k.assign(act=np.where(acts.max(1) == 1, acts.argmax(1), len(ff.ACTS)))[["session", "cs", "ev", "act"]]
    n = len(d)
    cols = {c: np.full(n, np.nan) for g in GRUPOS.values() for c in g}
    t_all, ti_all = d.t.values, d.task_index.values.astype(int)
    tend_all = t_all + d.f_task_end.values
    bs = d.groupby("block").t.transform("min").values
    be = d.groupby("block").t.transform("max").values
    ev_g = {key: g for key, g in k.sort_values("ev").groupby(["session", "cs"])}
    for key, idx in d.groupby(["session", "cs"]).indices.items():
        if key not in ev_g:
            for c in ("si2_nmatch3", "si2_nmatch_blk", "si2_ninstr_blk", "si2_nmatch60", "si2_nmatch180"):
                cols[c][idx] = 0
            continue
        g = ev_g[key]; e, a = g.ev.values, g.act.values.astype(int)
        t, ti, tend = t_all[idx], ti_all[idx], tend_all[idx]
        lo = np.searchsorted(e, t + 1.5)
        for j in range(3):
            ok = lo + j < len(e)
            cols[f"si2_seq{j + 1}"][idx[ok]] = a[lo[ok] + j]
            if j:
                cols[f"si2_t{j + 1}"][idx[ok]] = e[lo[ok] + j] - t[ok]
        cm = CMASK[ti][:, a]                                  # filas x eventos: instrucción compatible con la tarea
        dt = e[None, :] - t[:, None]
        rank = np.arange(len(e))[None, :] - lo[:, None]       # 0 = siguiente instrucción
        cols["si2_nmatch3"][idx] = (cm & (rank >= 0) & (rank < 3)).sum(1)
        fut = dt > 1.5
        cols["si2_nmatch60"][idx] = (cm & fut & (dt <= 60)).sum(1)
        cols["si2_nmatch180"][idx] = (cm & fut & (dt <= 180)).sum(1)
        inblk = (e[None, :] >= bs[idx][:, None] - 1.5) & (e[None, :] <= be[idx][:, None] + 5)
        cols["si2_ninstr_blk"][idx] = inblk.sum(1)
        cols["si2_nmatch_blk"][idx] = (cm & inblk).sum(1)
        # instrucción "resolutora": última compatible antes del fin de la tarea (+6 s)
        cand = cm & (e[None, :] <= tend[:, None] + 6)
        has = cand.any(1)
        last = np.where(has, len(e) - 1 - np.argmax(cand[:, ::-1], 1), 0)
        er = e[last]
        cols["si2_res"][idx[has]] = (er - t)[has]
        cols["si2_res_gap"][idx[has]] = (tend - er)[has]
    out = pd.DataFrame(cols, index=d.index)
    out["si2_match_end"] = d.si_match.values - d.f_task_end.values
    out["si2_gone_match"] = np.where(out.si2_match_end.isna(), np.nan,
                                     ((out.si2_match_end >= -3) & (out.si2_match_end <= 10)).astype(float))
    out["si2_res_rk"] = out.si2_res.abs().groupby(d.sample_id.values).rank(method="min")
    # orden entre aviones de la sesión
    S = {s: np.sort(g.ev.values) for s, g in k.groupby("session")}
    nb, fa = np.full(n, np.nan), np.full(n, np.nan)
    nxt_own = t_all + d.f_next_input.values
    for s, idx in d.groupby("session").indices.items():
        if s not in S:
            continue
        E = S[s]; t = t_all[idx]
        lo = np.searchsorted(E, t + 1.5)
        m = d.si_match.values[idx]; ok = ~np.isnan(m)
        nb[idx[ok]] = np.searchsorted(E, t[ok] + m[ok] - 0.5) - lo[ok]
        first = np.where(lo < len(E), E[np.minimum(lo, len(E) - 1)], np.inf)
        own = nxt_own[idx]
        fa[idx] = np.where(np.isnan(own), 0, (np.abs(own - first) < 0.5).astype(float))
    out["si2_nbefore_any"], out["si2_first_any"] = nb, fa
    cmin = d.si_match.groupby([d.sample_id.values, d.cs.values]).transform("min")
    out["si2_match_rk_cs"] = cmin.groupby(d.sample_id.values).rank(method="dense").values
    out["si2_ntie_flight"] = d.groupby([d.sample_id.values, d.cs.values, d.si_match.values]).si_match.transform("size") \
        .where(d.si_match.notna())
    return out


def datos_instr2():
    lab, te, fs = datos_instr()
    if CACHE2.exists():
        I = pd.read_parquet(CACHE2)
    else:
        t0 = time.time()
        D = pd.concat([lab, te], ignore_index=True)
        I = add_instr2(D)
        tmp = CACHE2.with_suffix(f".{time.time_ns()}.tmp"); I.to_parquet(tmp); tmp.rename(CACHE2)
        print(f"bloque instr2 calculado en {time.time() - t0:.0f}s", flush=True)
    fs["instr2"] = list(I.columns)
    lab = pd.concat([lab, I.iloc[:len(lab)].reset_index(drop=True)], axis=1)
    te = pd.concat([te, I.iloc[len(lab):].reset_index(drop=True)], axis=1)
    return lab, te, fs


def cv(variante="full", seeds=3):
    t0 = time.time()
    lab, te, fs = datos_instr2()
    extra = fs["instr2"] if variante == "full" else GRUPOS[variante]
    F = [c for b in ["base", "logic", "past", "future", "clear", "instr"] for c in fs[b]] + extra
    seeds = tuple(range(int(seeds)))
    P = {**PARAMS_FUT, "num_leaves": 63, "min_child_samples": 100, "extra_trees": True, "n_jobs": 4}
    oof, pred = lgb_cv(lab, F, P, seeds=seeds, test=te)
    nm = f"f11_instrx_{variante}_s{len(seeds)}"
    guardar(nm, lab, oof, te, pred)
    so, sp, _ = feat_session.sesion(lab, oof, te, pred)
    guardar(nm + "_ses", lab, so, te, sp)
    a, b = medir(nm, lab, oof), medir(nm, lab, so)
    for n_, (c, c23, mn) in ((nm, a), (nm + "_ses", b)):
        registrar(n_, hilo="f11", descripcion=f"N (instr l63 extra_trees) + instr2[{variante}] seeds={list(seeds)}",
                  n_features=len(F), extra=dict(cv_023=round(c23, 4)))
    print(f"{nm}: 1.er pase {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) · + sesión {b[0]:.4f} "
          f"(CV 023 {b[1]:.4f}, peor {b[2]:.3f})  {len(F)} features ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        lab, te, fs = datos_instr2()
        print(lab[fs["instr2"]].describe().T.to_string())
        y = lab.target_index == 1
        print(lab.groupby(y)[fs["instr2"]].mean().T.to_string())
    else:
        cv(*sys.argv[2:])

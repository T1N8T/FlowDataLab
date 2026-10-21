"""f11_orexit.py · hilo f11 · Oráculo de coordinaciones SOLO en exit_coordination (task_index 3).
Config de N (clear + instr, 63 hojas, min_child 100, extra_trees) + oracle_data "coord" con las or_* a NaN en todas las
filas que no son exit_coordination (por fold) y rangos or_*_rk recalculados entre las filas que quedan; + sesión.
    python futuro/pruebas/f11_orexit.py            # 3 semillas
Mide CV, CV 023, peor controlador y % de entry_coordination (task_index 2) en el top-3 del 020 en test,
y lo compara con f10_instr_l63_s3(_ses) y f10_instr_or_s3(_ses).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import guardar, registrar, OUT, PARAMS_FUT, lgb_fit  # noqa: E402
from f10_instr import datos_instr, medir  # noqa: E402
import feat_session, m_lgb_future as M  # noqa: E402

RK = ["or_ever", "or_near60", "or_near180", "or_cs_near60"]


def solo_exit(o, d):
    o = o.copy()
    o.loc[(d.task_index.values != 3)] = np.nan
    for c in RK:
        o[c + "_rk"] = o[c].round(6).groupby(d.sample_id.values).rank(ascending=False, method="min")
    o["or_dt_abs"] = o.or_dt.abs()
    return o


def top3_tipos(d, score, atco=20):
    x = d[["sample_id", "task_index", "atco"]].assign(s=score)
    x = x[x.atco == atco]
    x["rk"] = x.groupby("sample_id").s.rank(ascending=False, method="first")
    t = x[x.rk <= 3]
    return {k: round(float((t.task_index == k).mean() * 100), 2) for k in (2, 3)} | \
        {"top1_entry": round(float((x[x.rk == 1].task_index == 2).mean() * 100), 2)}


def main(seeds=3):
    t0 = time.time()
    lab, te, fs = datos_instr()
    F = [c for b in ["base", "logic", "past", "future", "clear", "instr"] for c in fs[b]]
    P = {**PARAMS_FUT, "num_leaves": 63, "min_child_samples": 100, "extra_trees": True, "n_jobs": 4}
    seeds = tuple(range(int(seeds)))
    O = M.oracle_data(dict(lab=lab, te=te), "coord")
    F = F + list(O["lab"].columns)
    oof = np.zeros(len(lab))
    for f in range(5):
        L = pd.concat([lab, solo_exit(O[f], lab)], axis=1)
        tr, va = (L.fold != f).values, (L.fold == f).values
        oof[va] = np.mean([lgb_fit(L[tr], F, {**P, "random_state": s}).predict(L.loc[va, F]) for s in seeds], axis=0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    L, T = pd.concat([lab, solo_exit(O["lab"], lab)], axis=1), pd.concat([te, solo_exit(O["te"], te)], axis=1)
    pred = np.mean([lgb_fit(L, F, {**P, "random_state": s}).predict(T[F]) for s in seeds], axis=0)
    nm = f"f11_orexit_s{len(seeds)}"
    guardar(nm, lab, oof, te, pred)
    so, sp, _ = feat_session.sesion(lab, oof, te, pred)
    guardar(nm + "_ses", lab, so, te, sp)
    a, b = medir(nm, lab, oof), medir(nm, lab, so)
    tp = top3_tipos(te, sp)
    for n_, (c, c23, mn) in ((nm, a), (nm + "_ses", b)):
        registrar(n_, hilo="f11", descripcion=f"N (clear+instr l63 et) + oráculo coord solo en exit_coord seeds={list(seeds)}",
                  n_features=len(F), extra=dict(cv_023=round(c23, 4), peor=round(mn, 4), top3_020_test=tp))
    print(f"{nm}: 1.er pase {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) · + sesión {b[0]:.4f} "
          f"(CV 023 {b[1]:.4f}, peor {b[2]:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    print("top-3 del 020 en test (% entry / exit coord):", tp, flush=True)
    for ref in ["f10_instr_l63_s3", "f10_instr_l63_s3_ses", "f10_instr_or_s3", "f10_instr_or_s3_ses", "f10_instr_l63_s10_ses"]:
        po, pp = OUT / "oof" / f"{ref}.csv", OUT / "preds" / f"{ref}.csv"
        if not po.exists():
            continue
        r = lab[["sample_id", "row_index"]].merge(pd.read_csv(po), how="left").score.values
        c, c23, mn = medir(ref, lab, r)
        extra = ""
        if pp.exists():
            rp = te[["sample_id", "row_index"]].merge(pd.read_csv(pp), how="left").score.values
            extra = f" top3 020 test {top3_tipos(te, rp)}"
        print(f"  ref {ref}: {c:.4f} (CV 023 {c23:.4f}, peor {mn:.3f}){extra}", flush=True)
    # por tipo de la respuesta: acierto según el tipo de la tarea correcta
    from metric import hits
    y = lab[lab.target_index == 1].set_index("sample_id").task_index
    for nmx, s in (("orexit_ses", so),):
        h = hits(lab[["sample_id", "row_index"]].assign(score=s))
        print(f"  {nmx} acierto por tipo de respuesta:", h.groupby(y.reindex(h.index)).mean().round(4).to_dict(), flush=True)
    for ref in ["f10_instr_l63_s3_ses", "f10_instr_or_s3_ses"]:
        r = lab[["sample_id", "row_index"]].merge(pd.read_csv(OUT / "oof" / f"{ref}.csv"), how="left").score.values
        h = hits(lab[["sample_id", "row_index"]].assign(score=r))
        print(f"  {ref} acierto por tipo:", h.groupby(y.reindex(h.index)).mean().round(4).to_dict(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])

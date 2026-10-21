"""f11_bagatco.py · Bagging por controladores (hilo f11).
Config de N (f10_instr_l63: base+logic+past+future+clear+instr, 63 hojas, min_child 100, extra_trees). En cada fold se
entrenan K LightGBM, el k-ésimo con semilla k y un subconjunto aleatorio del 75 % de los controladores de entrenamiento
(rng propio por fold y modelo), y se promedian. Modelo final para test: igual sobre los 18 controladores. + sesión.
Sin fuga: el fold que se predice nunca entra en el entrenamiento (ni en el sorteo de controladores).
Modos:
    python futuro/pruebas/f11_bagatco.py fit DIR K0 K1 ... # entrena los modelos k indicados, guarda oof/pred por modelo (npy)
    python futuro/pruebas/f11_bagatco.py eval DIR K     # promedia los modelos 0..K-1, + sesión, mide y guarda
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import guardar, registrar, PARAMS_FUT, lgb_fit, OUT  # noqa: E402
from f10_instr import datos_instr, VARIANTES, medir  # noqa: E402
import feat_session  # noqa: E402


FRAC = 0.75


def prep():
    lab, te, fs = datos_instr()
    bloques, p, _ = VARIANTES["f10_instr_l63"]
    F = [c for b in ["base", "logic", "past", "future"] + bloques for c in fs[b]]
    return lab, te, F, {**PARAMS_FUT, **p, "n_jobs": 4}


def sub_atcos(atcos, k, f):
    rng = np.random.default_rng(10_000 + 100 * k + (f + 1))
    atcos = np.sort(np.asarray(atcos))
    return rng.choice(atcos, size=int(round(FRAC * len(atcos))), replace=False)


def fit(ks, tmp):
    lab, te, F, P = prep()
    for k in ks:
        t0 = time.time()
        oof = np.zeros(len(lab))
        for f in range(5):
            tr_at = lab.loc[lab.fold != f, "atco"].unique()
            sel = sub_atcos(tr_at, k, f)
            tr, va = lab.atco.isin(sel).values & (lab.fold != f).values, (lab.fold == f).values
            oof[va] = lgb_fit(lab[tr], F, {**P, "random_state": k}).predict(lab.loc[va, F])
        sel = sub_atcos(lab.atco.unique(), k, -1)
        pred = lgb_fit(lab[lab.atco.isin(sel).values], F, {**P, "random_state": k}).predict(te[F])
        np.save(tmp / f"oof_{k}.npy", oof); np.save(tmp / f"pred_{k}.npy", pred)
        print(f"modelo {k} listo ({time.time() - t0:.0f}s)", flush=True)


def ev(K, tmp):
    lab, te, F, P = prep()
    K = int(K)
    from metric import hits  # noqa
    oofs = [np.load(tmp / f"oof_{k}.npy") for k in range(K)]
    preds = [np.load(tmp / f"pred_{k}.npy") for k in range(K)]
    for k in range(K):
        print(f"  modelo {k} solo: 1.er pase {medir('', lab, oofs[k])[0]:.4f}", flush=True)
    oof, pred = np.mean(oofs, 0), np.mean(preds, 0)
    nm = f"f11_bagatco_k{K}"
    guardar(nm, lab, oof, te, pred)
    so, sp, _ = feat_session.sesion(lab, oof, te, pred)
    guardar(nm + "_ses", lab, so, te, sp)
    a, b = medir(nm, lab, oof), medir(nm, lab, so)
    for n_, (c, c23, mn) in ((nm, a), (nm + "_ses", b)):
        registrar(n_, hilo="f11", descripcion=f"N (f10_instr_l63) bagging por controladores: {K} modelos, {FRAC:.0%} de "
                  f"atcos de entrenamiento cada uno, semillas 0..{K - 1}", n_features=len(F),
                  extra=dict(cv_023=round(c23, 4), k=K, frac=FRAC))
    print(f"{nm}: 1.er pase {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) · + sesión {b[0]:.4f} "
          f"(CV 023 {b[1]:.4f}, peor {b[2]:.3f})", flush=True)


if __name__ == "__main__":
    tmp = Path(sys.argv[2]); tmp.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "fit":
        fit([int(x) for x in sys.argv[3:]], tmp)
    else:
        ev(sys.argv[3], tmp)

"""
metricas.py · RAMA FUTURO · Métricas que se registran en TODO experimento (sección 14 de LGB_futuro.ipynb).

La media de la CV sola puede elegir un modelo que no generaliza (secciones 12d y 14): unas pocas sesiones la mueven.
Por eso, para cada OOF:
    cv, folds, peor controlador (min_atco y cuál), mediana por sesión, CV por escenario (sc1, sc2, sc3)
y, frente a una referencia (otro OOF):
    d_cv, Δ mediana por sesión, sesiones que suben / bajan, Δ media y mediana por controlador, controladores que suben / bajan
Lo usan m_lgb_future.py (registro en experiments/lgb_future_runs.csv), futuro/pruebas/comparar.py y el notebook.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cv"))
from metric import hits, labels  # noqa: E402

_INFO = None


def info():
    """sample_id -> atco, fold, session, scenario (train + validation)."""
    global _INFO
    if _INFO is None:
        s = pd.concat([pd.read_csv(ROOT / "Data" / f"{f}.csv", usecols=["sample_id", "session", "scenario"])
                       .drop_duplicates("sample_id") for f in ("train", "validation")])
        _INFO = labels().merge(s, on="sample_id").set_index("sample_id")[["atco", "fold", "session", "scenario"]]
    return _INFO


def metricas(oof_df, ref_df=None):
    """oof_df / ref_df: sample_id,row_index,score. Devuelve un dict plano (para el registro)."""
    i = info()
    h = hits(oof_df).rename("hit")
    i = i.reindex(h.index)
    at, ses = h.groupby(i.atco).mean(), h.groupby(i.session).mean()
    m = dict(cv=h.mean(), min_atco=at.min(), worst_atco=int(at.idxmin()), ses_median=ses.median(),
             **{f"fold{k}": v for k, v in h.groupby(i.fold).mean().items()},
             **{f"sc{k}": v for k, v in h.groupby(i.scenario).mean().items()})
    if ref_df is not None:
        r = hits(ref_df).reindex(h.index)
        ds = ses - r.groupby(i.session).mean()
        da = at - r.groupby(i.atco).mean()
        m.update(d_cv=h.mean() - r.mean(), d_ses_median=ds.median(), ses_up=int((ds > 0).sum()),
                 ses_down=int((ds < 0).sum()), d_atco_mean=da.mean(), d_atco_median=da.median(),
                 atco_up=int((da > 0).sum()), atco_down=int((da < 0).sum()))
    return m


def resumen(name, m, ref=None):
    """Una o dos líneas legibles con las métricas de metricas()."""
    folds = "  ".join(f"f{k[4:]}:{v:.3f}" for k, v in m.items() if k.startswith("fold"))
    esc = " / ".join(f"{m[k]:.3f}" for k in ("sc1", "sc2", "sc3") if k in m)
    s = (f"{name}: CV top3 = {m['cv']:.4f}   (min atco {m['min_atco']:.3f}, atco {m['worst_atco']:03d})   "
         f"mediana por sesión {m['ses_median']:.3f}   escenarios {esc}   {folds}")
    if "d_cv" in m:
        s += (f"\n  frente a {ref}: Δ {m['d_cv'] * 100:+.2f} · Δ mediana por sesión {m['d_ses_median'] * 100:+.2f} · sesiones "
              f"suben {m['ses_up']} / bajan {m['ses_down']} · por controlador Δ media {m['d_atco_mean'] * 100:+.2f}, "
              f"mediana {m['d_atco_median'] * 100:+.2f}, suben {m['atco_up']} / bajan {m['atco_down']}")
    return s

"""
features_real.py · Pablo (A) · RAMA "REALISTA" · Features para un modelo que funcionaría en tiempo real: el snapshot
actual y snapshots PASADOS de la sesión. Nada del futuro y nada de etiquetas (ni el oráculo de futuro/).

Ni main (regla 1: solo el snapshot) ni futuro/ (usa snapshots posteriores). Reutiliza las funciones de
futuro/features_future.py y futuro/feat_geo.py sin tocarlas, y se queda solo con las columnas que miran atrás:
    base    columnas originales (las a_*, x_*, k_* del reto ya miran solo atrás)
    logic   reglas dentro del snapshot (logic_*)
    past    p_*, blk_pos, gap_prev            (sin blk_len ni gap_next: miran al resto de la racha)
    causal  cz_*: segundos desde el último cambio del estado que resuelve cada tarea (realista/feat_causal.py)
    geo     gxa_*, gxb_*, gxc_* de futuro/feat_geo.py menos gxc_exitfl_next (siguiente cambio = futuro)
    hist    h_*: tareas resueltas / nuevas deducidas de los snapshots anteriores (realista/feat_hist.py)
    ctx     ctx_*: contexto de la lista dentro del mismo snapshot (realista/feat_ctx.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "realista"), str(ROOT / "futuro")]
import features_future as ff  # noqa: E402
from feat_geo import add_geo  # noqa: E402
from feat_causal import add_causal  # noqa: E402
from feat_hist import add_hist  # noqa: E402
from feat_ctx import add_ctx  # noqa: E402

FUTURE_COLS = {"blk_len", "gap_next", "gxc_exitfl_next"}


def build():
    d = ff.load()
    d = ff.add_folds(d)
    d = ff.add_blocks(d)
    d = ff.add_logic(d)
    d = ff.add_past(d)
    d = add_geo(d)
    d = add_causal(d)
    d = add_hist(d)
    return add_ctx(d)


def feature_sets(d):
    pref = ("logic_", "p_", "cz_", "gx", "h_", "ctx_")
    base = [c for c in d.columns if c not in ff.META and not c.startswith(pref)
            and c not in ("blk_pos", "blk_len", "gap_prev", "gap_next")]
    sets = dict(base=base,
                logic=[c for c in d.columns if c.startswith("logic_")],
                past=[c for c in d.columns if c.startswith("p_")] + ["blk_pos", "gap_prev"],
                causal=[c for c in d.columns if c.startswith("cz_")],
                geo=[c for c in d.columns if c.startswith("gx") and c not in FUTURE_COLS],
                hist=[c for c in d.columns if c.startswith("h_")],
                ctx=[c for c in d.columns if c.startswith("ctx_")])
    assert not FUTURE_COLS & set(sum(sets.values(), [])), "columna que mira al futuro entre las features"
    return sets

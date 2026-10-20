"""
m_nn_future.py · Pablo (A) · RAMA "futuro" · Transformer de snapshot + información de otros snapshots de la sesión.

!! ROMPE LA REGLA 1 DEL EQUIPO A PROPÓSITO: usa snapshots anteriores y posteriores de la misma sesión. !!
Vive en futuro/ y guarda en futuro/oof|preds|submissions: stacking/stack.py solo lo coge con --futuro.

Es la misma red que nn_tf_v2 (src/m_nn.py, ls=0, layers=4, lr=5e-4) con más columnas por tarea, sacadas de
futuro/features_future.py (no usan etiquetas, solo la propia sesión):
- f_* (futuro): segundos hasta la siguiente instrucción al avión y su tipo, si encaja con el tipo de tarea, cuándo
  desaparece la tarea / el avión de la lista, si sigue en el siguiente snapshot.
- p_* (pasado) y blk_*/gap_*: cuánto lleva la tarea en la lista, cambios respecto al snapshot anterior, posición en la
  racha de snapshots contiguos.
Los folds siguen siendo Data/folds.csv (por controlador), así que la CV es comparable con el resto.

Uso:  python futuro/m_nn_future.py                    # 3 semillas, guarda oof/preds/submissions de nn_future
      python futuro/m_nn_future.py --quick --folds 0  # prueba rápida
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import m_nn, features_future as features   # versión con pasado/futuro

m_nn.ROOT = ROOT / "futuro"                 # salidas en futuro/oof|preds|submissions (DATA y EXP no cambian)

TEMPORAL_EXTRA = ["blk_pos", "blk_len", "gap_prev", "gap_next"]
PREFIJOS = ("f_", "p_")
if "--clear" in sys.argv:                   # + bloque clear (sx_*) de feat_clear.py
    sys.argv.remove("--clear")
    PREFIJOS += ("sx_",)
_load_base = m_nn.load


def temporal_cols(d):
    return [c for c in d.columns if c.startswith(PREFIJOS)] + TEMPORAL_EXTRA


def load_future():
    d = _load_base()                                     # columnas originales + fold de Data/folds.csv
    t = features.build()                                 # sesión completa: pasado y futuro
    t = t[["sample_id", "row_index"] + temporal_cols(t)]
    n = len(d)
    d = d.merge(t, on=["sample_id", "row_index"], how="left")
    assert len(d) == n
    print(f"+{len(temporal_cols(t))} columnas temporales: {temporal_cols(t)}", flush=True)
    return d.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def main():
    m_nn.load = load_future
    a = m_nn.parse(["--name", "nn_future", "--ls", "0", "--layers", "4", "--lr", "5e-4",
                    "--tag", "RAMA FUTURO: + features de otros snapshots"] + sys.argv[1:])
    m_nn.run(a, m_nn.build_data())


if __name__ == "__main__":
    main()


def build_data_future():
    """Para el notebook: como m_nn.build_data() pero con las columnas temporales (no toca la red limpia)."""
    m_nn.load = load_future
    try:
        return m_nn.build_data()
    finally:
        m_nn.load = _load_base

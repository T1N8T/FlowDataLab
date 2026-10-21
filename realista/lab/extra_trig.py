"""
extra_trig.py · bloque trig (tg_*) de Adri, para el banco de pruebas: qué disparó la racha actual (tiempos de la tarea, del
cambio de estado y de la última instrucción relativos al primer snapshot de la racha). Reutiliza fe_trig de
realista/claude_mejoras.py sin copiarlo. Causal: la racha empezó en el pasado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claude_mejoras import fe_trig  # noqa: E402


def add(d):
    o = fe_trig(d)
    d = d.copy()
    for c in o.columns:
        d[c] = o[c].values
    return d, list(o.columns)

#!/bin/bash
# instr + 63 hojas + extra_trees con 10 semillas -> sesión -> suavizado con CV anidada -> comparar.py frente a A -> envío
cd "$(dirname "$0")/../.."
P=.venv/bin/python
$P futuro/pruebas/f10_instr.py cv f10_instr_l63 10 && \
$P futuro/pruebas/f5_stack.py suav f10_instr_l63_s10_ses && \
$P futuro/pruebas/comparar.py suav_lgb_future_clear10_ses suav_f10_instr_l63_s10_ses f10_instr_l63_s10_ses && \
$P -c "
import sys, pandas as pd; sys.path.insert(0,'cv'); from metric import save_outputs; from pathlib import Path
O=Path('futuro/pruebas')
for n in ('suav_f10_instr_l63_s10_ses','f10_instr_l63_s10_ses'): save_outputs(n, None, pd.read_csv(O/'preds'/f'{n}.csv'), out=O)
" && echo FIN

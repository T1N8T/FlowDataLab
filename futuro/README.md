# futuro/ · modelos con información de otros snapshots

**Rompen a propósito la regla 1** (cada snapshot solo con sus filas). Están aparte por si el organizador lo permite.
Todo lo de aquí guarda en `futuro/oof`, `futuro/preds` y `futuro/submissions`, así que `stacking/stack.py` sin
opciones (el flujo de `main`) no los ve. Los folds siguen siendo `Data/folds.csv`: la CV es comparable.

| Archivo | Qué es | Salida | CV |
|---|---|---|---|
| `features_future.py` | features `f_*` (futuro), `p_*`/`blk_*`/`gap_*` (pasado) de la sesión | — | — |
| `temporal.py` | `add_temporal(d)`: añade esas columnas a cualquier modelo | — | — |
| `m_lgb_future.py` | LightGBM LambdaRank (Pablo) | `lgb_future` | 0.901 |
| `m_nn_future.py` | transformer `nn_tf_v2` + columnas temporales (Pablo) | `nn_future` | 0.887 |
| `linear/m_linear.py --future` | logit condicional + columnas temporales (Juanma) | `linear_future` | 0.881 |
| `hierarchical/m_types.py --future` | tipos × tarea + columnas temporales (Juanma) | `types_future` | — |
| `stacking/stack.py --futuro` | stacking con los modelos de `oof/` y de `futuro/oof/` | `stack_avg`, `stack_meta` | 0.903 |

`clogit_future` (0.835) está en `futuro/oof|preds`, pero su código no está en el repo.

Uso (desde la raíz):
```
python futuro/m_lgb_future.py
python futuro/m_nn_future.py
python linear/m_linear.py --future
python stacking/stack.py --futuro      # -> futuro/submissions/sub_stack_avg.csv, sub_stack_meta.csv
```
Notebooks: `LGB_futuro.ipynb` (LightGBM con futuro, Pablo + Juanma; arriba el punto de partida actual), `NN_futuro_pablo.ipynb` (el de la red es `notebooks/NN_pablo.ipynb` + sección 8).

**Hilo f11 (Pablo, 21/10, sección 23 de `LGB_futuro.ipynb`)**, todo en `futuro/pruebas/`: `f11_lab.py` (banco de pruebas sobre
N: features, parámetros, oráculo, semillas, `--test`), `f11_cola2.py` (cola dinámica), `f11_feats.py` (bloques nuevos),
`f11_mezcla.py` / `f11_promedio.py` (mezcla aritmética y media de semillas), `f11_privado.py` (privado esperado y mejor de 2),
`f11_check.py` (valida un envío), `f11_tabla.py` (tabla del registro). Envíos finales: `futuro/pruebas/submissions/sub_f11_M30_30.csv`
(M30 = 0.7·C30 + 0.3·N30, CV 0.9290) y `sub_f10_instr_l63_s10_ses.csv` (N, público 0.951).

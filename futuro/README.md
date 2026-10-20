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
Notebooks: `LGB_futuro_pablo.ipynb`, `NN_futuro_pablo.ipynb` (el de la red es `notebooks/NN_pablo.ipynb` + sección 8).

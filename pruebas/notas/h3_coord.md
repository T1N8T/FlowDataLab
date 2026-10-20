# h3_coord: features dentro del snapshot para coordinaciones y tipos raros

Código: `pruebas/h3_coord.py` (grupos `fe_coord`, `fe_avion2`, `fe_carga`, `fe_inter`). Todos agregan solo por sample_id, (sample_id, cs) o (sample_id, task_index).
Referencia h3_ref (99 features de lgb_adri, 1 semilla): CV 0.6596, min atco 0.576, exit_coord 0.083, entry_coord 0.386.

| grupo (+ref, 1 semilla) | CV | min atco | exit_coord | entry_coord |
|---|---|---|---|---|
| coord (urgencia dentro del tipo + estado de coordinación) | 0.6600 | 0.572 | 0.120 | 0.377 |
| avion2 (estado del avión, nº de tareas, rank del avión por último input) | 0.6515 | 0.582 | 0.110 | 0.354 |
| carga (nº de urgentes del snapshot, "nada urgente") | 0.6465 | 0.559 | 0.086 | 0.350 |
| inter (tipo × x_task_age / a_since_flight / tts) | 0.6556 | 0.576 | 0.136 | 0.343 |

- Ningún grupo mejora la CV de forma clara: `coord` empata (+0.0004, dentro del ruido de semilla) y es el único que no baja nada.
  Sube exit_coordination de 0.08 a 0.12, pero a cambio pierde un poco en entry_coord y conflict.
- `avion2`, `carga` e `inter` hunden la CV (de -0.004 a -0.013): el modelo ya ve esa información a través de las columnas originales, y las features extra le hacen sobreajustar al controlador.
- La EDA de exit_coordination muestra poca señal propia: el target se distingue sobre todo porque es el avión "activo"
  (más tareas del avión, a_since_flight menor, k_outCoordMismatch / k_coordinationToolActivation más altos, exitConditionsAreMet más bajo).
  Es decir, el problema es elegir el avión, no la tarea dentro del tipo.
- Los grupos exit_coord/deadline/tipo_estado de base.py no llegaron a medirse (se acabó el tiempo).
- **h3_coord_best** = ref + coord, 2 semillas, con preds de test (pruebas/oof|preds/h3_coord_best.csv): CV **0.6637**, min atco 0.568, exit_coord 0.122, entry_coord 0.377, exit_conditions 0.210.
  Frente a lgb_adri (0.6569 con 2 semillas) son +0.007, pero h3_ref con 1 semilla ya daba 0.6596: parte de la mejora puede ser ruido de semilla. Habría que confirmarlo con un h3_ref de 2 semillas.
- Recomendación: añadir solo `fe_coord` (15 features `c3_*`) a lgb_adri, una vez confirmado con 2 semillas contra la referencia, y usar h3_coord_best en el stacking. El precio es min atco -0.004 a -0.008. Descartar avion2, carga e inter.

# f3_sel · selección de variables con futuro (script `futuro/pruebas/f3_sel.py`, tabla en `notas/f3_cv.csv`)

- Referencia lgb_future (125 col., 3 semillas): CV 0.9010, min_atco 0.769, std 0.022. f3_ref (1 semilla): 0.8996 / 0.746 / 0.023.
- Bloques (1 semilla): sin future 0.661 (todo el modelo); solo future+past+task_index (24 col.) 0.876; sin logic 0.897; sin past 0.897.
  Prefijos de base: sin a_* 0.902, sin l_* 0.900 (min 0.763), sin x_* 0.900, sin k_* 0.894, sin resto (originales) 0.896.
- Importancias (gain + permutación sobre el 20% de los snapshots OOF) en `notas/f3_importancias.csv`. Mandan task_index, f_end_in_block,
  f_next_input, f_task_end, logic_edf_rank_type, logic_window_dist. Unas 15 columnas tienen importancia 0 (p. ej. is_not_assumed, is_transferred, a_last_decontrol).
  Quitar el 30% / 50% menos importante: 0.898 / 0.896 → baja algo, no compensa.
- Validación adversaria (lab vs test 020/023): **AUC 0.97**. Delatan el volumen de tráfico de la sesión, no la tarea:
  a_n300, seq_len, blk_len, logic_n_window, k_outCoordRevision, a_n60, x_state_age, k_groundSpeed, k_clearedFL, l_frac_type (`notas/f3_adversaria.csv`).
  Sin las 5: 0.9008; sin las 10: 0.8997 con min 0.769 y std 0.018 (1 semilla).
- Con 3 semillas + preds: f3_best (sin las 10 delatoras, 115 col.) 0.9000 / 0.734 / 0.018; f3_best_sin_a (sin a_*, 113) 0.9000 / 0.749 / 0.021.
- Conclusión: ninguna selección mejora a lgb_future. Las diferencias de 1 semilla (±0.002 y min_atco ±0.03) son ruido. El modelo es
  robusto a quitar bloques de base: lo que cuenta es future (+ logic/past, ~0.003 cada uno). Quitar las 10 delatoras da la misma CV con menos
  varianza entre folds y quita el desplazamiento de covariables hacia test. Es una variante "segura" razonable para un stack, pero no hay pruebas de que mejore al 023.

# h4_sel · selección de variables y deriva entre controladores (solo snapshot)

Script: `pruebas/h4_sel.py`. Importancias (gain medio por fold y permutación sobre el 50 % de los snapshots de cada fold): `pruebas/notas/h4_importancias.csv`. Adversaria: `h4_adversaria.csv` y `h4_adv_auc.json`.

| nombre | CV | min atco | std folds | n_feat |
|---|---|---|---|---|
| h4_ref (99 features, 1 semilla) | 0.6596 | 0.576 | 0.025 | 99 |
| h4_sin_low20 (sin el 20 % menos importante) | 0.6591 | 0.580 | 0.024 | 79 |
| h4_sin_low40 | 0.6467 | 0.577 | 0.018 | 59 |
| h4_sin_deriva (sin a_n300, seq_len y cx_n_aircraft/entry_coord/assume_offer/zone) | 0.6521 | 0.571 | 0.021 | 93 |
| h4_sin_low20_n (low20 + a_n300, a_n60, seq_len) | 0.6596 | 0.591 | 0.023 | 76 |
| **h4_sel_best** (= low20_n, 2 semillas, con preds) | **0.6613** | 0.584 | 0.023 | 76 |

- Las 5 más importantes son task_index, distance_to_sector, time_to_sector, x_task_age y k_timeToExitPoint. Unas 20 columnas tienen permutación ≤ 0 y gain casi nulo; is_transferred, k_trackAgeS y a_last_decontrol tienen gain 0.
- Validación adversaria: AUC de test frente a lab 0.996, de 023 frente a lab 0.999 y de cada controlador frente al resto entre 0.998 y 1.000. El controlador se separa casi perfectamente. Delatan sobre todo las constantes por snapshot (a_n300, seq_len, los cx_n_*, a_n60), que reflejan la carga de tráfico de la sesión, y después k_outCoordRevision, a_since_any y x_state_age.
- Quitar los cx_n_* baja la CV (−0.0075): sirven como contexto de interacción. Quitar a_n300, a_n60 y seq_len no cuesta nada y sube min_atco.
- No llegué a probar la sustitución por rangos dentro del snapshot ni las ablaciones por prefijo, porque el plan cambió.

**Selección recomendada (quitar 23 columnas, lista en `h4_quitar.json`):** conflict_ind, a_last_assign_level_clearance, k_isMaintainingFL, k_isPassingCFL, CPA_exit_location, multiple_conflicts_ind, k_isClearedFLAssigned, k_out_COORDINATED, al_conflict_stca, k_isAtCFL, al_conflict_multi, k_isDescendingToCFL, k_isLocked, is_transferred, k_trackAgeS, a_last_decontrol, CPA_entry_location, k_inSectorVertically, x_entry_coord_not_done, what_if_potential_conflict, a_n300, seq_len, a_n60.

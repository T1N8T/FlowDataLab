# h1_gbm · CatBoost y XGBoost (solo snapshot)

Script: `pruebas/h1_gbm.py <variante> [--test]`. Las 99 features de `build(lab)` (lgb_adri), folds de Data/folds.csv, 1 semilla, 2 hilos.

| nombre | CV top-3 | peor atco | preds | tiempo |
|---|---|---|---|---|
| h1_xgb_ndcg | 0.6502 | 0.573 | sí | 42 min (CV+full, máquina cargada) |
| h1_cat_yeti | pendiente | – | – | ~21 min/fold → ~2.5 h en total |

- XGBRanker rank:ndcg (topk, 8 pares por muestra, ndcg_exp_gain=False), 600 it, lr 0.05, depth 6, subsample/colsample 0.8, task_index como category.
  Queda 0.007 por debajo de lgb_adri (0.6569), con un peor controlador parecido (0.573 frente a 0.570). Puede sumar en el stacking si su correlación con lgb_adri es menor; falta medirlo.
- CatBoostRanker YetiRank (800 it, lr 0.05, depth 6, task_index categórica): con 2 hilos y la máquina cargada (load ~10) cada fold tarda ~21 min.
  No acabó dentro del presupuesto. Se dejó corriendo (PID 161187); si termina, se registra solo como h1_cat_yeti con preds.
  YetiRankPairwise/QuerySoftMax y xgb rank:pairwise no se probaron por falta de tiempo (están en el script).
- Conclusión: XGBoost no supera al LightGBM. Su valor está en la diversidad para el stacking. CatBoost es demasiado lento con 2 hilos en 475k filas; para probarlo, mejor 300 it con lr 0.1 o más hilos.

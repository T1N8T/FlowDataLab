# f1_lgb · variantes de LightGBM sobre features de futuro (125 cols: base+logic+past+future)
Script: `futuro/pruebas/f1_lgb.py`. CV con Data/folds.csv; 1 semilla salvo f1_best_* (3 semillas + preds de test).
- Referencia f1_ref (PARAMS_FUT, 1 semilla): 0.8996 (min atco 0.746, std folds 0.023). lgb_future con 3 semillas: 0.9010 / 0.769.
- Objetivos: lambdarank sigue siendo el mejor. binary 0.894, rank_xendcg 0.886, truncation 3 0.894, truncation 10 0.899 (igual que 5).
- Capacidad: lr 0.03 + 800 árboles 0.9006 (nada); leaves 15 + mc200 + L2 5 + colsample 0.5 0.898; leaves 63 + mc100 0.9028 con 1 semilla pero min atco 0.728.
- **extra_trees=True**: 0.9016 con 1 semilla, el que más mejora en min atco (0.760) y std folds (0.015).
- 3 semillas + test: **f1_best_extratrees 0.9035** (min atco 0.769, std 0.013) > lgb_future 0.9010. f1_best_l63_mc100 0.9013 (min 0.749, std 0.018): su 0.9028 de 1 semilla era ruido.
- Medias de rangos (OOF): extratrees+binary 0.9018, +trunc10 0.9020, best_extratrees+best_l63 0.9046 pero min atco 0.754. Mezclar suma +0.001, poco.
- Por tipo (best_extratrees): exit_coord 0.57, entry_coord 0.71, return_to_route 0.86, zone 0.93: los tipos difíciles no cambian; la ganancia viene de conflict_resolution/qos/zone.
- Recomendación: PARAMS_FUT + extra_trees=True (3 semillas) como sustituto de lgb_future en el stack. oof/preds en futuro/pruebas/oof|preds/f1_best_extratrees.csv.
- No probado (se quedó sin tiempo): extra_trees con 800 árboles (`f1_et_800` está en el script).

# Búsqueda nocturna del modelo realista · 21/10 03:45

Base: `real_todo` de Pablo (snapshot + pasado; bloques base, logic, past, causal, geo). Folds de `Data/folds.csv`, métricas de `futuro/metricas.py`. *CV 023* = controladores parecidos al 023 (Juanma). Se relanza igual si se corta: las semillas ya calculadas se leen de `realista/noche/oof|pred`.

## 1 · Referencia

| modelo | CV | CV 023 | peor controlador | mediana por sesión | Δ CV vs ref | sesiones suben/bajan | controladores suben/bajan |
|---|---|---|---|---|---|---|---|
| ref_3s | 0.6697 | 0.6703 | 0.594 | 0.6799 | — | — | — |

(Pablo dio 0.6703 con estos mismos parámetros y 3 semillas.)

## 2 · Búsqueda de parámetros (1 semilla)

63 pruebas propias (mutaciones de las mejores) + 19 del Optuna de Pablo. Las 10 mejores:

| prueba | origen | CV (1 semilla) | parámetros |
|---|---|---|---|
| b030 | búsqueda | 0.6748 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 37, "lr": 0.01864, "max_bin": 63, "min_child": 16, "min_gain": 0.0005, "n_estimators": 1010, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 7}` |
| b056 | búsqueda | 0.6733 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 36, "lr": 0.02564, "max_bin": 63, "min_child": 25, "min_gain": 0.0005, "n_estimators": 539, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 8}` |
| b059 | búsqueda | 0.6728 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 37, "lr": 0.02312, "max_bin": 63, "min_child": 16, "min_gain": 0.0005, "n_estimators": 1010, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 5}` |
| b045 | búsqueda | 0.6721 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 37, "lr": 0.01864, "max_bin": 63, "min_child": 25, "min_gain": 0.0005, "n_estimators": 756, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 8}` |
| b013 | búsqueda | 0.6718 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": true, "leaves": 37, "lr": 0.02503, "max_bin": 63, "min_child": 16, "min_gain": 0.00079, "n_estimators": 899, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 7}` |
| b060 | búsqueda | 0.6717 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 37, "lr": 0.01985, "max_bin": 127, "min_child": 16, "min_gain": 0.0005, "n_estimators": 1010, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 7}` |
| b062 | búsqueda | 0.6717 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 47, "lr": 0.02564, "max_bin": 63, "min_child": 25, "min_gain": 0.0005, "n_estimators": 553, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 8}` |
| b035 | búsqueda | 0.6713 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 30, "lr": 0.01553, "max_bin": 63, "min_child": 16, "min_gain": 0.00059, "n_estimators": 1010, "reg_lambda": 15.91728, "subsample": 0.85102, "trunc": 7}` |
| b012 | búsqueda | 0.6711 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": true, "leaves": 37, "lr": 0.01864, "max_bin": 63, "min_child": 16, "min_gain": 0.00079, "n_estimators": 1103, "reg_lambda": 21.55722, "subsample": 0.93956, "trunc": 7}` |
| b034 | búsqueda | 0.6710 | `{"cat_smooth": 1.50382, "colsample": 0.77206, "extra_trees": false, "leaves": 30, "lr": 0.01553, "max_bin": 63, "min_child": 16, "min_gain": 0.0005, "n_estimators": 1010, "reg_lambda": 11.39067, "subsample": 0.93956, "trunc": 7}` |

## 3 · Confirmación con 3 semillas (frente a `real_todo` con 3 semillas)

| modelo | CV | CV 023 | peor controlador | mediana por sesión | Δ CV vs ref | sesiones suben/bajan | controladores suben/bajan |
|---|---|---|---|---|---|---|---|
| b045_3s | 0.6732 | 0.6719 | 0.601 | 0.6794 | +0.0036 | 18/15 | 12/6 |
| b030_3s | 0.6731 | 0.6675 | 0.594 | 0.6694 | +0.0034 | 19/13 | 10/7 |
| b056_3s | 0.6721 | 0.6690 | 0.597 | 0.6806 | +0.0024 | 18/15 | 12/5 |
| b059_3s | 0.6717 | 0.6699 | 0.593 | 0.6749 | +0.0020 | 19/17 | 9/9 |
| b060_3s | 0.6714 | 0.6673 | 0.594 | 0.6748 | +0.0017 | 17/17 | 7/8 |
| b013_3s | 0.6683 | 0.6602 | 0.591 | 0.6718 | -0.0014 | 16/19 | 9/8 |

## 4 · Mezclas (media de z-scores en el snapshot, pesos iguales)

| modelo | CV | CV 023 | peor controlador | mediana por sesión | Δ CV vs ref | sesiones suben/bajan | controladores suben/bajan |
|---|---|---|---|---|---|---|---|
| mezcla_top2 | 0.6737 | 0.6727 | 0.596 | 0.6721 | +0.0040 | 17/16 | 11/6 |
| mezcla_top3 | 0.6740 | 0.6736 | 0.598 | 0.6776 | +0.0043 | 18/17 | 12/5 |
| mezcla_top4 | 0.6734 | 0.6719 | 0.597 | 0.6776 | +0.0037 | 19/15 | 12/6 |
| mezcla_top5 | 0.6742 | 0.6730 | 0.597 | 0.6775 | +0.0046 | 18/15 | 11/7 |

**Elegida para la etapa 6:** `mezcla_top5` (mezcla, CV 0.6742). Criterio: la de mayor CV entre las que suben la CV ≥ 0.002 y suben más sesiones de las que bajan frente a la referencia (si ninguna, la referencia).

## 5 · Extras sobre la mejor configuración (2 semillas, frente a ella misma)

| modelo | CV | CV 023 | peor controlador | mediana por sesión | Δ CV vs ref | sesiones suben/bajan | controladores suben/bajan |
|---|---|---|---|---|---|---|---|
| b045_2s | 0.6735 | 0.6716 | 0.605 | 0.6785 | — | — | — |
| b045+hist_2s | 0.6676 | 0.6631 | 0.590 | 0.6710 | -0.0059 | 11/19 | 7/10 |
| b045+instr_atencion_2s | 0.6681 | 0.6635 | 0.598 | 0.6686 | -0.0055 | 9/21 | 4/10 |
| b045+peso_tramo_2s | 0.6711 | 0.6703 | 0.586 | 0.6763 | -0.0024 | 17/18 | 7/10 |

**Extra que se queda:** ninguno.

## 6 · Versión final

Miembros: `b045`, `b030`, `b056`, `b059`, `b060` · extra: — · peso por tramo: False · 4 semillas por miembro · segundo paso causal de Pablo (pesos {'lsm': np.float64(2.569), 'lPa': np.float64(0.121), 'p1': np.float64(-1.382), 'p2': np.float64(-0.905), 'Mp': np.float64(-5.008), 'Ep': np.float64(0.28), 'Pprev': np.float64(1.979), 'Pca': np.float64(-2.35), 'Mcp': np.float64(0.994), 'top_prev': np.float64(0.004)}).

| modelo | CV | CV 023 | peor controlador | mediana por sesión | Δ CV vs ref | sesiones suben/bajan | controladores suben/bajan |
|---|---|---|---|---|---|---|---|
| real_adri_noche | 0.6732 | 0.6734 | 0.595 | 0.6743 | +0.0036 | 17/18 | 12/6 |
| real_adri_noche_ses | 0.6722 | 0.6719 | 0.595 | 0.6763 | +0.0025 | 17/17 | 12/6 |

**Para Kaggle:** `realista/submissions/sub_real_adri_noche_ses.csv` (o `sub_real_adri_noche.csv` si el segundo paso no mejora). Submission comprobada: True.

## Resumen

Referencia `real_todo` (3 semillas): **0.6697** → versión final: **0.6732**, **0.6722** con el segundo paso. Tiempo total: 2.9 h.


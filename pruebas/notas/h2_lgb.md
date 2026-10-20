# h2_lgb · variantes de LightGBM sobre las 99 features de lgb_adri (1 semilla, solo OOF)

| variante | CV | min atco | std folds |
|---|---|---|---|
| h2_ref (lambdarank, PARAMS) | 0.6596 | 0.576 | 0.025 |
| h2_trunc3 | 0.6574 | 0.564 | 0.030 |
| h2_nonorm | 0.6514 | 0.546 | 0.025 |
| h2_binary (LGBMClassifier) | 0.6485 | 0.568 | 0.020 |
| h2_xendcg | 0.6392 | 0.566 | 0.022 |
| mezcla rangos ref+binary | 0.6524 | 0.578 | 0.023 |
| mezcla rangos ref+trunc3+binary | 0.6594 | 0.585 | 0.022 |

Pendientes (proceso en marcha, se registran solos en resultados.jsonl): reg, slow, big, extra, smooth.

Conclusión provisional: ningún objetivo alternativo mejora a lambdarank con truncation 5 (la referencia).
binary es peor en media pero el más estable entre folds; mezclado con ref+trunc3 sube el peor controlador
(0.585 frente a 0.576) a igual CV. Config recomendada por ahora: la de lgb_adri sin cambios.
Sin preds de test de 3 semillas todavía (no dio tiempo).

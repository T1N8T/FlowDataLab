# Plan de comprobaciones ISLR (rama `pruebas-islr`) · para ejecutar en un clúster

## Contexto

Reto AWARE (Kaggle, terminado): ordenar las tareas abiertas de un controlador aéreo y acertar en el top 3 la que está
trabajando. La métrica es el acierto top-3. El privado es un solo controlador (el 023). La entrega del equipo fue M30_30,
con CV 0.929, público 0.953 y privado 0.912.

La pregunta de esta rama: **si hubiéramos seguido el libro (ISLR) y una regla explícita de "solo adoptar lo que supera al
ruido", ¿qué habríamos enviado y qué nota habría sacado?** Y, de todo lo que probamos, ¿qué era información real y qué era
ruido de buscar mucho con los mismos 18 controladores?

Documentos y código:
- `entrega/justificacion_islr.md`: cada decisión del proyecto con su sección y página del ISLR.
- `entrega/codigo/islr_pruebas.py`: pruebas estadísticas por controlador sobre los OOF ya hechos. Ya ejecutado; salidas en
  `salidas/notas/islr_P*.csv`.
- `entrega/codigo/islr_escalera.py`: la escalera de modelos del libro (14 peldaños), stacking y regla de 1 SE.
- `entrega/codigo/islr_etapas.py`: la regla de adopción aplicada a la cadena histórica (`--historico`) y el recorrido por
  etapas con protocolo fijo (`--etapas`).

## Resultados ya obtenidos

- **Pruebas sobre los OOF.** Solo son significativos (con Holm) el salto al futuro (+21 puntos) y el segundo paso de sesión
  (+0.8). Entre A, B, N, C y M30_30 nada es significativo. La diferencia mínima detectable con 18 controladores es de ~1
  punto.
- **Caída de CV a privado.** Las ideas que añaden información se sostienen (lgb_future −0.1, A +0.2, OC +0.8). Los ajustes
  finales, que se encontraron tras ~620 configuraciones, pierden lo ganado (N −1.5, NOR −2.2, M30_30 −1.7). Es la señal de
  sobreajuste a la CV.
- **`--historico`.** La regla (Δ > σ·√(2 ln K) y permutación con Bonferroni) se queda en **B** (A + oráculo de
  coordinaciones) tanto con K = 620 como con K = 9. Su privado ya se conoce: **0.922**, mejor que la entrega (0.912).
- **Escalera, comprobación rápida** (fold 0, 20 % del train): el boosting es el mejor peldaño con snapshot (0.66) y con
  futuro (0.94). Los lineales se quedan en ~0.59 y ~0.84.

## Requisitos

- Python 3.12 con numpy, pandas, scipy, scikit-learn y lightgbm (`requirements.txt`).
- **Datos de Kaggle, que no están versionados:** copiar `train.csv`, `validation.csv` y `test.csv` a `entrega/datos/`.
- Las cachés (`entrega/cache/`) se generan solas la primera vez. **Hay que generarlas antes de lanzar trabajos en paralelo,**
  para que no se pisen al escribirlas:
  ```
  python entrega/codigo/features_futuro.py      # caché de variables con futuro (~1–3 min)
  python entrega/codigo/oraculo.py              # caché del oráculo (~1 min)
  python entrega/codigo/features_realista.py    # solo si se lanza --info realista
  ```

## Comprobaciones (cola). J1–J3 son independientes y pueden ir en paralelo

| # | Comando | Qué responde | Duración aprox. (12 núcleos) | Salidas |
|---|---|---|---|---|
| J1 | `python entrega/codigo/islr_escalera.py --info futuro --test --stack` | Siguiendo el libro con nuestras variables de futuro, ¿qué modelo se elige (regla de 1 SE) y qué CV tiene cada peldaño? | 6–10 h (ridge, lasso y GAM con CV anidada) | `salidas/notas/islr_escalera_futuro.csv`, `salidas/oof/escalera_futuro_*.csv`, `salidas/submissions/sub_escalera_futuro_*.csv` |
| J2 | `python entrega/codigo/islr_escalera.py --info snapshot --test --stack` | Lo mismo, solo con las columnas originales: el libro "tal cual", sin construir variables | 4–6 h | `islr_escalera_snapshot.csv`, `sub_escalera_snapshot_*.csv` |
| J3 | `python entrega/codigo/islr_etapas.py --etapas` | Recorrido por etapas con protocolo fijo y regla de adopción: qué ideas sobreviven y qué modelo queda | 3–6 h | `salidas/notas/islr_etapas.csv`, `salidas/submissions/sub_etapas_final.csv` |
| J4 (opcional) | `python entrega/codigo/islr_escalera.py --info realista --test --stack` | La escalera solo con pasado (asistente en tiempo real) | 4–6 h | `islr_escalera_realista.csv` |

Los tres scripts reutilizan lo ya calculado (OOF en `salidas/oof`, caché en `entrega/cache/etapas/`): si se cortan, al
relanzarlos continúan. `--rapido` hace una comprobación corta (fold 0) sin guardar.

### Detalle de J3 (`islr_etapas.py --etapas`)
Protocolo fijo: LightGBM LambdaRank con 400 árboles, lr 0.05, 31 hojas, min_child 50, subsample y colsample 0.8, 3
semillas, CV por controlador (`datos/folds.csv`). No se ajusta nada.

Regla, fijada de antemano: se adopta si Δ por controlador > σ_Δ·√(2 ln 30) **y** permutación con cambio de signo
p < 0.05/30.

- **E0 · ruido:** el modelo base con 3 grupos de semillas → σ_Δ y el umbral.
- **E1 · información**, en orden de prioridad: logic → past → future → clear → instr → oráculo. Para cada bloque, además:
  - la misma prueba con un **modelo simple sin ajustar** (logística condicional con λ fijo): si también mejora, es
    información y no ajuste;
  - una **nula por permutación**: el bloque barajado entre snapshots, 3 veces.
- **E2 · estructura:** el segundo paso de sesión.
- **E3 · capacidad:** 63 hojas + extra_trees.
- **E4 · conjunto:** 10 semillas y, si el oráculo entró, la mezcla con/sin oráculo en probabilidades (peso por CV anidada).
- **E5 · comprobación híbrida** (ISLR §6.1.2): quitar cada bloque adoptado y reintentar cada rechazado sobre el modelo
  final, para no dejarse una opción por el orden de entrada.
- **Final:** reentreno con todo y `sub_etapas_final.csv`.

## Qué hacer con los resultados

1. Para cada trabajo terminado, anotar en la sección **Resultados del clúster** (abajo):
   - J1/J2: la tabla final que imprime el resumen (CV, media por controlador, SE, peor controlador) y la línea
     `regla de 1 SE ⇒ …`.
   - J3: la tabla de decisiones (`islr_etapas.csv`) y la línea `MODELO FINAL`.
2. **Envíos tardíos a Kaggle** (si la competición los admite), sobre todo:
   - `sub_escalera_futuro_<elegido>.csv`
   - `sub_escalera_snapshot_<elegido>.csv`
   - `sub_etapas_final.csv`
   - y, como referencia, `sub_escalera_futuro_stack.csv`.

   Anotar el público y el privado de cada uno. El privado sirve para comprobar, no para volver a elegir: elegir mirando el
   privado es otra vez el problema de las comparaciones múltiples.
3. Commitear en la rama `pruebas-islr`: `salidas/notas/islr_*.csv`, `salidas/submissions/sub_escalera_*`, `sub_etapas_final.csv`
   y este documento con los resultados. Los OOF y preds de `salidas/oof` y `salidas/preds` también pesan poco y se pueden
   commitear. **No** las cachés (`entrega/cache/`).

## Resultados del clúster

_(por rellenar)_

# Entrega · FlowDataLab (AWARE Prioritisation Challenge, Engage 2 Zagreb)

Todo lo necesario para entender, reproducir y explicar las dos entregas, **ejecutable solo con esta carpeta**.

| Entrega | Qué información usa | Archivo | CV | Público (020) |
|---|---|---|---|---|
| **Futuro** | el snapshot, el pasado y el futuro de la sesión, y lo que hicieron otros controladores (oráculo) | `salidas/submissions/sub_f11_M30_30.csv` | 0.929 | 0.953 |
| **Realista** | solo el snapshot actual y los snapshots pasados (usable en tiempo real) | `salidas/submissions/sub_real_adri_noche.csv` | 0.673 | — |

La clasificación final de Kaggle es **solo el privado** (controlador 023). El público (controlador 020) solo orienta.

## Cómo se usa

```bash
# 1. datos de Kaggle: copiar train.csv, validation.csv y test.csv a entrega/datos/ (folds.csv, sample_submission.csv y scorer.py ya están)
# 2. entorno: Python 3.12 + numpy, pandas, scipy, scikit-learn, lightgbm, matplotlib, jupyter
# 3. notebooks (no entrenan: leen las salidas ya calculadas; la primera vez calculan y guardan las features en cache/, ~1–5 min)
jupyter nbconvert --to notebook --execute --inplace entrega/futuro.ipynb
jupyter nbconvert --to notebook --execute --inplace entrega/realista.ipynb
# 4. reentrenar las entregas desde cero (opcional; horas con 30 semillas)
python entrega/codigo/entrenar_futuro.py N --semillas 30
python entrega/codigo/entrenar_futuro.py C --semillas 30
python entrega/codigo/entrenar_futuro.py mezcla            # M30_30 = 0.7·C30 + 0.3·N30
python entrega/codigo/entrenar_realista.py                 # real_adri_noche
python datos/scorer.py --help                               # scorer oficial
```

| Carpeta / archivo | Qué hay |
|---|---|
| `futuro.ipynb` | la línea con futuro: datos, técnicas explicadas desde cero, de dónde sale la nota, por qué funciona, dónde falla, cuánto azar hay y cómo se eligió la entrega |
| `realista.ipynb` | la línea sin futuro: qué usa, cada bloque, la cadena de CV y por qué se eligió `real_adri_noche` |
| `codigo/` | solo el código de los modelos entregados: `comun.py` (datos, métrica, medidas), `features_futuro.py`, `oraculo.py`, `sesion.py`, `entrenar_futuro.py`, `features_realista.py`, `entrenar_realista.py`, `eleccion.py` |
| `salidas/` | OOF (predicciones fuera de fold sobre train+validation), preds de test y envíos de las entregas y de los modelos necesarios para explicar las decisiones |
| `datos/` | `folds.csv` (folds fijos por controlador), `sample_submission.csv`, `scorer.py`; aquí van los CSV de Kaggle |

---

## Lo que hay que entender y explicar

### 1 · El problema
- Qué es un **snapshot** (la lista de ~36 tareas abiertas del controlador en un instante) y qué se predice (las 3 tareas más probables
  de ser la que está trabajando). Por qué es un problema de **ordenar dentro de una lista** y no de clasificar filas sueltas.
- La métrica **top-3 hit rate** y sus referencias: azar 0.094, ejemplo oficial 0.352.
- **Público frente a privado:** son dos controladores distintos (020 y 023); la clasificación es solo el privado; el público son
  2 sesiones y engaña (el organizador avisó de sobreajuste al público).

### 2 · Cómo se valida
- **Validación cruzada por controlador** con folds fijos (`datos/folds.csv`): cada controlador se predice con un modelo que no lo vio,
  igual que el 023. Por qué un reparto aleatorio daría una nota inflada.
- **OOF**, **CV anidada** (elegir pesos o parámetros sin mirar el fold que se mide) y **semillas** (cuánto ruido hay: ±0.1–0.2).
- El **criterio robusto**: no solo la media, también la mediana por sesión, cuántos controladores suben o bajan, el peor
  controlador y la "CV 023" (controladores parecidos al 023).

### 3 · Los datos (`futuro.ipynb`, secciones 1–1d)
- El **lift por tipo de tarea** (transfer y assume se eligen mucho; las coordinaciones aparecen mucho y se eligen poco).
- La **estructura de las etiquetas**: dentro de una racha el controlador repite la tarea (95 %); tras un hueco casi nunca (0.2 %);
  cada tarea se trabaja en una sola racha (98 %).
- Qué variables separan la tarea elegida, cómo se relacionan las familias, el **embedding t-SNE** (por qué las coordinaciones son
  el techo), la **jerarquía entre tipos** y la **curva de la siguiente instrucción**.

### 4 · Por qué funciona el modelo con futuro (`futuro.ipynb`, secciones 2–3)
- **La información lo es todo:** solo el snapshot 0.66 → con pasado 0.67 → con futuro 0.90. El controlador decide y *después* actúa
  sobre el avión: la siguiente instrucción revela la tarea.
- **LightGBM LambdaRank**, qué hacen las 63 hojas y `extra_trees`, y por qué se promedian semillas.
- Los bloques que sí aportan: **`future`**, **`clear`** (cambios de nivel, rumbo, zona o ruta) e **`instr`** (la instrucción que
  *encaja* con la tarea).
- El **segundo paso de sesión**: un logit de 9 pesos que usa que cada tarea se trabaja una sola vez.
- El **oráculo de coordinaciones** (qué tareas trabajan otros controladores del mismo escenario) y **por qué se mezcla con N en
  probabilidades** (M30_30): en log, el oráculo vetaba coordinaciones raras (controlador 006, −4.5); mezclado, no.

### 5 · Dónde está el techo (`futuro.ipynb`, secciones 4–5b)
- La mitad de los fallos son **coordinaciones**, que no dejan huella en los datos; el 41 % acierta el avión pero no la tarea; los
  fallos son de tramo entero.
- Solo un **12.7 % de acuerdo** entre controladores ante la misma situación: parte de la decisión es preferencia personal.
- Lo que **no** funcionó y por qué (otros modelos, stacks, ajustes, post-procesos, adaptación al controlador, expertos sin futuro):
  solo lo necesario para justificar las decisiones.

### 6 · Cómo se eligió la entrega (`futuro.ipynb`, sección 6)
- Por qué no se decide por el público (B y el oráculo completo bajaron en el 020; la hipótesis H1 se rompió).
- Los cinco métodos para estimar el privado (regla validada, riesgo por controlador, desacuerdo en test, CV reponderada al 023,
  señal sin etiquetas) y por qué todos apuntan a **M30_30**.
- Lo que sabemos del 023 (coordina poco, listas más cortas, menos instrucciones inminentes, estilo atípico) y el nivel esperado
  (~0.91–0.94): el propio controlador pesa más que la elección del modelo.

### 7 · La línea realista (`realista.ipynb`)
- Por qué existe (un asistente en tiempo real no puede ver el futuro) y cuánto aporta el pasado (+1.5 puntos frente a +24 del futuro).
- Los bloques `causal` y `geo`, por qué el segundo paso de sesión no ayuda sin futuro y por qué se eligió `real_adri_noche`.

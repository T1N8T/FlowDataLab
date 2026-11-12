# Decisiones del proyecto y su justificación según ISLR

Referencia: James, Witten, Hastie y Tibshirani, *An Introduction to Statistical Learning* (2.ª ed., corrección de junio de 2023),
`ISLRv2_corrected_June_2023.pdf` en la raíz del repo. Las páginas son las **impresas** en el libro (página del PDF − 11).

Este documento no repite las cifras: cada decisión remite al sitio del repo donde está medida.

---

## 0 · Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Definición del problema | `README.md` (sección "El problema"), `Context/Overview.txt`, `Context/Data.txt` (las 96 columnas) |
| Decisiones y por qué | `entrega/README.md` (guion de lo que hay que explicar), `entrega/futuro.ipynb` (secciones 0–6), `entrega/realista.ipynb` (secciones 1–8) |
| Resultados de cada prueba | `notebooks/00_destilado.ipynb` (resumen), `futuro/LGB_futuro.ipynb` y `realista/LGB_realista.ipynb` (diarios), `futuro/pruebas/resultados.jsonl`, `futuro/pruebas/notas/`, `pruebas/notas/`, `realista/noche/INFORME.md`, `experiments/*.csv` |
| Métricas finales con público y privado | `entrega/salidas/notas/metricas_con_kaggle.csv`, `metricas_ranking.csv` |
| Pruebas de significación | `entrega/codigo/ttest_pareado.py` → `entrega/salidas/notas/ttest_pareado.csv`, `ttest_por_controlador.csv` |

---

## 1 · El problema, en el lenguaje de ISLR

- **Aprendizaje supervisado** (§2.1.4, p. 26): cada fila tiene una respuesta observada (`target_index`).
- **Clasificación** (§2.1.5, p. 28, y cap. 4): la respuesta es cualitativa. Con una diferencia: no se clasifica cada fila por
  separado, sino que se **elige una fila dentro de cada lista** (exactamente una es la correcta). Es la forma de la regresión
  logística multinomial con codificación *softmax* (§4.3.5, pp. 140–141), con la lista como conjunto de clases. Por eso el objetivo
  es un *ranking* dentro de la lista (LambdaRank) y la métrica es el acierto top-3. ISLR no trata los modelos de ranking: aquí el
  libro da el marco, pero no la técnica.
- **Predicción, no inferencia** (§2.1.1, pp. 17–21): el objetivo es estimar bien Y, aunque f̂ sea una caja negra. Esto justifica
  usar *boosting* con cientos de columnas en vez de un modelo interpretable. La parte de **inferencia** (qué hace que se elija una
  tarea) se cubre aparte con SHAP y diagnósticos (`futuro.ipynb` §3–4), sin pedirle al modelo que sea legible.
- **Error irreducible** (§2.1.1, ec. 2.3, p. 19): la mitad de los fallos son coordinaciones sin huella en los datos. Ante la misma
  situación, dos controladores coinciden solo el 12.7 % de las veces (`futuro.ipynb` §4). Es el término Var(ε) de ISLR: ningún
  modelo lo reduce, porque esa información no está en X.

---

## 2 · Validación

### 2.1 Validación cruzada con 5 folds por controlador
**Decisión.** CV de 5 folds fijos (`datos/folds.csv`) en la que todos los snapshots de un controlador caen en el mismo fold.

**ISLR.**
- El error de entrenamiento es optimista. Lo que interesa es el error en datos no vistos (§2.2.1, pp. 29–32; Lab 4.7.2, p. 175:
  *"the training error rate is often overly optimistic"*).
- k = 5 o 10 tiene mejor equilibrio sesgo-varianza que LOOCV (§5.1.4, pp. 205–206): *"one performs k-fold cross-validation using
  k = 5 or k = 10"*.
- **Por qué agrupar por controlador.** ISLR no trata la CV agrupada, pero da el argumento en §3.3.3, punto 2 (pp. 94–95). Con
  observaciones correlacionadas (series temporales, o individuos de una misma familia) el error estimado sale demasiado
  optimista, *"an unwarranted sense of confidence"*. Los snapshots tomados cada 3 s dentro de una sesión son ese caso: el
  controlador repite la tarea el 94.8 % de las veces. Con un reparto aleatorio, el fold de validación tendría "gemelos" de
  entrenamiento. El test son controladores nuevos (020 y 023), y la CV tiene que imitar ese reparto.

### 2.2 OOF y CV anidada para elegir pesos
**Decisión.** Los pesos de las mezclas (0.7 · C + 0.3 · N) y los parámetros se eligen con 4 folds y se miden en el 5.º.

**ISLR.** Elegir sobre los mismos datos con los que se mide introduce **sesgo de selección** (§10.6, p. 433, sobre el *relaxed
lasso*: *"since the model was selected on the training data, there would be selection bias"*). Es el mismo motivo por el que, en
§6.1.3 (pp. 232–237), el tamaño del modelo se elige con CV y no con el error de entrenamiento.

### 2.3 Varias semillas y la "CV 023"
**Decisión.** Toda mejora se confirma con 3 o más semillas. Además se mira la mediana por sesión, el peor controlador y el acierto
en los 6 controladores parecidos al 023.

**ISLR.** La estimación del error con un único conjunto de validación es muy **variable** (§5.1.1, pp. 198–200, fig. 5.2: cada
partición da una curva distinta). Aquí hay dos fuentes de azar: la semilla (±0.1–0.2 puntos) y qué controladores caen en cada
fold (±2 puntos, `futuro.ipynb` §2b). La lección del realista (*"con 1 semilla casi todo parece mejorar"*) es este fenómeno.

---

## 3 · Modelo

### 3.1 *Gradient boosting* de árboles (LightGBM) en vez de modelos lineales o redes
**Decisión.** LightGBM LambdaRank en las dos entregas. Se descartaron el modelo lineal (`linear/`), el logit condicional
(`clogit`), una red tipo transformer (`NN_pablo`, `m_nn_future.py`), XGBoost y CatBoost.

**ISLR.**
- Boosting *"learns slowly"* y *"statistical learning approaches that learn slowly tend to perform well"* (§8.2.3, p. 346).
  Recoge interacciones (tipo de tarea × tiempo hasta la siguiente instrucción × distancia) sin especificarlas a mano. Es la
  ventaja de los árboles frente a los modelos lineales cuando la relación no es lineal (§8.1.3, pp. 338–339).
- Ante métodos con rendimiento parecido, **navaja de Occam** (§10.6, pp. 432–433): la red no superó a LightGBM
  (0.887 frente a 0.901 con futuro, `futuro.ipynb` §5) y es más cara y opaca. ISLR dice que el *deep learning* gana sobre todo
  con señales muy grandes o estructuradas (imágenes, texto), que no es el caso de esta tabla.

### 3.2 Los tres parámetros de boosting
**Decisión.** Futuro: 400 árboles, `learning_rate` 0.05, 63 hojas, `min_child_samples` 100. Realista: tasas de 0.018–0.026 con
540–1010 árboles y 36–37 hojas, elegidos por búsqueda (Optuna y la búsqueda nocturna) evaluada con la CV.

**ISLR** (§8.2.3, pp. 346–347). Boosting tiene tres parámetros: el número de árboles **B** (*"we use cross-validation to select
B"*), la tasa **λ** (*"very small λ can require using a very large value of B"*: es lo que se ve en el realista, con λ menor y más
árboles) y el número de cortes **d**, que fija el orden de las interacciones. Pasar a 63 hojas es subir d.

### 3.3 Árboles más grandes con futuro y no sin él: el equilibrio sesgo-varianza
**Decisión.** 63 hojas + `extra_trees` suben +0.6 con futuro (junto con `instr`) y bajan −0.5 en el realista.

**ISLR** (§2.2.2, pp. 33–36). Un método más flexible baja el sesgo y sube la varianza. Que compense depende de la señal. Con
futuro hay una señal fuerte (la siguiente instrucción) que necesita interacciones de orden alto, y el sesgo que se quita pesa más
que la varianza que se añade. Sin futuro, la señal es débil y la flexibilidad extra solo ajusta ruido. Es la curva en U de la
fig. 2.9 (p. 31) con el mínimo en sitios distintos.

### 3.4 Promediar semillas y `extra_trees`
**Decisión.** Promediar 10–30 semillas (futuro) o 5 configuraciones × 4 semillas (realista). `extra_trees` (umbrales de corte
aleatorios) en la línea con futuro.

**ISLR.**
- *"Averaging a set of observations reduces variance"* (§8.2.1, p. 340): es la idea del *bagging*, aplicada aquí a modelos que
  solo difieren en el azar. Como en el bagging, más modelos no sobreajustan (*"using a very large value of B will not lead to
  overfitting"*, p. 341). Coincide con lo medido: de 3 a 30 semillas la CV de N no cambia (0.9233 / 0.9242 / 0.9234), y lo que
  baja es la variación por controlador.
- Promediar modelos muy correlacionados reduce poco la varianza. Los *random forests* lo resuelven **decorrelacionando** los
  árboles (§8.2.2, pp. 343–344). `extra_trees` aleatoriza de forma parecida, aunque no es la misma técnica: sortea el umbral de
  corte en vez del subconjunto de variables. La mezcla de **5 configuraciones distintas** del realista también va en esa
  dirección: modelos menos parecidos entre sí que 20 semillas de uno solo.

### 3.5 El segundo paso de sesión: un modelo pequeño, no otro LightGBM
**Decisión.** Un logit condicional de 9 pesos sobre los scores del primer paso. Un LightGBM en su lugar sobreajusta (0.906 frente a
0.914).

**ISLR.**
- Es una regresión logística (§4.3, pp. 133–140) sobre pocas variables muy informativas. Con pocos datos efectivos (37
  sesiones) un modelo de baja varianza gana (§2.2.2; §3.5, pp. 104–109: el método paramétrico supera al flexible cuando su forma
  se acerca a la verdadera).
- Su peso más grande es negativo y casi igual en los 5 folds (M ≈ −2.6). Que un coeficiente sea estable en los folds sugiere que
  el efecto es real y no ruido, aunque ISLR no formula esta comprobación de forma explícita.

### 3.6 Mezcla C + N en probabilidades, no en log
**Decisión.** M30_30 = 0.7 · C + 0.3 · N, mezclando probabilidades.

**ISLR.** No trata la mezcla de modelos (*stacking*) de forma explícita. Lo más cercano es §8.2.5 (p. 351): combinar modelos
reduce la varianza. El argumento de por qué en probabilidades es propio del repo (`futuro.ipynb` §3d): en log, una probabilidad
casi nula de C (el oráculo) veta la tarea aunque N la vea probable, y en probabilidades no. El peso 0.7 se eligió con CV anidada
(§2.2).

---

## 4 · Variables

**Decisión.** Las variables se añaden por bloques (`future`, `clear`, `instr`, `causal`, `geo`…) y cada bloque se adopta solo si
sube la CV con las mismas semillas. Los bloques que no suben se descartan (`hist` −0.6, `instr` hacia atrás −0.55…).

**ISLR.**
- Es **selección paso a paso hacia delante** por grupos (§6.1.2, pp. 229–232), con el criterio de §6.1.3: elegir con el error de
  validación o CV, no con el de entrenamiento (pp. 232–237).
- Con ventajas pequeñas e inciertas, ante dos modelos parecidos se prefiere el más simple. Es la **regla de un error estándar**
  (§6.1.3, p. 237) y es lo que se hizo con `ctx` y `sty` en el realista: +0.15 / +0.13 en CV global, dentro del ruido, y no se
  adoptaron.
- La **validación adversaria** (`pruebas/notas/h4_adversaria.csv`, `futuro/pruebas/notas/f3_adversaria.csv`), que quita
  variables que distinguen train de test, no está en ISLR.

---

## 5 · ¿Es significativa la diferencia entre modelos? (`ttest_pareado.py`)

**Decisión.** t de Student pareado entre los OOF, **con el controlador como unidad** (n = 18), Wilcoxon como alternativa no
paramétrica y corrección de Holm por comparaciones múltiples.

**ISLR.**
- El contraste de hipótesis y el p-valor: §13.1 (pp. 552–558). El t pareado aparece en el Lab 13.6 (p. 583, `t.test(..., paired = T)`).
- **Por qué no por snapshot:** con observaciones correlacionadas, los errores estándar se infravaloran y *"p-values associated
  with the model will be lower than they should be"* (§3.3.3, p. 94). El ejemplo de ISLR de duplicar los datos (p. 95) es lo que
  pasaría con 13 175 snapshots que son, en la práctica, unas decenas de rachas.
- **Comparaciones múltiples:** con muchos pares comparados alguno sale significativo por azar (el corredor de bolsa de §13.2,
  pp. 558–559). Holm controla el FWER y tiene más potencia que Bonferroni (§13.3.2, pp. 563–565, algoritmo 13.1). El riesgo de
  quedarse con el p-valor más pequeño (*cherry-picking*) es el ejercicio 13.8 (e) (p. 592).

**Resultado** (`salidas/notas/ttest_pareado.csv`). Después de Holm solo sale significativo pasar de `futuro_base` a cualquiera de los
modelos posteriores (18 de 18 controladores suben). **Entre A, B, N, C y M30_30 ninguna diferencia es significativa** (por
ejemplo A → M30_30: p = 0.008 sin corregir, 0.084 con Holm; N → M30_30: 0.017 sin corregir, 0.14 con Holm). El control (el mismo
modelo en dos máquinas) tampoco sale significativo, como debe ser.

---

## 6 · Elegir la entrega sin fiarse del público

**Decisión.** No se eligió por la nota pública (un controlador, 2 sesiones), sino por el privado esperado según la CV, con cinco
métodos (`futuro.ipynb` §6b).

**ISLR.** Una única partición de validación pequeña da una estimación del error de varianza muy alta (§5.1.1, pp. 198–200).
Elegir entre muchos envíos por el público es además un problema de comparaciones múltiples (§13.2): el mejor del público lo es
en parte por suerte. La "regla validada" (cada controlador de la CV hace por turnos de privado) es CV aplicada a la propia regla
de decisión, como se hace en §6.1.3 con la elección del modelo.

**Lo que pasó en el privado** (`salidas/notas/metricas_con_kaggle.csv`):

| Envío | CV | Público (020) | Privado (023) |
|---|---|---|---|
| limpia | 0.664 | 0.642 | 0.606 |
| realista (`real_adri_noche`) | 0.673 | 0.655 | 0.614 |
| `lgb_future` | 0.901 | 0.942 | 0.900 |
| A | 0.918 | 0.947 | 0.920 |
| B | 0.926 | 0.934 | 0.922 |
| OC (A + oráculo completo) | 0.924 | 0.940 | **0.932** |
| N | 0.924 | 0.951 | 0.909 |
| NOR | 0.928 | 0.948 | 0.906 |
| **M30_30 (entrega)** | **0.929** | **0.953** | 0.912 |

Lectura según ISLR:
- **El orden en el privado no sigue al del público.** M30_30 era la mejor del público (0.953) y queda en mitad de tabla en el
  privado. OC, de las peores del público, es la mejor del privado. Es la varianza de un único conjunto de validación (§5.1.1).
- **Lo que sí se mantiene** es lo que el t-test daba por significativo. El salto de no ver el futuro (~0.61) a verlo (~0.90–0.93)
  y la mejora sobre `lgb_future` se cumplen en el privado. Las diferencias entre A, B, N, C y M30_30 no eran significativas y en
  el privado se reordenan: ±1–2 puntos en un solo controlador, el ruido que ya se estimaba.
- El privado quedó dentro del rango que se esperaba (0.91–0.92 ± 5, `futuro.ipynb` §6c), unos 4 puntos por debajo del público.
  La previsión de que "el propio controlador pesa más que la elección del modelo" se cumplió.

---

## 7 · Lo que ISLR no cubre (y de dónde sale)

| Técnica del proyecto | Estado en ISLR |
|---|---|
| LambdaRank / objetivos de ranking | no aparece. Lo más cercano es la logística multinomial con *softmax* (§4.3.5) |
| CV agrupada por controlador | no aparece como tal. Se justifica con §3.3.3 (errores correlacionados) y con que la CV debe imitar el test |
| Corrección de Nadeau y Bengio (2003) para el t por fold | no aparece. Viene del artículo citado en `ttest_pareado.py` |
| `extra_trees` | no aparece. Es análoga a la decorrelación de los random forests (§8.2.2) |
| SHAP, t-SNE, validación adversaria | no aparecen. Para reducir dimensión ISLR da PCA (§12.2) |
| Mezcla de modelos (*stacking*) | solo de forma implícita (§8.2.5) |

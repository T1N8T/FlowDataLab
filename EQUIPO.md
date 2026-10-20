# FlowDataLab – Plan de equipo (AWARE Prioritisation Challenge, Engage 2 Zagreb)

> **Cómo usar este archivo con tu IA (Claude, Gemini, etc.):** pégale este archivo entero (o el repo) y dile:
> *"Soy <tu nombre>. Lee EQUIPO.md y ayúdame con mi parte."*
> La IA debe ceñirse a tu sección y respetar las **reglas comunes**.

## Quién es quién

| Persona | Rol | Recursos | IA |
|---|---|---|---|
| **Pablo** | **A** · Redes neuronales | RTX 3060 | Claude |
| **Adri** (Adrián) | **B** · GBM + `features.py` (modelo de referencia) | RTX 3050 | Claude |
| **Juanma** | **C** · Folds/métrica, modelos alternativos, stacking y envíos | GTX 1650 (trabaja en CPU) | Claude |
| **Pau** | **D** · EDA, análisis de errores, barridos de parámetros, slides | Sin GPU | Gemini Pro |

Todos pueden usar las máquinas del DSIC-UPV para lanzar lo pesado: bluenote.dsic.upv.es (2× RTX 4000 Ada) y gpu.dsic.upv.es (2× Quadro RTX 5000).

---

## 1. El reto en 1 minuto
- Datos de simulador de control aéreo (sector sueco ESMM_23). El sistema AWARE genera cada 3 s la lista de tareas abiertas del controlador.
- **Una muestra** (`sample_id`) es un snapshot de esa lista, con 9–59 tareas (~36 de media). **Una fila** es una tarea.
- Exactamente una fila por muestra tiene `target_index = 1`: la tarea en la que trabaja el controlador.
- **Predicción:** los 3 `row_index` más probables por muestra, el mejor primero.
- **Métrica:** top-3 hit rate, es decir, la fracción de muestras en las que la tarea correcta está entre las 3 propuestas. Aleatorio = 0.094, baseline oficial = 0.352.
- **Datos:** `train.csv` (11,923 muestras), `validation.csv` (1,252), `test.csv` (1,863, sin etiqueta).
- **Leaderboard público:** solo el controlador 020. **Privado (el que decide):** el controlador 023. Ninguno de los dos está en train.
- **Kaggle:** 5 envíos al día por equipo (el día empieza a las 02:00 de Zagreb). Cierre: **miércoles 21/10 a las 13:00**. Al final se eligen 2 entregas.
- Descripción de columnas: `Context/Data.txt`. Descripción del reto: `Context/Overview.txt`.

### Conceptos clave de los datos
- **14 tipos de tarea** (`task_index`), por ejemplo: 4 = assume_aircraft (aceptar el avión), 5 = transfer_aircraft (pasarlo al siguiente sector), 12 = zone_conflict (zona militar), 2 = entry_coordination. Es una **categoría**, no un número.
- La mayoría de columnas describen **el avión**, no la tarea, así que son iguales en todas las tareas de un mismo avión. Lo que las distingue: `task_index`, `x_task_age` y las columnas `l_*`.
- **Caso difícil:** un mismo avión con assume_aircraft y entry_coordination a la vez (`l_twin_AE = 1`). El controlador elige assume en unas 3 de cada 4 veces.
- **Celda vacía = "no aplica"**, nunca 0.
- Las etiquetas tienen ruido y los controladores no coinciden mucho entre sí: el techo está bastante por debajo de 1.
- Respuestas por tipo en train: transfer 27 %, assume 26 %, zone_conflict 16 %, entry_coordination 8 %, return_to_route 7 %, exit_coordination 6 %.

---

## 2. Reglas comunes (OBLIGATORIAS para todos y para todas las IAs)
1. **Cada snapshot se predice solo con sus propias filas** (las del mismo `sample_id`). Prohibido usar otros snapshots de la sesión, pasados o futuros, por ahora: nada de features temporales ni de suavizado entre snapshots. El organizador nos ha pedido primero la mejor nota así; si luego lo permite, se avisará.
2. **Validación:** GroupKFold de 5 folds por `atco`, siempre con `Data/folds.csv` (columnas `atco,fold`). Nunca un split aleatorio por filas o muestras.
3. **Se decide por la CV**, no por el leaderboard público.
4. **Columnas que no son features:** `sample_id`, `target_index`, `session`, `atco`, `scenario`, `timestamp`, `task_name`. `row_index` solo si el EDA demuestra que aporta.
5. Se entrena con `train.csv` + `validation.csv` juntos.
6. **Nadie sube a Kaggle salvo Juanma.**
7. El código con snapshots futuros/pasados se ha quitado; si se permite, está en el commit `67a15bc`.

### Formato común de salidas (para poder combinar modelos)
Cada modelo genera tres archivos:
- `oof/<modelo>.csv`: predicciones out-of-fold sobre train+validation, con columnas `sample_id,row_index,score` (score alto = más probable).
- `preds/<modelo>.csv`: predicciones de test con el modelo entrenado con todo, mismo formato.
- `submissions/sub_<modelo>.csv`: columnas `sample_id,prediction`, con 3 `row_index` separados por espacios.

Cada script imprime su **CV top-3** usando `cv/metric.py`. Se guarda el OOF aunque el modelo sea peor, porque en el stacking puede sumar.

### Estructura del repo
```
Data/          train/validation/test.csv, folds.csv, scorer.py   (CSV no versionados)
cv/metric.py                    ← Juanma (C) · top-3, report(), save_outputs()
src/
  folds.py                      ← Juanma (C) · genera Data/folds.csv
  features.py                   ← Adri (B) · features del snapshot (logic_*) -> models/features.parquet
  m_clogit.py                   ← Juanma (C) · logit condicional sobre logic_* (analítico)
  m_nn.py, nn_sweep.py          ← Pablo (A) · transformer de snapshot y barridos
hierarchical/m_types.py         ← Juanma (C) · P(tipo) × P(tarea | tipo)
linear/m_linear.py              ← Juanma (C) · logit condicional con todas las columnas y pendiente por tipo
stacking/stack.py               ← Juanma (C) · media ponderada y meta-ranker sobre oof/
notebooks/     01_eda.ipynb, LGB_adri.ipynb (LightGBM), NN_pablo.ipynb
experiments/   registro de ejecuciones (lgb_runs.csv, nn_runs.csv)
oof/ preds/ submissions/   salidas en formato común (no versionadas)
```
Pendiente: XGBoost / CatBoost (Adri) y dos etapas avión → tarea (Juanma).
Cada persona trabaja en su rama de git. Los CSV pesados se comparten por carpeta compartida, no por git.

### Resultados conocidos
- LightGBM LambdaRank con las columnas originales: **CV 0.655**.
- Rankings relativos simples dentro del snapshot: 0.652 (sin mejora todavía; hay que trabajar mejor esas features).
- Con snapshots futuros: 0.83, **no permitido por ahora**.

---

## 3. Tareas por persona

### Pablo · A · Redes neuronales (RTX 3060)
1. `src/m_nn.py`: transformer sobre el conjunto de tareas del snapshot.
   - Entrada: features numéricas (log + estandarizar, vacío → 0 + indicador de vacío) y un embedding de `task_index`.
   - Encoder sin codificación de posición (2–3 capas, d = 64, 4 cabezas, dropout 0.2) y una puntuación por tarea con máscara de padding.
   - Softmax sobre la lista + cross-entropy con label smoothing 0.1.
   - Early stopping con 3 controladores apartados dentro de cada fold. Media de 3 semillas.
2. Afinar tamaño, dropout, label smoothing y learning rate, una cosa cada vez.
3. Variante: usar la puntuación OOF del LightGBM de Adri como entrada extra.
4. Si se permiten otros snapshots: una versión que vea también la lista anterior y la siguiente.
- **Entregables:** `oof/nn_*.csv` y `preds/nn_*.csv`.
- **Punto de decisión a las 21:00:** si la CV queda a más de ~3 puntos del GBM, se queda como modelo secundario del ensemble y Pablo pasa a ayudar a Adri con features.
- **Miércoles:** apoyo a Pau con las slides (parte técnica).

### Adri · B · GBM y features (RTX 3050) — el núcleo
1. `src/features.py` (todos lo importan). Funciones `add_features(df)` y `feature_cols(df)`. Ideas por orden:
   - Rankings y diferencias respecto al mínimo **dentro del snapshot** de `a_since_flight`, `x_task_age`, `x_state_age`, `time_to_MD`, `k_timeToExitPoint`, `distance_to_zone` y `|time_to_sector|`.
   - Los mismos rankings **dentro del mismo tipo** de tarea.
   - Por avión: nº de tareas del avión y si esta tarea es la más antigua del avión.
   - Tipo × estado: assume y `x_transfer_in`, transfer y `k_aircraftNeedsToBeTransferred`, entry_coordination y `x_isc_coordinated`, zone y `distance_to_zone`.
   - Contexto del snapshot: nº de aviones y nº de transfers, assumes y zonas pendientes.
   - Añadir las ideas que lleguen de Pau.
2. `src/m_lgb.py`: LightGBM LambdaRank (modelo de referencia). Parámetros base: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.7`, `lambdarank_truncation_level=5` y `task_index` como categórica.
3. `src/m_xgb.py`: XGBoost `rank:pairwise` o `rank:ndcg` con `device='cuda'`.
4. `src/m_cat.py`: CatBoost `YetiRank` con `task_type='GPU'`.
5. Varias semillas de cada modelo.
- **Entregables:** `oof/` y `preds/` de lgb, xgb y cat. Push de `features.py` cada vez que la CV mejore.
- **Regla:** una feature solo entra si mejora la CV del LightGBM.

### Juanma · C · Infraestructura, modelos alternativos, stacking y envíos (CPU)
1. **Lo primero (antes de las 17:15):** `src/folds.py`, que genera `Data/folds.csv` asignando cada `atco` a un fold 0–4 de forma estable, y `src/metric.py`, con una función de top-3 a partir de un df `sample_id,row_index,score` y las etiquetas. Pasarlos a todos.
2. `src/m_types.py`: especialistas por tipo. Un modelo de P(tipo | snapshot) con features agregadas de la lista y un ranker dentro de cada tipo; puntuación = P(tipo) × P(tarea | tipo).
3. `src/m_two_stage.py`: primero ranking de **aviones** (una fila por avión) y luego la tarea dentro del avión (sobre todo assume vs entry_coordination). Cuando duda entre el gemelo, repartir el top-3 en consecuencia.
4. `src/stack.py`:
   - (a) Promedio ponderado de rankings dentro del snapshot, con pesos ajustados en la CV.
   - (b) Meta-modelo LightGBM ranker pequeño con los scores OOF de todos, su ranking dentro del snapshot, `task_index`, `l_twin_AE` y `seq_len`.
   - Usar **solo OOF** para entrenar el nivel 2.
5. **Envíos a Kaggle:** solo Juanma. Primer stacking antes de las 23:00; a las 02:00 se renuevan los envíos.
- **Entregables:** `Data/folds.csv`, `metric.py`, OOF/preds de types y two-stage, y las submissions finales.

### Pau · D · EDA, análisis de errores, barridos y slides (sin GPU, Gemini Pro)
1. **17:00–18:30 · `notebooks/EDA_pau.ipynb`:**
   - Estructura: tareas por snapshot, aviones por snapshot, tareas por avión. Comprobar que hay exactamente un `target_index = 1` por muestra.
   - ¿`row_index` aporta? Posición de la respuesta en la lista.
   - "Lift" de cada tipo: % como respuesta frente a % en la lista, y si cambia por escenario.
   - Para transfer, assume y zone_conflict: la tarea correcta frente a las demás del mismo tipo en el mismo snapshot, en `time_to_sector`, `x_task_age`, `x_state_age`, `a_since_flight` y flags (`x_transfer_in`, `k_aircraftNeedsToBeTransferred`).
   - Columnas vacías: cuáles, cuándo, y si estar vacía da información.
   - **Salida:** 3–5 ideas de features concretas, que se pasan a Adri.
2. **Desde las 19:00 · `notebooks/errores_pau.ipynb`** sobre `oof/lgb_rank.csv`:
   - Acierto por tipo de tarea correcta, por controlador, por longitud de lista y en snapshots con gemelo (`l_twin_AE`).
   - Ejemplos de fallos: qué tipo había en el top-3 y cuál era el correcto.
   - **Salida:** dónde atacar, que se pasa a Adri y Juanma.
3. **Barridos de parámetros** de `m_lgb.py` en CPU (`num_leaves` 15/31/63, `learning_rate` 0.03/0.05/0.1, `n_estimators` 300/600/1000, `min_child_samples` 20/50/100), uno cada vez.
4. **`RESULTADOS.md`:** tabla con fecha/hora, persona, modelo, cambio, CV top-3 y nota en el leaderboard público (si se subió).
5. **Miércoles desde las 10:00 · slides para el jurado:** el problema, por qué validamos por controlador, qué features funcionan, los modelos y el ensemble, el resultado y una versión desplegable en tiempo real.

---

## 4. Calendario
| Hora | Qué |
|---|---|
| 16:55–17:15 | Juanma: `folds.py` + `metric.py`. Adri: primera versión de `features.py` |
| 17:15–19:00 | Primera versión de cada modelo con su OOF. Pau: EDA |
| **19:00** | **Puesta en común (15 min):** CV de cada uno e ideas de features |
| 19:00–23:00 | Iterar. Pau: análisis de errores y barridos |
| **23:00** | **Primer stacking** (Juanma) |
| 02:00 | Se renuevan los 5 envíos: subir el mejor stacking |
| **Miér. 08:00** | **Puesta en común:** qué entra en el final. ¿Se permiten otros snapshots? |
| 08:00–11:00 | Últimas mejoras y reentrenar con todo |
| 11:00–12:30 | Elegir 2 entregas por CV (la mejor y una segura). Slides |
| 13:00 | Cierre de Kaggle. 13:00–13:30 evaluación del jurado y 13:30–14:30 entrevistas del top-3 |

## 5. Reglas para las IAs del equipo
- Antes de proponer una feature, comprobar que solo usa filas del mismo `sample_id`.
- Todo modelo usa `Data/folds.csv`, imprime su CV top-3 y guarda `oof/` y `preds/` en el formato común.
- Cambios mínimos en el código de otros; si hace falta tocar `features.py`, avisar a Adri.
- Código completo y listo para copiar.
- No inventar resultados: si algo no se ha ejecutado, decirlo.

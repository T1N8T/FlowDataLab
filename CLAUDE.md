# FlowDataLab – AWARE Prioritisation Challenge (Engage 2, Zagreb)

> **El plan de equipo vigente está en `EQUIPO.md`** (roles: Pablo = A redes neuronales, Adri = B GBM + features, Juanma = C folds/métrica, modelos alternativos, stacking y envíos, Pau = D EDA/errores/slides). Si este archivo y `EQUIPO.md` no coinciden, manda `EQUIPO.md`. Pregunta al usuario quién es si no lo dice.

Equipo de 4: Pablo, Adrián, Pau, Juanma. Hackathon de 24 h. Kaggle cierra el **miércoles 21/10 a las 13:00** (hora de Zagreb).

## El reto
- Cada `sample_id` es un snapshot de la lista de tareas abiertas del controlador aéreo (9–59 tareas, ~36 de media). Cada fila es una tarea.
- Exactamente una fila por snapshot tiene `target_index = 1`: la tarea en la que trabaja el controlador.
- Hay que predecir los 3 `row_index` más probables por snapshot, el mejor primero.
- **Métrica:** top-3 hit rate. Baselines en test: aleatorio 0.094, `sample_submission` 0.352.
- **Leaderboards:** el público es solo el controlador 020 (915 muestras). El privado, que decide, es el controlador 023 (948 muestras).
- **Envíos:** 5 al día por equipo; el día de Kaggle empieza a las 02:00 de Zagreb. Al final se eligen 2 entregas.
- Descripción completa de columnas: `Context/Data.txt`. Descripción del reto: `Context/Overview.txt`.

## Reglas que hemos decidido (no saltárselas)
1. **Cada snapshot se predice solo con sus propias filas.** Nada de usar otros snapshots de la misma sesión, ni pasados ni futuros (features, suavizado temporal, etc.). El organizador nos ha pedido conseguir primero la mejor nota así. Más adelante puede que se permita usar otros snapshots; hasta entonces, prohibido.
2. **Validación:** GroupKFold de 5 folds por `atco`, con los folds de `Data/folds.csv` (columnas `atco`, `fold`). Nadie crea sus propios folds ni hace splits aleatorios por filas.
3. **Las decisiones se toman por la CV**, no por el leaderboard público (es un solo controlador y el privado es otro).
4. **No usar como features:** `sample_id`, `target_index`, `session`, `atco`, `scenario`, `timestamp`, `task_name`. `row_index` solo si el EDA demuestra que aporta.
5. Las celdas vacías significan "no aplica", nunca 0. LightGBM las maneja de forma nativa; en redes neuronales se usa relleno + indicador de vacío.
6. Se entrena con `train.csv` + `validation.csv` juntos.

## Formato común de salidas (obligatorio para el stacking)
- `oof/<modelo>.csv`: predicciones out-of-fold de train+validation, con columnas `sample_id,row_index,score`.
- `preds/<modelo>.csv`: predicciones de test con el modelo entrenado con todo, mismo formato.
- `submissions/sub_<modelo>.csv`: columnas `sample_id,prediction`, con 3 `row_index` separados por espacios, el mejor primero.
- Comprobar en validación con: `python Data/scorer.py --labelled Data/validation.csv <sub>.csv`

## Estructura del repo
```
Data/          train, validation, test, folds.csv, scorer.py (los CSV no se versionan)
src/
  folds.py     común (Juanma)
  metric.py    común (Juanma)
  features.py  común: features dentro del snapshot (dueño: Pau)
  m_lgb.py     LightGBM LambdaRank (Pau)
  m_types.py   especialistas por tipo + dos etapas avión→tarea (Adrián)
  m_nn.py      transformer de conjunto sobre la lista (Pablo, GPU RTX 3060)
  m_xgb_cat.py XGBoost rank:pairwise, CatBoost YetiRank, logística (Juanma)
  stack.py     meta-modelo sobre los OOF y promedio de rankings (Juanma)
notebooks/     EDA_<nombre>.ipynb
oof/ preds/ submissions/
```
Cada persona trabaja en su rama de git. Los CSV grandes se comparten por carpeta compartida, no por git.

## Reparto
- **Pau:** EDA (estructura, lift por tipo, qué distingue a la tarea correcta, vacíos), `features.py` y `m_lgb.py`. Es el modelo de referencia.
- **Adrián:** EDA del caso assume vs entry_coordination (`l_twin_AE`), `m_types.py` (P(tipo) × ranker dentro del tipo) y la variante de dos etapas avión → tarea.
- **Pablo:** `m_nn.py`, el transformer sobre el conjunto de tareas del snapshot: embedding de `task_index`, encoder sin codificación de posición, softmax sobre la lista y cross-entropy con label smoothing. Límite de unas 4 h; a las 21:00, si su CV está a más de ~3 puntos del LightGBM, pasa a ayudar a otros. Mañana: slides para el jurado.
- **Juanma:** `folds.py` y `metric.py` lo primero; luego XGBoost/CatBoost, el stacking y la gestión de envíos (nadie más sube a Kaggle).

## Calendario
- 19:00 martes: puesta en común (CV de cada uno y features nuevas).
- Noche: primer stacking, que se sube después de las 02:00 (se renuevan los envíos).
- 08:00 miércoles: puesta en común. Si se permiten otros snapshots, añadir esas features.
- 11:00–12:30: elegir las 2 entregas finales por CV (la mejor y una segura). Slides.
- 13:00: cierre.

## Resultados conocidos
- LightGBM LambdaRank con las columnas originales, solo el snapshot: **CV 0.655** (GroupKFold por atco).
- Añadir rankings relativos sencillos dentro del snapshot: 0.652 (sin mejora; hay que trabajar mejor esas features).
- Usando snapshots futuros (siguiente input del controlador sobre el avión, a partir de `a_since_flight`): 0.83. **No permitido por ahora.**

## Ojo
- `src/baseline_ranker.py` usa la feature futura `f_next_input`: no usarlo tal cual.
- **No subir** `submissions/sub_lgb_temporal.csv`.
- Las etiquetas tienen ruido y los controladores no coinciden mucho entre sí: el techo está bastante por debajo de 1. Preferir modelos robustos a los sobreajustados.

## Cómo trabajar con Claude en este repo
- Antes de añadir una feature, comprobar que solo usa filas del mismo `sample_id`.
- Todo modelo nuevo debe usar `Data/folds.csv`, imprimir su CV top-3 y guardar `oof/` y `preds/` en el formato común.
- Cambios mínimos en el código existente; código completo y listo para copiar.

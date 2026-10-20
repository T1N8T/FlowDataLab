# FlowDataLab – AWARE Prioritisation Challenge

Modelo de Machine Learning para predecir **en qué tarea está trabajando un controlador aéreo** en cada instante, a partir de la lista de tareas abiertas que genera el sistema AWARE (problema de *ranking*).

Los datos proceden de ejercicios de simulador en el sector sueco de espacio aéreo superior **ESMM_23**: 19 controladores, hasta 3 escenarios de tráfico cada uno, 42 sesiones. Cada 3 segundos AWARE escribe su lista de tareas y unos expertos etiquetaron cuál estaba atendiendo el controlador.

## El problema

- **Muestra** (`sample_id`): una instantánea de la lista de tareas (9–59 tareas, ~36 de media).
- **Fila**: una tarea de esa lista, identificada por `row_index`. Tiene un tipo (`task_index`, 14 tipos) y pertenece a una aeronave.
- **Etiqueta**: exactamente una fila por muestra tiene `target_index = 1`.
- **Objetivo**: para cada muestra de test, proponer los 3 `row_index` más probables, el mejor primero.

### Métrica

*Top-3 hit rate*: proporción de muestras en las que la tarea etiquetada está entre las 3 primeras predicciones distintas.

| Método | Top-3 hit rate (test) |
|---|---|
| Aleatorio | 0.094 |
| `sample_submission.csv` (baseline por tipo de tarea + distancia al sector) | 0.352 |

## Datos

| Archivo | Contenido | Muestras | Filas | Sesiones |
|---|---|---|---|---|
| `train.csv` | Entrenamiento con `target_index` | 11,923 | 432,578 | 34 |
| `validation.csv` | Validación con `target_index` (019sc1, 019sc2, 020sc1) | 1,252 | 42,110 | 3 |
| `test.csv` | Test sin etiqueta (020sc2, 020sc3, 023sc1, 023sc2, 023sc3) | 1,863 | 66,409 | 5 |
| `sample_submission.csv` | Ejemplo de envío válido | 1,863 | | |

> Los CSV de datos **no se versionan** (train.csv supera el límite de 100 MB de GitHub). Descárgalos de la competición y colócalos en `Data/`.

La descripción completa de las 96 columnas está en [`Context/Data.txt`](Context/Data.txt) y la del reto en [`Context/Overview.txt`](Context/Overview.txt).

### A tener en cuenta

- **Validar por sesión**, nunca con split aleatorio: las muestras consecutivas están a 3 s y suelen compartir respuesta.
- Los controladores de test (020 y 023) no aparecen en `train.csv`.
- La mayoría de columnas describen la **aeronave**, no la tarea: para distinguir tareas de una misma aeronave hay que usar el tipo, `x_task_age` y las columnas `l_*`.
- Celdas vacías significan "no aplica / no calculado", nunca 0.
- Las etiquetas tienen ruido (anotación humana).

## Estructura

```
FlowDataLab/
├── Context/          # Descripción del reto y de las columnas
├── Data/             # CSVs (no versionados) + scorer.py
├── notebooks/        # Exploración y experimentos
├── src/              # Código reutilizable (features, modelos, envíos)
├── models/           # Modelos entrenados (no versionados)
└── submissions/      # Envíos generados (no versionados)
```

## Puesta en marcha

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Evaluación offline

```bash
python Data/scorer.py --labelled Data/validation.csv submissions/my_validation_submission.csv
```

### Formato de envío

```csv
sample_id,prediction
12797,9 6 41
12798,9 6 41
```

`prediction` contiene los `row_index` separados por espacios, el mejor primero. Solo cuentan los 3 primeros distintos.

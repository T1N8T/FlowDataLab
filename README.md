# FlowDataLab – AWARE Prioritisation Challenge (Engage 2, Zagreb)

Proyecto del equipo **FlowDataLab** (Pablo, Adrián, Juan Manuel y Pau, Universitat Politècnica de València) para el hackathon
Engage 2 en Zagreb (octubre de 2026).

El reto: predecir **en qué tarea está trabajando un controlador aéreo** en cada instante, a partir de la lista de tareas abiertas que
genera el sistema AWARE. Los datos proceden de ejercicios de simulador en el sector sueco de espacio aéreo superior **ESMM_23**:
19 controladores, hasta 3 escenarios de tráfico cada uno, 42 sesiones. Cada 3 segundos AWARE escribe su lista de tareas y unos
expertos etiquetaron cuál estaba atendiendo el controlador.

> **Por dónde empezar:** la carpeta [`entrega/`](entrega/) contiene todo lo necesario para entender y reproducir las entregas, con
> dos notebooks explicados paso a paso: [`entrega/futuro.ipynb`](entrega/futuro.ipynb) (la entrega final) y
> [`entrega/realista.ipynb`](entrega/realista.ipynb) (un modelo que funcionaría en tiempo real). El resto del repositorio es el
> trabajo de las 24 horas: experimentos, diarios y pruebas descartadas.

## El problema

- **Muestra** (`sample_id`): una instantánea de la lista de tareas (9–59 tareas, ~36 de media).
- **Fila**: una tarea de esa lista (`row_index`), con un tipo (`task_index`, 14 tipos) y una aeronave.
- **Etiqueta**: exactamente una fila por muestra tiene `target_index = 1`.
- **Objetivo**: para cada muestra de test, proponer los 3 `row_index` más probables, el mejor primero.
- **Métrica**: *top-3 hit rate*, la proporción de muestras en las que la tarea etiquetada está entre las 3 primeras.
- **Test**: dos controladores que no aparecen en entrenamiento: el 020 (público) y el 023 (privado, el que decide la clasificación).

## Resultados

Validación cruzada por controlador (5 folds fijos, `Data/folds.csv`): cada controlador se predice con un modelo que no lo ha visto.

| Modelo | Qué información usa | CV | Público |
|---|---|---|---|
| Aleatorio | — | — | 0.094 |
| `sample_submission.csv` (baseline oficial) | tipo de tarea + distancia al sector | — | 0.352 |
| Línea limpia (`lgb_adri_v2`) | solo el propio snapshot | 0.664 | 0.642 |
| Línea realista (`real_adri_noche`) | snapshot + snapshots pasados | 0.673 | — |
| N | + snapshots futuros de la sesión, segundo paso de sesión | 0.924 | 0.951 |
| C (NOR) | N + oráculo de coordinaciones (otras sesiones del escenario) | 0.928 | 0.948 |
| **M30_30 (entrega final)** | 0.7 · C + 0.3 · N, mezcla en probabilidades | **0.929** | **0.953** |

**Lo principal en tres ideas:**
1. **El futuro lo es casi todo.** Con solo el snapshot el techo está en ~0.66; viendo la sesión completa se llega a ~0.92. El
   controlador decide y *después* actúa: la siguiente instrucción que da al avión delata la tarea.
2. **La estructura de las etiquetas.** Dentro de una racha el controlador repite la tarea (95 %) y cada tarea se trabaja en un solo
   episodio (98 %). Un segundo paso de 9 pesos sobre toda la sesión lo aprovecha.
3. **El techo son las coordinaciones**, que no dejan huella en los datos: la mitad de los fallos. Ante la misma situación, dos
   controladores eligen la misma tarea solo el 12.7 % de las veces.

La entrega se eligió por su **privado esperado** (cinco métodos distintos sobre la CV), no por el público.

## Estructura

```
entrega/              ← EMPEZAR AQUÍ. Código, salidas y notebooks de las dos entregas, ejecutable solo con esta carpeta
  futuro.ipynb          línea con futuro: técnicas, cadena de CV, por qué funciona, dónde falla, elección de la entrega
  realista.ipynb        línea sin futuro (usable en tiempo real)
  codigo/               features, oráculo, segundo paso de sesión, entrenamiento y elección
  salidas/              OOF, predicciones de test y envíos
Context/              descripción del reto y de las 96 columnas
Data/                 folds.csv, scorer.py y sample_submission.csv (los CSV del reto no se versionan)
notebooks/            00_destilado (resumen de todo el proyecto), 01_eda, línea limpia (LGB_adri, NN_pablo)
futuro/               diario completo de la línea con futuro (LGB_futuro.ipynb) y todas las pruebas
realista/             diario de la línea sin futuro (LGB_realista.ipynb)
src/, cv/, linear/, hierarchical/, stacking/, pruebas/   código de la línea limpia, métricas y modelos descartados
experiments/          registros de ejecuciones
```

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copiar train.csv, validation.csv y test.csv de la competición a entrega/datos/
jupyter nbconvert --to notebook --execute --inplace entrega/futuro.ipynb
jupyter nbconvert --to notebook --execute --inplace entrega/realista.ipynb
```

Los notebooks no entrenan: leen las salidas ya calculadas de `entrega/salidas/`. Para reentrenar desde cero, ver
[`entrega/README.md`](entrega/README.md).

### Formato de envío

```csv
sample_id,prediction
12797,9 6 41
12798,9 6 41
```

`prediction` contiene los `row_index` separados por espacios, el mejor primero. Solo cuentan los 3 primeros distintos.

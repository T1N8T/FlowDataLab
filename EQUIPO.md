# FlowDataLab – Estado del proyecto (AWARE Prioritisation Challenge, Engage 2 Zagreb)

Equipo: Pablo, Adri, Juanma y Pau. Ya no hay reparto fijo de tareas: cualquiera puede trabajar en cualquier parte siguiendo las
**normas de trabajo** (sección 4). Este archivo es el punto de partida; el resumen razonado y ejecutable está en
**`notebooks/00_destilado.ipynb`**.

Kaggle cierra el **miércoles 21/10 a las 13:00** (hora de Zagreb). 5 envíos al día (el día empieza a las 02:00). Al cierre se marcan
**2 entregas** y el privado se puntúa con la mejor de las dos.

---

## 1. El reto en 1 minuto
- Cada `sample_id` es un **snapshot** de la lista de tareas abiertas del controlador aéreo (9–59 tareas, ~36). Cada fila es una tarea;
  exactamente una tiene `target_index = 1` (la tarea en la que trabaja). Se predicen los 3 `row_index` más probables.
- **Métrica:** top-3 hit rate (azar 0.094, `sample_submission` 0.352).
- **Público:** controlador 020 (2 sesiones, 915 snapshots). **Privado, el que decide:** controlador 023 (3 sesiones, 948).
- **Información permitida:** el organizador confirmó (21/10, 03:15) que se puede usar **cualquier información de los datasets**,
  incluidos los snapshots futuros y pasados de la sesión y las etiquetas de otras sesiones de train/validation (el "oráculo").
- Columnas: `Context/Data.txt`. Reto: `Context/Overview.txt`.

## 2. Lo que sabemos (no repetir)
Cifras: CV top-3 con `Data/folds.csv` (5 folds por controlador). Detalle en `notebooks/00_destilado.ipynb` y `futuro/LGB_futuro.ipynb`.

| Paso | CV | Público |
|---|---|---|
| Línea limpia: solo el propio snapshot (`lgb_adri_v2`) | 0.664 | 0.642 |
| + snapshots **pasados** (`realista/`, `real_adri_noche_ses`) | 0.672 | — |
| + snapshots **futuros** (`lgb_future`: siguiente instrucción al avión, desaparición de la tarea) | 0.901 | 0.942 |
| + `clear` + segundo paso de sesión + 10 semillas + suavizado = **A** | 0.918 | 0.947 |
| + bloque `instr` + 63 hojas + `extra_trees` = **N** | 0.924 | **0.951** |
| A + oráculo de coordinaciones = **B** | 0.925 | 0.934 |
| N + oráculo de coordinaciones = **NOR** / **C** | 0.928 | 0.948 |
| **M30_30** = 0.7·C30 + 0.3·N30 (mezcla de probabilidades, peso por CV anidada) | **0.929** | — |

**Por qué funciona:**
- **El futuro lo es casi todo** (+24 puntos; el pasado da ~1.5). La regla que aprende el modelo: el avión que recibe una instrucción
  en los próximos 10 s es el que se está trabajando.
- **Estructura de las etiquetas:** dentro de una racha el controlador repite la tarea (94.8 %); tras un hueco, casi nunca (0.2 %);
  cada tarea se trabaja en una sola racha (98.4 %). El **segundo paso de sesión** (`futuro/feat_session.py`) lo aprovecha (+0.7).
- **`clear`** (cuándo cambia el nivel, rumbo, zona o ruta del avión, +0.6) e **`instr`** (qué tipo de instrucción llega y si encaja
  con la tarea; solo funciona con 63 hojas + `extra_trees`, +0.6).
- **El oráculo** (qué tareas trabajan otros controladores del mismo escenario) es la única información entre sesiones: entre
  sesiones se transmite *si* una tarea se trabaja, no *cuándo*.

**Dónde está el techo:**
- La mitad de los fallos son **coordinaciones** (no dejan huella: coordinar no es una instrucción al avión). El 41 % de los fallos
  acierta el avión pero no la tarea. Los fallos son de tramo entero (~80 %): ningún post-proceso en la racha los arregla.
- Ante la misma situación, dos controladores eligen la misma tarea solo el **12.7 %** de las veces.
- El controlador pesa más que el modelo: en el privado se espera **~0.91–0.92** (±5 por el propio 023); las diferencias entre
  modelos son de ±0.5.
- **El oráculo en el público:** los envíos con oráculo sobre A bajaron (B 0.934). Sobre N aguanta mejor (NOR 0.948). La CV puede
  sobreestimar el oráculo en un controlador nuevo; el 023 coordina poco, así que debería afectarle menos.

**No funciona (ya probado; ver la sección 5 del notebook destilado y las secciones 2–24 de `LGB_futuro.ipynb`):** otros modelos
(XGBoost, CatBoost, red), stacks sin oráculo, ajuste de parámetros alrededor de N, bloques `coord`/`block`/`rel`/`geo`, post-procesos
en la racha (HMM, Kalman, modelo por racha, transiciones), variantes del segundo paso, adaptación al controlador (EM, covariate
shift, Bayes jerárquico, estilo), especialistas por tipo, pseudo-etiquetas, etiquetas graduadas, bagging por controladores,
oráculo recortado (solo exit o solo sin huella), más de ~20 semillas.

## 3. Entregas
| Envío | Archivo | CV | Público |
|---|---|---|---|
| **M30_30** (con oráculo) | `futuro/pruebas/submissions/sub_f11_M30_30.csv` | 0.929 | — |
| **N** (sin oráculo) | `futuro/pruebas/submissions/sub_f10_instr_l63_s10_ses.csv` | 0.924 | 0.951 |
| NOR (con oráculo) | `futuro/pruebas/submissions/sub_f10_instr_l63_or_s10_ses.csv` | 0.928 | 0.948 |
| A | `futuro/submissions/sub_suav_lgb_future_clear10_ses.csv` | 0.918 | 0.947 |
| B | `futuro/submissions/sub_suav_lgb_future_clear10_orcoord_ses.csv` | 0.925 | 0.934 |
| Limpia (solo el snapshot) | `submissions/sub_lgb_adri_v2.csv` | 0.664 | 0.642 |

**Pareja final recomendada: M30_30 + N** (una con oráculo y otra sin él; `LGB_futuro.ipynb`, 24l). Si el oráculo transfiere al
023 como dice la CV, cuenta M30; si falla como en el 020, cuenta N. Dos con oráculo (M30 + B) ganan algo más en la simulación, pero
no cubren ese fallo.

## 4. Normas de trabajo
1. **Validación:** siempre `Data/folds.csv` (GroupKFold por controlador). Nunca splits aleatorios por filas o snapshots.
2. **Se decide por la CV, no por el público** (son 2 sesiones de otro controlador; ±1–2 puntos de ruido).
3. **Criterio de adopción:** un cambio entra si mejora la media **y** la mediana por sesión, sube en más sesiones/controladores de los
   que baja y no hunde al peor controlador. Herramienta: `python futuro/pruebas/comparar.py <ref> <cand>`. Mirar también la "CV 023"
   (controladores 002, 011, 010, 006, 017, 015, los más parecidos al 023).
4. **Ruido:** con 1 semilla, menos de 0.005 es ruido; con 3 semillas, ±0.2. Comparar siempre con la referencia **con las mismas
   semillas**; confirmar con 10 antes de adoptar.
5. **Sin fuga:** lo que use etiquetas, solo de los folds de entrenamiento (el oráculo, por fold: `m_lgb_future.oracle_data`). Los
   parámetros de post-procesos y mezclas, con **CV anidada** (4 folds → 5º). Las cifras elegidas mirando todo el OOF son optimistas.
6. **Formato de salidas:** `oof/<m>.csv` y `preds/<m>.csv` con `sample_id,row_index,score` (score alto = más probable) y
   `submissions/sub_<m>.csv` con `sample_id,prediction` (3 `row_index` separados por espacios). Cada línea guarda en su carpeta:
   `oof|preds|submissions` (limpia), `realista/…`, `futuro/…` y `futuro/pruebas/…`.
7. **Registro:** cada ejecución se apunta (`registrar()` de `futuro/pruebas/base_fut.py` → `futuro/pruebas/resultados.jsonl`;
   `m_lgb_future.py` → `experiments/lgb_future_runs.csv`) y cada conclusión en el notebook de su línea.
8. **Git:** se trabaja en `main` (o en una rama y merge). Las salidas de experimentos de `futuro/pruebas/` no se versionan (están en
   `.gitignore`), salvo las de los envíos: añadir con `git add -f` solo lo que alimente una entrega. Los CSV que sobran se archivan
   fuera del repo.
9. **Kaggle:** antes de subir, avisar al equipo (5 envíos al día compartidos) y apuntar la nota en la tabla de la sección 3.

## 5. Estructura del repo
```
Data/                 train/validation/test.csv (no versionados), folds.csv, scorer.py, sample_submission.csv
cv/metric.py          métrica top-3, report(), save_outputs(), to_submission()
src/                  línea limpia: features.py (logic_*), m_clogit.py, m_nn.py (transformer), nn_sweep.py, folds.py
linear/, hierarchical/  logit condicional y P(tipo)×P(tarea|tipo) (también con --future)
stacking/stack.py     media ponderada con CV anidada y meta-ranker (--futuro para la línea futuro)
pruebas/              hilos h1–h5 de la línea limpia; base.py lo usa futuro/pruebas/base_fut.py
notebooks/            00_destilado (resumen), 01_eda, LGB_adri (línea limpia), NN_pablo, panel_modelos
realista/             snapshot + pasado (sin futuro): features_real.py, feat_causal.py, m_lgb_real.py, LGB_realista.ipynb
futuro/               línea con futuro:
  features_future.py    bloques base/logic/past/future y los feat_*.py (clear, coord, block, rel, geo)
  feat_session.py       segundo paso de sesión · feat_oracle.py oráculo · metricas.py métricas robustas
  m_lgb_future.py       LightGBM por bloques (--sesion, --oraculo) · m_gbm_future.py, m_nn_future.py
  LGB_futuro.ipynb      diario completo de la línea (secciones 0–24)
  pruebas/              base_fut.py, suavizado.py, comparar.py, f5_stack.py (sesión/suavizado/stack),
                        f10_instr.py (N, NOR), f11_lab.py + f11_mezcla.py (C, M30, banco de Pablo), diagnósticos f8/f9,
                        resultados.jsonl
experiments/          registros de ejecuciones (lgb_runs, nn_runs, lgb_future_runs, gbm_future_runs, lgb_real_runs)
```

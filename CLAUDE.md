# FlowDataLab – AWARE Prioritisation Challenge (Engage 2, Zagreb)

> Antes de proponer nada, lee **`EQUIPO.md`** (estado del proyecto, entregas y normas de trabajo) y, si hace falta contexto sobre
> por qué funcionan o no los modelos, **`notebooks/00_destilado.ipynb`**. El diario detallado de la línea con futuro es
> `futuro/LGB_futuro.ipynb` (secciones 0–24). Si este archivo y `EQUIPO.md` no coinciden, manda `EQUIPO.md`.

Equipo de 4 (Pablo, Adri, Juanma, Pau), sin reparto fijo: ayuda a quien te hable en lo que pida. Kaggle cierra el
**miércoles 21/10 a las 13:00** (hora de Zagreb).

## El reto
- Cada `sample_id` es un snapshot de la lista de tareas del controlador aéreo; una fila por tarea y exactamente una con
  `target_index = 1`. Se predicen los 3 `row_index` más probables. Métrica: top-3 hit rate.
- Público = controlador 020; privado (decide) = controlador 023. Se puede usar **cualquier información de los datasets**
  (snapshots pasados y futuros de la sesión, etiquetas de otras sesiones de train/validation), confirmado por el organizador.

## Cómo trabajar en este repo
- **Antes de probar algo, comprueba que no está ya probado** en `EQUIPO.md` (sección 2), en la sección 5 de
  `notebooks/00_destilado.ipynb` y en `futuro/pruebas/resultados.jsonl`. Casi todo lo razonable ya se ha medido.
- **Validación:** siempre `Data/folds.csv` (GroupKFold por controlador). Nada de splits aleatorios.
- **Comparar con la referencia con las mismas semillas** y con todas las métricas: `python futuro/pruebas/comparar.py <ref> <cand>`
  (media, mediana por sesión, sesiones/controladores que suben y bajan, bootstrap, escenarios) y la CV 023. Ruido: ±0.2 con 3
  semillas. Una mejora solo cuenta si sube media y mediana sin hundir al peor controlador.
- **Sin fuga:** etiquetas solo de los folds de entrenamiento; el oráculo, por fold; parámetros de post-procesos y mezclas con CV
  anidada. Di explícitamente cuándo una cifra es optimista (elegida con todo el OOF).
- **Reutiliza el código existente:** `futuro/pruebas/base_fut.py` (datos con caché, `lgb_cv`, `guardar`, `registrar`),
  `f10_instr.py` (bloque `instr`, modelos N/NOR), `f11_lab.py` (banco de pruebas de Pablo), `futuro/feat_session.py` (segundo paso),
  `futuro/pruebas/suavizado.py`, `cv/metric.py`. Crea scripts nuevos en lugar de modificar los de otros; si hay que tocarlos, cambios
  mínimos y opcionales (que no cambien el comportamiento por defecto).
- **Salidas:** formato común `sample_id,row_index,score` en la carpeta `oof|preds|submissions` de su línea; regístralas. En
  `futuro/pruebas/` las salidas no se versionan (`.gitignore`) salvo las de las entregas (`git add -f`).
- **Entrenamientos largos:** en segundo plano (nohup), sin bloquear la conversación; responde enseguida y da el estado cuando acabe.
- **Notebooks:** cada conclusión nueva va a la sección correspondiente de su notebook, con cifras, veredicto y qué se aprendió.
- **No subas nada a Kaggle ni hagas commit/push sin que te lo pidan.** No inventes resultados: si algo no se ha ejecutado, dilo.
- Código completo y listo para ejecutar; respuestas concisas y en español.

# f2_coord: features para coordinaciones y tipos raros, sobre futuro

Código: `futuro/pruebas/f2_coord.py` (grupos adri, adri2, coord, cfut y subconjuntos clean/cact). Log: `notas/f2_log.txt`.
cfut (s2_*, NUEVAS de sesión): segundos hasta el próximo cambio o desde el último cambio del estado de coordinación del avión (k_out_*, k_outCoordRevision,
mismatch, herramienta de coordinación, x_isc_*), el siguiente input "resuelve" la coordinación (salida->transfer, entrada->assume), avión activo
(rank por f_next_input), fin de la tarea frente a su tipo y su avión. Ninguna usa target_index.

| nombre (1 semilla salvo best) | CV | min atco | exit_coord | entry_coord |
|---|---|---|---|---|
| f2_ref (todos los bloques) | 0.8996 | 0.746 | 0.566 | 0.699 |
| + adri (alerta+contexto) | 0.8996 | 0.769 | 0.554 | 0.696 |
| + adri2 (exit_coord+deadline+tipo_estado) | 0.9000 | 0.734 | 0.556 | 0.706 |
| + coord (h3 c3_*) | 0.8999 | 0.743 | 0.538 | 0.699 |
| + cfut (17 s2_*) | 0.8933 | 0.754 | 0.528 | 0.650 |
| + adri + clean (5 s2_: ex_next_rev/tool/outc, en_next_isc, coord_next_rk) | 0.9016 | 0.754 | 0.545 | 0.688 |
| **f2_best** = adri+clean, 3 semillas, con preds | **0.9008** | 0.757 | 0.561 | 0.689 |
| lgb_future (referencia, 3 semillas) | 0.9010 | 0.769 | 0.569 | 0.706 |

- Nada mejora a lgb_future: todo cae en ±0.002, que es ruido de semilla. f2_best (0.9008) empata con lgb_future y tiene peor min atco.
- Ningún grupo sube exit/entry_coordination. Con futuro, f_next_input y f_task_end ya marcan el avión activo; las features de
  coordinación solo de snapshot (c3_*, ec_*) no añaden nada, y en exit_coord incluso bajan.
- cfut completo hunde la CV (-0.006): el EDA muestra señal (en entry_coord la correcta tiene el próximo cambio de x_isc a 159 s frente a 543 s),
  pero son 17 columnas muy redundantes con f_*, y el modelo sobreajusta. Las s2_ac_* (avión activo/fin de tarea) sobran.
- Recomendación: no tocar lgb_future. Si se quiere un modelo más para el stack, f2_best (futuro/pruebas/oof|preds/f2_best.csv) da diversidad barata.
  adri por sí solo sube min atco (0.746 -> 0.769) sin coste de CV: es lo único que merece confirmarse con 3 semillas.
- Pendiente (no medido por tiempo): clean y cact solos, sobre ref.

"""
stack_pruebas.py · Juanma (C) · stacking/stack.py sobre los modelos de main (oof/) + los de los hilos (pruebas/oof).

No toca oof/ ni preds/: las salidas van a pruebas/oof|preds/stackp_*.csv y se registran en pruebas/resultados.jsonl.

Uso:  python pruebas/stack_pruebas.py                               # todos los que tengan oof + preds
      python pruebas/stack_pruebas.py lgb_adri h1_cat_yeti ...      # solo esos
      python pruebas/stack_pruebas.py --tag sin_nuevos lgb_adri nn_tf_v2 linear types
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "pruebas"
sys.path[:0] = [str(ROOT / "stacking"), str(P)]
import stack  # noqa: E402
from base import registrar  # noqa: E402

args = sys.argv[1:]
tag = ""
if "--tag" in args:
    i = args.index("--tag"); tag = "_" + args[i + 1]; del args[i:i + 2]
stack.DIRS = [ROOT, P]
stack.OUT = P

# las salidas de stack.py se llaman stack_avg / stack_meta: se renombran a stackp<tag>_*
saved = {}
orig = stack.save_outputs
stack.save_outputs = lambda name, oof, preds, out: saved.update({name: (oof, preds)})
t0 = time.time()
sys.argv = ["stack.py"] + args
stack.main()
modelos = stack.available(args, False)
for name, (oof, preds) in saved.items():
    nombre = name.replace("stack", "stackp" + tag)
    orig(nombre, oof, preds, out=P)
    registrar(nombre, hilo="stack", descripcion="stacking de: " + ", ".join(modelos),
              oof_df=oof, segundos=time.time() - t0, n_features=len(modelos))

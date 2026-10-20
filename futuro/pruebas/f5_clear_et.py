"""f5_clear_et.py · lgb_future + bloque clear (y coord) con extra_trees=True (hilo f5_clear_et).
Combina las dos mejoras que funcionaron por separado: bloque clear (lgb_future_clear, 0.907) y extra_trees (f1, 0.9035).
Uso: python futuro/pruebas/f5_clear_et.py NOMBRE [...]     # 3 semillas + preds de test -> futuro/pruebas/oof|preds
     NOMBRE en VARIANTES (control_clear reproduce lgb_future_clear en esta máquina)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import datos, lgb_cv, guardar, registrar, OUT

BASE = ["base", "logic", "past", "future"]
VARIANTES = {
    "f5_clear_et": (BASE + ["clear"], dict(extra_trees=True)),
    "f5_coordclear_et": (BASE + ["coord", "clear"], dict(extra_trees=True)),
    "f5_control_clear": (BASE + ["clear"], {}),
    "f5_clear_et10": (BASE + ["clear"], dict(extra_trees=True)),   # 10 semillas, como lgb_future_clear10
}
# variantes con otro objetivo/forma: solas no ganan (f1), pero pueden aportar diversidad al stack + segundo paso
VARIANTES.update({
    "f5_clear_binary": (BASE + ["clear"], dict(objective="binary")),
    "f5_clear_xendcg": (BASE + ["clear"], dict(objective="rank_xendcg")),
    "f5_clear_l63": (BASE + ["clear"], dict(num_leaves=63, min_child_samples=100, extra_trees=True)),
})
VARIANTES.update({   # ¿de dónde sale la mejora de l63: tamaño del árbol o extra_trees?
    "f5_clear_l63_noet": (BASE + ["clear"], dict(num_leaves=63, min_child_samples=100)),
    "f5_clear_l127": (BASE + ["clear"], dict(num_leaves=127, min_child_samples=100, extra_trees=True)),
})
VARIANTES["f5_clear_l63_10"] = VARIANTES["f5_clear_l63"]
SEMILLAS = {"f5_clear_et10": tuple(range(10)), "f5_clear_l63_10": tuple(range(10))}


def log(msg):
    with open(OUT / "notas" / "f5_log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def correr(nombre, n_jobs=28):
    lab, te, fs = datos()
    bloques, p = VARIANTES[nombre]
    F = [c for b in bloques for c in fs[b]]
    t0 = time.time()
    seeds = SEMILLAS.get(nombre, (0, 1, 2))
    oof, pred = lgb_cv(lab, F, {**p, "n_jobs": n_jobs}, seeds=seeds, test=te)
    guardar(nombre, lab, oof, te, pred)
    cv = registrar(nombre, hilo="f5_clear_et", descripcion=f"{'+'.join(bloques)} {p} seeds={list(seeds)}",
                   segundos=time.time() - t0, n_features=len(F))
    log(f"{nombre}: CV {cv:.4f} ({time.time() - t0:.0f}s) {'+'.join(bloques)} {p}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        correr(n)

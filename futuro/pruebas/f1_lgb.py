"""f1_lgb.py · variantes de LightGBM sobre las features de futuro (hilo f1_lgb).
Uso: python futuro/pruebas/f1_lgb.py variantes            # 1 semilla cada una, solo OOF
     python futuro/pruebas/f1_lgb.py final NOMBRE [...]   # 3 semillas + preds de test -> f1_best_<nombre>
     python futuro/pruebas/f1_lgb.py mezcla A B [C]       # media de rangos dentro del snapshot (OOF)
"""
import sys, time
sys.path.insert(0, "/home/jm/proyectos/FlowDataLab/futuro/pruebas")
import pandas as pd
from base_fut import datos, lgb_cv, guardar, registrar, OUT

VARIANTES = {
    "f1_ref": {},
    "f1_xendcg": dict(objective="rank_xendcg"),
    "f1_binary": dict(objective="binary"),
    "f1_trunc3": dict(lambdarank_truncation_level=3),
    "f1_trunc10": dict(lambdarank_truncation_level=10),
    "f1_lr03_800": dict(learning_rate=0.03, n_estimators=800),
    "f1_l63_mc100": dict(num_leaves=63, min_child_samples=100),
    "f1_l15_reg": dict(num_leaves=15, min_child_samples=200, reg_lambda=5.0, colsample_bytree=0.5),
    "f1_extratrees": dict(extra_trees=True),
    "f1_et_800": dict(extra_trees=True, n_estimators=800),
}


def params_de(nombre):
    return dict(VARIANTES[nombre])   # truncation_level se ignora con binary / rank_xendcg


def log(msg):
    with open(OUT / "notas" / "f1_log.txt", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def correr(nombre, seeds=(0,), test=False, salida=None):
    lab, te, fs = datos()
    F = fs["base"] + fs["logic"] + fs["past"] + fs["future"]
    t0 = time.time()
    p = params_de(nombre)
    oof, pred = lgb_cv(lab, F, p, seeds=seeds, test=te if test else None)
    salida = salida or nombre
    guardar(salida, lab, oof, te if test else None, pred)
    cv = registrar(salida, hilo="f1_lgb", descripcion=f"{p} seeds={list(seeds)}", segundos=time.time() - t0,
                   n_features=len(F))
    log(f"{salida}: CV {cv:.4f} ({time.time() - t0:.0f}s) {p} seeds={list(seeds)}")


def mezcla(nombres):
    dfs = [pd.read_csv(OUT / "oof" / f"{n}.csv") for n in nombres]
    m = dfs[0][["sample_id", "row_index"]].copy()
    m["score"] = sum(d.groupby("sample_id").score.rank(pct=True) for d in dfs) / len(dfs)
    nombre = "f1_mix_" + "+".join(n.replace("f1_", "") for n in nombres)
    cv = registrar(nombre, hilo="f1_lgb", descripcion="media de rangos OOF de " + ", ".join(nombres), oof_df=m)
    log(f"{nombre}: CV {cv:.4f}")


if __name__ == "__main__":
    modo, args = sys.argv[1], sys.argv[2:]
    if modo == "variantes":
        for n in (args or VARIANTES):
            correr(n)
    elif modo == "final":
        for n in args:
            correr(n, seeds=(0, 1, 2), test=True, salida="f1_best_" + n.replace("f1_", ""))
    elif modo == "mezcla":
        mezcla(args)

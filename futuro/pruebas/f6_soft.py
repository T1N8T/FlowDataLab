"""f6_soft.py · 1.er pase entrenado con etiquetas ruidosas (hilo f6). Mismos parámetros/features que f5_control_clear.
Las etiquetas tienen ruido y el controlador sigue con la misma tarea varios snapshots seguidos:
- grado_k : relevancia graduada para LambdaRank: 2 = respuesta del snapshot, 1 = respuesta en algún snapshot a ±k
            de la misma racha (la misma tarea). Solo cambia el ENTRENAMIENTO; la CV se mide con la etiqueta real.
- aislada : quita del entrenamiento los snapshots cuya respuesta no coincide con la del anterior NI con la del siguiente
            de la racha (cuando existen): probable ruido.
Comparar con f5_control_clear (0.9071; + sesión 0.9142), 3 semillas.
Uso: python futuro/pruebas/f6_soft.py NOMBRE [...]   (NOMBRE en VARIANTES) -> oof/preds + <nombre>_ses
"""
import sys, time
from pathlib import Path
import numpy as np, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import datos, guardar, registrar, PARAMS_FUT, ROOT  # noqa: E402
import suavizado as S  # noqa: E402
sys.path.insert(0, str(ROOT / "futuro"))
import feat_session  # noqa: E402

VARIANTES = {"f6_grado1": ("grado", 1), "f6_grado3": ("grado", 3), "f6_aislada": ("aislada", 0)}


def etiquetas(lab, modo, k):
    y = lab.target_index.values.astype(float)
    keep = np.ones(len(lab), bool)
    nb = S.vecinos(lab[["sample_id", "row_index", "session", "block", "t", "task_name"]], max(k, 1))
    if modo == "grado":
        g = np.zeros(len(lab))
        for j in range(1, k + 1):
            for nom in ("prev", "next"):
                idx = nb[f"{nom}{j}"]; ok = ~np.isnan(idx)
                g[ok] = np.maximum(g[ok], y[idx[ok].astype(int)])
        return np.where(y == 1, 2, g).astype(int), keep
    # aislada: la fila respuesta; ¿es también respuesta en el snapshot anterior / siguiente?
    snap = lab.drop_duplicates("sample_id")[["sample_id", "block", "t"]].sort_values(["block", "t"])
    snap["n"] = snap.groupby("block").sample_id.transform("size")
    snap["pos"] = snap.groupby("block").cumcount()
    info = lab[["sample_id"]].merge(snap, on="sample_id", how="left")
    r = np.flatnonzero(y == 1)
    pv, nx = nb["prev1"][r], nb["next1"][r]
    same_p = np.where(np.isnan(pv), 0, y[np.nan_to_num(pv).astype(int)])
    same_n = np.where(np.isnan(nx), 0, y[np.nan_to_num(nx).astype(int)])
    has_p, has_n = info.pos.values[r] > 0, info.pos.values[r] < info.n.values[r] - 1
    iso = has_p & has_n & (same_p == 0) & (same_n == 0)
    bad = set(lab.sample_id.values[r[iso]])
    keep = ~lab.sample_id.isin(bad).values
    print(f"aislada: se quitan {len(bad)} de {lab.sample_id.nunique()} snapshots del entrenamiento", flush=True)
    return y.astype(int), keep


def correr(nombre, seeds=(0, 1, 2), n_jobs=10):
    lab, te, fs = datos()
    F = fs["base"] + fs["logic"] + fs["past"] + fs["future"] + fs["clear"]
    modo, k = VARIANTES[nombre]
    yt, keep = etiquetas(lab, modo, k)
    lab = lab.assign(_y=yt)
    P = {**PARAMS_FUT, "n_jobs": n_jobs}
    if modo == "grado":
        P["label_gain"] = [0, 1, 3]

    def fit(A, s):
        return lgb.LGBMRanker(**{**P, "random_state": s}).fit(
            A[F], A._y, group=A.groupby("sample_id", sort=False).size().values, categorical_feature=["task_index"])
    t0 = time.time()
    oof = np.zeros(len(lab))
    for f in range(5):
        tr, va = (lab.fold != f).values & keep, (lab.fold == f).values
        oof[va] = np.mean([fit(lab[tr], s).predict(lab.loc[va, F]) for s in seeds], axis=0)
    pred = np.mean([fit(lab[keep], s).predict(te[F]) for s in seeds], axis=0)
    guardar(nombre, lab, oof, te, pred)
    registrar(nombre, hilo="f6", descripcion=f"{modo} k={k}, clear, seeds={list(seeds)}", segundos=time.time() - t0,
              n_features=len(F))
    so, sp, _ = feat_session.sesion(lab, oof, te, pred)
    guardar(nombre + "_ses", lab, so, te, sp)
    registrar(nombre + "_ses", hilo="f6", descripcion=f"{nombre} + segundo paso de sesión")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        correr(n)

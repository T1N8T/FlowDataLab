"""
errores.py · Pablo (A) · RAMA FUTURO · Sección 19 de LGB_futuro.ipynb: dónde falla el mejor modelo sin oráculo y si
test se parece a train. Sin reentrenar (usa futuro/oof/<modelo>.csv).

    1. Fallos por tipo de respuesta, posición en la racha, cambios de respuesta y longitud del tramo.
    2. Tramos (racha con la misma respuesta seguida): cuántos se fallan enteros y cuánto pesan en los fallos.
    3. Huella observable de la tarea etiquetada frente a la top-1 del modelo, en el último snapshot del tramo:
       instrucción compatible en ≤ 60 s, la tarea desaparece en ≤ 30 s o cambia su estado ligado en ≤ 60 s (sc_/sx_).
       Y huella pasada en el primer snapshot: instrucción al avión en los 60 s anteriores o tarea aparecida hace ≤ 30 s.
    4. Test frente a train: tamaño de la lista, ritmo, mezcla de tipos de candidatas y cobertura de huella.

Uso: python futuro/pruebas/errores.py [modelo]   (por defecto lgb_future_clear10_ses)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro"), str(ROOT / "cv")]
import m_lgb_future as M, features_future as ff  # noqa: E402


def huella(R):
    return ((R.f_next_match == 1) & (R.f_next_input <= 60)) | (R.f_task_end <= 30) | (R.sc_task <= 60) | (R.sx_task <= 60)


def tabla(S, col, titulo):
    nf = (~S.hit).sum()
    t = S.groupby(col).agg(n=("hit", "size"), acierto=("hit", "mean"), fallos=("hit", lambda h: (~h).sum()))
    t["% de los fallos"] = (t.fallos / nf * 100).round(1)
    print(f"\n-- {titulo}\n{t.round(3).to_string()}")


def main(modelo="lgb_future_clear10_ses"):
    data = M.build_data()
    lab, te = data["lab"], data["te"]
    L = lab.merge(pd.read_csv(ROOT / "futuro" / "oof" / f"{modelo}.csv"), on=["sample_id", "row_index"], how="left")
    L["rk"] = L.groupby("sample_id").score.rank(ascending=False, method="first")
    S = L[L.target_index == 1].sort_values(["session", "t"]).reset_index(drop=True)
    S["hit"] = S.rk <= 3
    S["tipo"] = S.task_index.map(lambda i: ff.TYPES[i])
    prev, nxt = S.groupby("block").task_name.shift(), S.groupby("block").task_name.shift(-1)
    print(f"{modelo}: CV {S.hit.mean():.4f}, fallos {(~S.hit).sum()} de {len(S)}")
    # 1
    tabla(S, "tipo", "por tipo de respuesta")
    S["pos"] = np.select([S.blk_len == 1, S.blk_pos == 0], ["racha de 1", "1.º de racha"], "resto")
    tabla(S, "pos", "posición en la racha")
    S["cambio"] = np.select([prev.notna() & (prev != S.task_name), nxt.notna() & (nxt != S.task_name)],
                            ["justo tras un cambio", "justo antes de un cambio"], "tramo estable")
    tabla(S, "cambio", "cambios de respuesta dentro de la racha")
    S["tramo_len"] = pd.cut(S.n_tramo, [0, 1, 3, 10, 30, 1e9], labels=["1", "2-3", "4-10", "11-30", ">30"])
    tabla(S, "tramo_len", "longitud del tramo")
    # 2
    S["tramo"] = ((S.task_name != prev) | (S.blk_pos == 0)).cumsum()
    acc = S.groupby("tramo").hit.transform("mean")
    S["clase"] = np.select([acc == 0, acc == 1], ["fallado entero", "acertado entero"], "mixto")
    T = S.drop_duplicates("tramo")
    g = S.groupby("clase").agg(snapshots=("hit", "size"), fallos=("hit", lambda h: (~h).sum()))
    g["tramos"] = T.clase.value_counts()
    g["% de los fallos"] = (g.fallos / g.fallos.sum() * 100).round(1)
    print(f"\n-- tramos ({len(T)}); acierto por tramo {(T.clase == 'acertado entero').mean():.3f}\n{g.to_string()}")
    e = S.groupby("tramo").hit.apply(lambda h: (~h).sum()).sort_values(ascending=False)
    c = e.cumsum() / e.sum()
    print(f"tramos que acumulan el 50 % / 80 % de los fallos: {(c < 0.5).sum() + 1} / {(c < 0.8).sum() + 1}")
    print(S[S.clase == "fallado entero"].drop_duplicates("tramo").tipo.value_counts().head(6).to_dict())
    # 3
    top = L[L.rk == 1].set_index("sample_id")
    lab1 = L[L.target_index == 1].set_index("sample_id")
    last, first = S.groupby("tramo").tail(1), S.groupby("tramo").head(1)
    X = pd.DataFrame({"clase": last.clase.values, "etiqueta con huella": huella(lab1.loc[last.sample_id]).values,
                      "top-1 con huella": huella(top.loc[last.sample_id]).values})
    print(f"\n-- huella futura (último snapshot del tramo)\n{X.groupby('clase').mean().round(3).to_string()}")
    Ff = X[X.clase == "fallado entero"]
    print(f"fallados enteros: etiqueta sin huella y top-1 con huella {((~Ff['etiqueta con huella']) & Ff['top-1 con huella']).mean():.3f}")
    R = lab1.loc[first.sample_id]
    P = pd.DataFrame({"clase": first.clase.values, "huella futura": huella(R).values,
                      "instrucción ≤ 60 s antes": (R.a_since_flight <= 60).values, "tarea nueva ≤ 30 s": (R.p_task_start <= 30).values})
    print(f"\n-- huella pasada (primer snapshot del tramo)\n{P.groupby(['clase', 'huella futura']).agg(n=('clase', 'size'), instr=('instrucción ≤ 60 s antes', 'mean'), nueva=('tarea nueva ≤ 30 s', 'mean')).round(3).to_string()}")
    # 4
    D = pd.concat([lab.assign(g="train/val"), te.assign(g=te.session)])
    snap = D.drop_duplicates("sample_id")
    print("\n-- test frente a train (medianas por snapshot)")
    print(snap.groupby("g")[["seq_len", "blk_len", "a_n300"]].median().to_string())
    mix = D.groupby("g").task_index.value_counts(normalize=True).unstack().fillna(0) * 100
    mix.columns = [ff.TYPES[c][:12] for c in mix.columns]
    print(mix.round(1).to_string())
    cov = huella(D).groupby(D.sample_id).any().groupby(snap.set_index("sample_id").g).mean() * 100
    print("% de snapshots con alguna candidata con huella:", cov.round(1).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:])

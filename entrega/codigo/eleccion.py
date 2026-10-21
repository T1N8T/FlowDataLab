"""eleccion.py · línea "futuro" · elegir UNA entrega por su privado esperado. La clasificación final es solo el privado (controlador
023, el 51 % de las filas de test); el público (020, el 49 %) no cuenta y ya se vio que engaña.
Candidatas (OOF + preds + sub): N, N30, NOR, C30, M30_30, A, B. Todo sobre salidas existentes; no entrena (~2 min).
Métodos, todos como diferencia frente a N (la de mejor público conocido):
 1. regla de decisión validada: cada controlador de la CV hace de "privado"; se elige la candidata con mejor CV en los otros 17 y se
    mide en él (¿acierta el procedimiento "elegir por CV"?).
 2. riesgo a la baja por controlador: media de los 3 peores, P(perder > 1 punto frente a N), peor controlador.
 3. cota por desacuerdo en el test del 023 (y del 020): dos entregas solo pueden diferir en los snapshots con top-3 distinto.
 4. CV reponderada por snapshot hacia el 023: pesos = densidad del 023 / densidad de la CV con variables observables sin etiqueta
    (¿instrucción compatible ≤ 30 s?, ¿instrucción ≤ 10 s?, longitud de la lista, top-1 de N es coordinación), por regresión logística.
 5. señal sin etiquetas validada: estabilidad del top-1 dentro de la racha y repetición de la tarea favorita en otra racha; solo se
    usa si en la CV predice la diferencia de acierto entre modelos.
Público: real si la entrega está subida; si no, el público real de la entrega subida más parecida en el 020 + Δ esperado por la CV
(sesiones de los escenarios 2 y 3, como el 020).
Uso: python entrega/codigo/eleccion.py   -> imprime las tablas (lo llama la sección 6b de futuro.ipynb)
"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import hits, leer, SALIDAS  # noqa: E402
from features_futuro import construir  # noqa: E402

CAND = {  # nombre: (oof/preds en salidas/, envío en salidas/submissions/, público real o None)
    "N": ("N10_ses", "sub_N10_ses.csv", 0.951),
    "N30": ("N30_ses", "sub_N30_ses.csv", None),
    "NOR": ("C10_ses", "sub_C10_ses.csv", 0.948),
    "C30": ("C30_ses", "sub_C30_ses.csv", None),
    "M30_30": ("M30_30", "sub_M30_30.csv", 0.953),
    "A": ("A", "sub_A.csv", 0.947),
    "B": ("B", "sub_B.csv", 0.934),
}
W_PUB, W_PRIV, REF = 0.0, 1.0, "N"   # la nota final es solo el privado
K = ["sample_id", "row_index"]




def top3(df):
    df = df.sort_values(["sample_id", "score"], ascending=[True, False])
    return df.groupby("sample_id").row_index.apply(lambda r: frozenset(r.head(3)))


def top1(df, d):
    m = d[K + ["task_name"]].merge(df, on=K)
    return m.sort_values(["sample_id", "score"], ascending=[True, False]).drop_duplicates("sample_id").set_index("sample_id").task_name


def main():
    lab, te, _ = construir()
    snap = lab.drop_duplicates("sample_id").set_index("sample_id")[["session", "scenario", "atco", "block", "t"]]
    snapt = te.drop_duplicates("sample_id").set_index("sample_id")[["session", "scenario", "atco", "block", "t"]]
    H = pd.DataFrame({k: hits(leer(v[0], "oof")) for k, v in CAND.items()}).join(snap)
    G = H.groupby("atco")[list(CAND)].mean() * 100
    res = {}

    # 1 · regla de decisión validada (elegir por CV en 17, medir en el 18.º)
    eleg = []
    for c in G.index:
        otros = H[H.atco != c][list(CAND)].mean()
        k = otros.idxmax()
        eleg.append(dict(atco=c, elegida=k, acierto=G.loc[c, k], N=G.loc[c, REF]))
    E = pd.DataFrame(eleg)
    res["regla"] = dict(elegidas=E.elegida.value_counts().to_dict(), media_regla=E.acierto.mean(), media_N=E.N.mean(),
                        delta=E.acierto.mean() - E.N.mean())

    # 2 · riesgo a la baja por controlador (frente a N)
    D = G.sub(G[REF], axis=0)
    riesgo = pd.DataFrame({
        "cv": H[list(CAND)].mean() * 100,
        "media_por_controlador": G.mean(),
        "media_3_peores": G.apply(lambda s: s.nsmallest(3).mean()),
        "peor": G.min(),
        "delta_vs_N_media": D.mean(),
        "P(pierde >1 pt vs N)": (D < -1).mean(),
        "P(gana vs N)": (D > 0).mean(),
    })

    # 3 · cota por desacuerdo en test (020 y 023) frente a N
    t3 = {k: top3(leer(v[0], "preds")) for k, v in CAND.items()}
    at = snapt.atco
    cota = {}
    for k in CAND:
        for a in (20, 23):
            ids = at[at == a].index
            cota[(k, a)] = 1 - np.mean([t3[k][s] == t3[REF][s] for s in ids])
    riesgo["distinto_de_N_020"] = [cota[(k, 20)] * 100 for k in CAND]
    riesgo["distinto_de_N_023"] = [cota[(k, 23)] * 100 for k in CAND]

    # 4 · CV reponderada hacia el 023 con variables observables por snapshot
    def feats(d, n_df):
        g = d.groupby("sample_id")
        X = pd.DataFrame({"instr30": g.si_match.apply(lambda s: (s <= 30).any()).astype(float),
                          "instr10": g.f_next_input.apply(lambda s: (s <= 10).any()).astype(float),
                          "lista": g.size().astype(float)})
        t1 = d[K + ["task_index"]].merge(n_df, on=K).sort_values(["sample_id", "score"], ascending=[True, False]) \
            .drop_duplicates("sample_id").set_index("sample_id").task_index
        X["top1_coord"] = t1.reindex(X.index).isin([2, 3]).astype(float)
        X["lista2"] = X.lista ** 2 / 100
        return X
    XL = feats(lab, leer(CAND[REF][0], "oof"))
    XT = feats(te[te.atco == 23], leer(CAND[REF][0], "preds"))
    Z = pd.concat([XL.assign(y=0), XT.assign(y=1)])
    lr = LogisticRegression(max_iter=2000).fit(Z.drop(columns="y"), Z.y)
    p = lr.predict_proba(XL)[:, 1]
    w = np.clip(p / (1 - p), 0, np.quantile(p / (1 - p), 0.99))
    w = pd.Series(w / w.mean(), index=XL.index)
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    Hw = H[list(CAND)].mul(w.reindex(H.index), axis=0).sum() / w.reindex(H.index).sum() * 100
    riesgo["cv_reponderada_023"] = Hw
    riesgo["delta_reponderada_vs_N"] = Hw - Hw[REF]
    res["reponderacion"] = dict(ess=float(ess), medias_cv=XL.mean().round(3).to_dict(), medias_023=XT.mean().round(3).to_dict())

    # 5 · señal sin etiquetas: estabilidad del top-1 en la racha y repetición en otra racha
    def proxies(d, df, sn):
        t1 = top1(df, d).rename("t1").to_frame().join(sn).sort_values(["session", "t"])
        same = t1.t1.eq(t1.groupby("block").t1.shift()) & t1.block.eq(t1.block)
        mask = t1.groupby("block").t1.shift().notna()
        moda = t1.groupby(["session", "block"]).t1.agg(lambda s: s.mode().iloc[0]).reset_index()
        moda["rep"] = moda.groupby(["session", "t1"]).cumcount() > 0
        out = pd.DataFrame({"estab": same[mask].groupby(t1.atco[mask]).mean(),
                            "repite": moda.groupby(moda.session.map(sn.drop_duplicates("session").set_index("session").atco)).rep.mean()})
        return out
    PL = {k: proxies(lab, leer(v[0], "oof"), snap) for k, v in CAND.items()}
    PT = {k: proxies(te, leer(v[0], "preds"), snapt) for k, v in CAND.items()}
    filas = []
    for k in CAND:
        if k == REF: continue
        dd = pd.DataFrame({"dacc": D[k], "destab": PL[k].estab - PL[REF].estab, "drep": PL[k].repite - PL[REF].repite})
        filas.append(dd)
    DD = pd.concat(filas)
    r_estab, r_rep = DD.dacc.corr(DD.destab, method="spearman"), DD.dacc.corr(DD.drep, method="spearman")
    res["senal"] = dict(r_estab=float(r_estab), r_repite=float(r_rep), n=len(DD))
    riesgo["estab_023_vs_N"] = [(PT[k].estab.get(23) - PT[REF].estab.get(23)) * 100 for k in CAND]
    riesgo["repite_023_vs_N"] = [(PT[k].repite.get(23) - PT[REF].repite.get(23)) * 100 for k in CAND]

    # público: real o estimado (entrega subida más parecida en el 020 + Δ CV en sesiones de los escenarios 2 y 3)
    s = H.groupby(["scenario", "session"])[list(CAND)].mean()
    def dcv(x, y):
        d = (s[x] - s[y]) * 100
        return float((np.add.outer(d.loc[2].values, d.loc[3].values).ravel() / 2).mean())
    pub = {}
    for k, (_, _, real) in CAND.items():
        if real is not None:
            pub[k] = (real, "real")
            continue
        ids = at[at == 20].index
        cerca = max((j for j in CAND if CAND[j][2] is not None),
                    key=lambda j: np.mean([t3[k][i] == t3[j][i] for i in ids]))
        pub[k] = (CAND[cerca][2] + dcv(k, cerca) / 100 * 0.5, f"estimado desde {cerca}")   # la CV sobreestima: se toma la mitad
    riesgo["publico"] = [pub[k][0] for k in CAND]
    riesgo["publico_fuente"] = [pub[k][1] for k in CAND]

    # nota final esperada frente a N = Δprivado (media de CV reponderada y media por controlador); el público se informa, no cuenta
    dpriv = (riesgo.delta_reponderada_vs_N + riesgo.delta_vs_N_media) / 2
    riesgo["delta_privado_esperado"] = dpriv
    riesgo["delta_final_esperado"] = W_PUB * (riesgo.publico - riesgo.publico[REF]) * 100 + W_PRIV * dpriv
    riesgo = riesgo.round(3)
    pd.set_option("display.width", 250)
    print(riesgo.sort_values("delta_final_esperado", ascending=False).to_string())
    print("\nregla 'elegir por CV' validada:", res["regla"])
    print("reponderación:", res["reponderacion"])
    print("señal sin etiquetas (Spearman con Δacierto por controlador):", res["senal"])
    return riesgo, res


if __name__ == "__main__":
    main()

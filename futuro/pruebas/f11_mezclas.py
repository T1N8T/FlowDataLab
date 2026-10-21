"""
f11_mezclas.py · Juanma (C) · RAMA FUTURO · Mezclas de N con B y con el oráculo, sin reentrenar.

Parte de OOF/preds ya calculados:
  N   = f10_instr_l63_s10_ses            (referencia, público 0.951)
  B   = lgb_future_clear10_orcoord_ses   (B sin suavizar)
  SB  = suav_lgb_future_clear10_orcoord_ses (B suavizado; scores = probabilidades)
  OR  = f10_instr_or_s3_ses              (instr + oráculo coord, 3 semillas)
  SN  = suav_f10_instr_l63_s10_ses       (N suavizado)
Mezclas (todo sobre log-softmax por snapshot, o rangos normalizados dentro del snapshot):
  log    : (1-w)·lsm(N) + w·lsm(X)
  rank   : (1-w)·rank(N) + w·rank(X)
  coord  : lsm(N) + w·(lsm(X) - lsm(N)) solo en filas de coordinación (task_index 2/3); resto = N
  ecoord / xcoord: igual pero solo entry_coordination (2) / solo exit_coordination (3)
w en 0..1 (paso 0.05) elegido por CV anidada: para predecir el fold f se elige w con los otros 4 folds.
No usa etiquetas del fold que se predice (los OOF tampoco las vieron). El w final para test = el mejor con los 5 folds.
Uso: python futuro/pruebas/f11_mezclas.py
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402

CV023 = [2, 11, 10, 6, 17, 15]
FUENTES = dict(N="f10_instr_l63_s10_ses", B="lgb_future_clear10_orcoord_ses", SB="suav_lgb_future_clear10_orcoord_ses",
               OR="f10_instr_or_s3_ses", SN="suav_f10_instr_l63_s10_ses")
PROB = {"SB", "SN"}           # los suav_* guardan probabilidades
WS = np.round(np.arange(0, 1.0001, 0.05), 2)


def leer(nombre, carpeta):
    for b in (ROOT / "futuro", OUT):
        p = b / carpeta / f"{nombre}.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{carpeta}/{nombre}")


def lsm(code, s):
    s = np.asarray(s, float)
    mx = pd.Series(s).groupby(code).transform("max").values
    sh = s - mx
    return sh - np.log(pd.Series(np.exp(sh)).groupby(code).transform("sum").values)


def rnk(code, s):
    """rango normalizado dentro del snapshot (0 = peor, 1 = mejor)."""
    r = pd.Series(s).groupby(code).rank(method="average").values
    n = pd.Series(np.ones(len(s))).groupby(code).transform("sum").values
    return (r - 1) / np.maximum(n - 1, 1)


class Datos:
    def __init__(self, df, y=None):
        self.code = pd.factorize(df.sample_id)[0]
        self.sid = df.sample_id.values
        self.ti = df.task_index.values
        self.n = self.code.max() + 1
        if y is not None:
            self.tpos = np.flatnonzero(y)[self.code]

    def hits(self, s):
        """acierto top-3 por snapshot (empates: gana la fila anterior, como stack.hit3)."""
        ts = s[self.tpos]
        better = (s > ts) | ((s == ts) & (np.arange(len(s)) < self.tpos))
        return (np.bincount(self.code, weights=better, minlength=self.n) < 3).astype(float)

    def top3_tipo(self, s, tipo=2):
        """% de snapshots con alguna fila del tipo dado en el top-3."""
        o = np.lexsort((-s, self.code))
        pos = np.empty(len(s), int)
        pos[o] = np.arange(len(s))
        start = np.r_[0, np.cumsum(np.bincount(self.code))[:-1]]
        rk = pos - start[self.code]
        m = (rk < 3) & (self.ti == tipo)
        return np.bincount(self.code, weights=m, minlength=self.n).clip(0, 1)


def cargar(lab, te):
    S, T = {}, {}
    for k, nom in FUENTES.items():
        for dst, df, carp in ((S, lab, "oof"), (T, te, "preds")):
            x = df[["sample_id", "row_index"]].merge(leer(nom, carp), on=["sample_id", "row_index"], how="left")
            assert x.score.notna().all(), f"{nom} {carp} incompleto"
            v = x.score.values
            dst[k] = np.log(np.clip(v, 1e-12, None)) if k in PROB else v
    return S, T


def mezcla(D, a, b, w, modo):
    """a, b: scores crudos (ya en log si eran probabilidades)."""
    if modo == "rank":
        return (1 - w) * rnk(D.code, a) + w * rnk(D.code, b)
    la, lb = lsm(D.code, a), lsm(D.code, b)
    if modo == "log":
        return (1 - w) * la + w * lb
    m = {"coord": np.isin(D.ti, [2, 3]), "ecoord": D.ti == 2, "xcoord": D.ti == 3}[modo]
    return la + w * (lb - la) * m


def main():
    t0 = time.time()
    lab, te, _ = datos()
    S, T = cargar(lab, te)
    DL, DT = Datos(lab, lab.target_index.values), Datos(te)
    snap = lab.drop_duplicates("sample_id")
    fold, atco = snap.fold.values, snap.atco.values
    te_at = te.drop_duplicates("sample_id").atco.values

    def resumen(h):
        return dict(cv=h.mean(), cv023=h[np.isin(atco, CV023)].mean(), peor=pd.Series(h).groupby(atco).mean().min())

    hN = DL.hits(S["N"])
    ref = resumen(hN)
    ecN = DT.top3_tipo(lsm(DT.code, T["N"]))[te_at == 20].mean()
    print(f"N: CV {ref['cv']:.4f}  CV023 {ref['cv023']:.4f}  peor {ref['peor']:.4f}  entry_coord top3 en 020 {ecN:.3f}",
          flush=True)
    for k in S:
        r = resumen(DL.hits(S[k]))
        ec = DT.top3_tipo(T[k])[te_at == 20].mean()
        print(f"  fuente {k:3s}: CV {r['cv']:.4f}  CV023 {r['cv023']:.4f}  peor {r['peor']:.4f}  ec020 {ec:.3f}")

    filas = []
    for x in ["B", "SB", "OR", "SN"]:
        for modo in ["log", "rank", "coord", "ecoord", "xcoord"]:
            H = np.stack([DL.hits(mezcla(DL, S["N"], S[x], w, modo)) for w in WS])     # (nw, nsnap)
            # CV anidada: w elegido con 4 folds -> 5º
            hn = np.zeros(H.shape[1]); wsel = []
            for f in range(5):
                tr = fold != f
                wi = int(np.argmax(H[:, tr].mean(1)))
                hn[~tr] = H[wi, ~tr]; wsel.append(float(WS[wi]))
            wbest = float(WS[int(np.argmax(H.mean(1)))])
            rn = resumen(hn)
            sT = mezcla(DT, T["N"], T[x], wbest, modo)
            ec = DT.top3_tipo(sT)[te_at == 20].mean()
            fila = dict(x=x, modo=modo, cv_anidada=rn["cv"], cv023_anid=rn["cv023"], peor_anid=rn["peor"],
                        w_full=wbest, cv_full=H.mean(1).max(), w_folds=wsel, ec020=ec,
                        curva={float(w): round(float(v), 4) for w, v in zip(WS, H.mean(1))})
            filas.append(fila)
            print(f"{x:3s} {modo:6s}: anidada {rn['cv']:.4f} ({(rn['cv'] - ref['cv']) * 100:+.2f})  CV023 "
                  f"{rn['cv023']:.4f}  peor {rn['peor']:.4f}  w_full {wbest}  (CV {fila['cv_full']:.4f})  "
                  f"w_folds {wsel}  ec020 {ec:.3f}", flush=True)
    json.dump(filas, open(OUT / "f11_mezclas_res.json", "w"), indent=1, default=float)

    # guardar las 3 mejores por CV anidada (con w_full para test)
    filas.sort(key=lambda r: -r["cv_anidada"])
    for r in filas[:3]:
        nom = f"f11_mezclas_{r['x']}_{r['modo']}"
        so = mezcla(DL, S["N"], S[r["x"]], r["w_full"], r["modo"])
        sp = mezcla(DT, T["N"], T[r["x"]], r["w_full"], r["modo"])
        guardar(nom, lab, so, te, sp)
        registrar(nom, hilo="f11", segundos=time.time() - t0,
                  descripcion=f"mezcla {r['modo']} de N (f10_instr_l63_s10_ses) con {FUENTES[r['x']]}, w={r['w_full']} "
                              f"(CV anidada {r['cv_anidada']:.4f}; CV del OOF guardado = optimista)",
                  extra={k: r[k] for k in ("cv_anidada", "cv023_anid", "peor_anid", "w_full", "w_folds", "ec020")})
        print("guardado", nom, flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__" and not sys.argv[1:]:
    main()


def extra():
    """¿La ganancia de N+OR es por el oráculo o solo por mezclar dos modelos? Mezcla log con otros f10 sin oráculo,
    y mezcla a tres N + OR + SB(solo coordinación), todo con CV anidada."""
    global FUENTES
    FUENTES = dict(N="f10_instr_l63_s10_ses", OR="f10_instr_or_s3_ses", SB="suav_lgb_future_clear10_orcoord_ses",
                   REF="f10_ref_s3_ses", I31="f10_instr_s3_ses", REFOR="f10_ref_or_s3_ses", B="lgb_future_clear10_orcoord_ses")
    lab, te, _ = datos()
    S, T = cargar(lab, te)
    DL, DT = Datos(lab, lab.target_index.values), Datos(te)
    snap = lab.drop_duplicates("sample_id")
    fold, atco = snap.fold.values, snap.atco.values
    te_at = te.drop_duplicates("sample_id").atco.values
    base = DL.hits(S["N"]).mean()

    def anid(H):
        hn = np.zeros(H.shape[1]); ws = []
        for f in range(5):
            tr = fold != f
            wi = int(np.argmax(H[:, tr].mean(1))); hn[~tr] = H[wi, ~tr]; ws.append(wi)
        return hn, ws, int(np.argmax(H.mean(1)))

    for x in ["REF", "I31", "REFOR"]:
        H = np.stack([DL.hits(mezcla(DL, S["N"], S[x], w, "log")) for w in WS])
        hn, ws, wb = anid(H)
        ec = DT.top3_tipo(mezcla(DT, T["N"], T[x], WS[wb], "log"))[te_at == 20].mean()
        print(f"N+{x} log: anidada {hn.mean():.4f} ({(hn.mean() - base) * 100:+.2f}) CV023 {hn[np.isin(atco, CV023)].mean():.4f} "
              f"peor {pd.Series(hn).groupby(atco).mean().min():.4f} w_full {WS[wb]} w_folds {[WS[i] for i in ws]} ec020 {ec:.3f}",
              flush=True)
    # a tres: primero N+OR log con w1, luego coordinación hacia SB con w2
    grid = [(w1, w2) for w1 in np.arange(0, 0.81, 0.1) for w2 in np.arange(0, 1.01, 0.1)]
    def tres(D, A, w1, w2):
        m = lsm(D.code, mezcla(D, A["N"], A["OR"], w1, "log"))
        return mezcla(D, m, A["SB"], w2, "coord")
    H = np.stack([DL.hits(tres(DL, S, w1, w2)) for w1, w2 in grid])
    hn, ws, wb = anid(H)
    w1, w2 = grid[wb]
    sT = tres(DT, T, w1, w2)
    ec = DT.top3_tipo(sT)[te_at == 20].mean()
    print(f"N+OR log + SB coord: anidada {hn.mean():.4f} ({(hn.mean() - base) * 100:+.2f}) CV023 "
          f"{hn[np.isin(atco, CV023)].mean():.4f} peor {pd.Series(hn).groupby(atco).mean().min():.4f} "
          f"full {grid[wb]} (CV {H[wb].mean():.4f}) folds {[tuple(np.round(grid[i], 2)) for i in ws]} ec020 {ec:.3f}", flush=True)
    nom = "f11_mezclas_N_OR_SBcoord"
    guardar(nom, lab, tres(DL, S, w1, w2), te, sT)
    registrar(nom, hilo="f11", descripcion=f"N+OR log (w1={w1:.1f}) y luego coord hacia SB (w2={w2:.1f}); "
              f"CV anidada {hn.mean():.4f}", extra=dict(cv_anidada=round(hn.mean(), 4), ec020=round(ec, 3)))


if __name__ == "__main__" and sys.argv[1:] == ["extra"]:
    extra()

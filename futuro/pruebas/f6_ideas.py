"""f6_ideas.py · post-procesos nuevos sobre oof/preds de futuro (hilo f6). No reentrena nada.
- hmm    : la sesión como cadena oculta. En cada racha (block) el controlador sigue con la misma tarea con prob. s o cambia
           (uniforme). Emisión = p^tau (p = softmax del score en el snapshot). Forward-backward -> P(tarea | racha entera).
           s y tau se eligen con CV anidada (4 folds -> 5º). Sustituye al suavizado (que promedia a ojo).
- prior  : adaptación al controlador nuevo (prior shift, EM de Saerens). Con las predicciones de SUS snapshots (sin
           etiquetas) se estima su reparto por tipo de tarea y se reescala p(tarea) por (pi_atco / pi_resto)^lam.
           En OOF cada controlador es "nuevo" (su OOF no lo vio). lam con CV anidada.
- escen  : por escenario, el modelo (de una lista) que mejor va en los folds de entrenamiento; se mide en el 5º.
Cada salida se compara con REF por sesión y por escenario (criterio de LGB_futuro.ipynb, 12d).
Uso: python futuro/pruebas/f6_ideas.py hmm MODELO [...]     -> hmm_<modelo>
     python futuro/pruebas/f6_ideas.py prior MODELO [...]   -> prior_<modelo>
     python futuro/pruebas/f6_ideas.py escen NOMBRE M1 M2 [...] -> escen_<nombre>
"""
import sys, itertools, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_fut import OUT, ROOT, datos, guardar, registrar  # noqa: E402
import suavizado as S  # noqa: E402

REF = "suav_lgb_future_clear10_ses"
K = ["sample_id", "row_index"]
_lab, _te, _ = datos()
_cols = K + ["session", "block", "t", "task_name", "task_index", "scenario", "atco"]
LS = _lab[_cols + ["target_index", "fold"]].assign(y=_lab.target_index).reset_index(drop=True)
TS = _te[_cols].copy().reset_index(drop=True)


def softmax(d, s):
    return np.exp(S.lsm(d, s))


def cargar(d, modelo, carpeta):
    v = d[K].merge(S.leer(modelo, carpeta), on=K, how="left").score.values
    assert not np.isnan(v).any(), f"{modelo}/{carpeta} incompleto"
    return v


def comparar(nombre, o):
    """Δ frente a REF por sesión y escenario (en puntos)."""
    L = LS.assign(a=o, b=cargar(LS, REF, "oof"))
    code = pd.factorize(L.sample_id)[0]
    tpos = np.flatnonzero(L.y.values)[code]
    def hit(s):
        ts = s[tpos]; better = (s > ts) | ((s == ts) & (np.arange(len(s)) < tpos))
        return np.bincount(code, weights=better) < 3
    snap = L.drop_duplicates("sample_id")[["sample_id", "session", "scenario"]].assign(ha=hit(L.a.values), hb=hit(L.b.values))
    ses = snap.groupby(["scenario", "session"])[["ha", "hb"]].mean()
    ses["d"] = (ses.ha - ses.hb) * 100
    g = ses.groupby("scenario").d.agg(media="mean", mediana="median", suben=lambda x: (x > 0).sum(), bajan=lambda x: (x < 0).sum())
    print(f"{nombre} - {REF}: Δ medio {(snap.ha.mean() - snap.hb.mean()) * 100:+.2f}  mediana por sesión {ses.d.median():+.2f}  "
          f"suben/bajan {(ses.d > 0).sum()}/{(ses.d < 0).sum()}\n" + g.round(2).to_string(), flush=True)
    return dict(d_medio=round(float((snap.ha.mean() - snap.hb.mean()) * 100), 3), d_mediana_sesion=round(float(ses.d.median()), 3),
                por_escenario={str(k): round(float(v), 3) for k, v in g.media.items()})


def anidado(L, cands, base=None):
    """cands: {params: score}. Elige en 4 folds, mide en el 5º. Devuelve (cv_anidada, params_final, por_fold)."""
    n, tot, el = L.sample_id.nunique(), 0, []
    for f in range(5):
        tr, va = (L.fold != f).values, (L.fold == f).values
        g = max(cands, key=lambda g: S.hit3(L, cands[g], tr))
        tot += S.hit3(L, cands[g], va) * L[va].sample_id.nunique(); el.append(g)
    return tot / n, max(cands, key=lambda g: S.hit3(L, cands[g])), el


def salvar(nombre, o, p, desc, extra):
    guardar(nombre, LS, o, TS, p)
    extra.update(comparar(nombre, o))
    registrar(nombre, hilo="f6", descripcion=desc, extra=extra)


# ----------------------------------------------------------------------------- hmm
def pos_en_racha(d):
    snap = d.drop_duplicates("sample_id")[["sample_id", "block", "t"]].sort_values(["block", "t"])
    snap["pos"] = snap.groupby("block").cumcount()
    return d[["sample_id"]].merge(snap[["sample_id", "pos"]], on="sample_id", how="left").pos.values


def preparar_hmm(d):
    nb = S.vecinos(d, 1)
    code = pd.factorize(d.sample_id)[0]
    return dict(pos=pos_en_racha(d), prev=nb["prev1"], next=nb["next1"], code=code,
                kcnt=np.bincount(code)[code].astype(float))


def _norm(x, code):
    return x / np.bincount(code, weights=x)[code]


def hmm(p, H, s, tau):
    e = p ** tau
    pos, prev, nxt, code, kc = H["pos"], H["prev"], H["next"], H["code"], H["kcnt"]
    order = [np.flatnonzero(pos == j) for j in range(int(pos.max()) + 1)]
    a = np.zeros_like(p)
    for j, r in enumerate(order):
        if j == 0:
            a[r] = e[r]
        else:
            pr = prev[r]; ok = ~np.isnan(pr)
            same = np.zeros(len(r)); same[ok] = a[pr[ok].astype(int)]
            a[r] = e[r] * (s * same + (1 - s) / kc[r])
        a[r] = a[r] / np.bincount(code[r], weights=a[r], minlength=code.max() + 1)[code[r]]
    b = np.ones_like(p)
    eb = np.zeros_like(p)
    for j in range(len(order) - 1, -1, -1):
        r = order[j]
        if j < len(order) - 1:
            nx = nxt[r]; ok = ~np.isnan(nx)
            same = np.zeros(len(r)); same[ok] = eb[nx[ok].astype(int)]
            # suma de e*b del snapshot siguiente (el de la misma racha); si no hay siguiente, b = 1
            r2 = order[j + 1]
            tot = np.bincount(code[r2], weights=eb[r2], minlength=code.max() + 1)
            k2 = np.bincount(code[r2], minlength=code.max() + 1)
            # snapshot siguiente de cada fila: el de su vecino si la tarea sigue; si no, cualquiera de la racha
            nxt_snap = _snap_siguiente(H, r)
            has = nxt_snap >= 0
            bb = np.ones(len(r))
            bb[has] = s * same[has] + (1 - s) * tot[nxt_snap[has]] / np.maximum(k2[nxt_snap[has]], 1)
            b[r] = bb
        b[r] = b[r] / np.bincount(code[r], weights=b[r], minlength=code.max() + 1)[code[r]]
        eb[r] = e[r] * b[r]
    return _norm(a * b, code)


def _snap_siguiente(H, r):
    return H["nsnap"][r]


def con_snap_siguiente(d, H):
    """code del snapshot siguiente en la misma racha (o -1)."""
    snap = d.drop_duplicates("sample_id")[["sample_id", "block", "t"]].assign(code=lambda x: pd.factorize(d.sample_id)[0][x.index])
    snap = snap.sort_values(["block", "t"])
    snap["ncode"] = snap.groupby("block").code.shift(-1).fillna(-1).astype(int)
    H["nsnap"] = d[["sample_id"]].merge(snap[["sample_id", "ncode"]], on="sample_id", how="left").ncode.values
    return H


GRID_HMM = list(itertools.product((0.5, 0.8, 0.9, 0.95, 0.98, 0.99), (0.15, 0.25, 0.35, 0.5, 0.75, 1.0)))   # (s, tau)


def correr_hmm(modelos):
    HL, HT = con_snap_siguiente(LS, preparar_hmm(LS)), con_snap_siguiente(TS, preparar_hmm(TS))
    for m in modelos:
        t0 = time.time()
        pl, pt = softmax(LS, cargar(LS, m, "oof")), softmax(TS, cargar(TS, m, "preds"))
        cands = {g: hmm(pl, HL, *g) for g in GRID_HMM}
        cv, g, el = anidado(LS, cands)
        h0 = S.hit3(LS, pl)
        print(f"hmm_{m}: {h0:.4f} -> CV anidada {cv:.4f} ({cv - h0:+.4f}); por fold {el}; final {g} ({time.time() - t0:.0f}s)", flush=True)
        salvar(f"hmm_{m}", cands[g], hmm(pt, HT, *g), f"HMM forward-backward en la racha sobre {m}, (s, tau)={g}",
               dict(cv_anidada=round(cv, 4), cv_sin=round(h0, 4), params=list(g), params_por_fold=[list(x) for x in el]))


# ----------------------------------------------------------------------------- prior shift
def prior_shift(d, p, lam, iters=10):
    """Reescala p por controlador: (pi_atco(tipo) / pi_resto(tipo))^lam, con pi estimado por EM sobre sus snapshots."""
    if lam == 0:
        return p
    ty, at, code = d.task_index.values, d.atco.values, pd.factorize(d.sample_id)[0]
    nsnap = pd.Series(code).groupby(at).nunique()
    def masa(q, mask):
        return np.bincount(ty[mask], weights=q[mask], minlength=14) / nsnap.reindex(np.unique(at[mask])).sum()
    out = p.copy()
    for a in np.unique(at):
        m = at == a
        ref = masa(p, ~m) + 1e-6
        q = p.copy()
        for _ in range(iters):
            w = (masa(q, m) + 1e-6) / ref
            q_m = p[m] * w[ty[m]] ** lam
            q[m] = q_m / np.bincount(code[m], weights=q_m, minlength=code.max() + 1)[code[m]]
        out[m] = q[m]
    return out


def correr_prior(modelos):
    for m in modelos:
        t0 = time.time()
        pl, pt = softmax(LS, cargar(LS, m, "oof")), softmax(TS, cargar(TS, m, "preds"))
        cands = {lam: prior_shift(LS, pl, lam) for lam in (0, 0.25, 0.5, 0.75, 1.0)}
        cv, g, el = anidado(LS, cands)
        print(f"prior_{m}: {S.hit3(LS, pl):.4f} -> CV anidada {cv:.4f}; por fold {el}; final {g} ({time.time() - t0:.0f}s)", flush=True)
        # test: el reparto de cada controlador de test frente a TODOS los etiquetados
        both = pd.concat([LS[TS.columns], TS], ignore_index=True)
        pb = prior_shift(both, np.concatenate([pl, pt]), g)[len(LS):] if g else pt
        salvar(f"prior_{m}", cands[g], pb, f"prior shift por controlador (EM) sobre {m}, lam={g}",
               dict(cv_anidada=round(cv, 4), params=[g], params_por_fold=el))


# ----------------------------------------------------------------------------- por escenario
def correr_escen(nombre, modelos):
    O = {m: softmax(LS, cargar(LS, m, "oof")) for m in modelos}
    P = {m: softmax(TS, cargar(TS, m, "preds")) for m in modelos}
    sc, sct = LS.scenario.values, TS.scenario.values
    elec = {}
    o = np.zeros(len(LS))
    for f in range(5):
        tr, va = (LS.fold != f).values, (LS.fold == f).values
        for e in np.unique(sc):
            me = sc == e
            best = max(modelos, key=lambda m: S.hit3(LS, O[m], tr & me))
            o[va & me] = O[best][va & me]
            elec.setdefault(str(e), []).append(best)
    final = {e: max(modelos, key=lambda m: S.hit3(LS, O[m], sc == e)) for e in np.unique(sc)}
    p = np.zeros(len(TS))
    for e, m in final.items():
        p[sct == e] = P[m][sct == e]
    cv = S.hit3(LS, o)
    print(f"escen_{nombre}: CV anidada {cv:.4f}; elegidos por fold {elec}; final {final}", flush=True)
    salvar(f"escen_{nombre}", o, p, f"modelo por escenario entre {modelos}: {final}",
           dict(cv_anidada=round(cv, 4), params={str(k): v for k, v in final.items()}))


if __name__ == "__main__":
    modo, args = sys.argv[1], sys.argv[2:]
    {"hmm": correr_hmm, "prior": correr_prior}.get(modo, lambda a: correr_escen(a[0], a[1:]))(args)

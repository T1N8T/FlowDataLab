"""f11_esp_instr.py · especialista en tareas CON INSTRUCCIÓN (hilo f11).

El 023 coordina poco: sus respuestas son sobre todo tareas que se resuelven con una instrucción al avión (zone 12,
return_route 13, exit_conditions 10, conflict_resolution 6, assume 4, transfer 5; y entry/exit_conflict 0/1).
Se entrena un LambdaRank SOLO con las filas de esos tipos en los snapshots cuya respuesta es de esos tipos, con las
features de N + contrastes entre candidatas con instrucción (rango y diferencia al mínimo dentro del snapshot, del tipo
y del avión). Su score reordena DENTRO del grupo instr; la masa instr vs no-instr la pone N (más un desplazamiento b):
    m = P_N(instr) ; q = softmax((1-w)·log q_N + w·log q_S) en filas instr ; m' = m·e^b / (m·e^b + 1 - m)
    p = m'·q (instr) | (1-m')·p_N/(1-m) (resto)  ->  score = log p  ->  segundo paso de sesión (feat_session.sesion)
(w, b) se eligen con CV anidada: para el fold f, con la sesión ajustada y medida SOLO en los otros 4 folds.
Sin fuga: el especialista del fold f se entrena con los otros folds; N es f10_instr_l63_s3 (OOF).
Modos:  python futuro/pruebas/f11_esp_instr.py train [semillas]   # especialista (CV + test) -> f11_esp_instr_spec
        python futuro/pruebas/f11_esp_instr.py combine            # mezcla con N + sesión, CV anidada
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "futuro" / "pruebas"), str(ROOT / "futuro"), str(ROOT / "cv")]
from base_fut import guardar, registrar, OUT, PARAMS_FUT, lgb_fit  # noqa: E402
from f10_instr import datos_instr, medir  # noqa: E402
import feat_session  # noqa: E402

INSTR = [0, 1, 4, 5, 6, 10, 12, 13]
NOMBRE = "f11_esp_instr"
REF = "f10_instr_l63_s3"
P_N = dict(num_leaves=63, min_child_samples=100, extra_trees=True)
KEY = ["si_match", "si_prevmatch", "si_n60", "si_next_nivel", "si_next_lateral", "si_next_transfer", "si_next_assume",
       "f_next_input", "f_next_match", "f_task_end", "f_cs_end", "sx_task", "p_task_start", "x_task_age", "x_state_age",
       "time_to_sector", "distance_to_sector", "a_since_flight", "a_since_any", "logic_deadline", "time_to_MD",
       "distance_to_MD", "k_timeToExitPoint", "distance_to_zone"]


def contrastes(d):
    """Features de contraste entre candidatas instr del mismo snapshot / tipo / avión (solo filas del snapshot)."""
    m = d.task_index.isin(INSTR).values
    x = d.loc[m, ["sample_id", "cs", "task_index"] + KEY].copy()
    out = pd.DataFrame(index=x.index)
    gs, gt, gc = x.sample_id, [x.sample_id, x.task_index], [x.sample_id, x.cs]
    out["ci_n"] = x.groupby(gs).sample_id.transform("size")
    out["ci_n_type"] = x.groupby(gt).sample_id.transform("size")
    out["ci_n_cs"] = x.groupby(gc).sample_id.transform("size")
    for c in KEY:
        v = x[c]
        out[f"ci_{c}_rk"] = v.groupby(gs).rank(method="min")
        out[f"ci_{c}_d"] = v - v.groupby(gs).transform("min")
        out[f"ci_{c}_rkt"] = v.groupby(gt).rank(method="min")
        out[f"ci_{c}_rkc"] = v.groupby(gc).rank(method="min")
    return out.reindex(d.index)


def preparar():
    lab, te, fs = datos_instr()
    C = contrastes(pd.concat([lab, te], ignore_index=True))
    lab = pd.concat([lab, C.iloc[:len(lab)].reset_index(drop=True)], axis=1)
    te = pd.concat([te, C.iloc[len(lab):].reset_index(drop=True)], axis=1)
    F = [c for b in ["base", "logic", "past", "future", "clear", "instr"] for c in fs[b]] + list(C.columns)
    return lab, te, F


def train(seeds=3):
    t0 = time.time()
    lab, te, F = preparar()
    seeds = tuple(range(int(seeds)))
    P = {**PARAMS_FUT, **P_N, "n_jobs": 4}
    mi = lab.task_index.isin(INSTR).values
    tgt_instr = lab.sample_id.map(lab[lab.target_index == 1].set_index("sample_id").task_index).isin(INSTR).values
    mt = te.task_index.isin(INSTR).values
    print(f"filas instr {mi.mean():.1%}; snapshots con respuesta instr {lab[tgt_instr].sample_id.nunique()}"
          f"/{lab.sample_id.nunique()}; {len(F)} features", flush=True)
    oof = np.full(len(lab), np.nan)
    for f in range(5):
        tr = (lab.fold != f).values & mi & tgt_instr
        va = (lab.fold == f).values & mi
        oof[va] = np.mean([lgb_fit(lab[tr], F, {**P, "random_state": s}).predict(lab.loc[va, F]) for s in seeds], 0)
        print(f"  fold {f} ({time.time() - t0:.0f}s)", flush=True)
    tr = mi & tgt_instr
    pred = np.full(len(te), np.nan)
    pred[mt] = np.mean([lgb_fit(lab[tr], F, {**P, "random_state": s}).predict(te.loc[mt, F]) for s in seeds], 0)
    # sin instr: muy negativo (el especialista no las puntúa)
    guardar(NOMBRE + "_spec", lab, np.nan_to_num(oof, nan=-30), te, np.nan_to_num(pred, nan=-30))
    # acierto del especialista solo: dentro de los snapshots con respuesta instr
    h = _hit3(lab.sample_id.values, np.nan_to_num(oof, nan=-30), lab.target_index.values)
    hn = _hit3(lab.sample_id.values, _leer(REF, lab), lab.target_index.values)
    sel = pd.Series(tgt_instr, index=lab.sample_id).groupby(level=0).first().reindex(h.index).values
    print(f"especialista: top-3 en snapshots con respuesta instr {h[sel].mean():.4f} (N: {hn[sel].mean():.4f}) "
          f"({time.time() - t0:.0f}s)", flush=True)


def _hit3(sid, s, y):
    x = pd.DataFrame(dict(sid=sid, s=s, y=y))
    ts = x[x.y == 1].set_index("sid").s
    above = (x.s.values > ts.reindex(x.sid).values).astype(int)
    return pd.Series(above, index=x.sid).groupby(level=0).sum().reindex(ts.index) < 3


def _leer(nombre, d, carpeta="oof"):
    o = pd.read_csv(OUT / carpeta / f"{nombre}.csv")
    return d[["sample_id", "row_index"]].merge(o, on=["sample_id", "row_index"], how="left").score.values


def mezcla(d, sN, sS, w, b):
    sid = d.sample_id.values
    mi = d.task_index.isin(INSTR).values
    g = pd.Series(sN).groupby(sid)
    lp = sN - g.transform("max").values
    p = np.exp(lp); p /= pd.Series(p).groupby(sid).transform("sum").values
    m = pd.Series(np.where(mi, p, 0)).groupby(sid).transform("sum").values
    lqN = np.log(np.clip(p / np.maximum(m, 1e-12), 1e-12, None))
    sSm = np.where(mi, sS, -np.inf)
    lqS = sSm - pd.Series(sSm).groupby(sid).transform("max").values
    lqS = lqS - np.log(pd.Series(np.exp(lqS)).groupby(sid).transform("sum").values)
    z = np.where(mi, (1 - w) * lqN + w * np.where(mi, lqS, 0), -np.inf)
    z = z - pd.Series(z).groupby(sid).transform("max").values
    q = np.exp(z); q /= pd.Series(q).groupby(sid).transform("sum").values
    q = np.nan_to_num(q)
    m2 = m * np.exp(b) / (m * np.exp(b) + 1 - m)
    rest = np.where(m < 1 - 1e-12, p / np.maximum(1 - m, 1e-12), 0) * (1 - m2)
    out = np.where(mi, m2 * q, rest)
    return np.log(out + 1e-9)


def combine():
    t0 = time.time()
    lab, te, fs = datos_instr()
    sN, pN = _leer(REF, lab), _leer(REF, te, "preds")
    sS, pS = _leer(NOMBRE + "_spec", lab), _leer(NOMBRE + "_spec", te, "preds")
    y, sid = lab.target_index.values, lab.sample_id.values
    W, B = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0], [0.0, 0.3, 0.6]
    grid = [(w, b) for w in W for b in B]
    S = {k: mezcla(lab, sN, sS, *k) for k in grid}
    at = lab.drop_duplicates("sample_id").set_index("sample_id").atco
    fold_s = lab.drop_duplicates("sample_id").set_index("sample_id").fold
    # 1.er pase por fold
    tab1 = {k: _hit3(sid, s, y) for k, s in S.items()}
    print("1.er pase (CV / CV023):", flush=True)
    for k in grid:
        h = tab1[k]
        print(f"  w={k[0]:.2f} b={k[1]:.1f}: {h.mean():.4f} / {h[at.reindex(h.index).isin([2, 11, 10, 6, 17, 15])].mean():.4f}", flush=True)
    # sesión: (a) cruzada normal en 5 folds para cada (w,b) -> tabla; (b) anidada: para el fold f, sesión con los otros 4
    ses5 = {}
    for k in grid:
        so, _, _ = feat_session.sesion(lab, S[k])
        ses5[k] = so
        h = _hit3(sid, so, y)
        print(f"  ses w={k[0]:.2f} b={k[1]:.1f}: {h.mean():.4f} / {h[at.reindex(h.index).isin([2, 11, 10, 6, 17, 15])].mean():.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    final = np.zeros(len(lab)); elegidos = {}
    for f in range(5):
        inn = (lab.fold != f).values
        L = lab[inn]
        best, bk = -1, None
        for k in grid:
            so, _, _ = feat_session.sesion(L, S[k][inn])
            v = _hit3(sid[inn], so, y[inn]).mean()
            if v > best:
                best, bk = v, k
        elegidos[f] = bk
        final[~inn] = ses5[bk][~inn]
        print(f"  fold {f}: elegido w={bk[0]} b={bk[1]} (interna {best:.4f})", flush=True)
    # test: (w,b) elegido con todo el OOF + sesión
    kb = max(grid, key=lambda k: _hit3(sid, ses5[k], y).mean())
    first = mezcla(lab, sN, sS, *kb); pt = mezcla(te, pN, pS, *kb)
    so, sp, wts = feat_session.sesion(lab, first, te, pt)
    nm = NOMBRE
    guardar(nm, lab, first, te, pt)
    guardar(nm + "_ses", lab, final, te, sp)
    a, bb, ref = medir(nm, lab, first), medir(nm, lab, final), medir(REF, lab, _leer(REF + "_ses", lab))
    k0 = (0.0, 0.0)
    c0 = medir("", lab, ses5[k0])
    registrar(nm, hilo="f11", descripcion=f"N + especialista instr, (w,b) global {kb}", extra=dict(cv_023=round(a[1], 4)))
    registrar(nm + "_ses", hilo="f11", descripcion=f"N + especialista instr + sesión, (w,b) por CV anidada {elegidos}",
              extra=dict(cv_023=round(bb[1], 4), peor=round(bb[2], 4)))
    print(f"\n{nm}: 1.er pase {a[0]:.4f} (CV 023 {a[1]:.4f}, peor {a[2]:.3f}) con {kb}")
    print(f"{nm}_ses (anidada): {bb[0]:.4f} (CV 023 {bb[1]:.4f}, peor {bb[2]:.3f})  elegidos {elegidos}")
    print(f"control w=0,b=0 + sesión: {c0[0]:.4f} (CV 023 {c0[1]:.4f}, peor {c0[2]:.3f})")
    print(f"ref {REF}_ses: {ref[0]:.4f} (CV 023 {ref[1]:.4f}, peor {ref[2]:.3f})  delta {100 * (bb[0] - ref[0]):+.2f} pts "
          f"(CV 023 {100 * (bb[1] - ref[1]):+.2f})  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    {"train": train, "combine": combine}[sys.argv[1]](*sys.argv[2:])

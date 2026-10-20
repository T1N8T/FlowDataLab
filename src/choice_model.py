"""
Modelo analítico: logit condicional (elección discreta, McFadden).
El controlador elige la tarea i del snapshot con P(i) = softmax_i(u_i), con utilidad lineal e interpretable:
    u_i = prior[tipo] + urgencia[tipo, tramo de deadline] + reglas lógicas (+ señales futuras)
    python src/choice_model.py   -> CV por controlador, coeficientes en models/choice_coefs_*.csv,
                                    utilidades OOF/test en models/choice_util_*.npy (para apilar en LightGBM)
"""
import numpy as np, pandas as pd, torch
from features import MODELS, TYPES
from train import load_feats, hit3, smooth, softmax_by_sample

DL_EDGES = [-np.inf, -600, -300, -120, -60, 0, 60, 120, 240, 480, 900, np.inf]
NI_EDGES = [0, 5, 10, 20, 40, 80, 160, np.inf]
END_EDGES = [-0.1, 0, 3, 6, 10, 15, 30, 60, 120, 300, np.inf]
CLOSING = [4, 5, 7, 9, 13]          # tareas que desaparecen al completarse
RULES = ["logic_feasible", "logic_offered", "logic_due_transfer", "logic_stca", "logic_twin_assume", "a_is_last_cs",
         "a_is_last2_cs", "conflict_ind", "multiple_conflicts_ind", "x_transfer_in", "is_assumed", "logic_in_window"]


def onehot(name, codes, n):
    m = np.zeros((len(codes), n), np.float32)
    ok = codes >= 0
    m[np.where(ok)[0], codes[ok]] = 1
    return pd.DataFrame(m, columns=[f"{name}={i}" for i in range(n)])


def design(d, future):
    ty = d.task_index.values
    blocks = [onehot("type", ty, len(TYPES))]
    # curva de urgencia por tipo: tramo del deadline (tramo extra = deadline desconocido)
    b = np.digitize(d.logic_deadline.values, DL_EDGES[1:-1])
    b = np.where(np.isnan(d.logic_deadline.values), len(DL_EDGES) - 1, b)
    nb = len(DL_EDGES)
    blocks.append(onehot("urg", ty * nb + b, len(TYPES) * nb))
    blocks.append(d[RULES].fillna(0).astype(np.float32).reset_index(drop=True))
    if future:
        ni = np.digitize(d.f_next_input.values, NI_EDGES[1:-1]); ni = np.where(np.isnan(d.f_next_input.values), -1, ni)
        blocks.append(onehot("next_input", ni, len(NI_EDGES) - 1))
        blocks.append(pd.DataFrame({"next_rk1": (d.f_next_input_rk == 1).values, "next_match": d.f_next_match.fillna(0).values,
                                    "next_match_soon": d.f_next_match_soon.fillna(0).values,
                                    "end_rk1": (d.f_task_end_rk == 1).values, "gone_next": d.f_gone_next.values,
                                    "end_in_block": d.f_end_in_block.values}).astype(np.float32))
        e = np.digitize(d.f_task_end.values, END_EDGES[1:-1])
        grp = np.isin(ty, CLOSING).astype(int)
        blocks.append(onehot("end", grp * (len(END_EDGES) - 1) + e, 2 * (len(END_EDGES) - 1)))
    X = pd.concat(blocks, axis=1)
    return X


def fit_clogit(X, sid, y, l2=1e-3, iters=200):
    Xt = torch.tensor(X.values, dtype=torch.float64)
    g = torch.tensor(pd.factorize(sid)[0], dtype=torch.long)
    yt = torch.tensor(y == 1)
    ns = int(g.max()) + 1
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def loss_fn():
        opt.zero_grad()
        u = Xt @ w
        mx = torch.zeros(ns, dtype=u.dtype).scatter_reduce(0, g, u, "amax", include_self=False)
        lse = torch.log(torch.zeros(ns, dtype=u.dtype).index_add(0, g, torch.exp(u - mx[g]))) + mx
        loss = -(u[yt].sum() - lse.sum()) / ns + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(loss_fn)
    return w.detach().numpy()


def run(d, future, tag):
    lab, te = d[d.split != "test"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    X, Xte = design(lab, future), design(te, future)
    oof = np.zeros(len(lab))
    for f in sorted(lab.fold.unique()):
        tr, va = (lab.fold != f).values, (lab.fold == f).values
        w = fit_clogit(X[tr], lab.sample_id.values[tr], lab.target_index.values[tr])
        oof[va] = X[va].values @ w
    w = fit_clogit(X, lab.sample_id.values, lab.target_index.values)
    ps = smooth(lab, softmax_by_sample(lab, oof))
    print(f"clogit {tag:14s} top3={hit3(lab, oof):.4f}  +suavizado={hit3(lab, ps):.4f}  ({X.shape[1]} coeficientes)", flush=True)
    pd.Series(w, index=X.columns).to_csv(MODELS / f"choice_coefs_{tag}.csv", header=["coef"])
    np.save(MODELS / f"choice_util_{tag}_oof.npy", oof)
    np.save(MODELS / f"choice_util_{tag}_test.npy", Xte.values @ w)
    return oof, w


if __name__ == "__main__":
    torch.set_num_threads(12)
    d = load_feats()
    run(d, future=False, tag="logic")
    run(d, future=True, tag="logic_future")

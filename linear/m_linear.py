"""
m_linear.py · Juanma (C) · Modelo lineal base: logit condicional (softmax sobre las tareas del snapshot).

score(tarea) = b[tipo] + x·w + x·W[tipo]      P(tarea) = softmax de score dentro del sample_id
- x: columnas originales (añadir los rankings por tipo de m_types.add_row_feats no mejora: 0.631 vs 0.632).
  Transformación: log con signo si |x| > 50, estandarizado con la media/desv. del train del fold,
  vacío -> 0 + indicador de vacío (regla 5: vacío = "no aplica").
- La pendiente por tipo (x·W[tipo]) es la clave: sin ella la CV baja ~0.10 (las columnas significan cosas
  distintas según el tipo de tarea).
- L2 separada para w (común) y W (por tipo). Ajuste full-batch con L-BFGS (torch, GPU si hay).
- Regla 1: todo usa solo filas del mismo sample_id. CV con Data/folds.csv. Test: entrenado con train+validation.

Uso:  python linear/m_linear.py                 # CV + oof/linear.csv, preds/linear.csv, submissions/sub_linear.csv
      python linear/m_linear.py --quick         # solo CV
      python linear/m_linear.py --lw 1e-3 --lW 1e-2
"""
import sys, time, argparse, numpy as np, pandas as pd, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cv"))
sys.path.insert(0, str(ROOT / "hierarchical"))
from metric import report, save_outputs
from m_types import load, NOT_FEATS, N_TYPES

NAME = "linear"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- diseño
def design(d, cols, tr):
    """Matriz X (float32) con estadísticos de las filas tr. Devuelve X y nombres de columnas."""
    X, names = [], []
    for c in cols:
        x = d[c].values.astype(np.float64)
        if np.nanstd(x[tr]) == 0:
            continue
        if np.nanmax(np.abs(x)) > 50:
            x = np.sign(x) * np.log1p(np.abs(x))
        na = np.isnan(x)
        mu, sd = np.nanmean(x[tr]), np.nanstd(x[tr]) + 1e-9
        X.append(np.where(na, 0, (x - mu) / sd)); names.append(c)
        if na[tr].any() and not na[tr].all():
            X.append(na.astype(np.float64)); names.append(c + "_na")
    return np.stack(X, 1).astype(np.float32), names


# ----------------------------------------------------------------------------- modelo
class CondLogit:
    def __init__(self, lw=1e-3, lW=1e-2, iters=200):
        self.lw, self.lW, self.iters = lw, lW, iters

    def _score(self, X, ty):
        return self.b[ty] + X @ self.w + (X * self.W[ty]).sum(1)

    def fit(self, X, ty, sid, y):
        X = torch.tensor(X, device=DEV); ty = torch.tensor(ty, device=DEV)
        _, g = np.unique(sid, return_inverse=True)
        g = torch.tensor(g, device=DEV); G = int(g.max()) + 1
        y = torch.tensor(y, device=DEV, dtype=torch.float32)
        P = X.shape[1]
        self.b = torch.zeros(N_TYPES, device=DEV, requires_grad=True)
        self.w = torch.zeros(P, device=DEV, requires_grad=True)
        self.W = torch.zeros(N_TYPES, P, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([self.b, self.w, self.W], lr=1, max_iter=self.iters, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            s = self._score(X, ty)
            m = torch.zeros(G, device=DEV).scatter_reduce(0, g, s.detach(), "amax", include_self=False)
            lse = torch.log(torch.zeros(G, device=DEV).index_add(0, g, torch.exp(s - m[g]))) + m
            loss = (lse.sum() - (s * y).sum()) / G + self.lw * (self.w ** 2).sum() + self.lW * (self.W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    @torch.no_grad()
    def predict(self, X, ty):
        return self._score(torch.tensor(X, device=DEV), torch.tensor(ty, device=DEV)).cpu().numpy()


def fit_predict(d, F, tr, te, args):
    X, _ = design(d, F, tr)
    m = CondLogit(args.lw, args.lW).fit(X[tr], d.task_index.values[tr], d.sample_id.values[tr],
                                        d.target_index.values[tr])
    return m.predict(X[te], d.task_index.values[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lw", type=float, default=1e-3)
    ap.add_argument("--lW", type=float, default=1e-2)
    args = ap.parse_args()

    d = load()
    F = [c for c in d.columns if c not in NOT_FEATS + ["task_index"]]
    lab = (d.split != "test").values
    print(f"{len(F)} features, dispositivo {DEV}, lw={args.lw} lW={args.lW}", flush=True)

    oof = np.full(len(d), np.nan)
    for f in range(5):
        t0 = time.time()
        tr, te = lab & (d.fold.values != f), d.fold.values == f
        oof[te] = fit_predict(d, F, tr, te, args)
        print(f"  fold {f}: {time.time() - t0:.0f}s", flush=True)
    oof_df = pd.DataFrame({"sample_id": d.sample_id, "row_index": d.row_index, "score": oof})[lab]
    report(oof_df, NAME)
    if args.quick:
        return

    te = ~lab
    preds_df = pd.DataFrame({"sample_id": d.sample_id[te], "row_index": d.row_index[te],
                             "score": fit_predict(d, F, lab, te, args)})
    save_outputs(NAME, oof_df, preds_df)


if __name__ == "__main__":
    main()

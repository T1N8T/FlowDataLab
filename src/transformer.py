"""
Set-transformer sobre las tareas de un snapshot (FlowDataLab).
Cada snapshot es un conjunto de <=59 tareas; self-attention entre tareas (con sesgo aprendido para tareas del mismo
avión), score por tarea y cross-entropy sobre el softmax del snapshot.
    python src/transformer.py   -> CV por controlador (mismos folds), guarda models/tf_oof.npy y models/tf_test.npy
"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from features import MODELS, TYPES, feature_sets
from train import load_feats, hit3, smooth, softmax_by_sample

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 59


def prep(d, F):
    """Transformación robusta: signed-log en columnas de cola larga, estandarizar, NaN -> 0 + indicador."""
    X = d[F].astype(np.float32).copy()
    lab = d.split != "test"
    for c in F:
        if X[c].abs().max() > 100: X[c] = np.sign(X[c]) * np.log1p(X[c].abs())
    mu, sd = X[lab].mean(), X[lab].std().replace(0, 1)
    X = ((X - mu) / sd).clip(-5, 5)
    nan_cols = [c for c in F if X[c].isna().any()]
    M = X[nan_cols].isna().astype(np.float32).add_suffix("_nan")
    return pd.concat([X.fillna(0), M], axis=1).values.astype(np.float32)


def to_tensors(d, X):
    """Empaqueta por sample: X [S,L,F], tipo [S,L], avión [S,L], y [S], máscara [S,L]."""
    sid = d.sample_id.values
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    S = len(starts); n = np.diff(np.r_[starts, len(d)])
    pos = np.arange(len(d)) - np.repeat(starts, n)
    s_idx = np.repeat(np.arange(S), n)
    XT = np.zeros((S, L, X.shape[1]), np.float32); XT[s_idx, pos] = X
    T = np.zeros((S, L), np.int64); T[s_idx, pos] = d.task_index.values
    cs = pd.factorize(d.session + d.cs)[0]
    A = np.full((S, L), -1, np.int64); A[s_idx, pos] = cs
    M = np.zeros((S, L), bool); M[s_idx, pos] = True
    y = np.full(S, -1, np.int64)
    t = d.target_index.values == 1
    y[s_idx[t]] = pos[t]
    return dict(X=torch.tensor(XT), T=torch.tensor(T), A=torch.tensor(A), M=torch.tensor(M), y=torch.tensor(y),
                s_idx=s_idx, pos=pos)


class Block(nn.Module):
    def __init__(self, dm, h, p):
        super().__init__()
        self.h = h; self.q = nn.Linear(dm, 3 * dm); self.o = nn.Linear(dm, dm)
        self.same_bias = nn.Parameter(torch.zeros(h))
        self.n1, self.n2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.ff = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Dropout(p), nn.Linear(4 * dm, dm))
        self.drop = nn.Dropout(p)

    def forward(self, x, mask, same):
        B, N, D = x.shape
        q, k, v = self.q(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        bias = same[:, None] * self.same_bias[None, :, None, None]
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class SetRanker(nn.Module):
    def __init__(self, nf, dm=128, h=4, layers=3, p=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, dm), nn.GELU(), nn.Dropout(p), nn.Linear(dm, dm))
        self.temb = nn.Embedding(len(TYPES), dm)
        self.blocks = nn.ModuleList([Block(dm, h, p) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 1))

    def forward(self, X, T, A, M):
        x = self.inp(X) + self.temb(T)
        same = ((A[:, :, None] == A[:, None, :]) & M[:, :, None]).float()
        for b in self.blocks: x = b(x, M, same)
        return self.out(x).squeeze(-1).masked_fill(~M, -1e4)


def train_one(tr, ep=30, seed=0, bs=128, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SetRanker(tr["X"].shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    S = len(tr["y"]); steps = ep * ((S + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for e in range(ep):
        m.train()
        for i in torch.randperm(S).split(bs):
            b = {k: tr[k][i].to(DEV) for k in "XTAMy"}
            # ruido de features como regularización
            loss = nn.functional.cross_entropy(m(b["X"] + 0.05 * torch.randn_like(b["X"]), b["T"], b["A"], b["M"]), b["y"],
                                               label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    return m


@torch.no_grad()
def predict(m, te, bs=512):
    m.eval()
    out = [m(*(te[k][i].to(DEV) for k in "XTAM")).cpu() for i in torch.arange(len(te["y"])).split(bs)]
    s = torch.cat(out).numpy()
    return s[te["s_idx"], te["pos"]]


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    d = load_feats()
    fs = feature_sets(d)
    F = [c for c in fs["base"] + fs["logic"] + fs["past"] + fs["future"] if c != "task_index"]
    X = prep(d, F)
    lab_m = (d.split != "test").values
    lab, te = d[lab_m].reset_index(drop=True), d[~lab_m].reset_index(drop=True)
    TL, TT = to_tensors(lab, X[lab_m]), to_tensors(te, X[~lab_m])
    sfold = lab.drop_duplicates("sample_id").fold.values
    oof, tst = np.zeros(len(lab)), np.zeros(len(te))
    for f in sorted(np.unique(sfold)):
        tri, vai = np.flatnonzero(sfold != f), np.flatnonzero(sfold == f)
        sub = lambda D, idx: {k: D[k][idx] for k in "XTAMy"}
        va = sub(TL, vai); va["s_idx"], va["pos"] = None, None
        rows = np.isin(TL["s_idx"], vai)
        va["s_idx"] = np.searchsorted(vai, TL["s_idx"][rows]); va["pos"] = TL["pos"][rows]
        for s in range(seeds):
            m = train_one(sub(TL, tri), seed=s)
            oof[rows] += predict(m, va) / seeds
            tst += predict(m, TT) / (seeds * len(np.unique(sfold)))
        print(f"fold {f}: top3={hit3(lab[rows], oof[rows]):.4f}", flush=True)
    ps = smooth(lab, softmax_by_sample(lab, oof))
    print(f"transformer top3={hit3(lab, oof):.4f}  +suavizado={hit3(lab, ps):.4f}")
    np.save(MODELS / "tf_oof.npy", oof); np.save(MODELS / "tf_test.npy", tst)


if __name__ == "__main__":
    main()

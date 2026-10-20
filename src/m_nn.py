"""
m_nn.py · Pablo (A) · Transformer de conjunto sobre las tareas de un snapshot.

- Cada snapshot se predice SOLO con sus propias filas (regla 1): no hay features temporales ni de otras muestras.
- Entrada por tarea: columnas originales (signed-log en colas largas, estandarizar, vacío -> 0 + indicador de vacío)
  + embedding de task_index. Sin codificación de posición (row_index no se usa).
- Encoder transformer con máscara de padding -> una puntuación por tarea -> softmax sobre la lista,
  cross-entropy con label smoothing.
- CV: GroupKFold por atco con Data/folds.csv. Dentro de cada fold se apartan 3 controladores para early stopping.
- Test: modelos entrenados con train+validation completo durante la mediana de épocas óptimas de la CV.

Opciones de las ideas del EDA (bloque 9 del notebook), desactivadas por defecto:
    --rel       features relativas: z-score de columnas clave dentro del snapshot y dentro de snapshot×tipo
    --same-ac   sesgo de atención aprendido entre tareas del mismo avión (avión sacado de task_name, solo para agrupar)
    --drop-red  quitar una columna de cada par casi redundante (|rho| > 0.95, sección 6 del EDA)

Uso:
    python src/m_nn.py                                  # configuración base, 3 semillas, guarda salidas
    python src/m_nn.py --quick --layers 2 --drop 0.3    # prueba rápida (1 semilla, no guarda salidas)
    python src/nn_sweep.py ...                          # barridos de parámetros, ver ese archivo
Salidas: oof/<name>.csv, preds/<name>.csv, submissions/sub_<name>.csv
Cada ejecución se apunta en experiments/nn_runs.csv.
"""
import argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
EXP = ROOT / "experiments"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TYPES = 14
# Regla 4: nunca son features. task_index entra por embedding, no como número.
NOT_FEATS = ["sample_id", "target_index", "session", "atco", "scenario", "timestamp", "task_name", "row_index",
             "task_index", "split", "aircraft"]
TYPES = ['entry_conflict_resolution', 'exit_conflict_resolution', 'entry_coordination', 'exit_coordination',
         'assume_aircraft', 'transfer_aircraft', 'conflict_resolution', 'non_conformance_resolution',
         'qos_improvement', 'entry_conditions', 'exit_conditions', 'aircraft_request', 'zone_conflict',
         'return_to_route']
# --rel: columnas que se comparan dentro del snapshot (EDA: importa la posición relativa, no el valor absoluto)
REL_COLS = ["time_to_sector", "abs_time_to_sector", "x_task_age", "x_state_age", "a_since_flight",
            "k_timeToExitPoint", "distance_to_zone", "time_to_MD"]
# --drop-red: una columna de cada par con |rho| > 0.95 (EDA, sección 6); se quita la segunda
REDUNDANT = ["k_clearedFLDelta", "k_isLocked", "distance_to_sector", "is_not_assumed",
             "non_conformance_agreed_route_deviation_fl", "k_distanceToExitPoint", "x_in_sector_h", "l_rank_f0_type",
             "a_last_initiate_transfer_of_control", "distance_to_MD", "approaching_sector"]


# ----------------------------------------------------------------------------- datos
def load():
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    va = pd.read_csv(DATA / "validation.csv").assign(split="validation")
    te = pd.read_csv(DATA / "test.csv").assign(split="test", target_index=-1)
    d = pd.concat([tr, va, te], ignore_index=True).sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    fp = DATA / "folds.csv"
    if fp.exists():
        folds = pd.read_csv(fp)
    else:  # mismo esquema que el antiguo features.py; se sustituye en cuanto Juanma suba Data/folds.csv
        print("AVISO: no existe Data/folds.csv -> folds provisionales (atco ordenado, i % 5). NO comparar CV hasta tenerlo.")
        atcos = sorted(d.loc[d.split != "test", "atco"].unique())
        folds = pd.DataFrame({"atco": atcos, "fold": [i % 5 for i in range(len(atcos))]})
    d["fold"] = d.atco.map(dict(zip(folds.atco, folds.fold))).fillna(-1).astype(int)
    d.loc[d.split == "test", "fold"] = -1
    # avión de cada tarea (para --same-ac); conflictos nombran dos aviones: el primero es el de la tarea
    d["aircraft"] = [n[len(TYPES[t]) + 1:].split("_")[0] for n, t in zip(d.task_name, d.task_index)]
    return d


def add_relative(d):
    """Columnas rel_s_* (z-score dentro del snapshot) y rel_t_* (dentro de snapshot × tipo). Solo filas del sample_id."""
    d = d.copy()
    d["abs_time_to_sector"] = d.time_to_sector.abs()
    new = {}
    for key, by in (("s", [d.sample_id]), ("t", [d.sample_id, d.task_index])):
        g = d.groupby(by)[REL_COLS]
        z = (d[REL_COLS] - g.transform("mean")) / g.transform("std")
        for c in REL_COLS:
            new[f"rel_{key}_{c}"] = z[c].where(d[c].notna()).fillna(0).clip(-5, 5)   # sin valor o solo -> 0
    return pd.concat([d.drop(columns="abs_time_to_sector"), pd.DataFrame(new, index=d.index)], axis=1)


def prep(d):
    """Matriz de features por fila: signed-log, estandarizar con filas etiquetadas, clip, vacío -> 0 + indicador."""
    F = [c for c in d.columns if c not in NOT_FEATS + ["fold"]]
    X = d[F].astype(np.float32)
    X = X.loc[:, X.notna().any()]                       # columnas siempre vacías fuera
    for c in X.columns:
        if X[c].abs().max() > 100:
            X[c] = np.sign(X[c]) * np.log1p(X[c].abs())
    lab = (d.split != "test").values
    mu, sd = X[lab].mean(), X[lab].std().replace(0, 1).fillna(1)
    X = ((X - mu) / sd).clip(-5, 5)
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    M = X[nan_cols].isna().astype(np.float32).add_suffix("_nan")
    X = pd.concat([X.fillna(0), M], axis=1)
    X = X.loc[:, X.std() > 0]                           # constantes fuera
    return X.values.astype(np.float32), list(X.columns)


def pack(d, X, L):
    """Empaqueta por muestra: X [S,L,F], tipo [S,L], máscara [S,L], y [S] (posición de la respuesta)."""
    sid = d.sample_id.values
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    n = np.diff(np.r_[starts, len(d)])
    S = len(starts)
    s_idx = np.repeat(np.arange(S), n)
    pos = np.arange(len(d)) - np.repeat(starts, n)
    XT = np.zeros((S, L, X.shape[1]), np.float32); XT[s_idx, pos] = X
    T = np.zeros((S, L), np.int64); T[s_idx, pos] = d.task_index.values
    M = np.zeros((S, L), bool); M[s_idx, pos] = True
    A = np.full((S, L), -1, np.int64)                   # id de avión dentro del snapshot (-1 = padding)
    A[s_idx, pos] = d.groupby("sample_id", sort=False).aircraft.transform(lambda s: pd.factorize(s)[0]).values
    y = np.full(S, -1, np.int64)
    t = d.target_index.values == 1
    y[s_idx[t]] = pos[t]
    first = d.iloc[starts]
    return dict(X=torch.tensor(XT), T=torch.tensor(T), M=torch.tensor(M), A=torch.tensor(A), y=torch.tensor(y),
                s_idx=s_idx, pos=pos, atco=first.atco.values, fold=first.fold.values, sample_id=first.sample_id.values)


def subset(P, idx):
    return {k: P[k][idx] for k in "XTMAy"}


# ----------------------------------------------------------------------------- modelo
class SetRanker(nn.Module):
    def __init__(self, nf, d=64, heads=4, layers=3, drop=0.2, same_ac=False):
        super().__init__()
        self.heads = heads
        # --same-ac: un sesgo aprendido por cabeza que se suma a la atención entre tareas del mismo avión
        self.same_bias = nn.Parameter(torch.zeros(heads)) if same_ac else None
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        self.temb = nn.Embedding(N_TYPES, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, X, T, M, A):
        x = self.inp(X) + self.temb(T)
        if self.same_bias is None:
            x = self.enc(x, src_key_padding_mask=~M)
        else:
            B, L = M.shape
            same = ((A[:, :, None] == A[:, None, :]) & M[:, None, :]).to(x.dtype)
            bias = same[:, None] * self.same_bias.to(x.dtype)[None, :, None, None]          # [B, h, L, L]
            bias = bias.masked_fill(~M[:, None, None, :], float("-inf"))                    # padding como clave
            x = self.enc(x, mask=bias.reshape(B * self.heads, L, L))
        return self.out(x).squeeze(-1).masked_fill(~M, -1e4)


@torch.no_grad()
def logits(m, P, bs=1024):
    m.eval()
    idx = torch.arange(len(P["y"]))
    return torch.cat([m(*(P[k][i].to(DEV) for k in "XTMA")).float().cpu() for i in idx.split(bs)])


def top3_from_logits(lg, y):
    top = lg.topk(3, dim=1).indices
    return (top == y[:, None]).any(1).float().mean().item()


def train(tr, a, seed, es=None, epochs=None):
    """Entrena; con `es` hace early stopping y devuelve (modelo, mejor_época). Sin `es`, entrena `epochs` épocas."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = SetRanker(tr["X"].shape[-1], a.d, a.heads, a.layers, a.drop, a.same_ac).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    n_ep = epochs or a.epochs
    S = len(tr["y"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=n_ep * ((S + a.bs - 1) // a.bs),
                                              pct_start=0.1)
    best, best_ep, best_state, bad = -1.0, n_ep, None, 0
    for ep in range(1, n_ep + 1):
        m.train()
        for i in torch.randperm(S).split(a.bs):
            b = {k: tr[k][i].to(DEV, non_blocking=True) for k in "XTMAy"}
            x = b["X"] + a.noise * torch.randn_like(b["X"]) if a.noise else b["X"]
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                out = m(x, b["T"], b["M"], b["A"])
            loss = nn.functional.cross_entropy(out.float(), b["y"], label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if es is not None:
            h = top3_from_logits(logits(m, es), es["y"])
            if h > best + 1e-4:
                best, best_ep, bad = h, ep, 0
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience: break
    if best_state is not None: m.load_state_dict(best_state)
    return m, best_ep


def softmax_rows(lg, P):
    """Logits [S,L] -> probabilidad por fila original (softmax dentro del snapshot)."""
    return torch.softmax(lg, dim=1).numpy()[P["s_idx"], P["pos"]]


# ----------------------------------------------------------------------------- utilidades de salida
def hit3(df, score):
    r = pd.Series(score, index=df.index).groupby(df.sample_id.values).rank(ascending=False, method="first")
    return float((r[df.target_index.values == 1] <= 3).mean())


def write_sub(df, score, path):
    x = df.assign(s=score).sort_values(["sample_id", "s"], ascending=[True, False])
    sub = x.groupby("sample_id").row_index.apply(lambda r: " ".join(map(str, r.iloc[:3]))).rename("prediction")
    sub.reset_index().to_csv(path, index=False)
    return path


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nn_tf")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.0, help="ruido gaussiano en features al entrenar")
    ap.add_argument("--n-es", type=int, default=3, help="controladores apartados para early stopping")
    ap.add_argument("--folds", default="", help="solo estos folds, p.ej. '0' o '0,1' (pruebas rápidas)")
    ap.add_argument("--quick", action="store_true", help="1 semilla y sin guardar oof/preds/submission")
    ap.add_argument("--tag", default="", help="nota libre para el registro de experimentos")
    ap.add_argument("--rel", action="store_true", help="features relativas dentro del snapshot y del tipo")
    ap.add_argument("--same-ac", action="store_true", help="sesgo de atención entre tareas del mismo avión")
    ap.add_argument("--drop-red", action="store_true", help="quitar columnas casi redundantes")
    a = ap.parse_args(argv)
    if a.quick: a.seeds = 1
    return a


def build_data():
    """Carga y empaqueta una vez (el sweep lo reutiliza entre ejecuciones)."""
    d = add_relative(load())                            # las rel_* solo se usan con --rel (ver select)
    X, cols = prep(d)
    L = int(d.seq_len.max())
    lab_m = (d.split != "test").values
    lab, te = d[lab_m].reset_index(drop=True), d[~lab_m].reset_index(drop=True)
    PL, PT = pack(lab, X[lab_m], L), pack(te, X[~lab_m], L)
    print(f"{len(cols)} features por tarea, {len(PL['y'])} muestras etiquetadas, {len(PT['y'])} de test, L={L}", flush=True)
    assert (PL["y"] >= 0).all(), "hay muestras sin exactamente una respuesta"
    return dict(lab=lab, te=te, PL=PL, PT=PT, cols=cols, folds_file=(DATA / "folds.csv").exists())


def select(data, a):
    """Columnas que usa esta ejecución según --rel y --drop-red (los datos se cargan una vez con todas)."""
    cols = data["cols"]
    keep = [i for i, c in enumerate(cols) if (a.rel or not c.startswith("rel_"))
            and not (a.drop_red and c.removesuffix("_nan") in REDUNDANT)]
    print(f"  {len(keep)} de {len(cols)} columnas", flush=True)
    if len(keep) == len(cols):
        return data["PL"], data["PT"]
    k = torch.tensor(keep)
    return {**data["PL"], "X": data["PL"]["X"][..., k]}, {**data["PT"], "X": data["PT"]["X"][..., k]}


def log_run(a, data, cv, by_atco, fold_cv, best_eps, secs, saved):
    """Añade una fila a experiments/nn_runs.csv (una por ejecución)."""
    EXP.mkdir(exist_ok=True)
    row = dict(when=time.strftime("%Y-%m-%d %H:%M"), name=a.name, tag=a.tag, cv=round(cv, 4),
               min_atco=round(by_atco.min(), 4), **{f"fold{k}": round(v, 4) for k, v in fold_cv.items()},
               best_epochs=" ".join(map(str, best_eps)), secs=int(secs), saved=saved, folds_csv=data["folds_file"],
               **{k: v for k, v in vars(a).items() if k not in ("name", "tag")})
    f = EXP / "nn_runs.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)


def run(a, data):
    t0 = time.time()
    print(f"\n=== {a.name} {a.tag}  device={DEV}  " + " ".join(f"{k}={v}" for k, v in vars(a).items()
                                                            if k not in ("name", "tag")), flush=True)
    lab, te = data["lab"], data["te"]
    PL, PT = select(data, a)
    folds = sorted(np.unique(PL["fold"]))
    if a.folds: folds = [int(f) for f in a.folds.split(",")]
    oof = np.full(len(lab), np.nan); best_eps, fold_cv = [], {}
    for f in folds:
        va = np.flatnonzero(PL["fold"] == f)
        rest = np.flatnonzero(PL["fold"] != f)
        rows = np.isin(PL["s_idx"], va)
        pv = subset(PL, va)
        p_fold = np.zeros(rows.sum())
        for s in range(a.seeds):
            # controladores del resto apartados para early stopping (cambian con la semilla)
            atcos = np.unique(PL["atco"][rest])
            es_atcos = np.random.RandomState(100 * f + s).choice(atcos, a.n_es, replace=False)
            es_m = np.isin(PL["atco"][rest], es_atcos)
            m, ep = train(subset(PL, rest[~es_m]), a, seed=s, es=subset(PL, rest[es_m]))
            best_eps.append(ep)
            lg = logits(m, pv)
            p_fold += torch.softmax(lg, 1).numpy()[np.searchsorted(va, PL["s_idx"][rows]), PL["pos"][rows]] / a.seeds
            print(f"  fold {f} seed {s}: best_ep={ep:2d}  top3={top3_from_logits(lg, pv['y']):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        oof[rows] = p_fold
        fold_cv[f] = hit3(lab[rows], p_fold)
        print(f"fold {f}: top3={fold_cv[f]:.4f}", flush=True)

    done = ~np.isnan(oof)
    sub_lab = lab[done]
    cv = hit3(sub_lab, oof[done])
    by_atco = pd.Series({k: hit3(g, oof[g.index]) for k, g in sub_lab.groupby("atco")})
    print(f"{a.name}: CV top3 = {cv:.4f}   (min atco {by_atco.min():.3f}, media por atco {by_atco.mean():.3f})"
          f"   épocas óptimas {best_eps}", flush=True)
    if not data["folds_file"]: print("  (folds provisionales: no comparable con el resto del equipo)")

    saved = not (a.quick or a.folds)
    if saved:
        for sub in ("oof", "preds", "submissions"): (ROOT / sub).mkdir(exist_ok=True)
        lab[["sample_id", "row_index"]].assign(score=oof).to_csv(ROOT / "oof" / f"{a.name}.csv", index=False)
        vmask = (lab.split == "validation").values
        write_sub(lab[vmask], oof[vmask], ROOT / "oof" / f"sub_{a.name}_validation.csv")
        # test: modelos con todo train+validation durante la mediana de épocas óptimas
        n_ep = max(1, int(np.median(best_eps)))
        ptest = np.zeros(len(te))
        for s in range(a.seeds):
            m, _ = train(subset(PL, np.arange(len(PL["y"]))), a, seed=s, epochs=n_ep)
            ptest += softmax_rows(logits(m, PT), PT) / a.seeds
            print(f"  test seed {s}: {n_ep} épocas ({time.time() - t0:.0f}s)", flush=True)
        te[["sample_id", "row_index"]].assign(score=ptest).to_csv(ROOT / "preds" / f"{a.name}.csv", index=False)
        write_sub(te, ptest, ROOT / "submissions" / f"sub_{a.name}.csv")
        print(f"Guardado oof/{a.name}.csv, preds/{a.name}.csv, submissions/sub_{a.name}.csv")
        print(f"Comprobar: python Data/scorer.py --labelled Data/validation.csv oof/sub_{a.name}_validation.csv")
    log_run(a, data, cv, by_atco, fold_cv, best_eps, time.time() - t0, saved)
    return cv


def main():
    a = parse()
    run(a, build_data())


if __name__ == "__main__":
    main()

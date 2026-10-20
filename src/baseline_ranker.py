"""
Baseline FlowDataLab: LightGBM LambdaRank + features temporales.
Uso (desde la raíz del repo):  python src/baseline_ranker.py
- CV agrupada por controlador (atco), 5 folds -> imprime top-3 hit rate
- Entrena con train+validation y escribe submissions/sub_lgb_temporal.csv
"""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "Data", ROOT / "submissions"
TYPES = ['entry_conflict_resolution', 'exit_conflict_resolution', 'entry_coordination', 'exit_coordination',
         'assume_aircraft', 'transfer_aircraft', 'conflict_resolution', 'non_conformance_resolution',
         'qos_improvement', 'entry_conditions', 'exit_conditions', 'aircraft_request', 'zone_conflict',
         'return_to_route']
DROP = ['sample_id', 'target_index', 'session', 'atco', 'scenario', 'timestamp', 'task_name', 'row_index', 'cs', 't', 'ev']


def load():
    tr = pd.read_csv(DATA / "train.csv")
    va = pd.read_csv(DATA / "validation.csv")
    te = pd.read_csv(DATA / "test.csv")
    te["target_index"] = -1
    d = pd.concat([tr, va, te], ignore_index=True)
    d["cs"] = [n[len(TYPES[t]) + 1:].split("_")[0] for n, t in zip(d.task_name, d.task_index)]
    d["t"] = pd.to_timedelta(d.timestamp).dt.total_seconds()
    return d.sort_values(["session", "t", "row_index"]).reset_index(drop=True)


def add_temporal(d):
    """Momento del siguiente input del controlador sobre el mismo avión (usa snapshots futuros de la sesión)."""
    d["ev"] = np.where(d.a_since_flight < 3600, d.t - d.a_since_flight, np.nan)
    nxt = np.full(len(d), np.nan)
    for _, g in d.groupby("session"):
        ev = g.dropna(subset=["ev"]).groupby("cs").ev.apply(lambda s: np.unique(np.round(s.values)))
        for i, cs, t in zip(g.index, g.cs.values, g.t.values):
            e = ev.get(cs)
            if e is not None:
                j = np.searchsorted(e, t + 1.5)
                if j < len(e):
                    nxt[i] = e[j] - t
    d["f_next_input"] = nxt
    d["f_next_input_rk"] = d.groupby("sample_id").f_next_input.rank(method="min")
    return d


def ranker():
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                          lambdarank_truncation_level=5, verbose=-1)


def fit(m, A, F):
    m.fit(A[F], A.target_index, group=A.groupby("sample_id", sort=False).size().values,
          categorical_feature=["task_index"])
    return m


def top3(B, s):
    B = B.assign(s=s).sort_values(["sample_id", "s"], ascending=[True, False])
    return B.groupby("sample_id").row_index.apply(lambda r: " ".join(map(str, r.iloc[:3])))


def main():
    d = add_temporal(load())
    F = [c for c in d.columns if c not in DROP]
    tr, te = d[d.target_index >= 0], d[d.target_index < 0]

    hits = []
    for a, b in GroupKFold(5).split(tr, groups=tr.atco):
        A, B = tr.iloc[a], tr.iloc[b]
        pred = top3(B, fit(ranker(), A, F).predict(B[F]))
        lab = B[B.target_index == 1].set_index("sample_id").row_index
        hits += [str(lab[s]) in p.split()[:3] for s, p in pred.items()]
    print(f"CV top-3 (GroupKFold por atco): {np.mean(hits):.4f}")

    m = fit(ranker(), tr, F)
    sub = top3(te, m.predict(te[F])).rename("prediction").reset_index()
    OUT.mkdir(exist_ok=True)
    sub.to_csv(OUT / "sub_lgb_temporal.csv", index=False)
    print("Escrito", OUT / "sub_lgb_temporal.csv", len(sub), "filas")


if __name__ == "__main__":
    main()

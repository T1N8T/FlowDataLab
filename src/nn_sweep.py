"""
nn_sweep.py · Pablo (A) · Barridos de parámetros del transformer, una cosa cada vez.

Carga los datos una sola vez y lanza m_nn.run() para la configuración base y para cada variación,
cambiando UN parámetro respecto a la base. Por defecto en modo rápido (1 semilla, 5 folds, sin guardar salidas).
Todo queda apuntado en experiments/nn_runs.csv.

Uso:
    python src/nn_sweep.py drop=0.1,0.3 layers=2,4 lr=5e-4,2e-3
    python src/nn_sweep.py --base "--d 128 --ls 0.05" drop=0.1,0.3      # otra base
    python src/nn_sweep.py --folds 0,1 d=32,128                        # aún más rápido (2 folds)
    python src/nn_sweep.py --no-base ls=0,0.2                          # no repetir la base
    python src/nn_sweep.py --show                                       # ver el registro ordenado por CV
Nombres de parámetro: los de m_nn.py sin guiones (d, heads, layers, drop, ls, lr, wd, bs, epochs, patience,
noise, n_es).
"""
import argparse, shlex, sys, pandas as pd
import m_nn


def show(n=25):
    f = m_nn.EXP / "nn_runs.csv"
    if not f.exists():
        print("Aún no hay experimentos en", f); return
    r = pd.read_csv(f)
    cols = ["when", "name", "tag", "cv", "min_atco", "seeds", "folds", "d", "heads", "layers", "drop", "ls", "lr",
            "wd", "bs", "noise", "best_epochs", "secs"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(r[[c for c in cols if c in r]].sort_values("cv", ascending=False).head(n).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("changes", nargs="*", help="param=v1,v2,...")
    ap.add_argument("--base", default="", help="argumentos de m_nn.py para la base, entre comillas")
    ap.add_argument("--folds", default="", help="limitar a estos folds")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--show", action="store_true")
    s = ap.parse_args()
    if s.show or not s.changes and s.no_base:
        return show()

    common = shlex.split(s.base) + ["--quick", "--seeds", str(s.seeds)] + (["--folds", s.folds] if s.folds else [])
    base = m_nn.parse(common + ["--name", "sweep_base"])
    base.seeds = s.seeds                     # --quick fuerza 1; aquí manda --seeds
    runs = [] if s.no_base else [("base", base)]
    for ch in s.changes:
        k, vals = ch.split("=", 1)
        key = k.replace("-", "_")
        if not hasattr(base, key): sys.exit(f"Parámetro desconocido: {k}")
        for v in vals.split(","):
            a = m_nn.parse(common + ["--name", f"sweep_{key}={v}", f"--{k.replace('_', '-')}", v])
            a.seeds = s.seeds
            runs.append((f"{key}={v}", a))
    for _, a in runs: a.tag = a.tag or s.base

    data = m_nn.build_data()
    res = []
    for label, a in runs:
        res.append((label, m_nn.run(a, data)))
    ref = dict(res).get("base")
    print("\n===== resumen" + (f" (base {ref:.4f})" if ref else ""))
    for label, cv in sorted(res, key=lambda x: -x[1]):
        print(f"  {label:20s} {cv:.4f}" + (f"  ({cv - ref:+.4f})" if ref is not None and label != "base" else ""))
    print("Ojo: con 1 semilla, diferencias < ~0.005 son ruido. Confirmar lo que gane con --seeds 3.")


if __name__ == "__main__":
    main()

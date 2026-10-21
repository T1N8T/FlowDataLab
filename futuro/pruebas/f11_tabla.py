"""f11_tabla.py · tabla markdown de las pruebas del hilo f11 (resultados.jsonl): 1.er pase, + sesión y las métricas de la
sección 14 frente a su referencia. Uso: python futuro/pruebas/f11_tabla.py [filtro]"""
import json, sys
from pathlib import Path

R = Path(__file__).resolve().parent / "resultados.jsonl"
filas = {}
for l in open(R):
    x = json.loads(l)
    if x.get("hilo") == "f11":
        filas[x["nombre"]] = x
filtro = sys.argv[1] if len(sys.argv) > 1 else ""
print("| prueba | 1.er pase | + sesión | Δ | Δ mediana por sesión | sesiones ↑/↓ | controladores ↑/↓ | peor controlador | CV 023 | escenarios 1/2/3 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n, x in filas.items():
    if not n.endswith("_ses") or filtro not in n:
        continue
    p = filas.get(n[:-4], {})
    d = lambda k: f"{x[k] * 100:+.2f}" if k in x else "—"
    ud = lambda a, b: f"{x[a]} / {x[b]}" if a in x else "—"
    print(f"| `{n[:-4]}` | {p.get('cv', float('nan')):.4f} | **{x['cv']:.4f}** | {d('d_cv')} | {d('d_ses_median')} | "
          f"{ud('ses_up', 'ses_down')} | {ud('atco_up', 'atco_down')} | {x['min_atco']:.3f} | {x.get('cv_023', float('nan')):.4f} | "
          f"{x.get('sc1', 0):.3f} / {x.get('sc2', 0):.3f} / {x.get('sc3', 0):.3f} |")

"""f11_nb.py · escribe/actualiza celdas del cuaderno LGB_futuro.ipynb identificadas por una marca (<!-- f11:clave -->),
insertándolas antes de «Historial completo». Uso: python futuro/pruebas/f11_nb.py clave fichero.md [code]"""
import json, sys, hashlib
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "LGB_futuro.ipynb"


def poner(clave, texto, tipo="markdown", despues=None):
    nb = json.loads(NB.read_text())
    marca = f"<!-- f11:{clave} -->" if tipo == "markdown" else f"# f11:{clave}"
    src = (marca + "\n" + texto).splitlines(keepends=True)
    cell = dict(cell_type=tipo, id="f11" + hashlib.md5(clave.encode()).hexdigest()[:5], metadata={}, source=src)
    if tipo == "code":
        cell.update(execution_count=None, outputs=[])
    for c in nb["cells"]:
        if "".join(c["source"]).startswith(marca):
            c.clear(); c.update(cell)
            break
    else:
        if despues:      # justo después de la celda que empieza por `despues`
            i = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith(despues)) + 1
        else:
            i = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("## Historial completo"))
        nb["cells"].insert(i, cell)
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    poner(sys.argv[1], Path(sys.argv[2]).read_text(), sys.argv[3] if len(sys.argv) > 3 else "markdown",
          sys.argv[4] if len(sys.argv) > 4 else None)

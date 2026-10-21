"""f11_cola.py · cola de pruebas de f11_lab.py: una por línea en el fichero dado ("nombre | argumentos"), N a la vez.
Uso: python futuro/pruebas/f11_cola.py futuro/pruebas/notas/f11_tanda1.txt 4"""
import sys, shlex, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/alumno.upv.es/ptorsim/venv-flow/bin/python"


def uno(linea):
    nombre, args = [x.strip() for x in linea.split("|", 1)]
    log = ROOT / "logs" / f"f11_{nombre}.log"
    with open(log, "w") as f:
        subprocess.run([PY, str(ROOT / "futuro" / "pruebas" / "f11_lab.py"), nombre] + shlex.split(args),
                       stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    res = [l for l in open(log) if l.startswith("RESUMEN") or "Traceback" in l or "error:" in l]
    print(f"terminado {nombre}: {res[-1].strip() if res else '¿?'}", flush=True)


if __name__ == "__main__":
    lineas = [l for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
        list(ex.map(uno, lineas))

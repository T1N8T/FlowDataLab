"""f11_cola2.py · cola dinámica de f11_lab.py: relee el fichero cada 15 s (se pueden añadir líneas mientras corre) y lanza
la siguiente prueba no lanzada cuando hay menos de N procesos f11_lab.py en la máquina (cuenta también los de otras colas).
Líneas "nombre | argumentos"; '#' comenta. Termina cuando el fichero contiene una línea 'FIN' y todo está lanzado.
Uso: python futuro/pruebas/f11_cola2.py futuro/pruebas/notas/f11_cola.txt 4"""
import sys, shlex, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/alumno.upv.es/ptorsim/venv-flow/bin/python"


def corriendo():
    r = subprocess.run(["pgrep", "-f", "futuro/pruebas/f11_lab.py"], capture_output=True, text=True)
    return len(r.stdout.split())


if __name__ == "__main__":
    cola, N = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lanzadas, procs = set(), {}
    while True:
        lineas = [l.strip() for l in cola.read_text().splitlines() if l.strip() and not l.startswith("#")]
        fin = "FIN" in lineas
        pend = [l for l in lineas if l != "FIN" and l.split("|")[0].strip() not in lanzadas]
        for n, p in list(procs.items()):
            if p.poll() is not None:
                res = [l for l in open(ROOT / "logs" / f"f11_{n}.log") if l.startswith("RESUMEN") or "Traceback" in l]
                print(f"{time.strftime('%H:%M')} terminado {n}: {res[-1].strip()[:220] if res else '¿?'}", flush=True)
                del procs[n]
        while pend and corriendo() < N:
            l = pend.pop(0)
            n, args = [x.strip() for x in l.split("|", 1)]
            log = open(ROOT / "logs" / f"f11_{n}.log", "w")
            procs[n] = subprocess.Popen([PY, str(ROOT / "futuro" / "pruebas" / "f11_lab.py"), n] + shlex.split(args),
                                        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            lanzadas.add(n)
            print(f"{time.strftime('%H:%M')} lanzado {n}", flush=True)
            time.sleep(2)
        if fin and not pend and not procs:
            break
        time.sleep(15)

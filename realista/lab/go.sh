#!/bin/bash
# go.sh <nombre_log> <args de lab.py...>  · lanza lab.py en segundo plano con nice (log en models/real_lab/logs)
# PY=ruta/a/python y OMP=hilos opcionales; mirad antes la carga de la máquina (uptime) y no la saturéis
cd "$(dirname "$0")"
LOGS=../../models/real_lab/logs; mkdir -p $LOGS
name=$1; shift
OMP_NUM_THREADS=${OMP:-4} nohup nice -n 10 ${PY:-python} lab.py "$@" > $LOGS/$name.log 2>&1 &
echo "lanzado $name pid $!"

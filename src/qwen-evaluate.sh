#!/bin/bash
#PBS -N qwen-evaluate
#PBS -l select=1:ncpus=64:mem=64gb:scratch_local=256gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-evaluate.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

export TMPDIR=${SCRATCHDIR}

cd ${SCRATCHDIR}

cp ${PBS_O_WORKDIR}/data.tar .
tar -xf data.tar

cp -a /storage/brno2/home/xmachu05/knn-mis/outputs/qwen-lora-2026-04-11_09-20-37 .

cp ${PBS_O_WORKDIR}/src/qwen-evaluate.py .
cp ${PBS_O_WORKDIR}/test.json .
cp ${PBS_O_WORKDIR}/requirements.txt .

module load python

python -m venv venv
source venv/bin/activate

pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

python qwen-evaluate.py

cp report.json ${PBS_O_WORKDIR}

clean_scratch

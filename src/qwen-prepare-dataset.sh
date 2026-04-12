#!/bin/bash
#PBS -N qwen-prepare-dataset
#PBS -l select=1:ncpus=1:mem=8gb:scratch_local=128gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-prepare-dataset.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

cd ${SCRATCHDIR}

cp ${PBS_O_WORKDIR}/data.tar .
tar -xf data.tar

cp ${PBS_O_WORKDIR}/src/qwen-prepare-dataset.py .
cp ${PBS_O_WORKDIR}/metadata.json .
cp ${PBS_O_WORKDIR}/requirements.txt .

module load python

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python qwen-prepare-dataset.py

cp train.json ${PBS_O_WORKDIR}
cp eval.json ${PBS_O_WORKDIR}
cp test.json ${PBS_O_WORKDIR}

clean_scratch

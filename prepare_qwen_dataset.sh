#!/bin/bash
#PBS -N prepare-qwen-dataset
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=64gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: prepare_qwen_dataset.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

cd "${SCRATCHDIR}"

[ -f "${PBS_O_WORKDIR}/data.tar" ] \
  && cp "${PBS_O_WORKDIR}/data.tar" . \
  && tar -xf data.tar

cp "${PBS_O_WORKDIR}/prepare_qwen_dataset.py" .
cp "${PBS_O_WORKDIR}/requirements.txt" .

module load python

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python prepare_qwen_dataset.py

cp train.json "${PBS_O_WORKDIR}/"
cp eval.json "${PBS_O_WORKDIR}/"

clean_scratch

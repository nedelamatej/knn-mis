#!/bin/bash
#PBS -N qwen-prepare-dataset
#PBS -l select=1:ncpus=1:mem=8gb:scratch_local=64gb
#PBS -l walltime=12:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-prepare-dataset.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

# Run this script in knn-mis directory

VENV_DIR="${PBS_O_WORKDIR}/../venv"

export TMPDIR="${SCRATCHDIR}"

mkdir -p "${TMPDIR}"

cd ${SCRATCHDIR}

# Copy data
cp -r ${PBS_O_WORKDIR}/data .
mkdir -p data/jpg
tar -xf data/jpg.tar -C data/jpg

# Copy scripts
cp ${PBS_O_WORKDIR}/src/qwen-prepare-dataset.py .

# Load modules
module load python

source "${VENV_DIR}/bin/activate"

python qwen-prepare-dataset.py

cp data/train.json ${PBS_O_WORKDIR}/data
cp data/eval.json ${PBS_O_WORKDIR}/data
cp data/test.json ${PBS_O_WORKDIR}/data

clean_scratch

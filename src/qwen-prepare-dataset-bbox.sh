#!/bin/bash
#PBS -N qwen-prepare-dataset-bbox
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=128gb
#PBS -l walltime=12:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-prepare-dataset-bbox.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

# Run this script in knn-mis directory

VENV_DIR="${PBS_O_WORKDIR}/../venvs/knn-prepare-bbox"
PERO_OCR_DIR="${PBS_O_WORKDIR}/../pero-ocr"

export TMPDIR="${SCRATCHDIR}"

mkdir -p "${TMPDIR}"

cd ${SCRATCHDIR}

# Copy scripts
cp ${PBS_O_WORKDIR}/src/qwen-prepare-dataset-bbox.py .
cp ${PBS_O_WORKDIR}/src/alto_utils.py .

# Copy data
cp -r ${PBS_O_WORKDIR}/data .
mkdir -p data/jpg
tar -xf data/jpg.tar -C data/jpg
mkdir -p data/alto
tar -xzf data/alto.tar.gz -C data/alto

# Load modules
module load python

source "${VENV_DIR}/bin/activate"

# Make local scripts and pero-ocr available for imports
export PYTHONPATH="${SCRATCHDIR}:${PERO_OCR_DIR}:${PYTHONPATH:-}"

python qwen-prepare-dataset-bbox.py

cp data/train_bbox.json ${PBS_O_WORKDIR}/data
cp data/eval_bbox.json ${PBS_O_WORKDIR}/data
cp data/test_bbox.json ${PBS_O_WORKDIR}/data

clean_scratch

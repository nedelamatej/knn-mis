#!/bin/bash
#PBS -N qwen-evaluate-gpu
#PBS -l select=1:ncpus=16:mem=64gb:scratch_local=256gb:ngpus=1:gpu_mem=16gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-evaluate-gpu.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

VENV_DIR="${PBS_O_WORKDIR}/../venv"

export TMPDIR="${SCRATCHDIR}/tmp"
export HF_HOME="${SCRATCHDIR}/hf-home"
export HUGGINGFACE_HUB_CACHE="${SCRATCHDIR}/hf-cache"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

cd ${SCRATCHDIR}

cp ${PBS_O_WORKDIR}/data.tar .
tar -xf data.tar

cp -a /storage/brno2/home/xmachu05/knn-mis/outputs/qwen-lora-2026-04-11_09-20-37 .

cp ${PBS_O_WORKDIR}/src/qwen-evaluate.py .
cp ${PBS_O_WORKDIR}/test.json .
cp ${PBS_O_WORKDIR}/requirements.txt .

module load python
module load cuda

source "${VENV_DIR}/bin/activate"

python qwen-evaluate.py -c 5 -b 5

cp report.json ${PBS_O_WORKDIR}/better-report-gpu.json

clean_scratch

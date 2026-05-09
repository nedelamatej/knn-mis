#!/bin/bash
#PBS -N qwen-evaluate
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=128gb:ngpus=1:gpu_mem=16gb
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

BASE=Qwen2.5-VL-3B-Instruct
LORA=Qwen2.5-VL-3B-Instruct-lora-19611240
BBOX=false

VENV_DIR="${PBS_O_WORKDIR}/../venv"

export TMPDIR="${SCRATCHDIR}/tmp"
export HF_HOME="${SCRATCHDIR}/hf-home"
export HUGGINGFACE_HUB_CACHE="${SCRATCHDIR}/hf-cache"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

cd ${SCRATCHDIR}

cp -r ${PBS_O_WORKDIR}/data .

mkdir -p data/jpg

tar -xf data/jpg.tar -C data/jpg

cp -a /storage/brno2/home/xmachu05/knn-mis/outputs/${LORA} .

cp ${PBS_O_WORKDIR}/src/qwen-evaluate.py .

module load python
module load cuda

source "${VENV_DIR}/bin/activate"

if [ "${BBOX}" = "true" ]; then
  cp /storage/brno2/home/xmachu05/knn-mis/data/test_bbox_1.json data/test.json

  python qwen-evaluate.py -m ${BASE} -l ${LORA} --bbox
  cp report.json ${PBS_O_WORKDIR}/${BASE}-bbox.json
else
  python qwen-evaluate.py -m ${BASE} -l ${LORA}
  cp report.json ${PBS_O_WORKDIR}/${BASE}.json
fi

clean_scratch

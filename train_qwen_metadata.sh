#!/bin/bash
#PBS -N qwen-metadata
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=80gb:ngpus=1
#PBS -l walltime=08:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: train_qwen_metadata.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail
trap 'clean_scratch' EXIT TERM

TRAIN_FILE="train.json"
EVAL_FILE="eval.json"
OUTPUT_DIR="qwen_metadata_lora"

cd "${SCRATCHDIR}"

echo "=== JOB INFO ==="
echo "PBS_JOBID=${PBS_JOBID}"
echo "PBS_O_WORKDIR=${PBS_O_WORKDIR}"
echo "SCRATCHDIR=${SCRATCHDIR}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "EVAL_FILE=${EVAL_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
hostname

# 1) unpack data
[ -f "${PBS_O_WORKDIR}/data.tar" ] \
  && cp "${PBS_O_WORKDIR}/data.tar" . \
  && tar -xf data.tar

# 2) copy dataset
cp "${PBS_O_WORKDIR}/${TRAIN_FILE}" .
cp "${PBS_O_WORKDIR}/${EVAL_FILE}" .

# 3) clone framework
git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git

# 4) python environment
module load python

python -m venv venv
source venv/bin/activate

pip install --upgrade pip

# 5) training
cd Qwen-VL-Series-Finetune
pip install -r requirements.txt

bash scripts/finetune_lora.sh \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path "${SCRATCHDIR}/${TRAIN_FILE}" \
  --eval_path "${SCRATCHDIR}/${EVAL_FILE}" \
  --image_folder "${SCRATCHDIR}/data/png" \
  --output_dir "${SCRATCHDIR}/${OUTPUT_DIR}"

# 6) copy results
mkdir -p "${PBS_O_WORKDIR}/outputs"
cp -r "${SCRATCHDIR}/${OUTPUT_DIR}" "${PBS_O_WORKDIR}/outputs/"

echo "=== DONE ==="

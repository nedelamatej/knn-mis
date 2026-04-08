#!/bin/bash
#PBS -N qwen-metadata-train
#PBS -l select=1:ncpus=4:mem=64gb:scratch_local=100gb:ngpus=1:gpu_mem=16gb
#PBS -l walltime=24:00:00

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

cd "${SCRATCHDIR}"

echo "=== JOB INFO ==="
echo "PBS_JOBID=${PBS_JOBID}"
echo "PBS_O_WORKDIR=${PBS_O_WORKDIR}"
echo "SCRATCHDIR=${SCRATCHDIR}"
hostname

# 1) unpack data
[ -f "${PBS_O_WORKDIR}/data.tar" ] \
  && cp "${PBS_O_WORKDIR}/data.tar" . \
  && tar -xf data.tar

# 2) copy dataset
cp "${PBS_O_WORKDIR}/train.json" .
cp "${PBS_O_WORKDIR}/eval.json" .

# 3) clone framework
git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git

# 4) python enviroment
module load python

python -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

#pip install torch torchvision
#pip install git+https://github.com/huggingface/transformers accelerate
#pip install qwen-vl-utils pillow
#pip install deepspeed liger-kernel datasets sentencepiece

# 5) training
cd Qwen-VL-Series-Finetune

bash scripts/finetune_lora.sh \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path "${SCRATCHDIR}/train.json" \
  --eval_path "${SCRATCHDIR}/eval.json" \
  --image_folder "${SCRATCHDIR}/data/png" \
  --output_dir "${SCRATCHDIR}/qwen_metadata_lora"

# 6) copy results
mkdir -p "${PBS_O_WORKDIR}/outputs"
cp -r "${SCRATCHDIR}/qwen_metadata_lora" "${PBS_O_WORKDIR}/outputs/"

echo "=== DONE ==="

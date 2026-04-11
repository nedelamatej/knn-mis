#!/bin/bash
#PBS -N qwen-metadata
#PBS -l select=1:ncpus=4:mem=96gb:scratch_local=250gb:ngpus=1:gpu_mem=24gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: train_qwen.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail
trap 'clean_scratch' EXIT TERM

# settings
XLOGIN="xmachu05"
TRAIN_FILE="train.json"
DATA_TAR_FILE="data.tar"
DATA_DIR="${DATA_TAR_FILE%.tar}"
OUTPUT_DIR="qwen-lora-$(date +%F_%H-%M-%S)"
VENV="knn-qwen"
REPO_DIR="/storage/brno2/home/${XLOGIN}/Qwen-VL-Series-Finetune"
VENV_DIR="/storage/brno2/home/${XLOGIN}/venvs/${VENV}"
OUTPUT_BASE_DIR="${PBS_O_WORKDIR}/outputs"

cd "${SCRATCHDIR}"

echo "=== JOB INFO ==="
echo "PBS_JOBID=${PBS_JOBID}"
echo "PBS_O_WORKDIR=${PBS_O_WORKDIR}"
echo "SCRATCHDIR=${SCRATCHDIR}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "DATA_TAR_FILE=${DATA_TAR_FILE}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "VENV_DIR=${VENV_DIR}"
hostname

export TMPDIR="${SCRATCHDIR}/tmp"
export TRITON_CACHE_DIR="${SCRATCHDIR}/triton-cache"
export HF_HOME="${SCRATCHDIR}/hf-home"
export HUGGINGFACE_HUB_CACHE="${SCRATCHDIR}/hf-cache"
export TORCH_EXTENSIONS_DIR="${SCRATCHDIR}/torch-extensions"

mkdir -p \
  "${TMPDIR}" \
  "${TRITON_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${OUTPUT_BASE_DIR}"

if [ ! -f "${PBS_O_WORKDIR}/${DATA_TAR_FILE}" ]; then
  echo "ERROR: Missing ${PBS_O_WORKDIR}/${DATA_TAR_FILE}"
  exit 1
fi

if [ ! -f "${PBS_O_WORKDIR}/${TRAIN_FILE}" ]; then
  echo "ERROR: Missing ${PBS_O_WORKDIR}/${TRAIN_FILE}"
  exit 1
fi

cp "${PBS_O_WORKDIR}/${DATA_TAR_FILE}" .
tar -xf "${DATA_TAR_FILE}"
cp "${PBS_O_WORKDIR}/${TRAIN_FILE}" .

if [ ! -d "${SCRATCHDIR}/${DATA_DIR}" ]; then
  echo "ERROR: Expected unpacked directory ${SCRATCHDIR}/${DATA_DIR} not found"
  ls -la
  exit 1
fi

if [ ! -d "${SCRATCHDIR}/${DATA_DIR}/png" ]; then
  echo "ERROR: Expected image directory ${SCRATCHDIR}/${DATA_DIR}/png not found"
  find "${SCRATCHDIR}/${DATA_DIR}" -maxdepth 2 -type d | sort
  exit 1
fi

module add python/3.11
module add cuda
module add ffmpeg
module add pkgconf
module add gcc

source "${VENV_DIR}/bin/activate"

echo "=== ENV CHECK ==="
which python
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

echo "CUDA_VISIBLE_DEVICES before=${CUDA_VISIBLE_DEVICES:-<unset>}"
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES after=${CUDA_VISIBLE_DEVICES}"

nvidia-smi || true

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RESUME_ARG=()
LATEST_CHECKPOINT="$(find "${OUTPUT_BASE_DIR}" -maxdepth 2 -type d -name "checkpoint-*" | sort -V | tail -n 1 || true)"
if [ -n "${LATEST_CHECKPOINT}" ]; then
  echo "Resuming from checkpoint: ${LATEST_CHECKPOINT}"
  RESUME_ARG=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
fi

python src/train/train_sft.py \
  --use_liger_kernel True \
  --lora_enable True \
  --use_dora False \
  --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_lora_modules -1 \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path "${SCRATCHDIR}/${TRAIN_FILE}" \
  --image_folder "${SCRATCHDIR}/${DATA_DIR}/png" \
  --remove_unused_columns False \
  --freeze_vision_tower False \
  --freeze_llm True \
  --freeze_merger False \
  --bf16 False \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir "${SCRATCHDIR}/${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --image_min_pixels $((256 * 28 * 28)) \
  --image_max_pixels $((1280 * 28 * 28)) \
  --learning_rate 1e-4 \
  --merger_lr 1e-5 \
  --vision_lr 2e-6 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 False \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 3 \
  --dataloader_num_workers 2 \
  "${RESUME_ARG[@]}"

cp -r "${SCRATCHDIR}/${OUTPUT_DIR}" "${OUTPUT_BASE_DIR}/"

echo "=== DONE ==="

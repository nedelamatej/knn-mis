#!/bin/bash
#PBS -N qwen-metadata
#PBS -l select=1:ncpus=8:mem=96gb:scratch_local=250gb:ngpus=1:gpu_mem=24gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-train.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

# Run this script in knn-mis directory

set -euo pipefail
trap 'clean_scratch || true' EXIT TERM

# Model
MODEL_NAME="Qwen2-VL-2B-Instruct"
#MODEL_NAME="Qwen2.5-VL-3B-Instruct"
#MODEL_NAME="Qwen2.5-VL-7B-Instruct"
#MODEL_NAME="Qwen2-VL-7B-Instruct"

# Files
TRAIN_FILE="train.json"
EVAL_FILE="eval_small.json"
JPG_TAR_FILE="jpg.tar"
OUTPUT_DIR="${MODEL_NAME}-lora-$(date +%F_%H-%M-%S)-${PBS_JOBID}"

# Paths
TRAIN_PATH="${PBS_O_WORKDIR}/data/${TRAIN_FILE}"
EVAL_PATH="${PBS_O_WORKDIR}/data/${EVAL_FILE}"
JPG_TAR_PATH="${PBS_O_WORKDIR}/data/${JPG_TAR_FILE}"
OUTPUT_DIR_PATH="${PBS_O_WORKDIR}/outputs/${OUTPUT_DIR}"
FRAMEWORK_DIR_PATH="$(dirname "${PBS_O_WORKDIR}")/knn-framework"
VENV_PATH="$(dirname "${PBS_O_WORKDIR}")/venvs/knn-qwen"

cd "${SCRATCHDIR}"

# Cache
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
  "${OUTPUT_DIR}"

# Copy data
cp "${JPG_TAR_PATH}" .
mkdir -p jpg
tar -xf jpg.tar -C jpg
cp "${TRAIN_PATH}" .
cp "${EVAL_PATH}" .

# Load modules
module add python/3.11
module add cuda
module add ffmpeg
module add pkgconf
module add gcc

source "${VENV_PATH}/bin/activate"

echo "=== ENV CHECK ==="
echo "CUDA_VISIBLE_DEVICES before=${CUDA_VISIBLE_DEVICES:-<unset>}"
export CUDA_VISIBLE_DEVICES=0

nvidia-smi || true

cd "${FRAMEWORK_DIR_PATH}"
export PYTHONPATH="${FRAMEWORK_DIR_PATH}/src:${PYTHONPATH:-}"

python src/train/train_sft.py \
  --use_liger_kernel True \
  --lora_enable True \
  --use_dora False \
  --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_lora_modules -1 \
  --model_id "Qwen/${MODEL_NAME}" \
  --data_path "${SCRATCHDIR}/${TRAIN_FILE}" \
  --prediction_loss_only True \
  --eval_path "${SCRATCHDIR}/${EVAL_FILE}" \
  --eval_strategy steps \
  --eval_steps 1000 \
  --per_device_eval_batch_size 1 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --image_folder "${SCRATCHDIR}/jpg" \
  --remove_unused_columns False \
  --freeze_vision_tower False \
  --freeze_llm True \
  --freeze_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir "${SCRATCHDIR}/${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
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
  --save_steps 1000 \
  --save_total_limit 3 \
  --dataloader_num_workers 2

mkdir -p "$(dirname "${OUTPUT_DIR_PATH}")"
cp -r "${SCRATCHDIR}/${OUTPUT_DIR}" "${OUTPUT_DIR_PATH}"

echo "=== DONE ==="

#!/bin/bash

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: setup_venv.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

XLOGIN="xmachu05"

set -euo pipefail

module add python/3.11
module add cuda
module add ffmpeg
module add pkgconf
module add gcc

export TMPDIR="${SCRATCHDIR}/tmp"
export PIP_CACHE_DIR="${SCRATCHDIR}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

VENV_BASE="/storage/brno2/home/${XLOGIN}/venvs"
VENV="${VENV_BASE}/knn-qwen"
mkdir -p "${VENV_BASE}"

python -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja packaging psutil

python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
python -m pip install qwen-vl-utils

# MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation

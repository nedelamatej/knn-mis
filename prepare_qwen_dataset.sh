#!/bin/bash
#PBS -N prepare-qwen-dataset
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=64gb
#PBS -l walltime=01:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: prepare_qwen_dataset.sh
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

# 2) copy script
cp "${PBS_O_WORKDIR}/prepare_qwen_dataset.py" .

# 3) python environment
module load python

python -m venv venv
source venv/bin/activate

python prepare_qwen_dataset.py

# 4) copy results back
cp train.json "${PBS_O_WORKDIR}/"
cp eval.json "${PBS_O_WORKDIR}/"
cp test.json "${PBS_O_WORKDIR}/"

echo "=== DONE ==="

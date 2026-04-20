#!/bin/bash
#PBS -N arxiv-prepare-dataset
#PBS -l select=1:ncpus=2:mem=16gb:scratch_local=128gb
#PBS -l walltime=36:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: arxiv-prepare-dataset.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

set -euo pipefail

VENV_DIR="${PBS_O_WORKDIR}/../venv"

export TMPDIR="${SCRATCHDIR}"
export SINGULARITY_CACHEDIR="${SCRATCHDIR}/singularity_cache"
export SINGULARITY_TMPDIR="${SCRATCHDIR}/singularity_tmp"

mkdir -p "${TMPDIR}" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
mkdir -p "${SCRATCHDIR}/grobid_data/logs" "${SCRATCHDIR}/grobid_data/tmp"

cd ${SCRATCHDIR}

[[ -d "${PBS_O_WORKDIR}/data" ]] \
  && cp -r ${PBS_O_WORKDIR}/data . \
  && mkdir -p data/pdf data/jpg \
  && tar -xf data/pdf.tar -C data/pdf \
  && tar -xf data/jpg.tar -C data/jpg

cp ${PBS_O_WORKDIR}/src/arxiv-prepare-dataset.py .
cp ${PBS_O_WORKDIR}/arxiv-metadata-snapshot.json .
cp ${PBS_O_WORKDIR}/grobid.sif .

module load python

nohup singularity run \
  --bind "${SCRATCHDIR}/grobid_data/logs:/opt/grobid/logs" \
  --bind "${SCRATCHDIR}/grobid_data/tmp:/opt/grobid/grobid-home/tmp" \
  --pwd /opt/grobid grobid.sif &

GROBID_PID=$!

while ! curl -s -o /dev/null http://localhost:8070/api/isalive; do
  sleep 5
done

source "${VENV_DIR}/bin/activate"

python arxiv-prepare-dataset.py -c 100000

kill $GROBID_PID

tar -cf data/pdf.tar -C data/pdf .
tar -cf data/jpg.tar -C data/jpg .

rm -rf data/pdf data/jpg

cp -r data ${PBS_O_WORKDIR}

clean_scratch

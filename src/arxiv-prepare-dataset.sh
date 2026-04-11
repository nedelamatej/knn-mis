#!/bin/bash
#PBS -N arxiv-prepare-dataset
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=64gb
#PBS -l walltime=24:00:00

# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: arxiv-prepare-dataset.sh
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

cd ${SCRATCHDIR}

[ -f "${PBS_O_WORKDIR}/data.tar" ] \
  && cp ${PBS_O_WORKDIR}/data.tar . \
  && tar -xf data.tar

cp ${PBS_O_WORKDIR}/arxiv-prepare-dataset.py .
cp ${PBS_O_WORKDIR}/arxiv-metadata-snapshot.json .
cp ${PBS_O_WORKDIR}/requirements.txt .

module load python

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python arxiv-prepare-dataset.py -c 20000

tar -cf data.tar data

cp data.tar ${PBS_O_WORKDIR}

clean_scratch

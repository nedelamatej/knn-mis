Virtual environment setup:
+ `python3 -m venv venv`
+ `source venv/bin/activate`
+ `pip install -r requirements.txt`

Dataset preparation (step 1, download from arXiv.org):
+ `python3 src/arxiv-prepare-dataset.py`
- use: `python3 src/arxiv-prepare-dataset.py --help` to see more options
- use: `qsub src/arxiv-prepare-dataset.sh` to submit the job to cluster
- sample output saved to: `data/jpg/*`, `data/pdf/*` and `data/metadata.json`

Dataset preparation (step 2a, prepare for Qwen without OCR):
+ `python3 src/qwen-prepare-dataset.py`
- sample output saved to: `data/eval.json`, `data/test.json` and `data/train.json`

Dataset preparation (step 2b, prepare for Qwen with OCR):
+ `python3 src/qwen-prepare-dataset-bbox.py`
- sample output saved to: `data/eval_bbox.json`, `data/test_bbox.json` and `data/train_bbox.json`

Qwen training:
+ `bash src/qwen-train.sh`
- use: `qsub src/qwen-train.sh` to submit the job to cluster

Qwen evaluation:
+ `python3 src/qwen-evaluate.py`
- use: `python3 src/qwen-evaluate.py --help` to see more options
- use: `qsub src/qwen-evaluate.sh` to submit the job to cluster
- sample output saved to: `results/*`

See variables on top of each bash script for customization of paths and parameters.

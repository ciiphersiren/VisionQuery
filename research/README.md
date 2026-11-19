# Research code

Everything behind the paper's numbers: training, evaluation, counting, conformal intervals and the VLM comparison.
The web application lives at the repository root; this folder holds the experiments.

All GPU work ran in Kaggle notebooks (2× NVIDIA T4). The CPU steps (scoring, conformal intervals, statistics, figures)
run on any machine from the cached outputs in [`experiments/`](experiments/).

## Layout

| Folder | What is in it |
|---|---|
| [`dronekit/`](dronekit/) | Shared Python package, one implementation of the evaluation protocol used by every notebook: dataset loaders (VisDrone, UAVDT, CARPK), tiling, sliced inference engine, COCO scorer, count records, conformal intervals, VLM prompts and parsers |
| [`tests/`](tests/) | Unit tests for `dronekit` |
| [`tools/`](tools/) | Scripts that generate the Kaggle notebooks (`build_*_notebook.py`) and run the CPU analyses (`phase3_score.py`, `phase4_conformal.py`, figures, application calibration) |
| [`notebooks/`](notebooks/) | The Kaggle notebooks, one per GPU job (generated from `tools/`) |
| [`notebooks/original/`](notebooks/original/) | The first training and evaluation notebooks, with their outputs |
| [`notebooks/openvocab/`](notebooks/openvocab/) | Open-vocabulary experiments (zero-shot detectors, held-out classes, SAM 3 labels → YOLOE-26s) |
| [`countset/`](countset/) | Ground-truth count records per image and class for every split, the fixed 300-image subset and the VLM queries |
| [`experiments/`](experiments/) | Result tables (CSV), figures, training logs and run settings for each phase |

## Notebooks

| Notebook | What it does | Results |
|---|---|---|
| `original/visdrone-training.ipynb` | Builds the tiled training set and trains YOLO26m-Tile on VisDrone | `notebooks/original/` outputs |
| `original/yoloe-1.ipynb`, `original/yoloe-2.ipynb` | YOLOE-26m linear probe and full fine-tune on VisDrone | |
| `original/prompt1.ipynb`, `original/prompt2.ipynb` | YOLO26s on VisDrone (slice-aided fine-tuning and sliced inference) | |
| `original/evaluation.ipynb` | Sliced inference engine, zoom, merge and ensemble ablations | |
| `original/visdrone-prompt-app.ipynb` | The first question-answering prototype (Gradio) | |
| `phase0_metrics.ipynb`, `phase0_yolo26s.ipynb` | Basic metrics for every VisDrone model: precision, recall, mAP50, mAP50-95, per-class AP, confusion matrices, PR curves | `experiments/phase0/` |
| `phase1_train.ipynb`, `phase1_retrain.ipynb`, `phase1_retrain_yolo26s.ipynb` | YOLO26m-Tile and YOLO26s trained on UAVDT | `experiments/phase1/` |
| `phase1_epochscan.ipynb` | Scores every saved UAVDT epoch with the paper's scorer | `experiments/phase1/` |
| `phase1_eval.ipynb` | Cross-dataset transfer matrix (VisDrone ↔ UAVDT, CARPK) and count records | `experiments/phase1/eval/` |
| `seeds_train.ipynb`, `seeds_eval.ipynb` | YOLO26m-Tile retrained with two more seeds and scored | `experiments/seeds/` |
| `phase3_vlm.ipynb` | Qwen2.5-VL-7B counting: direct, localize-then-count, and given detector boxes | `experiments/phase3/` |

CPU steps, run from this folder (each script's first lines give its exact arguments):

| Script | What it does | Needs |
|---|---|---|
| `tools/build_countset.py` | Ground-truth count records per image and class | the three datasets' labels |
| `tools/phase4_conformal.py` | Conformal intervals: coverage in-domain, under shift, few-shot recalibration | labels + merged detections from `phase1_eval.ipynb` |
| `tools/phase3_score.py` | Detector vs VLM comparison | the VLM answers in `experiments/phase3/raw/` |

```bash
pip install -r requirements.txt
python -m pytest tests
```

The merged detections (parquet files, written by `phase1_eval.ipynb`) are not in the repository because of their size.
The count records in `countset/` and all result tables, figures and training logs in `experiments/` are included.

## Datasets

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset), [UAVDT](https://sites.google.com/view/grli-uavdt/) and
[CARPK](https://lafi.github.io/LPN/) are not redistributed here; the notebooks download them. All three are licensed for
academic, non-commercial use.

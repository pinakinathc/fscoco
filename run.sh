#!/bin/bash

cd /vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text

source /vol/research/sketchcaption/miniconda/bin/activate pytorch

python -m src.tbir_baseline.main --use_coco

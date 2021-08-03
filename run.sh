#!/bin/bash

cd /vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text

source /vol/research/sketchcaption/miniconda/bin/activate pytorch

python -m cnn_baseline.main

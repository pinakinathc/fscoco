#!/bin/bash

cd /vol/research/sketchcaption/phd/code/partial-associativity/baselines/pvse-sketch/

/user/HS500/pc00725/miniconda3/envs/pvse/bin/python main.py --mode train --img_finetune --batch_size=16 --margin 0.1 --max_violation --num_embeds 1 --img_attention --mmd_weight 0.01 --div_weight 0.1 --batch_size_eval=3 --workers=12 --log_file=condor_pvse_sketch-triplet --ckpt=condor_ckpt-triplet.pth.tar

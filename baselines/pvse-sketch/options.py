# -*- coding: utf-8 -*-
# author: pinakinathc

import os, argparse
parser = argparse.ArgumentParser(description='Parameters for training PVSE-Sketch.')

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
print ('Current directory: ', CUR_DIR)
parser.add_argument('--curr_dir', default=CUR_DIR, type=str, help='Get current working directory.')
parser.add_argument('--log_file', default=CUR_DIR+'/logs/', help='Path to save result logs.')
parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log')
parser.add_argument('--mode', default='evaluate', type=str, help='Run main.py in train/evaluate mode.')

# Data parameters
parser.add_argument('--sketchyCOCO', default='/vol/research/sketchcaption/phd/dataset/SketchyCOCO', type=str, help='Path to SketchyCOCO.')
parser.add_argument('--workers', default=12, type=int, help='Number of data loader workers')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input')

# Model parameters
parser.add_argument('--cnn_type', default='resnet152', help='The CNN used for image encoder')
parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding')
parser.add_argument('--wemb_type', default=None, choices=('glove','fasttext'), type=str, help='Word embedding (glove|fasttext)')
parser.add_argument('--margin', default=0.1, type=float, help='Rank loss margin')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')
parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss')
parser.add_argument('--order', action='store_true', help='Enable order embedding')

# Attention parameters
parser.add_argument('--img_attention', action='store_true', help='Use self attention on images/videos')
parser.add_argument('--txt_attention', action='store_true', help='Use self attention on text')
parser.add_argument('--num_embeds', default=1, type=int, help='Number of embeddings for MIL formulation')

# Loss weights
parser.add_argument('--mmd_weight', default=.0, type=float, help='Weight term for the MMD loss')
parser.add_argument('--div_weight', default=.0, type=float, help='Weight term for the log-determinant divergence loss')

# Training / optimizer setting
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image embedding')
parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embedding')
parser.add_argument('--val_metric', default='rsum', choices=('rsum','med_rsum','mean_rsum'), help='Validation metric to use (rsum|med_rsum|mean_rsum)')
parser.add_argument('--num_epochs', default=700, type=int, help='Number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch')
parser.add_argument('--batch_size_eval', default=16, type=int, help='Size of a evaluation mini-batch')
parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay (l2 norm) for optimizer')
parser.add_argument('--lr', default=.0002, type=float, help='Initial learning rate')
parser.add_argument('--ckpt', default='ckpt.pth.tar', type=str, metavar='PATH', help='path to latest ckpt (default: none)')
parser.add_argument('--eval_on_gpu', action='store_true', help='Evaluate on GPU (default: CPU)')
parser.add_argument('--legacy', action='store_true', help='Turn this on to reproduce results in CVPR2018 paper')
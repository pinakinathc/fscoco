# -*- coding: utf-8 -*-
# author: pinakinathc

import os
import argparse
parser = argparse.ArgumentParser(description='Parameters for training PVSE on partial sketches.')

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
print ('Current directory: ', CUR_DIR)
# parser.add_argument('--root_dir', default='/vol/research/sketchcaption/datasets/sketchyscene/SketchyScene-7k',
# 	type=str, help='Enter root path to SketchScene dataset.')
parser.add_argument('--root_dir', default='/vol/research/sketchcaption/datasets/SketchyCOCO/',
	type=str, help='Enter root path to SketchScene dataset.')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input')
parser.add_argument('--embed_size', type=int, default=1024, help='Dimensionality of joint embedding')
parser.add_argument('--num_embeds', type=int, default=5, help='Number of embeddings for MIL formulation')
parser.add_argument('--img_attention', action='store_true', help='Use self attention on images')
parser.add_argument('--cnn_type', default='resnet18', help='The CNN used for image encoder')
parser.add_argument('--order', action='store_true', help='Enable order embedding')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')

# Logging
parser.add_argument('--log_freq', default=100, type=int, help='Frequency of logging')
parser.add_argument('--log_dir', default='saved_model/log_net', type=str, help='Saved log path')
parser.add_argument('--save_freq', default=2000, type=int, help='Frequency of saving model')
parser.add_argument('--model_path', default='saved_model/net.pth', type=str, help='Saved model path')
parser.add_argument('--eval_freq', default=2000, type=int, help='Frequency of evaluating model')

# Training
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image embedding')
parser.add_argument('--batch_size', default=32, type=int, help='Size of a training mini-batch')
parser.add_argument('--workers', default=12, type=int, help='Number of data loader workers')
parser.add_argument('--lr', default=.0002, type=float, help='Initial learning rate')
parser.add_argument('--num_epochs', default=3000, type=int, help='Number of training epochs')

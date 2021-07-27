import argparse

parser = argparse.ArgumentParser(description='Partial Associativity Experiment')

# ----------------------------
# Dataloader Options
# ----------------------------

# For SketchyScene:
# ------------------

parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/datasets/sketchyscene/SketchyScene-7k',
	help='Enter root directory of SketchyScene')
parser.add_argument('--p_mask', type=float, default=0.3, help='Probability of an instance being masked')
parser.add_argument('--max_len', type=int, default=224, help='Max Edge length of images')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--workers', type=int, default=12, help='Num of workers in dataloader')

opts = parser.parse_args()

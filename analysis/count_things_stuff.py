import os
# from scipy.io import loadmat
import cv2
import numpy as np
import glob

if __name__ == '__main__':
	coco_dir = '/vol/research/sketchcaption/datasets/COCO-stuff/thing-stuff-anns/train2017'

	file_ids_list = glob.glob('/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/sketchycoco/*/*.png')
	file_ids_list = [os.path.split(filename)[-1][:-4] for filename in file_ids_list]

	things_stuff = []
	for file_id in file_ids_list:
		# for folder_dir in [sketchycoco_dir_train, sketchycoco_dir_valInTrain, sketchycoco_dir_val]:
		filepath = os.path.join(coco_dir, file_id+'.png')
		# if os.path.exists(filepath):
		# 	break
		# data = loadmat(filepath)
		# unique = np.unique(data['CLASS_GT'])
		data = cv2.imread(filepath)[:,:,0]
		unique = np.unique(data)
		things_stuff.extend(unique)

	things_stuff = np.unique(things_stuff)
	things = things_stuff[things_stuff<=91]
	things = things[things>=1]
	stuff = things_stuff[things_stuff>91]

	print ('Number of things: {}, stuff: {}'.format(len(things), len(stuff)))

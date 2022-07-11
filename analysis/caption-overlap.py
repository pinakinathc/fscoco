import os
import re
import glob
import json
import numpy as np
from collections import Counter
from nltk import word_tokenize


if __name__ == '__main__':
	coco_dir = '/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/coco.json'
	our_text_dir = '/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/text'

	coco_data = json.load(open(coco_dir))

	sketch_overlap = []
	image_overlap = []

	list_text_files = glob.glob(os.path.join(our_text_dir, '*', '*.txt'))
	for text_filepath in list_text_files:
		cocoid = int(os.path.split(text_filepath)[-1][:-4])

		sketch_caption = open(text_filepath).read().lower()
		list_image_caption = [item for item in coco_data['images'] if item['cocoid'] == cocoid][0]

		# word tokenise
		sketch_caption = word_tokenize(
			re.sub('[^0-9a-z]+', ' ',  sketch_caption))

		
		list_image_caption = [word_tokenize(
			re.sub('[^0-9a-z]+', ' ',  item['raw'])) for item in list_image_caption['sentences']]

		# print (sketch_caption, list_image_caption)
		# input ('check')

		# overlap_len = max([len(list((Counter(sketch_caption) & Counter(image_caption)).elements())) \
		# 		for image_caption in list_image_caption ])
		
		overlap_len = len(list((
			Counter(sketch_caption) & Counter(sum(list_image_caption, []))).elements()))

		sketch_overlap.append(overlap_len*1.0/len(sketch_caption))

		# input (list_image_caption)

		overlap_img = []
		for i in range(5):
			if len(list_image_caption[i]) == 0:
				continue

			all_image_captions = []

			for j in range(5):
				if i == j or len(list_image_caption[j]) == 0:
					continue
				all_image_captions.append(list_image_caption[j])

			overlap_len = len(list((
				Counter(list_image_caption[i]) & Counter(sum(all_image_captions, []))).elements()))

			# image_overlap.append(overlap_len*1.0/len(list_image_caption[i]))

			overlap_img.append(overlap_len*1.0/len(list_image_caption[i]))

		image_overlap.append(np.mean(overlap_img))

		# print (sketch_overlap)
		# print (image_overlap)
		# input ('check')

	print ('Sketch-Image caption overlap: ', np.mean(sketch_overlap))
	print ('Within Image caption overlap: ', np.mean(image_overlap))

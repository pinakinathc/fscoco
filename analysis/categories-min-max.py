# -*- coding: utf-8 -*-
#
# Find the minimum and maximum times a category has occured in FSCOCO
# Since our sketches do not have labels, we can get maximum estimate from images
# and minimum estimate from captions.

import os
import glob
import numpy as np
import cv2
import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import  matplotlib.pyplot as plt

coco_stuff_dir = '/vol/research/sketchcaption/datasets/COCO-stuff/'
fscoco_dir = '/vol/vssp/datasets/still/sketchx-scenesketch/'

all_coco_ids = [os.path.split(item)[-1][:-4] for item in 
    glob.glob(os.path.join(fscoco_dir, 'raster_sketches', '*', '*.jpg'))]

tmp = open(os.path.join(coco_stuff_dir, 'labels.txt'), 'r').readlines()
coco_labels_list = {}
for item in tmp:
    key, value = item.strip().split(':')
    if int(key.strip()) == 0:
        continue
    coco_labels_list[int(key.strip())-1] = value.strip()

upper_bound_images = {}
num_categories_per_sketch = []
for coco_id in tqdm.tqdm(all_coco_ids):
    sketch_gt_filename = glob.glob(os.path.join(coco_stuff_dir, 'thing-stuff-anns', '*', '%s.png'%coco_id))[0]
    sketch_gt = cv2.imread(sketch_gt_filename)[:, :, 0]
    classes = np.unique(sketch_gt)
    num_categories_per_sketch.append(len(classes))
    for class_id in classes:
        if class_id == 255:
            continue
        key = coco_labels_list[class_id]
        if key in upper_bound_images.keys():
            upper_bound_images[key] += 1
        else:
            upper_bound_images[key] = 1

print ('Upper bound: # categories per sketch: {} +- {}, min: {}, max: {}'.format(
    np.mean(num_categories_per_sketch), np.std(num_categories_per_sketch),
    np.min(num_categories_per_sketch), np.max(num_categories_per_sketch)))

print ('UB Sketches per category: mean: {}, std: {}, min: {}, max: {}'.format(
    np.mean(list(upper_bound_images.values())), np.std(list(upper_bound_images.values())),
    np.min(list(upper_bound_images.values())), np.max(list(upper_bound_images.values()))
))

list_categories = np.unique(list(upper_bound_images.keys()))

lower_bound_text = {}
num_categories_per_sketch = []
all_text_files = glob.glob(os.path.join(fscoco_dir, 'text', '*', '*.txt'))
for txt_filepath in tqdm.tqdm(all_text_files):
    txt_data = open(txt_filepath, 'r').read()
    txt_data = word_tokenize(txt_data)
    txt_data = [lemmatizer.lemmatize(word) for word in txt_data]
    num_categories = 0
    for word in txt_data:
        word = word.lower()
        if word in list(coco_labels_list.values()):
            num_categories += 1
            if word in lower_bound_text.keys():
                lower_bound_text[word] += 1
            else:
                lower_bound_text[word] = 1
    if num_categories == 0:
        continue
    num_categories_per_sketch.append(num_categories)

print ('Lower bound: # categories per sketch: {} +- {}, min: {}, max: {}, num_categories: {}'.format(
    np.mean(num_categories_per_sketch), np.std(num_categories_per_sketch),
    np.min(num_categories_per_sketch), np.max(num_categories_per_sketch), len(list(upper_bound_images.keys())) ))

print ('LB Sketches per category: mean: {}, std: {}, min: {}, max: {}, num_categories: {}'.format(
    np.mean(list(lower_bound_text.values())), np.std(list(lower_bound_text.values())),
    np.min(list(lower_bound_text.values())), np.max(list(lower_bound_text.values())), len(list(lower_bound_text.keys()))
))

print ('lower_bound_text: ', lower_bound_text)

# common_classes = list(set(list(upper_bound_images.keys())) & set(list(lower_bound_text.keys())))
common_classes = list(upper_bound_images.keys())
common_classes = sorted(common_classes)
print ('FSCOCO Categories: ', common_classes)

x = np.arange(len(common_classes))
tick_label = common_classes
image_count = [upper_bound_images[item] for item in common_classes] # Maximum value
text_count = [] # Minimum value
for key in common_classes:
    if key in lower_bound_text.keys():
        text_count.append(lower_bound_text[key])
    else:
        text_count.append(0)

image_count = np.array(image_count)
text_count = np.array(text_count)

print (upper_bound_images)

print (lower_bound_text)

indoors = open('analysis/indoors.txt', 'r').readlines()
outdoors = open('analysis/outdoors.txt', 'r').readlines()
indoors = [item.split(':')[-1].strip() for item in indoors]
outdoors = [item.split(':')[-1].strip() for item in outdoors]

print ('\n\n\n\n\nHiiiiiiiiii')

print (len(list(set.intersection(set(list(upper_bound_images.keys())), set(list(indoors)) ))))
print (len(list(set.intersection(set(list(lower_bound_text.keys())), set(list(indoors)) ))))
print (len(list(set.intersection(set(list(upper_bound_images.keys())), set(list(outdoors)) ))))
print (len(list(set.intersection(set(list(lower_bound_text.keys())), set(list(outdoors)) ))))

print (list(set.intersection(set(list(upper_bound_images.keys())), set(list(indoors)) )))
print (list(set.intersection(set(list(lower_bound_text.keys())), set(list(indoors)) )))
print (list(set.intersection(set(list(upper_bound_images.keys())), set(list(outdoors)) )))
print (list(set.intersection(set(list(lower_bound_text.keys())), set(list(outdoors)) )))


plt.style.use('ggplot')
plt.figure(figsize=(23, 3))
plt.bar(x, height=np.log(image_count+1), width=0.8, bottom=np.log(text_count+1), align='center', color='#007acc', alpha=0.5)
plt.xticks(np.arange(len(common_classes)), tick_label, rotation='vertical')
plt.grid(visible=True, which='major', axis='y', linestyle='-.')
plt.savefig('rebuttal-original-scale.pdf', bbox_inches='tight')
plt.show()
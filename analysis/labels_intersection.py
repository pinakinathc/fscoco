# -*- coding-utf -*-

"""
    Find the following:
    * List of categories in FSCOCO
    * List of categories in SketchyCOCO
    * List of categories in SketchyScene
    * List of categories in SketchyCOCO and FSCOCO
    * List of categories in SketchyCOCO but not in FSCOCO
    * List of categories in SketchyScene and FSCOCO
    * List of categories in SketchyScene but not in FSCOCO
"""

import os
import glob
import numpy as np
import cv2
import tqdm
from scipy.io import loadmat
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import  matplotlib.pyplot as plt

coco_stuff_dir = '/vol/research/sketchcaption/datasets/COCO-stuff/'
fscoco_dir = '/vol/vssp/datasets/still/sketchx-scenesketch/'
sketchycoco_dir = '/vol/research/sketchcaption/datasets/SketchyCOCO/'
sketchyscene_dir = '/vol/research/sketchcaption/datasets/sketchyscene/SketchyScene-7k/'


## Get FSCOCO
all_coco_ids = [os.path.split(item)[-1][:-4] for item in 
    glob.glob(os.path.join(fscoco_dir, 'raster_sketches', '*', '*.jpg'))]

tmp = open(os.path.join(coco_stuff_dir, 'labels.txt'), 'r').readlines()
coco_labels_list = {}
for item in tmp:
    key, value = item.strip().split(':')
    if int(key.strip()) == 0:
        continue
    coco_labels_list[int(key.strip())] = value.strip()

fscoco_categories = {}
num_categories_per_sketch = []
for coco_id in tqdm.tqdm(all_coco_ids):
    sketch_gt_filename = glob.glob(os.path.join(coco_stuff_dir, 'thing-stuff-anns', '*', '%s.png'%coco_id))[0]
    sketch_gt = cv2.imread(sketch_gt_filename)[:, :, 0]
    classes = np.unique(sketch_gt)
    num_categories_per_sketch.append(len(classes))
    for class_id in classes:
        if class_id == 255:
            continue
        key = coco_labels_list[class_id+1]
        if key in fscoco_categories.keys():
            fscoco_categories[key] += 1
        else:
            fscoco_categories[key] = 1
fscoco_classes = fscoco_categories
fscoco_categories = np.unique(list(fscoco_categories.keys()))

# indoors = open('analysis/indoors.txt', 'r').readlines()
# outdoors = open('analysis/outdoors.txt', 'r').readlines()
# indoors = [item.split(':')[-1].strip() for item in indoors]
# outdoors = [item.split(':')[-1].strip() for item in outdoors]
# print (indoors, outdoors)


lower_bound_text = {}
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


## Get SketchyCOCO categories
sketchycoco_2_cocostuff = {1:2, 2:3, 3:4, 4:5, 5:10, 6:11, 7:17, 8:18, 9:19, 10:20,\
    11:21, 12:22, 14:24, 15:25, 16:106, 17:124, 18:106, 19:169}

sketchycoco_categories = sorted(np.unique([coco_labels_list[item] for item in sketchycoco_2_cocostuff.values()]))


## Get SketchyScene categories
tmp = open(os.path.join(sketchyscene_dir, 'colorMap_46.txt'), 'r').readlines()
sketchyscene_categories = {}
for idx, row in enumerate(tmp):
    row = row.split()[0].strip()
    sketchyscene_categories[idx+1] = row
sketchyscene_categories_val = sorted(np.unique([item.split()[0].strip() for item in sketchyscene_categories.values()]))

## Print all categories
print ("Number of categories in FSCOCO: {}, categories: {}\n".format(len(fscoco_categories), fscoco_categories),
    "Number of categories in SketchyCOCO: {}, categories: {}\n".format(len(sketchycoco_categories), sketchycoco_categories),
    "Number of categories in SketchyScene: {}, categories: {}".format(len(sketchyscene_categories_val), sketchyscene_categories_val))

# top50 = sorted(list(fscoco_classes.values()))[:-50]
# top50_categories = []
# for key, value in fscoco_classes.items():
#     if value in top50:
#         top50_categories.append(key)

# all_categories = list(set(fscoco_categories).union(set(sketchycoco_categories)).union(set(sketchyscene_categories_val)))
# all_categories = list(set.intersection(set(top50_categories), set(all_categories)))
# all_categories = sorted(all_categories)
all_categories = fscoco_categories

## 
sketchycoco_classes = {} # id --> category
sketchyscene_classes = {} # id --> category

## Load SketchyCOCO
sketchycoco_count = 0
num_categories = []
for sketch_path in tqdm.tqdm(glob.glob(os.path.join(sketchycoco_dir, 'Scene', 'Annotation', 'paper_version', '*', 'CLASS_GT', '*.mat'))):
    classes = np.unique(loadmat(sketch_path)['CLASS_GT'])
    if len(classes)<1 or sketchycoco_count>1225:
        continue
    num_categories.append(len(classes))
    for key in classes:
        if key not in sketchycoco_2_cocostuff.keys():
            continue
        key = sketchycoco_2_cocostuff[key]
        key = coco_labels_list[key]
        if key in sketchycoco_classes.keys():
            sketchycoco_classes[key] += 1
        else:
            sketchycoco_classes[key] = 1
    sketchycoco_count += 1

sketchycoco_classes = {k: v for k, v in sorted(sketchycoco_classes.items(), key=lambda item: item[1], reverse=True)}
print ('Dataset: {} # sketches per category -- min: {}, max: {}, mean: {}, std: {}\n # categories per sketch -- min: {}, max: {}, mean: {}, std: {}, num_categories: {}'.format(
   'sketchycoco', np.min(list(sketchycoco_classes.values())), np.max(list(sketchycoco_classes.values())),
   np.mean(list(sketchycoco_classes.values())), np.std(list(sketchycoco_classes.values())),
   np.min(num_categories), np.max(num_categories), np.mean(num_categories), np.std(num_categories), len(list(sketchycoco_classes.keys()))))
tmp = []
for key, value in sketchycoco_classes.items():
    tmp.append([key, value, value/1225*100])
print (tmp)

## Load SketchyScene
sketchyscene_count = 0
num_categories = []
for sketch_path in tqdm.tqdm(glob.glob(os.path.join(sketchyscene_dir, '*', 'CLASS_GT', '*.mat'))):
    classes = np.unique(loadmat(sketch_path)['CLASS_GT'])
    if sketchyscene_count > 2724:
        continue
    num_categories.append(len(classes))
    for key in classes:
        if key == 0:
            continue
        key = sketchyscene_categories[key]
        if key in sketchyscene_classes.keys():
            sketchyscene_classes[key] += 1
        else:
            sketchyscene_classes[key] = 1
    sketchyscene_count += 1

sketchyscene_classes = {k: v for k, v in sorted(sketchyscene_classes.items(), key=lambda item: item[1], reverse=True)}
print ('Dataset: {} # sketches per category -- min: {}, max: {}, mean: {}, std: {}\n # categories per sketch -- min: {}, max: {}, mean: {}, std: {}, num_categories: {}'.format(
   'sketchyscene', np.min(list(sketchyscene_classes.values())), np.max(list(sketchyscene_classes.values())),
   np.mean(list(sketchyscene_classes.values())), np.std(list(sketchyscene_classes.values())),
   np.min(num_categories), np.max(num_categories), np.mean(num_categories), np.std(num_categories), len(list(sketchyscene_classes.keys())) ))
tmp = []
for key, value in sketchyscene_classes.items():
    tmp.append([key, value, value/2724*100])
print (tmp)


# ## Manual edits to make labels consistent across datasets
# sketchyscene_classes['person'] = sketchyscene_classes.pop('people')
# sketchyscene_classes['couch'] = sketchyscene_classes.pop('sofa')

## all 0 to missing categories
for key in all_categories:
    if key not in fscoco_classes.keys():
        fscoco_classes[key] = 0
    # if key not in sketchycoco_classes.keys():
    #     sketchycoco_classes[key] = 0
    # if key not in sketchyscene_classes.keys():
    #     sketchyscene_classes[key] = 0
    if key not in lower_bound_text.keys():
        lower_bound_text[key] = 0

## Plot
tmp = []
fscoco_classes = {k: v for k, v in sorted(fscoco_classes.items(), key=lambda item: item[1], reverse=True)}
for key, value in fscoco_classes.items():
    tmp.append([key, value, value*100/10000, lower_bound_text[key], lower_bound_text[key]*100/10000])
print (tmp)

plt.figure(figsize=(23, 1))
N = len(all_categories)
# plt.bar(list(range(0, 2*N, 2)), height=[fscoco_classes[item]*100/1000 for item in all_categories], width=1.0, align='center', color='#AFA893', alpha=1.0, label='FSCOCO-UB')
# plt.bar(list(range(1, 2*N, 2)), height=[lower_bound_text[item]*100/1000 for item in all_categories], width=1.0, align='center', color='#74867C', alpha=1.0, label='FSCOCO-LB')
# plt.bar(np.arange(1, 2*N, 2), height=[sketchycoco_classes[item]*100/14081 for item in all_categories], width=1.0, align='center', color='#B6A6A8', alpha=1.0, label='SketchyCOCO')
plt.bar(np.arange(1, 2*N, 2), height=[sketchyscene_classes[item]**0.1 for item in all_categories], width=1.0, align='center', color='#8A959B', alpha=1.0, label='SketchyScene')
# plt.xticks(np.arange(0.5, 2*N, 2), all_categories, rotation='vertical')
plt.xticks([])
plt.legend()
# plt.grid(visible=True, which='major', axis='y', linestyle='-.')
plt.savefig('sketchyscene.pdf', bbox_inches='tight')
plt.show()

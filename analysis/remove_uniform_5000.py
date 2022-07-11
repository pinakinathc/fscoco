import os
import glob
import tqdm
import cv2
import numpy as np

coco_stuff_dir = '/vol/research/sketchcaption/datasets/COCO-stuff/'
fscoco_dir = '/vol/vssp/datasets/still/sketchx-scenesketch/'

all_coco_ids = glob.glob(os.path.join(fscoco_dir, 'raster_sketches', '*', '*.jpg'))
all_coco_ids = [os.path.split(item)[-1][:-4] for item in all_coco_ids]

val_set = np.loadtxt(os.path.join(fscoco_dir, 'val_normal.txt'), dtype=str)

training_coco_ids = list(set(all_coco_ids) - set(val_set))

tmp = open(os.path.join(coco_stuff_dir, 'labels.txt'), 'r').readlines()
coco_labels_list = {}
for item in tmp:
    key, value = item.strip().split(':')
    if int(key.strip()) == 0:
        continue
    coco_labels_list[int(key.strip())-1] = value.strip()

categories_list = {}
num_categories_per_sketch = []
for coco_id in tqdm.tqdm(training_coco_ids):
    sketch_gt_filename = glob.glob(os.path.join(coco_stuff_dir, 'thing-stuff-anns', '*', '%s.png'%coco_id))[0]
    sketch_gt = cv2.imread(sketch_gt_filename)[:, :, 0]
    classes = np.unique(sketch_gt)
    num_categories_per_sketch.append(len(classes))
    for class_id in classes:
        if class_id == 255:
            continue
        key = coco_labels_list[class_id]
        if key in categories_list.keys():
            categories_list[key] += 1
        else:
            categories_list[key] = 1

# set max removed from each category
for key in categories_list.keys():
    categories_list[key] = int(0.27 * categories_list[key])

deleted_ids = []
for coco_id in training_coco_ids:
    sketch_gt_filename = glob.glob(os.path.join(coco_stuff_dir, 'thing-stuff-anns', '*', '%s.png'%coco_id))[0]
    sketch_gt = cv2.imread(sketch_gt_filename)[:, :, 0]
    classes = np.unique(sketch_gt)
    try:
        classes = [coco_labels_list[item] for item in classes if item != 255]
    except:
        print ('checking--', classes, sketch_gt)
        raise ValueError('check')

    flag = False
    for class_ in classes:
        if categories_list[class_] == 0:
            flag = True
            break
    
    if flag:
        continue

    for class_ in classes:
        categories_list[class_] -= 1

    deleted_ids.append(coco_id)

print (len(deleted_ids))

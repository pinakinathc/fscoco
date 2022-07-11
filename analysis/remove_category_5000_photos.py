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

training_coco_ids = np.loadtxt('/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/category_5000_photos.txt', dtype=str)

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

categories_list = list(categories_list.items())
categories_list.sort(key=lambda x: x[1])


for category_name in tqdm.tqdm([item[0] for item in categories_list]):
    delete_ids = []
    for coco_id in training_coco_ids:
        filename = glob.glob(os.path.join(coco_stuff_dir, 'thing-stuff-anns', '*', '%s.png'%coco_id))[0]
        anns = np.unique(cv2.imread(filename)[:, :, 0])
        for class_id in anns:
            if class_id == 255:
                continue
            key = coco_labels_list[class_id]
            if key == category_name:
                delete_ids.append(coco_id)
    
    for coco_ids in delete_ids:
        training_coco_ids.remove(coco_ids)
        if len(training_coco_ids) <= 5000:
            break

    if len(training_coco_ids) <= 5000:
        break

print (training_coco_ids)
np.savetxt('remove_category_5000_photos.txt', training_coco_ids, fmt='%s')
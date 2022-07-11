# -*- coding: utf-8 -*-
# Finds the object distribution in scene images

import os
import glob
import json
import tqdm
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform analysis on collected sketches')
    parser.add_argument('--data_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/',
        help='path to directory with sketches')
    opt = parser.parse_args()

    all_filepath = glob.glob(os.path.join(opt.data_dir, 'images', '*', '*.jpg'))
    all_ids = [int(os.path.split(filepath)[-1][:-4]) for filepath in all_filepath]

    coco_labels = open(os.path.join(opt.data_dir, 'stuff_labels.md'), 'r').readlines()
    coco_labels_dic = {}
    for label in coco_labels[13:196]:
        label = [item.strip() for item in label.split('|')]
        coco_labels_dic[int(label[0])] = label[1]

    stuff_anns = []
    print ('loading COCO-Stuff annotation. Please wait ...')
    ## Background
    for stuff_path in glob.glob(os.path.join(opt.data_dir, 'stuff*.json')):
        stuff_anns.extend(json.load(open(stuff_path, 'r'))['annotations'])
    
    ## Foreground
    # for stuff_path in glob.glob(os.path.join(opt.data_dir, 'instances*.json')):
    #     stuff_anns.extend(json.load(open(stuff_path, 'r'))['annotations'])

    print ('Processing data ...')
    fscoco_category = {}
    for coco_id in tqdm.tqdm(all_ids[:]):
        all_categories = [item['category_id'] \
            for item in stuff_anns if item['image_id']==coco_id]
        all_categories = np.unique(all_categories)
        for category in all_categories:

            ## Skip Clause indoors: 
            ## 1-43 ==> indoors
            ## 44 - 91 ==> outdoors
            if False:
                continue

            if category>90:
                category = category - 1
            key = coco_labels_dic[category]
            if key in fscoco_category.keys():
                fscoco_category[key] += 1
            else:
                fscoco_category[key] = 1

    data = list(fscoco_category.values())
    keys = list(fscoco_category.keys())

    # Reorder / reshuffle
    order = np.argsort(data)[::-1]
    N = int(len(order)*1.0)

    data = [data[i] for i in order[:N]]
    keys = [keys[i] for i in order[:N]]

    plt.figure(figsize=(15,3))
    g = sns.barplot(x=np.arange(len(data)), y=data)
    plt.xticks(np.arange(len(data)), keys, rotation='vertical')
    # for p in g.patches:
    #     g.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
    #                 ha='center', va='bottom',
    #                 color= 'black')

    
    # plt.savefig('rebuttal-data-foreground.png', bbox_inches='tight')
    plt.savefig('rebuttal-data-background.png', bbox_inches='tight')

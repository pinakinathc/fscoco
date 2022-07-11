# -*- coding: utf-8 -*-
# Finds the object distribution in scene images

import os
import glob
import json
import tqdm
import argparse
from scipy.io import loadmat
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
sns.set_theme()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform analysis on collected sketches')
    parser.add_argument('--fscoco_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/',
        help='path to directory with FSCOCO sketches')
    parser.add_argument('--sketchycoco_dir', type=str, default='/vol/research/sketchcaption/datasets/SketchyCOCO/Scene',
        help='path to directory with SketchyCOCO sketches')
    parser.add_argument('--sketchyscene_dir', type=str, default='/vol/research/sketchcaption/datasets/sketchyscene/SketchyScene-7k',
        help='path to directory with SketchyScene sketches')
    opt = parser.parse_args()

    coco_classes = {} # id --> category
    fscoco_classes = {} # id --> category
    sketchycoco_classes = {} # id --> category
    sketchyscene_classes = {} # id --> category

    all_classes = {} # common category --> frequency

    num_obj_sketchycoco, num_obj_sketchyscene, num_obj_fscoco = [], [], []

    ## Get classes for coco
    register = open(os.path.join(opt.fscoco_dir, 'stuffthings_categories.txt'), 'r').readlines()
    for row in register:
        row = [item.strip() for item in row.split(':')]
        coco_classes[int(row[0])] = lemmatizer.lemmatize(row[1])

    ## Get classes for sketchycoco
    print ('Processing SketchyCOCO. Please wait ...')
    sketchycoco_2_coco = {1:2, 2:3, 3:4, 4:5, 5:10, 6:11, 7:17, 8:18, 9:19, 10:20,\
        11:21, 12:22, 14:24, 15:25, 16:106, 17:124, 18:106, 19:169}
    list_all_sketches = glob.glob(os.path.join(opt.sketchycoco_dir, 'Annotation', 'paper_version', '*', 'CLASS_GT', '*.mat'))
    for sketch_path in tqdm.tqdm(list_all_sketches):
        classes = np.unique(loadmat(sketch_path)['CLASS_GT'])
        num_obj_sketchycoco.append(len(classes))
        for key in classes:
            if key not in sketchycoco_2_coco.keys():
                continue
            key = sketchycoco_2_coco[key]
            key = coco_classes[key]
            if key in sketchycoco_classes.keys():
                sketchycoco_classes[key] += 1
            else:
                sketchycoco_classes[key] = 1

    ## Get classes for sketchyscene
    print ('Processing SketchyScene. Please wait ...')
    register = open(os.path.join(opt.sketchyscene_dir, 'colorMap_46.txt'), 'r').readlines()
    colorMap_sketchyscene = {}
    for idx, row in enumerate(register):
        row = row.split()[0].strip()
        colorMap_sketchyscene[idx+1] = row

    list_all_sketches = glob.glob(os.path.join(opt.sketchyscene_dir, '*', 'CLASS_GT', '*.mat'))
    for sketch_path in tqdm.tqdm(list_all_sketches):
        classes = np.unique(loadmat(sketch_path)['CLASS_GT'])
        num_obj_sketchyscene.append(len(classes))
        for key in classes:
            if key == 0:
                continue
            key = colorMap_sketchyscene[key]
            if key in sketchyscene_classes.keys():
                sketchyscene_classes[key] += 1
            else:
                sketchyscene_classes[key] = 1

    ## Get classes for fscoco
    print ('Processing FSCOCO. Please wait ...')
    categories = open(os.path.join(opt.fscoco_dir, 'stuffthings_categories.txt'), 'r').readlines()
    all_categories = []
    for cat in categories:
        cat = [item.strip() for item in cat.split(':')]
        all_categories.append(lemmatizer.lemmatize(cat[1]))

    all_text_files = glob.glob(os.path.join(opt.fscoco_dir, 'text', '*', '*.txt'))
    for txt_filepath in tqdm.tqdm(all_text_files[:]):
        txt_data = open(txt_filepath, 'r').read()
        txt_data = word_tokenize(txt_data)
        txt_data = [lemmatizer.lemmatize(word) for word in txt_data]
        count = 0
        for word in txt_data:
            if word in all_categories:
                if word in fscoco_classes.keys():
                    fscoco_classes[word] += 1
                    count += 1
                else:
                    fscoco_classes[word] = 1
        if count > 0:
            num_obj_fscoco.append(count)

    ## Stats
    print ('SketchyCOCO => Mean: {}, SD: {}, Max: {}, Min: {}'.format(
        np.mean(num_obj_sketchycoco), np.std(num_obj_sketchycoco),
        np.max(num_obj_sketchycoco), np.min(num_obj_sketchycoco)))

    print ('SketchyScene => Mean: {}, SD: {}, Max: {}, Min: {}'.format(
        np.mean(num_obj_sketchyscene), np.std(num_obj_sketchyscene),
        np.max(num_obj_sketchyscene), np.min(num_obj_sketchyscene)))

    print ('FSCOCO => Mean: {}, SD: {}, Max: {}, Min: {}'.format(
        np.mean(num_obj_fscoco), np.std(num_obj_fscoco),
        np.max(num_obj_fscoco), np.min(num_obj_fscoco)))

    ## Combine all categories from SketchyCOCO, SketchyScene, FSCOCO
    all_classes = list(
        set(list(sketchycoco_classes.keys())) |\
        set(list(sketchyscene_classes.keys())) |\
        set(list(fscoco_classes.keys())))

    new_sketchycoco_classes = {}
    new_sketchyscene_classes = {}
    new_fscoco_classes = {}
    for key in all_classes:
        if key in sketchycoco_classes.keys():
            new_sketchycoco_classes[key] = sketchycoco_classes[key]
        else:
            new_sketchycoco_classes[key] = 0
        
        if key in sketchyscene_classes.keys():
            new_sketchyscene_classes[key] = sketchyscene_classes[key]
        else:
            new_sketchyscene_classes[key] = 0
        
        if key in fscoco_classes.keys():
            new_fscoco_classes[key] = fscoco_classes[key]
        else:
            new_fscoco_classes[key] = 0

    ## Sort x-axis
    order = np.argsort([new_fscoco_classes[key] for key in new_fscoco_classes.keys()])[::-1]
    all_classes = [all_classes[idx] for idx in order]

    ## Plot bargraph
    plt.figure(figsize=(15, 3))
    for idx, (color, dict_data, legend) in enumerate([\
                ('r', new_sketchycoco_classes, 'sketchycoco'),
                ('g', new_sketchyscene_classes, 'sketchyscene'),
                ('b', new_fscoco_classes, 'fscoco')
            ]):
        
        values = [dict_data[key] for key in all_classes]

        g = sns.barplot(
            x=np.arange(len(values)),
            y = values,
            color=color)
    
    plt.xticks(np.arange(len(values)), all_classes, rotation='vertical')
    plt.savefig('rebuttal-original-scale.pdf', bbox_inches='tight')

    plt.clf()
          
    all_categories = []
    all_values = []
    all_dataset = []

    ## Generate Gamma corrected barplot
    plt.figure(figsize=(15, 3))

    for idx, (color, dict_data, data_name) in enumerate([\
                ('r', new_sketchycoco_classes, 'sketchycoco'),
                ('g', new_sketchyscene_classes, 'sketchyscene'),
                ('b', new_fscoco_classes, 'fscoco')
            ]):
        
        values = np.array([dict_data[key] for key in all_classes])

        all_categories.extend(all_classes)
        all_values.extend(values)
        all_dataset.extend([data_name]*len(all_classes))

        print ('Dataset: {}, Mean: {}, SD: {}, length: {}, max: {}, min: {}'.format(
            data_name,
            np.mean(values[values!=0]),
            np.std(values[values!=0]),
            len(values[values!=0]),
            max(values[values!=0]),
            min(values[values!=0])))

        g = sns.barplot(
            x=np.arange(len(values)),
            y = np.array(values)**0.1,
            color=color)
    
    plt.xticks(np.arange(len(values)), all_classes, rotation='vertical')
    plt.savefig('rebuttal-01-scale.pdf', bbox_inches='tight')

    plt.clf()
    ## Generate Gamma corrected barplot
    plt.figure(figsize=(15, 3))
    print (all_categories)
    data = {
        'categories': np.array(all_categories),
        'values': np.array(all_values),
        'dataset': np.array(all_dataset)
    }
    df = pd.DataFrame(data)
    sns.catplot(x='categories', y='values', hue='dataset', kind='bar', data=data)
    plt.savefig('rebuttal-01-scale-group.pdf', bbox_inches='tight')

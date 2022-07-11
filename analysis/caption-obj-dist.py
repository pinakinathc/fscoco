# -*- coding: utf-8 -*-
# Finds the object distribution in scene images

import os
import glob
import json
import tqdm
import argparse
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
sns.set_theme()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform analysis on collected sketches')
    parser.add_argument('--data_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/',
        help='path to directory with sketches')
    opt = parser.parse_args()

    ## Load categories
    categories = open(os.path.join(opt.data_dir, 'stuffthings_categories.txt'), 'r').readlines()
    all_categories = []
    for cat in categories:
        cat = [item.strip() for item in cat.split(':')]
        all_categories.append(lemmatizer.lemmatize(cat[1]))
    
    fscoco_categories = {}
    all_text_files = glob.glob(os.path.join(opt.data_dir, 'text', '*', '*.txt'))
    for txt_filepath in tqdm.tqdm(all_text_files[:]):
        txt_data = open(txt_filepath, 'r').read()
        txt_data = word_tokenize(txt_data)
        txt_data = [lemmatizer.lemmatize(word) for word in txt_data]
        for word in txt_data:
            if word in all_categories:
                if word in fscoco_categories.keys():
                    fscoco_categories[word] += 1
                else:
                    fscoco_categories[word] = 1

    print (fscoco_categories)

    data = list(fscoco_categories.values())
    keys = list(fscoco_categories.keys())

    # Reorder / reshuffle
    order = np.argsort(data)[::-1]
    N = int(len(order)*1.0)
    data = [data[i] for i in order[:N]]
    keys = [keys[i] for i in order[:N]]

    plt.figure(figsize=(15,3))
    g = sns.barplot(x=np.arange(len(data)), y=data)
    plt.xticks(np.arange(len(data)), keys, rotation='vertical')
    
    # plt.savefig('rebuttal-data-foreground.png', bbox_inches='tight')
    plt.savefig('rebuttal-data-check.png', bbox_inches='tight')


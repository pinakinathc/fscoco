# -*- coding: utf-8 -*-
import argparse
import json
import glob
import re
import os
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from nltk import word_tokenize, pos_tag, ngrams

""" Analysis Done on Captions:

- Average Length of Sketch captions v/s Image Captions

- Vocabulary Size of Sketch Captions v/s Image Captions corpus

- How much overlap is there between Sketch Captions and Image Captions?
    - Percentage of sketch caption works present in Image captions.
    - ROUGE-L metric using Sketch Caption as prediction and Image caption as GT. [Didn't do]

- Uniqueness of sketch captions and image captions [Didn't do]

- Probability distribution of various caption components:
    - Nouns, Verbs, Adjectives
    - Outcome: sketch captions has higher probability of verbs which means they
        focus more on action an object is doing instead on its attributes.
    - 
"""

def probability_dist_fn(tokens, all_count=0):
    pos = pos_tag(tokens, tagset='universal')
    all_count = len(pos)
    noun_prob = len([elem[0] for elem in pos if elem[1] == 'NOUN'])/all_count
    verb_prob = len([elem[0] for elem in pos if elem[1] == 'VERB'])/all_count
    adj_prob = len([elem[0] for elem in pos if elem[1] == 'ADJ'])/all_count

    return noun_prob, verb_prob, adj_prob, all_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform analysis on collected captions.')
    parser.add_argument('--coco', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/coco.json',
        help='path to directory with train and test captions in json format')
    parser.add_argument('--text_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/text',
        help='path to directory with sketch captions')
    opt = parser.parse_args()

    # get coco dataset json
    coco_data = json.load(open(opt.coco))
    # coco_data = [y['tokens'] for x in coco_data['images'] for y in x['sentences']]
    # coco_data = [y['tokens'] for x in coco_data['images'][:20000] for y in x['sentences']]
    # coco_data = [y['tokens'] for x in coco_data['images'][:] for y in x['sentences']]
    coco_data = [word_tokenize(re.sub('[^0-9a-z]+', ' ', y['raw'])) for x in coco_data['images'][:] for y in x['sentences']]
    
    # get our sketch captions
    our_data = []
    for text_file in glob.glob(os.path.join(opt.text_dir, '*', '*.txt')):
        # our_data.append(
        #     re.sub('[^0-9a-z]+', ' ',  open(text_file).read().lower()).strip().split())
        our_data.append(
            word_tokenize(re.sub('[^0-9a-z]+', ' ',  open(text_file).read().lower())))
        # our_data.append(open(text_file).read().split())

    # average length
    avg_coco_len = sum([len(sentence) for sentence in coco_data])/len(coco_data)
    avg_our_len = sum([len(sentence) for sentence in our_data])/len(our_data)
    print ('Average COCO caption length: {} | Our caption length: {}'.format(
        avg_coco_len, avg_our_len))

    # Vocabulary size
    merged_our_data = np.concatenate(np.array(our_data), axis=None)
    print ('Sketch caption vocabulary size: ', np.unique(merged_our_data).shape)
    merged_coco_data = np.concatenate(np.array(coco_data), axis=None)
    print ('COCO caption vocabulary size: ', np.unique(merged_coco_data).shape)

    # Overlap between sketch and image captions   
    overlap = list((Counter(merged_our_data) & Counter(merged_coco_data)).elements())

    percentage = len(overlap) / len(merged_our_data)
    print ('Percentage of overlap: {}/{} = {}'.format(
        len(overlap), len(merged_our_data), percentage))

    # print (overlap)
    # WordCloud for Sketch Caption, Image Caption, Overalap
    WordCloud(background_color='white', max_font_size=40, scale=10).generate(' '.join(np.unique(merged_our_data))).to_image().save('sketch-caption-wc.png')
    WordCloud(background_color='white', max_font_size=40, scale=10).generate(' '.join(np.unique(merged_coco_data))).to_image().save('image-caption-wc.png')
    WordCloud(background_color='white', max_font_size=40, scale=10).generate(' '.join(np.unique(overlap))).to_image().save('overlap-caption-wc.png')

    # Probability distribution of various caption components
    noun_prob, verb_prob, adj_prob, _ = probability_dist_fn(overlap)
    print ('Probability of overlapping Nouns: {}, Verbs: {}, Adjectives: {}'.format(
        noun_prob, verb_prob, adj_prob))

    noun_prob, verb_prob, adj_prob, _ = probability_dist_fn(merged_our_data)
    print ('Probability of Our Data Nouns: {}, Verbs: {}, Adjectives: {}'.format(
        noun_prob, verb_prob, adj_prob))

    noun_prob, verb_prob, adj_prob, _ = probability_dist_fn(merged_coco_data)
    print ('Probability of COCO data Nouns: {}, Verbs: {}, Adjectives: {}'.format(
        noun_prob, verb_prob, adj_prob))

import os
import json
from collections import Counter

if __name__ == '__main__':
    max_len = 50
    min_word_freq = 5

    data = json.load(open('dataset_coco.json'))
    word_freq = Counter()

    for img in data['images']:
        for c in img['sentences']:
            word_freq.update(c['tokens'])

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Save word map to a JSON
    with open(os.path.join('word_map.json'), 'w') as j:
        json.dump(word_map, j)
    
    print (word_map)
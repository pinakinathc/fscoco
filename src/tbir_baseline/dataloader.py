# -*- coding: utf-8 -*-

import os
import glob
import json
from collections import Counter
from nltk import word_tokenize
import numpy as np
from PIL import Image, ImageOps
import torch

class OursScene(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train',
            transform=None, return_orig=False, max_len=50, use_coco=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig
        self.max_len = max_len
        self.use_coco = use_coco
        self.mode = mode

        if mode == 'train':
            self.all_image_files = glob.glob('/vol/research/sketchcaption/datasets/COCO-stuff/images/train2017/*.jpg')
        else:
            self.all_image_files = glob.glob('/vol/research/sketchcaption/datasets/COCO-stuff/images/val2017/*.jpg')

        self.all_ids = [os.path.split(x)[-1][:-4] for x in self.all_image_files]

        word_corpus = json.load(open(os.path.join(self.opt.root_dir, 'coco.json')))

        self.coco_anns = {}
        for ann in word_corpus['images']:
            self.coco_anns[ann['cocoid']] = ann['sentences'][0]['raw']

        word_map = [tokens['tokens'] for sentences in word_corpus['images']
                                        for tokens in sentences['sentences']]
        all_tokens = []
        for tokens in word_map:
            all_tokens.extend(tokens)
        token_freq = Counter(all_tokens)
        self.word_map = {'<start>': 0, '<end>': 1, '<unk>': 2, '<pad>': 3}
        count = 4
        for (word, freq) in token_freq.items():
            if freq >= 5:
                self.word_map[word] = count
                count += 1

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]

        local_state = np.random.RandomState()

        if self.mode == 'train':
            image_file = os.path.join('/vol/research/sketchcaption/datasets/COCO-stuff/images/train2017/', '%s.jpg'%filename)
            negative_shirtname  = local_state.choice(self.all_ids, 1)[0]
            negative_file = os.path.join('/vol/research/sketchcaption/datasets/COCO-stuff/images/train2017/', '%s.jpg'%negative_shirtname)
        else:
            image_file = os.path.join('/vol/research/sketchcaption/datasets/COCO-stuff/images/val2017/', '%s.jpg'%filename)
            negative_shirtname  = local_state.choice(self.all_ids, 1)[0]
            negative_file = os.path.join('/vol/research/sketchcaption/datasets/COCO-stuff/images/val2017/', '%s.jpg'%negative_shirtname)

        # captions from our dataset
        text_data = word_tokenize(self.coco_anns[int(filename)].lower())[:self.max_len]

        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        # encode caption
        word_encode = [self.word_map['<start>']]
        word_encode += [self.word_map.get(token, self.word_map['<unk>']) for token in text_data]
        word_encode += [self.word_map['<end>']]

        word_length = len(word_encode)

        # pad the rest of encoded caption
        word_encode += [self.word_map['<pad>']]*(max(0, self.max_len - len(word_encode)))

        if self.transform:
            txt_tensor = torch.tensor(word_encode)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return txt_tensor, word_length, img_tensor, neg_tensor, text_data, word_encode, image_data, negative_data
        else:
            return txt_tensor, word_length, img_tensor, neg_tensor


if __name__ == '__main__':
    from options import opts
    from torchvision import transforms

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = OursScene(opts, mode='val',
        transform=dataset_transforms, return_orig=True, use_coco=True)

    for idx, (txt_tensor, txt_length, img_tensor, neg_tensor,
        text_data, text_emb, image_data, negative_data) in enumerate(dataset):

        with open(os.path.join(output_dir, '%d_text.txt'%idx), 'w') as fp:
            fp.write(' '.join(text_data))
        with open(os.path.join(output_dir, '%d_emb.txt'%idx), 'w') as fp:
            fp.write(' '.join(list(map(str, text_emb))))
        image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
        negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

        print('Shape of txt_tensor: {} | txt_length: {} | img_tensor: {} | neg_tensor: {}'.format(
            txt_tensor.shape, txt_length, img_tensor.shape, neg_tensor.shape))

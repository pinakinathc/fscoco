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

        self.all_image_files = glob.glob(os.path.join(
            self.opt.root_dir, 'images', '*', '*.jpg'))

        self.all_ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                self.opt.root_dir, 'text', '*', '*.txt'))])

        split = int(len(self.all_ids) * 0.7)

        if mode == 'train':
            self.all_ids = self.all_ids[:split]
        else:
            self.all_ids = self.all_ids[split:]

        word_corpus = json.load(open(os.path.join(self.opt.root_dir, 'coco.json')))

        if self.use_coco:
            print ('using COCO captions for %s set'%mode)
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

        text_file = glob.glob(os.path.join(self.opt.root_dir, 'text', '*', '%s.txt'%filename))[0]
        sketch_file = glob.glob(os.path.join(self.opt.root_dir, 'raster_sketches', '*', '%s.jpg'%filename))[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '%s.jpg'%filename))[0]
        negative_file = np.random.choice(self.all_image_files, 1)[0]

        assert os.path.split(text_file)[-1][:-4] == os.path.split(image_file)[-1][:-4], ValueError('file mismatch')
        assert os.path.split(sketch_file)[-1] == os.path.split(image_file)[-1], ValueError('file mismatch')

        # captions from our dataset
        if self.use_coco:
            text_data = word_tokenize(self.coco_anns[int(filename)].lower())[:self.max_len]
        else:
            text_data = word_tokenize(open(text_file, 'r+', encoding='utf-8').read().lower())[:self.max_len]

        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
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
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return txt_tensor, word_length, sk_tensor, img_tensor, neg_tensor, text_data, word_encode, sketch_data, image_data, negative_data
        else:
            return txt_tensor, word_length, sk_tensor, img_tensor, neg_tensor


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

    dataset = OursScene(opts, mode='train',
        transform=dataset_transforms, return_orig=True, use_coco=True)

    for idx, (txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor,
        text_data, text_emb, sketch_data, image_data, negative_data) in enumerate(dataset):

        with open(os.path.join(output_dir, '%d_text.txt'%idx), 'w') as fp:
            fp.write(' '.join(text_data))
        with open(os.path.join(output_dir, '%d_emb.txt'%idx), 'w') as fp:
            fp.write(' '.join(list(map(str, text_emb))))
        
        sketch_data.save(os.path.join(output_dir, '%d_sk.jpg'%idx))
        image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
        negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

        print('Shape of txt_tensor: {} | txt_length: {} | \
            sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
            txt_tensor.shape, txt_length, sk_tensor.shape, img_tensor.shape, neg_tensor.shape))

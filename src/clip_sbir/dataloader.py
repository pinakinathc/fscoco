# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat
import clip as clip
import torch

class OursScene(torch.utils.data.Dataset):

    def __init__(self, opt, img_preprocess, mode='train', return_orig=False, use_coco=False):
        self.opt = opt
        self.img_preprocess = img_preprocess
        self.return_orig = return_orig
        self.use_coco = use_coco

        self.all_image_files = glob.glob(os.path.join(
            self.opt.root_dir, 'images', '*', '*.jpg'))

        self.all_ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                self.opt.root_dir, 'raster_sketches', '*', '*.jpg'))])

        val_ids = np.loadtxt(os.path.join(self.opt.root_dir, 'val_normal.txt'), dtype=str)

        if mode == 'train':
            self.all_ids = list(set(self.all_ids) - set(val_ids))
        else:
            self.all_ids = np.random.choice(val_ids, 210)
        print ('total ids: ', len(self.all_ids))


        if self.use_coco:
            word_corpus = json.load(open(os.path.join(self.opt.root_dir, 'coco.json')))
            print ('using COCO captions for %s set'%mode)
            self.coco_anns = {}
            for ann in word_corpus['images']:
                self.coco_anns[ann['cocoid']] = np.random.choice(ann['sentences'], 1)[0]['raw']

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]

        text_file = glob.glob(os.path.join(self.opt.root_dir, 'text', '*', '%s.txt'%filename))[0]
        sketch_file = glob.glob(os.path.join(self.opt.root_dir, 'raster_sketches', '*', '%s.jpg'%filename))[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '%s.jpg'%filename))[0]
        negative_file = np.random.choice(self.all_image_files, 1)[0]

        # captions from our dataset
        if self.use_coco:
            text_data = self.coco_anns[int(filename)]
        else:
            text_data = open(text_file, 'r+', encoding='utf-8').read()

        txt_tensor = clip.tokenize(text_data)[0]
        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sk_tensor = self.img_preprocess(sketch_data)
        img_tensor = self.img_preprocess(image_data)
        neg_tensor = self.img_preprocess(negative_data)


        if self.return_orig:
            return txt_tensor, sk_tensor, img_tensor, neg_tensor, text_data, sketch_data, image_data, negative_data
        else:
            return txt_tensor, sk_tensor, img_tensor, neg_tensor


class SketchyScene(torch.utils.data.Dataset):

    def __init__(self, opt, img_preprocess, mode='train', transform=None, return_orig=False):

        self.opt = opt
        self.img_preprocess = img_preprocess
        self.transform = transform
        self.return_orig = return_orig


        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'

        self.all_ids = [os.path.split(filepath)[-1][:-4] for filepath in glob.glob(
                os.path.join(self.opt.root_dir, self.mode, 'reference_image', '*.jpg'))]
        print ('Found %d samples in %s'%(len(self.all_ids), self.mode))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]
        sketch_file = os.path.join(self.opt.root_dir, self.mode, 'DRAWING_GT', 'L0_sample%s.png'%filename)
        image_file = os.path.join(self.opt.root_dir, self.mode, 'reference_image', '%s.jpg'%filename)
        negative_file = os.path.join(self.opt.root_dir, self.mode, 'reference_image', '%s.jpg'%np.random.choice(self.all_ids, 1)[0])

        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sk_tensor = self.img_preprocess(sketch_data)
        img_tensor = self.img_preprocess(image_data)
        neg_tensor = self.img_preprocess(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


class SketchyCOCO(torch.utils.data.Dataset):

    def __init__(self, opt, img_preprocess, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.img_preprocess = img_preprocess
        self.transform = transform
        self.return_orig = return_orig

        if mode == 'train':
            self.mode = 'trainInTrain'
        else:
            self.mode = 'val'

        self.all_ids = glob.glob(os.path.join(
            self.opt.root_dir, 'Annotation', 'paper_version', self.mode, 'CLASS_GT', '*.mat'))

        self.all_ids = [os.path.split(filepath)[-1][:-4] 
            for filepath in self.all_ids if (np.unique(loadmat(filepath)['CLASS_GT']) <16).sum() >= 1]
        print ('total %s samples: %d'%(self.mode, len(self.all_ids)))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]
        sketch_file = os.path.join(self.opt.root_dir, 'Sketch', 'paper_version', self.mode, '%s.png'%filename)
        image_file = os.path.join(self.opt.root_dir, 'GT', self.mode, '%s.png'%filename)
        negative_file = os.path.join(self.opt.root_dir, 'GT', self.mode, '%s.png'%np.random.choice(self.all_ids, 1)[0])

        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sk_tensor = self.img_preprocess(sketch_data)
        img_tensor = self.img_preprocess(image_data)
        neg_tensor = self.img_preprocess(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


class COCO(torch.utils.data.Dataset):

    def __init__(self, opt, img_preprocess, mode='train', return_orig=False):
        self.opt = opt
        self.img_preprocess = img_preprocess
        self.return_orig = return_orig

        if mode == 'train':
            self.mode = 'train2017'
            all_captions = json.load(open(os.path.join(self.opt.root_dir, 'annotations', 'captions_train2017.json')))
        else:
            self.mode = 'val2017'
            all_captions = json.load(open(os.path.join(self.opt.root_dir, 'annotations', 'captions_val2017.json')))

        self.all_ids = glob.glob(os.path.join(
            self.opt.root_dir, 'images', self.mode, '*.jpg'))

        self.all_ids = [os.path.split(filepath)[-1][:-4] for filepath in self.all_ids]
        print ('total %s samples: %d'%(self.mode, len(self.all_ids)))

        self.coco_anns = {}
        for coco_id in self.all_ids:
            self.coco_anns[int(coco_id)] = np.random.choice(
                [item for item in all_captions['annotations'] if item['image_id']==int(coco_id)], 1)[0]['caption']

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]

        image_file = os.path.join(self.opt.root_dir, 'images', self.mode, '%s.jpg'%filename)
        negative_file = os.path.join(
            self.opt.root_dir, 'images', self.mode, '%s.jpg'%np.random.choice(self.all_ids, 1)[0])

        # captions from our dataset
        text_data = self.coco_anns[int(filename)]

        txt_tensor = clip.tokenize(text_data)[0]
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        img_tensor = self.img_preprocess(image_data)
        neg_tensor = self.img_preprocess(negative_data)


        if self.return_orig:
            return txt_tensor, img_tensor, neg_tensor, text_data, image_data, negative_data
        else:
            return txt_tensor, img_tensor, neg_tensor


if __name__ == '__main__':
    from options import opts
    from torchvision import transforms

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    _, img_preprocess = clip.load(opts.pretrained_clip, jit=False)

    # dataset = OursScene(opts, img_preprocess, mode='train', return_orig=True, use_coco=True)
    # dataset = SketchyCOCO(opts, img_preprocess, mode='val', return_orig=True)
    # dataset = Sketchy(opts, img_preprocess, mode='val', return_orig=True)
    dataset = COCO(opts, img_preprocess, mode='val', return_orig=True)

    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3)

    # for idx, (txt_tensor, sk_tensor, img_tensor, neg_tensor, \
    #     text_data, sketch_data, image_data, negative_data) in enumerate(dataset):

    # for idx, (sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data) in enumerate(dataset):
    
    for idx, (txt_tensor, img_tensor, neg_tensor, text_data, image_data, negative_data) in enumerate(dataset):

        with open(os.path.join(output_dir, '%d_text.txt'%idx), 'w') as fp:
            fp.write(text_data)

        image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
        negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

        print('Shape of txt_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
            txt_tensor.shape, img_tensor.shape, neg_tensor.shape))

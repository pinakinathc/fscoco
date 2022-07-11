# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageOps
import torch

class OursScene(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig

        self.all_image_files = glob.glob(os.path.join(
            self.opt.root_dir, 'images', '*', '*.jpg'))

        self.all_ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                self.opt.root_dir, 'raster_sketches', '*', '*.jpg'))])

        # # TODO remove this next line
        # self.all_ids = np.loadtxt(os.path.join(self.opt.root_dir, 'category_5000_photos.txt'), dtype=str)

        val_ids = np.loadtxt(os.path.join(self.opt.root_dir, 'val_normal.txt'), dtype=str)
        # val_ids = np.loadtxt(os.path.join(self.opt.root_dir, 'val_unseen_user.txt'), dtype=str)

        if mode == 'train':
            self.all_ids = list(set(self.all_ids) - set(val_ids))
            # self.all_ids = np.random.choice(self.all_ids, 5000)
        else:
            # self.all_ids = np.random.choice(val_ids, 210)
            self.all_ids = val_ids
        print ('total ids: ', len(self.all_ids))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        filename = self.all_ids[index]

        sketch_file = glob.glob(os.path.join(self.opt.root_dir, 'raster_sketches', '*', '%s.jpg'%filename))[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '%s.jpg'%filename))[0]
        negative_file = np.random.choice(self.all_image_files, 1)[0]

        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


class SketchyScene(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None, return_orig=False):

        self.opt = opt
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

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


class SketchyCOCO(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
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

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig

        self.category = None
        self.all_categories = os.listdir(os.path.join(self.opt.root_dir, 'sketch', 'tx_000000000000'))

        self.all_ids = {}
        for category in self.all_categories:
            if mode == 'train':
                self.mode = 'train'
                self.all_ids[category] = sorted([os.path.split(item)[-1][:-4] for item in glob.glob(os.path.join(
                    self.opt.root_dir, 'photo', '*', category, '*.jpg')) ])[:180]
            else:
                self.mode = 'val'
                self.all_ids[category] = sorted([os.path.split(item)[-1][:-4] for item in glob.glob(os.path.join(
                    self.opt.root_dir, 'photo', '*', category, '*.jpg')) ])[180:]

    def __len__(self):
        if self.mode == 'train':
            return 180
        else:
            return 20

    def __getitem__(self, index):
        if self.category is None:
            category = np.random.choice(self.all_categories, 1)[0]
        else:
            category = self.category

        filename = self.all_ids[category][index]
        sketch_file = np.random.choice(glob.glob(
            os.path.join(self.opt.root_dir, 'sketch', '*', category, filename+'*.png')), 1)[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'photo', '*', category, filename+'*.jpg'))[0]
        negative_file = glob.glob(os.path.join(self.opt.root_dir, 'photo', '*', \
            category, np.random.choice(self.all_ids[category], 1)[0] +'*.jpg'))[0]

        sketch_data = Image.open(sketch_file).convert('RGB')
        image_data = Image.open(image_file).convert('RGB')
        negative_data = Image.open(negative_file).convert('RGB')

        sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
        negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            sk_tensor = self.transform(sketch_data)
            img_tensor = self.transform(image_data)
            neg_tensor = self.transform(negative_data)

        if self.return_orig:
            return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
        else:
            return sk_tensor, img_tensor, neg_tensor


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

    dataset = SketchyCOCO(opts, mode='train',
        transform=dataset_transforms, return_orig=True)

    # dataset = Sketchy(opts, mode='val', transform=dataset_transforms, return_orig=True)

    # for category in dataset.all_categories:
    # dataset.category = category
    category = 'sketchycoco'
    for idx, (sk_tensor, img_tensor, neg_tensor,
            sketch_data, image_data, negative_data) in enumerate(dataset):

        sketch_data.save(os.path.join(output_dir, '%s_%d_sk.jpg'%(category, idx)))
        image_data.save(os.path.join(output_dir, '%s_%d_img.jpg'%(category, idx)))
        negative_data.save(os.path.join(output_dir, '%s_%d_neg.jpg'%(category, idx)))

        print('Shape of sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
            sk_tensor.shape, img_tensor.shape, neg_tensor.shape))

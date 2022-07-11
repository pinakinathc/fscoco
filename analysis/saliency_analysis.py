# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
from bresenham import bresenham
import cv2
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from src.sbir_baseline.model import TripletNetwork

def drawPNG (vector_images, side=256, time_frac=None, skip_front=False):
    raster_image = np.ones((side, side), dtype=np.uint8);
    prevX, prevY = None, None;
    begin_time = vector_images[0]['timestamp']
    start_time = vector_images[0]['timestamp']
    end_time = vector_images[-1]['timestamp']

    if time_frac:
        if skip_front:
            start_time = (end_time - start_time)*time_frac
        else:
            end_time -= (end_time - start_time)*time_frac

    for points in vector_images:
        time = points['timestamp'] - begin_time
        if not (time >= start_time and time <= end_time):
            continue

        x, y = map(float, points['coordinates'])
        x = int(x * side); y = int(y * side)
        pen_state = list(map(int, points['pen_state']))
        if not (prevX is None or prevY is None):
            cordList = list(bresenham(prevX, prevY, x, y))
            for cord in cordList:
                    if (cord[0] > 0 and  cord[1] > 0) and (cord[0] < side and  cord[1] < side):
                        raster_image[cord[1], cord[0]] = 0
            if pen_state == [0, 1, 0]:
                    prevX = x; prevY = y
            elif pen_state == [1, 0, 0]:
                    prevX = None; prevY = None;
            else:
                raise ValueError('pen_state not accounted for')            
        else:
            prevX = x; prevY = y;
    # invert black and white pixels and dialate
    raster_image = (1 - cv2.dilate(1-raster_image, np.ones((3,3),np.uint8), iterations=1))*255
    return raster_image


class OursScene(torch.utils.data.Dataset):

    def __init__(self, opt, transform=None, return_orig=False, mask=None, skip_front=None):
        self.opt = opt
        self.return_orig = return_orig
        self.mask = mask
        self.skip_front = skip_front
        self.transform = transform
        print('Init: ', self.mask, self.skip_front)

        self.all_image_files = glob.glob(os.path.join(
            self.opt.root_dir, 'images', '*', '*.jpg'))

        self.all_ids = sorted([
            os.path.split(idx)[-1][:-4] for idx in self.all_image_files])

        split = int(len(self.all_ids) * 0.7)
        self.all_ids = self.all_ids[split:]

    def __len__(self):
        return len(self.all_ids)

    def update(self, mask, skip_front):
        self.mask = mask
        self.skip_front = skip_front

    def __getitem__(self, index):
        filename = self.all_ids[index]

        sketch_file = glob.glob(os.path.join(self.opt.root_dir, 'raw_data', '*', '%s.json'%filename))[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '%s.jpg'%filename))[0]
        negative_file = np.random.choice(self.all_image_files, 1)[0]

        sketch_data = json.load(open(sketch_file))

        # Partial Sketch
        sketch_data = drawPNG(sketch_data, time_frac=self.mask, skip_front=self.skip_front)
        sketch_data = Image.fromarray(sketch_data).convert('RGB')

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
            return 1, 1, sk_tensor, img_tensor, neg_tensor, 1, 1, sketch_data, image_data, negative_data
        else:
            return 1, 1, sk_tensor, img_tensor, neg_tensor


if __name__ == '__main__':
    from src.sbir_baseline.options import opts
    from torchvision import transforms

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = TripletNetwork().load_from_checkpoint('/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/lightning_logs/version_0/checkpoints/epoch=413-step=181331.ckpt')
    trainer = Trainer(gpus=1)

    dataset = OursScene(opts,
        transform=dataset_transforms, return_orig=False, mask=None, skip_front=None)


    for skip_front in [False, True]:
        for mask in np.arange(0, 1.0, 0.1):
            print ('Evaluation for mask: {} skip_front: {}'.format(mask, skip_front))
            dataset.update(mask, skip_front)
            val_loader = DataLoader(
                dataset=dataset, batch_size=opts.batch_size, num_workers=opts.workers)
            trainer.validate(model, val_loader)


    # for idx, (txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor,
    #     text_data, text_emb, sketch_data, image_data, negative_data) in enumerate(dataset):
        
    #     sketch_data.save(os.path.join(output_dir, '%d_sk.jpg'%idx))
    #     image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
    #     negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

    #     print('sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
    #         sk_tensor.shape, img_tensor.shape, neg_tensor.shape))

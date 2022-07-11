import os
import glob
import json
from collections import Counter
import numpy as np
from PIL import Image, ImageOps
import torch


def convert_to_five_point(vector_sketch):
    """ Convert 3-point sketch to 5-point """
    N, _ = vector_sketch.shape
    new_vector = np.zeros((N, 5))
    new_vector[:, :2] = vector_sketch[:, :2]
    split_idx = np.where(vector_sketch[:, 2]==1)[0]

    new_vector[:, 2] = 1
    new_vector[(split_idx, 2)] = 0
    new_vector[(split_idx, 3)] = 1
    new_vector[0, 2:] = [1, 0, 0]
    new_vector[-1, 2:] = [0, 0, 1]
    return new_vector

class OursScene(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None):
        self.opt = opt
        self.transform = transform

        self.all_image_files = glob.glob(os.path.join(
            self.opt.root_dir, 'images', '*', '*.jpg'))

        self.all_ids = sorted([os.path.split(imagepath)[-1][:-4]
            for imagepath in self.all_image_files])

        val_ids = np.loadtxt(os.path.join(self.opt.root_dir, 'val_normal.txt'), dtype=str)

        # evaluate
        # train_ids = np.loadtxt(os.path.join('output', 'train_ids.txt'), dtype=str)
        # val_ids = np.loadtxt(os.path.join('output', 'val_ids.txt'), dtype=str)
        # self.all_ids = list(train_ids) + list(val_ids)
        # np.random.shuffle(self.all_ids)
        # np.random.shuffle(val_ids)
        # self.all_ids = self.all_ids[:100]
        # val_ids = val_ids[:100]
        # with open ('output_small_sketch/train_ids.txt', 'w') as fp:
        #     fp.write('\n'.join(self.all_ids))
        # with open ('output_small/val_ids.txt', 'w') as fp:
        #     fp.write('\n'.join(val_ids))

        if mode == 'train':
            self.all_ids = list(set(self.all_ids) - set(val_ids))
        else:
            # self.all_ids = val_ids
            self.all_ids = self.all_ids

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        file_id = self.all_ids[index]

        raster_sketch = glob.glob(os.path.join(self.opt.root_dir, 'raster_sketches', '*', '%s.jpg'%file_id))[0]
        vector_sketch = glob.glob(os.path.join(self.opt.root_dir, 'vector_sketches', '*', '%s.npy'%file_id))[0]
        image_file = glob.glob(os.path.join(self.opt.root_dir, 'images', '*', '%s.jpg'%file_id))[0]

        raster_sketch = Image.open(raster_sketch).convert('RGB')
        vector_sketch = np.load(vector_sketch).astype(np.float32) # shape: (nbatch x 3)
        image_data = Image.open(image_file).convert('RGB')

        raster_sketch = ImageOps.pad(raster_sketch, size=(self.opt.max_len, self.opt.max_len))
        image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))

        if self.transform:
            raster_sk_tensor = self.transform(raster_sketch)
            img_tensor = self.transform(image_data)
        else:
            raise NotImplementedError

        vector_sketch = convert_to_five_point(vector_sketch)
        vector_sketch, list_stroke_len = self.process_vector_sketch(vector_sketch)
        
        return raster_sk_tensor, img_tensor, vector_sketch, list_stroke_len

    def process_vector_sketch(self, vector_sketch):
        """ split 5 point vector sketch into strokes """
        split_idx = np.where(vector_sketch[:, 3]==1)[0] + 1
        vector_sketch = np.split(vector_sketch, split_idx)
        stroke_len = [len(stroke) for stroke in vector_sketch]
        required_len = sorted(stroke_len, reverse=True)[:150]

        new_vector = []
        new_stroke_len = []
        for idx, s_len in enumerate(stroke_len):
            # if s_len > 1:
            if s_len in required_len:
                new_vector.append(vector_sketch[idx])
                new_stroke_len.append(s_len)
        return new_vector, new_stroke_len


if __name__ == '__main__':
    from options import opts
    from torchvision import transforms
    from utils import draw_tensor_sketch, collate_fn

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = OursScene(opts, mode='train', transform=dataset_transforms)   

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=3, collate_fn=collate_fn)

    for batch_id, (raster_sketch, image, vector_sketch, batch_num_strokes, batch_stroke_len)  in enumerate(dataloader):
        # print ('shape of raster_sketch: {}, image: {}, vector_sketch: {},\
        #     batch_num_strokes: {}, batch_stroke_len: {}'.format(
        #     raster_sketch.shape, image.shape, vector_sketch.shape, batch_num_strokes, batch_stroke_len))

        draw_tensor_sketch(image, raster_sketch, vector_sketch, batch_num_strokes, batch_stroke_len, batch_idx=batch_id)

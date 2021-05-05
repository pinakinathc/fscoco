# -*- coding: utf-8 -*-
# author: pinakinathc

import os, glob
from PIL import Image
import torch
from torchvision import transforms

class SketchyCOCO_Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.mode = mode
        if self.mode == 'train':
            self.img_dir = os.path.join(opt.sketchyCOCO, 'Scene', 'GT', 'trainInTrain')
            self.sk_dir = os.path.join(opt.sketchyCOCO, 'Scene', 'Sketch', 'paper_version', 'trainInTrain')
        elif self.mode == 'val':
            self.img_dir = os.path.join(opt.sketchyCOCO, 'Scene', 'GT', 'valInTrain')
            self.sk_dir = os.path.join(opt.sketchyCOCO, 'Scene', 'Sketch', 'paper_version', 'valInTrain')
        elif self.mode == 'test':
            self.img_dir = os.path.join(opt.sketchyCOCO, 'Scene', 'GT', 'val')
            self.sk_dir = os.path.join(opt.sketchyCOCO, 'Scene', 'Sketch', 'paper_version', 'val')
        else:
            raise ValueError('use either train/test as mode value')
        self.img_ids = list()
        self.transform = transform
        self.return_orig = return_orig

        self.img_ids = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.img_ids = list(map(int, [os.path.split(path)[-1][:-4] for path in self.img_ids]))
        self.length = len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        img = Image.open(os.path.join(self.img_dir, '%012d'%img_id+'.png')).convert('RGB')
        sk = Image.open(os.path.join(self.sk_dir, '%012d'%img_id+'.png')).convert('RGB')

        if self.transform:
            img_tensor = self.transform(img)
            sk_tensor = self.transform(sk)

        if self.return_orig:
            return img_tensor, sk_tensor, img, sk
        return img_tensor, sk_tensor

    def __len__(self):
        return self.length

    
def get_dataloaders(opt):
    dataset_transforms = transforms.Compose([
        transforms.RandomResizedCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SketchyCOCO_Dataset(opt, mode='train', transform=dataset_transforms)
    val_dataset = SketchyCOCO_Dataset(opt, mode='test', transform=dataset_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=opt.workers,
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        batch_size=opt.batch_size_eval,
        shuffle=False,
        pin_memory=True,
        num_workers=opt.workers,
        drop_last=True)

    return train_loader, val_loader


if __name__== '__main__': # Test get_dataloaders()
    from options import parser

    opt = parser.parse_args() # get parameters
    train_loader, val_loader = get_dataloaders(opt)

    for img, sk in train_loader:
        print ('Shape of img: {}, sk: {}'.format(img.shape, sk.shape))

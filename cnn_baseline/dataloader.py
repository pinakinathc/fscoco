import os
import glob
import numpy as np
from PIL import Image, ImageOps
import torch

class OursScene(torch.utils.data.Dataset):

    def __init__(self, opt, mode='train', transform=None, return_orig=False):
        self.opt = opt
        self.transform = transform
        self.return_orig = return_orig

        self.all_sketch_files = glob.glob(os.path.join(
            self.opt.root_dir, 'sketches', '*', '*.jpg'))
        self.all_image_files = glob.glob(os.path.join(
            self.opt.root_dir, 'images', '*', '*.jpg'))

        split = int(len(self.all_sketch_files) * 0.7)
        if mode == 'train':
            self.all_sketch_files = self.all_sketch_files[:split]
            self.all_image_files = self.all_image_files[:split]
        else:
            self.all_sketch_files = self.all_sketch_files[split:]
            self.all_image_files = self.all_image_files[split:]

    def __len__(self):
        return len(self.all_sketch_files)

    def __getitem__(self, index):
        sketch_file = self.all_sketch_files[index]
        image_file = self.all_image_files[index]
        negative_file = np.random.choice(self.all_image_files, 1)[0]

        assert os.path.split(sketch_file)[-1] == os.path.split(image_file)[-1], ValueError('file mismatch')

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

    dataset = OursScene(opts, mode='train', transform=dataset_transforms, return_orig=True)
    for idx, (sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data) in enumerate(dataset):
        sketch_data.save(os.path.join(output_dir, '%d_sk.jpg'%idx))
        image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
        negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

        print ('shape of sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
            sk_tensor.shape, img_tensor.shape, neg_tensor.shape))

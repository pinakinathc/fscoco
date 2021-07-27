''' This dataloader should work for both SketchyScene and SketchyCOCO.
First, we need to filter these noisy datasets following instructions mentioned in each paper.

-------------------------------------------------
Some Quantitative values claimed in their paper.
-------------------------------------------------
* SketchyScene -- 32.13% Acc@1 | 69.48% Acc@10

* SketchyCOCO -- 31.91% Acc@1 | 86.19% Acc@10

None of these papers have released their code or the sequences of images for which they got this result.

'''

import os
import glob
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageOps
import torch

class SketchyScene(torch.utils.data.Dataset):
	def __init__(self, opt, mode='train', transform=None, return_orig=False):
		self.opt = opt
		self.transform = transform
		self.return_orig = return_orig

		self.sketch_dir = os.path.join(self.opt.root_dir, mode, 'INSTANCE_GT')
		self.image_dir = os.path.join(self.opt.root_dir, mode, 'reference_image')

		self.list_ids = self.filter(self.sketch_dir, self.image_dir)

	def filter(self, sketch_dir, image_dir):
		''' Images and Sketches have some inconsistency, hence filtering is required -- although heuristic '''
		list_sk_ids = [int(os.path.split(x)[-1].split('_')[1]) for x in glob.glob(os.path.join(sketch_dir, '*.mat'))]
		list_img_ids = [int(os.path.split(x)[-1][:-4]) for x in glob.glob(os.path.join(image_dir, '*.jpg'))]

		return [x for x in list_sk_ids if x in list_img_ids] # intersection

	def __len__(self):
		return len(self.list_ids)

	def __getitem__(self, index):
		index = self.list_ids[index]
		sketch_data = loadmat(os.path.join(self.sketch_dir, 'sample_%d_instance.mat'%index))['INSTANCE_GT']
		image_data = Image.open(os.path.join(self.image_dir, '%d.jpg'%index))
		negative_data = Image.open(os.path.join(self.image_dir, '%d.jpg'%np.random.choice(self.list_ids, 1)[0]))
		
		# Partial data
		sketch_data = self.partial_data(sketch_data, p_mask=self.opt.p_mask)
		sketch_data = Image.fromarray(sketch_data).convert('RGB')

		sketch_data = ImageOps.pad(sketch_data, size=(self.opt.max_len, self.opt.max_len))
		image_data = ImageOps.pad(image_data, size=(self.opt.max_len, self.opt.max_len))
		negative_data = ImageOps.pad(negative_data, size=(self.opt.max_len, self.opt.max_len))

		if self.transform:
			img_tensor = self.transform(image_data)
			sk_tensor = self.transform(sketch_data)
			neg_tensor = self.transform(negative_data)

		if self.return_orig:
			return sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data
		else:
			return sk_tensor, img_tensor, neg_tensor

	def partial_data(self, sketch_data, p_mask):
		partial_sketch = np.zeros_like(sketch_data)
		instances = np.unique(sketch_data)[1:] # Remove 0-th element
		for obj in instances:
			if np.random.random_sample() > p_mask:
				partial_sketch[sketch_data == obj] = 255
		return partial_sketch


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

	dataset = SketchyScene(opts, mode='train', transform=dataset_transforms, return_orig=True)
	for idx, (sk_tensor, img_tensor, neg_tensor, sketch_data, image_data, negative_data) in enumerate(dataset):
		sketch_data.save(os.path.join(output_dir, '%d_sk.jpg'%idx))
		image_data.save(os.path.join(output_dir, '%d_img.jpg'%idx))
		negative_data.save(os.path.join(output_dir, '%d_neg.jpg'%idx))

		print ('Shape of sk_tensor: {} | img_tensor: {} | neg_tensor: {}'.format(
			sk_tensor.shape, img_tensor.shape, neg_tensor.shape))

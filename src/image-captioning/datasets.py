import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class OursScene(torch.utils.data.Dataset):

    def __init__(self, data_folder, split='train', transform=None):
        self.return_orig = False

        self.all_image_files = glob.glob(os.path.join(
            data_folder, 'raster_sketches', '*', '*.jpg'))

        self.all_ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                data_folder, 'raster_sketches', '*', '*.jpg'))])

        val_ids = np.loadtxt(os.path.join(data_folder, 'val_normal.txt'), dtype=str)

        if mode == 'train':
            self.all_ids = list(set(self.all_ids) - set(val_ids))
        else:
            self.all_ids = val_ids
        print ('total ids: ', len(self.all_ids))

        self.word_map = json.load(open('word_map.json', 'r'))

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

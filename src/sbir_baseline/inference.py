import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import opts
from model import TripletNetwork
from dataloader import OursScene, SketchyScene, SketchyCOCO, Sketchy

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Our Dataset
    train_dataset = OursScene(opts, mode='train',
        transform=dataset_transforms)
    val_dataset = OursScene(opts, mode='val',
        transform=dataset_transforms)

    # # SketchyScene Dataset
    # train_dataset = SketchyScene(opts, mode='train',
    #     transform=dataset_transforms)
    # val_dataset = SketchyScene(opts, mode='val',
    #     transform=dataset_transforms)

    # # SketchyCOCO Dataset
    # train_dataset = SketchyCOCO(opts, mode='train',
    #     transform=dataset_transforms)
    # val_dataset = SketchyCOCO(opts, mode='val',
    #     transform=dataset_transforms)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    model = TripletNetwork()#.load_from_checkpoint(checkpoint_path="saved_model/our-dataset-epoch=103-top10=0.52.ckpt")


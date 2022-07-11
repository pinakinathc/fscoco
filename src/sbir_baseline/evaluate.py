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

    rank1, rank10 = [], []
    for i in range(10):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_len, opts.max_len)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Our Dataset
        val_dataset = OursScene(opts, mode='val',
            transform=dataset_transforms)

        # SketchyScene Dataset
        # val_dataset = SketchyScene(opts, mode='val',
        #     transform=dataset_transforms)

        # SketchyCOCO Dataset
        # val_dataset = SketchyCOCO(opts, mode='val',
        #     transform=dataset_transforms)

        # Sketchy Dataset
        # val_dataset = Sketchy(opts, mode='val', transform=dataset_transforms)

        # rank10 = []
        # for category in val_dataset.all_categories[:10]:
        #     print ('sketch category: ', category)
        #     val_dataset.category = category

        val_loader = DataLoader(
            dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

        model = TripletNetwork().load_from_checkpoint('saved_model/our-dataset-epoch=1167-top10=0.53.ckpt')
        
        trainer = Trainer(logger=False, gpus=-1)

        metrics = trainer.validate(model, val_loader)
        rank1.append(metrics[0]['top1'])
        rank10.append(metrics[0]['top10'])

    print ("mean top1: ", np.mean(rank1), np.std(rank1))
    print ("mean top10: ", np.mean(rank10), np.std(rank10))
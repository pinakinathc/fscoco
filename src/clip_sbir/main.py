import os
from torch.utils.data import DataLoader
from options import opts
from model import CLIPNetwork
from dataloader import OursScene, SketchyScene, SketchyCOCO

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':

    model = CLIPNetwork().load_from_checkpoint(checkpoint_path="saved_models/our_dataset-epoch=29-top10=0.26.ckpt")
    img_preprocess = model.img_preprocess

    # Our Dataset
    train_dataset = OursScene(opts, img_preprocess=img_preprocess, mode='train', use_coco=True)
    val_dataset = OursScene(opts, img_preprocess=img_preprocess, mode='val', use_coco=True)

    # # SketchyScene Dataset
    # train_dataset = SketchyScene(opts, img_preprocess=img_preprocess, mode='train')
    # val_dataset = SketchyScene(opts, img_preprocess=img_preprocess, mode='val')

    # # SketchyCOCO Dataset
    # train_dataset = SketchyCOCO(opts, img_preprocess=img_preprocess, mode='train')
    # val_dataset = SketchyCOCO(opts, img_preprocess=img_preprocess, mode='val')

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger("tb_logs", name="our_dataset")

    checkpoint_callback = ModelCheckpoint(
        monitor='top10',
        dirpath='saved_models/',
        filename="our_dataset-{epoch:02d}-{top10:.2f}",
        mode='max')

    trainer = Trainer(gpus=-1, 
        min_epochs=1, max_epochs=200,
        benchmark=True,
        logger=logger,
        # profiler="advanced",
        # val_check_interval=1, 
        accumulate_grad_batches=8,
        check_val_every_n_epoch=3,
        resume_from_checkpoint=None, #"saved_models/sketchycoco-epoch=98-top10=0.44.ckpt",
        callbacks=[checkpoint_callback])

    print ('validating the pre-trained model ...')
    trainer.validate(model, val_loader)

    input('press any key to continue training')

    print ('Beginning training...')
    trainer.fit(model, train_loader, val_loader)

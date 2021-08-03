from torchvision import transforms
from torch.utils.data import DataLoader

from cnn_baseline.options import opts
from cnn_baseline.model import TripletNetwork
from cnn_baseline.dataloader import OursScene

from pytorch_lightning import Trainer

if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = OursScene(opts, mode='train', transform=dataset_transforms)
    val_dataset = OursScene(opts, mode='val', transform=dataset_transforms)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    model = TripletNetwork()
    trainer = Trainer(gpus=1)
    trainer.fit(model, train_loader, val_loader)

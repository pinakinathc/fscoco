import torch
from torchvision import transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from dataloader import SketchyCOCO
from options import parser
from model import Model

if __name__ == '__main__':
    opt = parser.parse_args() # get parameters

    dataset_transforms = transforms.Compose([
        transforms.Resize((opt.crop_size, opt.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  

    train_dataset = SketchyCOCO(opt, mode='train', transform=dataset_transforms)
    val_dataset = SketchyCOCO(opt, mode='val', transform=dataset_transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=opt.batch_size,
        shuffle=True, pin_memory=True,
        num_workers=opt.workers, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=opt.batch_size,
        shuffle=True, pin_memory=True,
        num_workers=opt.workers, drop_last=True)

    multi_gpu = torch.cuda.device_count() > 1
    model = Model(opt)
    # model.load_model()
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda() if multi_gpu else model.cuda()
        cudnn.benchmark = True

    # Sanity check
    for name, param in model.img_encoder.named_parameters():
        try:
            assert param.requires_grad == True
        except Exception as err:
            print (name, param.requires_grad)
            raise ValueError(err)
    for name, param in model.sketch_encoder.named_parameters():
        assert param.requires_grad == True

    for epoch in range(opt.num_epochs):
        model.train_epoch(train_loader, val_loader, epoch)

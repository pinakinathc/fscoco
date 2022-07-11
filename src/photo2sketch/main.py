from options import opts
import torch
from torchvision import transforms
from utils import collate_fn
from model import Photo2Sketch
from dataloader import OursScene


dataset_transforms = transforms.Compose([
    transforms.Resize((opts.max_len, opts.max_len)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = OursScene(opts, mode='train', transform=dataset_transforms)   
dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=opts.batch_size, collate_fn=collate_fn)

dataset_val = OursScene(opts, mode='val', transform=dataset_transforms)   
dataloader_val = torch.utils.data.DataLoader(
    dataset=dataset_val, batch_size=opts.batch_size, collate_fn=collate_fn)


model = Photo2Sketch().cuda()
model.load_state_dict(torch.load('model_small_sketch.ckpt'))

for epoch in range(1600, 100000):
    if epoch % 100 == 0:
        # First evaluate
        print ('evaluation started...')
        output_dir = 'output_small_sketch/%d'%epoch
        model.eval()
        for batch_idx, batch in enumerate(dataloader_val):
           model.evaluate(batch, batch_idx, output_dir)
        print ('evaluation done. Check %s'%output_dir)

    # Train model
    model.train()
    for params in model.img_encoder.parameters():
        params.require_grad = False

    for batch_idx, batch in enumerate(dataloader_train):
        loss = model.train_batch(batch)
        if batch_idx % 30 == 0:
            print ('Epoch: {}, Iter: {}/{}, Loss: {}'.format(
                epoch, batch_idx, len(dataloader_train), loss))

    if epoch % 200 == 0:
        torch.save(model.state_dict(), 'model_small_sketch.ckpt')

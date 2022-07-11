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

dataset_val = OursScene(opts, mode='val', transform=dataset_transforms)   
dataloader_val = torch.utils.data.DataLoader(
    dataset=dataset_val, batch_size=opts.batch_size, collate_fn=collate_fn)


model = Photo2Sketch().cuda()

model.load_state_dict(torch.load('model.ckpt'))

# First evaluate
print ('evaluation started...')
output_dir = 'output_evaluate/'
model.eval()
for batch_idx, batch in enumerate(dataloader_val):
   model.evaluate(batch, batch_idx, output_dir)
print ('evaluation done. Check %s'%output_dir)

import torchvision
import torch.nn as nn
import torch.nn.functional as F

class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)

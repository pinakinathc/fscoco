import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)

class Txt_Encoder(nn.Module):
    def __init__(self, vocab_size, word_dim=512, output_dim=512, num_layers=2):
        super().__init__()
        print 
        self.emb_layer = nn.Embedding(vocab_size, word_dim)
        self.gru_layer = nn.GRU(word_dim, 512,
            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(512*2*num_layers, output_dim)

    def forward(self, x, length):
        embedded = self.emb_layer(x) # B x max_len x word_dim
        packed = pack_padded_sequence(embedded, length.cpu(),
            batch_first=True, enforce_sorted=False)
        # packed = embedded
        # last_hidden shape: 2*2 x B x 512
        _, last_hidden = self.gru_layer(packed)
        last_hidden = torch.cat([feat for feat in last_hidden], dim=1)
        output = self.fc_layer(last_hidden)
        return output

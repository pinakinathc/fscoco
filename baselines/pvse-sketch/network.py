# -*- coding: utf-8 -*-
# author: pinakinathc
# Reference: github.com/yalesong/pvse.git

import torch
import torch.nn as nn
import torch.nn.init
import torchtext
import torchvision
from torch.autograd import Variable

def get_cnn(arch, pretrained):
  return torchvision.models.__dict__[arch](pretrained=pretrained) 


def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)

class MultiHeadSelfAttention(nn.Module):
  """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

  def __init__(self, n_head, d_in, d_hidden):
    super(MultiHeadSelfAttention, self).__init__()

    self.n_head = n_head
    self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
    self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)
    self.init_weights()

  def init_weights(self):
    nn.init.xavier_uniform_(self.w_1.weight)
    nn.init.xavier_uniform_(self.w_2.weight)

  def forward(self, x, mask=None):
    # This expects input x to be of size (b x seqlen x d_feat)
    attn = self.w_2(self.tanh(self.w_1(x)))
    if mask is not None:
      mask = mask.repeat(self.n_head, 1, 1).permute(1,2,0)
      attn.masked_fill_(mask, -np.inf)
    attn = self.softmax(attn)

    output = torch.bmm(attn.transpose(1,2), x)
    if output.shape[1] == 1:
      output = output.squeeze(1)
    return output, attn


class PIENet(nn.Module):
  """Polysemous Instance Embedding (PIE) module"""

  def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
    super(PIENet, self).__init__()

    self.num_embeds = n_embeds
    self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
    self.fc = nn.Linear(d_in, d_out)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_out)
    self.init_weights()

  def init_weights(self):
    nn.init.xavier_uniform_(self.fc.weight)
    nn.init.constant_(self.fc.bias, 0.0)

  def forward(self, out, x, pad_mask=None):
    residual, attn = self.attention(x, pad_mask)
    residual = self.dropout(self.sigmoid(self.fc(residual)))
    if self.num_embeds > 1:
      out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
    out = self.layer_norm(out + residual)
    return out, attn, residual


class PVSE(nn.Module):
  """Polysemous Visual-Semantic Embedding (PVSE) module"""

  def __init__(self, opt):
    super(PVSE, self).__init__()

    self.mil = opt.num_embeds > 1
    self.img_enc = EncoderImage(opt)
    self.sk_enc = EncoderImage(opt)

  def forward(self, sketch, image):
    sk_emb, sk_attn, sk_residual, sk_map = self.sk_enc(Variable(sketch))
    img_emb, img_attn, img_residual, img_map = self.sk_enc(Variable(image))
    return sk_emb, img_emb, sk_map, img_map, sk_residual, img_residual


class EncoderImage(nn.Module):

  def __init__(self, opt):
    super(EncoderImage, self).__init__()

    embed_size, num_embeds = opt.embed_size, opt.num_embeds
    self.use_attention = opt.img_attention
    self.abs = True if hasattr(opt, 'order') and opt.order else False

    # Backbone CNN
    self.cnn = get_cnn(opt.cnn_type, True)
    cnn_dim = self.cnn_dim = self.cnn.fc.in_features

    self.avgpool = self.cnn.avgpool
    self.cnn.avgpool = nn.Sequential()

    self.fc = nn.Linear(cnn_dim, embed_size)
    self.cnn.fc = nn.Sequential()

    self.dropout = nn.Dropout(opt.dropout)

    if self.use_attention:
      self.pie_net = PIENet(num_embeds, cnn_dim, embed_size, cnn_dim//2, opt.dropout)

    for idx, param in enumerate(self.cnn.parameters()):
      param.requires_grad = opt.img_finetune

  def init_weights(self):
    nn.init.xavier_uniform_(self.fc.weight)
    nn.init.constant_(self.fc.bias, 0.0)

  def forward(self, images):
    out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
    out = self.avgpool(out_7x7).view(-1, self.cnn_dim)
    out = self.fc(out)
    out = self.dropout(out)

    # compute self-attention map
    attn, residual = None, None
    if self.use_attention:
      out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)
      out, attn, residual = self.pie_net(out, out_7x7.transpose(1,2))
    
    out = l2norm(out)
    if self.abs:
      out = torch.abs(out)

    return out, attn, residual, out_7x7
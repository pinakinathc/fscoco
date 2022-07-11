import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.clip_sbir.model import CLIPNetwork
from src.clip_sbir.dataloader import OursScene
from src.clip_sbir.options import opts

distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)

if __name__ == '__main__':
	clip_network = CLIPNetwork('ViT-B/32').cuda()
	img_preprocess = clip_network.img_preprocess
	train_dataset = OursScene(opts, img_preprocess, mode='train', use_coco=True)
	val_dataset = OursScene(opts, img_preprocess, mode='val', use_coco=True)

	train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
	val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

	all_sk_features = []
	all_img_features = []

	for (txt_tensor, sk_tensor, img_tensor, neg_tensor) in val_loader:
		# print (txt_tensor.shape, sk_tensor.shape, img_tensor.shape, neg_tensor.shape)		

		sk_feature = clip_network(txt_tensor.cuda(), dtype='text')
		img_feature = clip_network(img_tensor.cuda(), dtype='image')

		# print (sk_feature.shape, img_feature.shape)
		all_sk_features.append(sk_feature.detach())
		all_img_features.append(img_feature.detach())

	all_sk_features = torch.cat(all_sk_features, dim=0)
	all_img_features = torch.cat(all_img_features, dim=0)

	print ('shape of all_sk_features: {}, all_img_features:{}'.format(
		all_sk_features.shape, all_img_features.shape))

	rank = []

	for idx, sk_feature in enumerate(all_sk_features):
		dist = distance_fn(sk_feature.unsqueeze(0), all_img_features)
		target = distance_fn(sk_feature.unsqueeze(0), all_img_features[idx].unsqueeze(0))
		rank.append(dist.le(target).sum().item())


	rank = np.array(rank)
	print ('Top1: %.2f | Top10: %.2f | MeanK: %.2f'%(
		np.sum(rank[rank<=1])/len(rank),
		np.sum(rank[rank<=10])/len(rank),
		np.mean(rank)))



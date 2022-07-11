import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import pytorch_lightning as pl

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        if p.data is None or p.grad is None:
            continue
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class CLIPNetwork(pl.LightningModule):

	def __init__(self):
		super(CLIPNetwork, self).__init__()
		
		# get a ViT
		self.ViT, self.img_preprocess = clip.load("ViT-B/32",device="cuda:0",jit=False)
		convert_models_to_fp32(self.ViT)
		self.ViT.apply(freeze_all_but_bn)

		self.img_emb = nn.Sequential(
			nn.Linear(512, 512, dtype=torch.float16),
			nn.ReLU())

		self.txt_emb = nn.Sequential(
			nn.Linear(512, 512, dtype=torch.float16),
			nn.ReLU())

		self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
		self.loss = nn.TripletMarginWithDistanceLoss(
			distance_function=self.distance_fn, margin=0.2)

	def forward(self, x, dtype):
		if dtype == 'image':
			features = self.ViT.encode_image(x)
			features = self.img_emb(features)
		if dtype == 'text':
			features = self.ViT.encode_text(x)
			features = self.txt_emb(features)
		return features

	def configure_optimizers(self):
		optimizer = torch.optim.Adam([
				{'params': self.ViT.parameters()},
				{'params': self.img_emb.parameters(), 'lr': 1e-4},
				{'params': self.txt_emb.parameters(), 'lr': 1e-4}
			],
			lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
		return optimizer

	def training_step(self, batch, batch_idx):
		self.ViT.apply(freeze_all_but_bn)
		# sk_tensor, img_tensor, neg_tensor =  batch
		txt_tensor, sk_tensor, img_tensor, neg_tensor =  batch
		txt_feature = self.forward(txt_tensor, dtype='text')
		# sk_feature = self.forward(sk_tensor, dtype='image')
		img_feature = self.forward(img_tensor, dtype='image')
		neg_feature = self.forward(neg_tensor, dtype='image')

		query_feature = txt_feature
		loss = self.loss(query_feature, img_feature, neg_feature)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		# sk_tensor, img_tensor, neg_tensor =  val_batch
		txt_tensor, sk_tensor, img_tensor, neg_tensor = val_batch
		txt_feature = self.forward(txt_tensor, dtype='text')
		# sk_feature = self.forward(sk_tensor, dtype='image')
		img_feature = self.forward(img_tensor, dtype='image')
		neg_feature = self.forward(neg_tensor, dtype='image')
		
		query_feature = txt_feature
		loss = self.loss(query_feature, img_feature, neg_feature)
		self.log('val_loss', loss)
		return query_feature, img_feature

	def validation_epoch_end(self, validation_step_outputs):
		Len = len(validation_step_outputs)
		query_feature_all = torch.cat([validation_step_outputs[i][0] for i in range(Len)])
		image_feature_all = torch.cat([validation_step_outputs[i][1] for i in range(Len)])

		rank = torch.zeros(len(query_feature_all))
		for idx, query_feature in enumerate(query_feature_all):
			distance = self.distance_fn(query_feature.unsqueeze(0), image_feature_all)
			target_distance = self.distance_fn(
				query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
			rank[idx] = distance.le(target_distance).sum()

		top1 = rank.le(2).sum().numpy() / rank.shape[0]
		top5 = rank.le(5).sum().numpy() / rank.shape[0]
		top10 = rank.le(10).sum().numpy() / rank.shape[0]
		meanK = rank.mean().numpy()

		self.log('top1', top1)
		self.log('top5', top5)
		self.log('top10', top10)
		self.log('meanK', meanK)

		print ('Evaluation metrics: top1 %.4f | top5 %.4f top10 %.4f | meanK %.4f'%(
			top1, top5, top10, meanK))

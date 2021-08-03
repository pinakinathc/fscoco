import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_baseline.network import VGG_Network
import pytorch_lightning as pl

class TripletNetwork(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.sk_embedding_network = VGG_Network()
        self.img_embedding_network = VGG_Network()
        self.loss = nn.TripletMarginLoss(margin=0.2)

    def forward(self, x):
        feature = self.embedding_network(x)
        return feature

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # defines the train loop
        sk_tensor, img_tensor, neg_tensor = batch
        sk_feature = self.sk_embedding_network(sk_tensor)
        img_feature = self.img_embedding_network(img_tensor)
        neg_feature = self.img_embedding_network(neg_tensor)
        loss = self.loss(sk_feature, img_feature, neg_feature)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # defines the validation loop
        sk_tensor, img_tensor, neg_tensor = val_batch
        sk_feature = self.sk_embedding_network(sk_tensor)
        img_feature = self.img_embedding_network(img_tensor)
        neg_feature = self.img_embedding_network(neg_tensor)
        loss = self.loss(sk_feature, img_feature, neg_feature)
        self.log('val_loss', loss)
        return sk_feature, img_feature

    def validation_epoch_end(self, validation_step_outputs):
        Len = len(validation_step_outputs)
        sketch_feature_all = torch.cat([validation_step_outputs[i][0] for i in range(Len)])
        image_feature_all = torch.cat([validation_step_outputs[i][1] for i in range(Len)])

        rank = torch.zeros(len(sketch_feature_all))
        for idx, sketch_feature in enumerate(sketch_feature_all):
            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), image_feature_all)
            target_distance = F.pairwise_distance(
                sketch_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
            rank[idx] = distance.le(target_distance).sum()

        self.log('top1', rank.le(1).sum().numpy() / rank.shape[0])
        self.log('top10', rank.le(10).sum().numpy() / rank.shape[0])
        self.log('meanK', rank.mean().numpy())

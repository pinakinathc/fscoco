import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stbir_baseline.network import VGG_Network, Txt_Encoder, Combine_Network
import pytorch_lightning as pl

class TripletNetwork(pl.LightningModule):

    def __init__(self, vocab_size, combine_type='concat'):
        super().__init__()
        self.txt_embedding_network = Txt_Encoder(vocab_size=vocab_size)
        self.img_embedding_network = VGG_Network()
        self.combine_network = Combine_Network(input_dim=512, output_dim=512, mode=combine_type)
        self.loss = nn.TripletMarginLoss(margin=0.2)

    def forward(self, x):
        feature = self.embedding_network(x)
        return feature

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # defines the train loop
        txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor = batch
        txt_feature = self.txt_embedding_network(txt_tensor, txt_length)
        sk_feature = self.img_embedding_network(sk_tensor)
        img_feature = self.img_embedding_network(img_tensor)
        neg_feature = self.img_embedding_network(neg_tensor)
        query_feature = self.combine_network(sk_feature, txt_feature)
        loss = self.loss(query_feature, img_feature, neg_feature)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # defines the validation loop
        txt_tensor, txt_length, sk_tensor, img_tensor, neg_tensor = val_batch
        txt_feature = self.txt_embedding_network(txt_tensor, txt_length)
        sk_feature = self.img_embedding_network(sk_tensor)
        img_feature = self.img_embedding_network(img_tensor)
        neg_feature = self.img_embedding_network(neg_tensor)
        query_feature = self.combine_network(sk_feature, txt_feature)
        loss = self.loss(query_feature, img_feature, neg_feature)
        self.log('val_loss', loss)
        return query_feature, img_feature

    def validation_epoch_end(self, validation_step_outputs):
        Len = len(validation_step_outputs)
        query_feature_all = torch.cat([validation_step_outputs[i][0] for i in range(Len)])
        image_feature_all = torch.cat([validation_step_outputs[i][1] for i in range(Len)])

        rank = torch.zeros(len(query_feature_all))
        for idx, query_feature in enumerate(query_feature_all):
            distance = F.pairwise_distance(query_feature.unsqueeze(0), image_feature_all)
            target_distance = F.pairwise_distance(
                query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
            rank[idx] = distance.le(target_distance).sum()

        self.log('top1', rank.le(1).sum().numpy() / rank.shape[0])
        self.log('top10', rank.le(10).sum().numpy() / rank.shape[0])
        self.log('meanK', rank.mean().numpy())

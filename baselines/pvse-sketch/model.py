# -*- coding: utf-8 -*-
# author: pinakinathc

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from network import EncoderImage
from pvse_loss import PVSELoss

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        # define models
        self.img_encoder = EncoderImage(opt)
        self.sketch_encoder = EncoderImage(opt)

        self.optimizer = torch.optim.Adam(
            list(self.img_encoder.parameters()) + list(self.sketch_encoder.parameters()))
        # self.criterion = torch.nn.TripletMarginWithDistanceLoss(
        #   distance_function=torch.nn.CosineSimilarity())
        # self.criterion = torch.nn.TripletMarginLoss(margin=0.5)
        self.criterion = PVSELoss(self.opt)

        # record training
        self.total_iter = 0
        self.logger = SummaryWriter(self.opt.log_dir)

    def train_epoch(self, dataloader, val_loader, epoch):
        N = len(dataloader)
        start_time = time.time()
        data_time = 0
        iter_time = 0
        for itr, (img, sketch, negative) in enumerate(dataloader):
            self.total_iter += 1

            img = img.cuda()
            sketch = sketch.cuda()
            negative = negative.cuda()
            data_time += time.time() - start_time

            self.optimizer.zero_grad()
            
            emb_img, _, _ = self.img_encoder(img)
            emb_sketch, _, _ = self.sketch_encoder(sketch)
            emb_negative, _, _ = self.img_encoder(negative)
            
            anchor = emb_sketch
            positive = emb_img
            negative = emb_negative

            loss = self.criterion(anchor, positive, negative)

            loss.backward()
            self.optimizer.step()

            iter_time += time.time() - start_time

            # logging
            if self.total_iter % self.opt.log_freq == 0:
                print ('Epoch: [%d] Iteration: [%d/%d] Triplet Loss: %.7f dataT: %.3f iterT: %.3f' % (
                    epoch, itr, N, float(loss), data_time/self.opt.log_freq, iter_time/self.opt.log_freq))
                self.logger.add_scalar('Loss/train', float(loss), self.total_iter)
                start_time = time.time()
                data_time = 0
                iter_time = 0

            # saving model
            if self.total_iter % self.opt.save_freq == 0:
                torch.save({
                    'total_iter': self.total_iter,
                    'img_encoder_state_dict': self.img_encoder.state_dict(),
                    'sketch_encoder_state_dict': self.sketch_encoder.state_dict()
                    }, self.opt.model_path)

            # evaluating model
            # if (self.total_iter-1) % self.opt.eval_freq == 0:
            #     self.evaluate(dataloader, val_loader)

    def calculate_meanK(self, dataloader):
        rank = []
        all_emb_img = []
        all_emb_sketch = []
        all_img = []
        all_sketch = []

        for (img, sketch, _) in dataloader:
            img = img.cuda()
            sketch = sketch.cuda()

            emb_img, _, _ = self.img_encoder(img)
            emb_sketch, _, _ = self.sketch_encoder(sketch)

            emb_img = emb_img.detach().cpu()
            emb_sketch = emb_sketch.detach().cpu()
            img = img.detach().cpu()
            sketch = sketch.detach().cpu()

            all_emb_img.append(emb_img.cpu())
            all_emb_sketch.append(emb_sketch.cpu())
            
            all_img.append(img.cpu())
            all_sketch.append(sketch.cpu())

        all_emb_img = torch.cat(all_emb_img, dim=0)
        all_emb_sketch = torch.cat(all_emb_sketch, dim=0)
        all_img = torch.cat(all_img, dim=0)
        all_sketch = torch.cat(all_sketch, dim=0)

        retrieval_vis = []
        rand_idx = np.random.choice(len(all_emb_sketch), 3)

        for idx, emb_sketch in enumerate(all_emb_sketch):
            distance = torch.nn.functional.pairwise_distance(
                emb_sketch.unsqueeze(0), all_emb_img)
            target_distance = torch.nn.functional.pairwise_distance(
                emb_sketch.unsqueeze(0), all_emb_img[idx].unsqueeze(0))

            rank.append(distance.le(target_distance).sum())
            # meanK += rank

            if idx in rand_idx:
                retrieval_vis.append(all_sketch[idx].unsqueeze(0))
                retrieval_vis.append(all_img[torch.argsort(distance)[:10]])

        retrieval_vis = torch.cat(retrieval_vis, dim=0)
        retrieval_vis = ((retrieval_vis - retrieval_vis.min()) / (retrieval_vis.max() - retrieval_vis.min()) * 255).type(torch.uint8)

        rank = torch.tensor(rank, dtype=torch.float32)
        # meanK = meanK / all_emb_sketch.shape[0]
        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]
        meanK = rank.mean()
        return top1, top10, meanK, retrieval_vis

    def evaluate(self, trainloader, valloader):
        self.eval()
        train_t1, train_t10, train_meanK, train_vis = self.calculate_meanK(trainloader)
        val_t1, val_t10, valid_meanK, val_vis = self.calculate_meanK(valloader)

        self.logger.add_scalars('Top-1',
            {'training': train_t1, 'validation': val_t1},
            self.total_iter)

        self.logger.add_scalars('Top-10',
            {'training': train_t10, 'validation': val_t10},
            self.total_iter)

        self.logger.add_scalars('MeanK',
            {'training': train_meanK, 'validation': valid_meanK},
            self.total_iter)
        self.logger.add_images('retrieval-train', train_vis, self.total_iter)
        self.logger.add_images('retrieval-val', val_vis, self.total_iter)

        self.train()

    def load_model(self):
        checkpoint = torch.load(self.opt.model_path)
        self.total_iter = checkpoint['total_iter']
        self.img_encoder.load_state_dict(checkpoint['img_encoder_state_dict'])
        self.sketch_encoder.load_state_dict(checkpoint['sketch_encoder_state_dict'])

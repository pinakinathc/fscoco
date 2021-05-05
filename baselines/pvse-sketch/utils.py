# -*- coding: utf-8 -*-
# author: pinakinathc
# Reference: github.com/Jeff-Zilence/Explain_Metric_Learning.git

import cv2, torch, numpy as np
from loss import cosine_sim

def metrics(sk_embs, img_embs, num_embeds):
    # compute image-sentence score matrix
    if num_embeds > 1:
      scores = cosine_sim(sk_embs.view(-1, sk_embs.size(-1)), img_embs.view(-1, img_embs.size(-1)))
      scores = torch.nn.functional.max_pool2d(scores.unsqueeze(0), num_embeds).squeeze()
    else:
      scores = cosine_sim(sk_embs, img_embs)

    N = scores.shape[0]
    _, ind_all = scores.topk(N, 1, True, True)
    correct = ind_all.eq(torch.arange(N).view(-1, 1).expand_as(ind_all))
    rank_ref = torch.arange(N).expand_as(ind_all) + 1
    rank = torch.where(correct, rank_ref, torch.tensor(0)).sum(dim=1).numpy()

    r1 = len(rank[rank <= 1]) / len(rank) * 100
    r5 = len(rank[rank <= 5]) / len(rank) * 100
    r10 = len(rank[rank <= 10]) / len(rank) * 100
    r50 = len(rank[rank <= 50]) / len(rank) * 100
    r100 = len(rank[rank <= 100]) / len(rank) * 100
    medr = np.median(rank)
    meanr = np.mean(rank)

    return r1, r5, r10, r50, r100, medr, meanr


def save_ckpt(opt, model, iterations):
    print ('saving model at iteration: ', iterations)
    torch.save({
        'model': model.state_dict(),
        'iterations': iterations
    }, opt.ckpt)


def imshow_convert(raw):
    '''
        convert the heatmap for imshow
    '''
    heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET))
    return heatmap


def GradCAM(map, size=(224, 224)):
    gradient = map.grad.cpu().numpy()
    map = map.detach().cpu().numpy()

    # compute the average value
    weights = np.mean(gradient[0], axis=(1, 2), keepdims=True)
    grad_CAM_map = np.sum(np.tile(
        weights, [1, map.shape[-2], map.shape[-1]])*map[0], axis=0)
    
    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = grad_CAM_map / np.max(grad_CAM_map)
    cam = cv2.resize(cam, (size[1], size[0]))
    return cam

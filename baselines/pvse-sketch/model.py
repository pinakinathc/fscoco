# -*- coding: utf-8 -*-
# author: pinakinathc

import tqdm, numpy as np, torch, torchvision

from loss import PVSELoss, cosine_sim, order_sim, l2norm
from data import get_dataloaders
from utils import *

def train(opt, model, log_writer, iterations=0):
    train_loader, val_loader = get_dataloaders(opt)

    # Loss and optimizer
    criterion = PVSELoss(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, 
        weight_decay=opt.weight_decay, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-10, verbose=True)

    total_iter = iterations
    for epoch in range(iterations//len(train_loader), opt.num_epochs):
        for itr, (img, sk) in enumerate(train_loader):
            img = img.cuda() if torch.cuda.is_available() else img
            sk = sk.cuda() if torch.cuda.is_available() else sk

            sk_emb, img_emb, sk_map, img_map, sk_r, img_r = model(sk, img)
            loss, loss_dict = criterion(img_emb, sk_emb, img_r, sk_r)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            # Print log info
            if itr > 0 and (itr % opt.log_step == 0 or itr + 1 == len(train_loader)):
                print('[%d][%d/%d] Loss: %.4f' %(
                    epoch, itr, len(train_loader), loss.item()))
                log_writer.add_scalar('Train/Loss', loss.item(), total_iter)
                log_writer.add_scalar('Train/ranking_loss', loss_dict['ranking_loss'].item(), total_iter)
                if opt.num_embeds > 1:
                    log_writer.add_scalar('Train/div_loss', loss_dict['div_loss'].item(), total_iter)
                    log_writer.add_scalar('Train/mmd_loss', loss_dict['mmd_loss'].item(), total_iter)
                log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], total_iter)
            
            total_iter += 1

        val_score = evaluate(opt, model, val_loader, log_writer, total_iter=total_iter)
        model.train() # switch to train mode
        lr_scheduler.step(val_score)


def evaluate(opt, model, val_loader, log_writer, total_iter):
    model.eval() # switch to evaluate mode

    # store all embeddings
    img_embs = list()
    sk_embs = list()

    print ('Evaluating model...')
    for (img, sk) in tqdm.tqdm(val_loader):
        img = img.cuda() if torch.cuda.is_available() else img
        sk = sk.cuda() if torch.cuda.is_available() else sk
        sk_emb, img_emb, sk_map, img_map, sk_r, img_r = model(sk, img)
        
        img_embs.append(img_emb.cpu().detach())
        sk_embs.append(sk_emb.cpu().detach())
    
    img_embs = torch.cat(img_embs, axis=0)
    sk_embs = torch.cat(sk_embs, axis=0)

    r1, r5, r10, r50, r100, medr, meanr = metrics(sk_embs, img_embs, opt.num_embeds)
    print ('r1: {}, r5: {}, r10: {}, r50:{}, r100:{}, medr: {}, meanr: {}'.format(r1, r5, r10, r50, r100, medr, meanr))
    log_writer.add_scalar('Validation/r1', r1, total_iter)
    log_writer.add_scalar('Validation/r5', r5, total_iter)
    log_writer.add_scalar('Validation/r10', r10, total_iter)
    log_writer.add_scalar('Validation/medianR', medr, total_iter)
    log_writer.add_scalar('Validation/meanR', meanr, total_iter)

    if total_iter % 100 == 0 and total_iter: # save every 1000 iterations
        save_ckpt(opt, model, total_iter)

    return r1 + r5 + r10


def testing(opt, model):
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from data import SketchyCOCO_Dataset

    dataset_transforms = transforms.Compose([
        transforms.RandomResizedCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SketchyCOCO_Dataset(opt, mode='test', transform=dataset_transforms, return_orig=True)

    img_embs, sk_embs, img_maps, imgs, sks, img_tensors, sk_tensors = list(), list(), list(), list(), list(), list(), list()

    model.train()
    print ('Testing model...')
    for (img_tensor, sk_tensor, img, sk) in tqdm.tqdm(test_dataset):
        img_tensor = img_tensor.unsqueeze(0) # 1x3x224x224
        sk_tensor = sk_tensor.unsqueeze(0)
        img_tensor = img_tensor.cuda() if torch.cuda.is_available() else img_tensor
        sk_tensor = sk_tensor.cuda() if torch.cuda.is_available() else sk_tensor

        img_emb, _, _, img_map = model(img_tensor)
        sk_emb, _, _ , sk_map = model(sk_tensor)

        img_map.retain_grad()

        img_embs.append(img_emb.cpu().detach())
        sk_embs.append(sk_emb.cpu().detach())
        imgs.append(img)
        sks.append(sk)
        img_tensors.append(img_tensor.cpu())
        sk_tensors.append(sk_tensor.cpu())

    img_embs = torch.cat(img_embs, axis=0)
    sk_embs = torch.cat(sk_embs, axis=0)

    if opt.num_embeds > 1:
      scores = cosine_sim(sk_embs.view(-1, sk_embs.size(-1)), img_embs.view(-1, img_embs.size(-1)))
      scores = torch.nn.functional.max_pool2d(scores.unsqueeze(0), opt.num_embeds).squeeze()
    else:
      scores = cosine_sim(sk_embs, img_embs)

    _, indices = scores.topk(5, 1, True, True) # Get top 5 retrieval

    for i in range(len(sk_embs)):
        sk_tensor = sk_tensors[i]
        sk_tensor = sk_tensor.cuda() if torch.cuda.is_available() else sk_tensor
        sk_emb, _, _ , sk_map= model(sk_tensor)

        sk = sks[i]

        fig, ax = plt.subplots(1, 7)
        ax[0].set_title('GT Img')
        ax[0].imshow(imgs[i])
        ax[1].set_title('Query')
        ax[1].imshow(sks[i])

        for ind, j in enumerate(indices[i]):
            img_tensor = img_tensors[j.item()]
            img_tensor = img_tensor.cuda() if torch.cuda.is_available() else img_tensor
            img_emb, _, _, img_map = model(img_tensor)
            img = np.array(imgs[j.item()])
            img = cv2.resize(img, (224, 224))

            img_map.retain_grad()

            product_vector = torch.mul(img_emb, sk_emb)
            product = torch.sum(product_vector)
            product.backward(torch.tensor(1.).cuda(), retain_graph=True)
            img_gradcam = GradCAM(img_map)

            image_overlay =  cv2.addWeighted(imshow_convert(img_gradcam), 0.3, img, 0.7, 0)
            # image_overlay = np.array(img) * 0.7 + imshow_convert(img_gradcam) / 255.0 * 0.3
            ax[ind+2].set_title('R-%d'%(ind+1))
            ax[ind+2].imshow(image_overlay)
            del img_tensor, img_emb, img_map

        plt.show()

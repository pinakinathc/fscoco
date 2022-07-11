import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from options import opts
from dataloader import OursScene, SketchyScene, SketchyCOCO, COCO
import clip

class TaskEmbedding(nn.Module):
    def __init__(self):
        super(TaskEmbedding, self).__init__()
        self.linear = nn.Linear(512, 512, dtype=torch.half)

    def forward(self, x):
        return self.linear(x)

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

def eval(dataloader):
    all_image_features = []
    all_query_features = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            txt_tensor, sk_tensor, img_tensor, neg_tensor = batch
            # sk_tensor, img_tensor, neg_tensor = batch
            # txt_tensor, img_tensor, neg_tensor = batch
            
            # query_tensor = txt_tensor
            image_features = model.encode_image(img_tensor.to(device))
            query_features = model.encode_image(sk_tensor.to(device))
            image_features = task_embedding(image_features)
            query_features = task_embedding(query_features)
            # query_features = model.encode_text(txt_tensor.to(device))

            all_image_features.append(image_features)
            all_query_features.append(query_features)

    all_image_features = torch.cat(all_image_features, dim=0)
    all_query_features = torch.cat(all_query_features, dim=0)

    # Pick the top 5 most similar labels for the image
    all_image_features /= all_image_features.norm(dim=-1, keepdim=True)
    all_query_features /= all_query_features.norm(dim=-1, keepdim=True)

    rank = torch.zeros(len(all_query_features))
    for q_id in range(len(all_query_features)):
        query_features = all_query_features[q_id]
        similarity = (100.0 * query_features.unsqueeze(0) @ all_image_features.T).softmax(dim=-1)
        rank[q_id] = similarity[0].ge(similarity[0, q_id]).sum()

    rank1 = rank.le(1).sum().numpy() / rank.shape[0]
    rank5 = rank.le(5).sum().numpy() / rank.shape[0]
    rank10 = rank.le(10).sum().numpy() / rank.shape[0]
    rankM = rank.mean().numpy()

    print ('Metrics -- rank1: {}, rank5: {}, rank10: {}, meanK: {}'.format(
        rank1, rank5, rank10, rankM))


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load('ViT-B/32', device)
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    task_embedding = TaskEmbedding().to(device)

    # Our Dataset
    train_dataset = OursScene(opts, img_preprocess=preprocess, mode='train', use_coco=True)
    val_dataset = OursScene(opts, img_preprocess=preprocess, mode='val', use_coco=True)

    # # SketchyScene Dataset
    # train_dataset = SketchyScene(opts, img_preprocess=preprocess, mode='train')
    # val_dataset = SketchyScene(opts, img_preprocess=preprocess, mode='val')

    # # SketchyCOCO Dataset
    # train_dataset = SketchyCOCO(opts, img_preprocess=preprocess, mode='train')
    # val_dataset = SketchyCOCO(opts, img_preprocess=preprocess, mode='val')

    # # COCO Dataset
    # # train_dataset = COCO(opts, img_preprocess=preprocess, mode='train')
    # val_dataset = COCO(opts, img_preprocess=preprocess, mode='val')

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    optimizer = optim.Adam([ {'params': model.parameters()}, {'params': task_embedding.parameters(), 'lr': 1e-4}],
        lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    for epoch in range(200):
        eval(val_dataloader)
        model.apply(freeze_all_but_bn)
        for batch in tqdm.tqdm(train_dataloader) :
            optimizer.zero_grad()
            txt_tensor, sk_tensor, img_tensor, neg_tensor = batch

            # calculate features
            image_features = model.encode_image(img_tensor.to(device))
            query_features = model.encode_image(sk_tensor.to(device))
            image_features = task_embedding(image_features)
            query_features = task_embedding(query_features)
            
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ query_features.t()
            logits_per_text = logits_per_image.t()

            # logits_per_image, logits_per_text = model(img_tensor.to(device), txt_tensor.to(device))

            ground_truth = torch.arange(img_tensor.shape[0], dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                convert_models_to_fp32(task_embedding)
                optimizer.step()
                clip.model.convert_weights(model)   
                clip.model.convert_weights(task_embedding)
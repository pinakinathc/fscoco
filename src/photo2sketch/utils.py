import os
import cv2
import numpy as np
from bresenham import bresenham
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def collate_fn(batch):
    """ List of tuples: (raster_sketch, image, vector_sketch, list_stroke_len) 
    Return:
        raster_sketch: (nbatch, 3, h, w)
        image: (nbatch, 3, h, w)
        vector_sketch: (nbatch, max_num_strokes, max_stroke_len, 5)
        batch_num_strokes: (nbatch, 1)
        batch_stroke_len: (nbatch, max_num_strokes)

    """

    nbatch = len(batch)
    list_raster_sketch = [item[0] for item in batch]
    list_image = [item[1] for item in batch]
    list_list_vector = [item[2] for item in batch]
    list_list_stroke_len = [item[3] for item in batch]

    raster_sketch = torch.stack(list_raster_sketch) # nbatch x 3 x h x w
    image = torch.stack(list_image) # nbatch x 3 x h x w

    batch_num_strokes = torch.LongTensor([len(list_stroke_len) \
        for list_stroke_len in list_list_stroke_len]) # nbatch x 1

    max_num_strokes = batch_num_strokes.max()
    batch_stroke_len = torch.zeros(nbatch, max_num_strokes).long()
    
    for b_id, list_stroke_len in enumerate(list_list_stroke_len):
        batch_stroke_len[b_id, :len(list_stroke_len)] = torch.LongTensor(list_stroke_len)
        
    max_stroke_len = batch_stroke_len.max()
    vector_sketch = torch.zeros(nbatch, max_num_strokes, max_stroke_len, 5).float()
    for b_id in range(nbatch):
        list_vector = list_list_vector[b_id]
        for stroke_id in range(len(list_vector)):
            vector_sketch[b_id, stroke_id, :batch_stroke_len[b_id, stroke_id], :] = \
                                            torch.FloatTensor(list_vector[stroke_id])

    # normalise vector_sketch
    vector_sketch[:, :, :, :2] = vector_sketch[:, :, :, :2] / vector_sketch[:, :, :, :2].max()
    return raster_sketch, image, vector_sketch, batch_num_strokes, batch_stroke_len



def draw_tensor_sketch(
    image, raster_sketch, vector_sketch, batch_num_strokes, batch_stroke_len, output_dir='output_evaluate', batch_idx=0):
    """
    Input
        image: Tensor (nbatch, 3, 224, 224)
        raster_sketch: Tensor(nbatch, 3, 224, 224)
        vector_sketch: Tensor (nbatch, max_num_strokes, max_stroke_len, 5)
        batch_num_strokes: Tensor (nbatch, 1)
        batch_stroke_len: Tensor (nbatch, max_num_strokes)

    Return
        None
    
    """

    # batch_id = 0 # draw only first image
    vector_sketch = vector_sketch[:, :, :, :2]
    nbatch = vector_sketch.shape[0]
    os.makedirs(output_dir, exist_ok=True)

    vector_sketch = (vector_sketch - vector_sketch.min())/(vector_sketch.max() - vector_sketch.min()) * 255.0

    for batch_id in range(nbatch):
        max_xy = int(vector_sketch.max().item()) + 1
        canvas = np.ones((256, 256), dtype=np.uint8) * 255 # white canvas
        vector_output =[]
        for stroke_id in range(batch_num_strokes[batch_id]):
            for xy_id in range(1, batch_stroke_len[batch_id, stroke_id]):
                xy = vector_sketch[batch_id, stroke_id, xy_id][:2].long()
                prev_xy = vector_sketch[batch_id, stroke_id, xy_id-1][:2].long()

                canvas[xy[1], xy[0]] = 0

                # cordList = list(bresenham(prev_xy[0], prev_xy[1], xy[0], xy[1]))
                # for cord in cordList:
                #     canvas[cord[1], cord[0]] = 0

                vector_output.append(list(xy.cpu().numpy()))

        canvas = cv2.dilate(255 - canvas, np.ones((3,3),np.uint8), iterations=1)
        canvas = 255 - canvas
        cv2.imwrite(os.path.join(output_dir, '%d%d_%s.jpg'%(batch_idx, batch_id, 'pred')), canvas)
        np.save(os.path.join(output_dir, '%d%d_%s.npy'%(batch_idx, batch_id, 'pred')), vector_output)
        torchvision.utils.save_image([image[batch_id], raster_sketch[batch_id]],
            os.path.join(output_dir, '%d%d_%s.jpg'%(batch_idx, batch_id, 'gt')))
        
        # plt.imshow(canvas, cmap='gray')
        # plt.show()


def decode_vector_sketch(vector_sketch, batch_num_strokes, batch_stroke_len):
    """
        Input
            vector_sketch: Tensor (nbatch, max_num_strokes, max_stroke_len, 5)
            batch_num_strokes: Tensor (nbatch, 1)
            batch_stroke_len: Tensor (nbatch, max_num_strokes)

        Return:
            Tensor (N, 5)

    """
    nbatch, max_num_strokes, max_stroke_len, P = vector_sketch.shape
    mask = torch.zeros(nbatch, max_num_strokes, max_stroke_len).cuda()
    for batch_id in range(nbatch):
        for stroke_id in range(batch_num_strokes[batch_id]):
            stroke_len = batch_stroke_len[batch_id, stroke_id]
            mask[batch_id, stroke_id, :stroke_len] = 1

    return vector_sketch.view(-1, P)[mask.view(-1) > 0]


if __name__ == '__main__':
    seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
    lens = [2, 1, 3]

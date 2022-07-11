import torch
import torch.nn as nn
import torch.optim as optim
from network import VGG_Network, RNN_Decoder
from utils import draw_tensor_sketch, decode_vector_sketch

class Photo2Sketch(nn.Module):
	def __init__(self):
		super(Photo2Sketch, self).__init__()
		self.img_encoder = VGG_Network()
		self.sketch_decoder = RNN_Decoder()

		self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
		self.criterion_mse = nn.MSELoss()
		self.criterion_ce = nn.CrossEntropyLoss()

	def forward(self, image, vector_sketch, batch_num_strokes, batch_stroke_len):

		enc_features = self.img_encoder(image)
		out_coords = self.sketch_decoder(
			enc_features, vector_sketch, batch_num_strokes, batch_stroke_len)
		return out_coords, batch_num_strokes, batch_stroke_len

	def train_batch(self, batch):
		self.optimizer.zero_grad()

		raster_sketch, image, vector_sketch, batch_num_strokes, batch_stroke_len = batch
		raster_sketch = raster_sketch.cuda()
		image = image.cuda()
		vector_sketch = vector_sketch.cuda()

		out_coords, batch_num_strokes, batch_stroke_len = self.forward(
				image, vector_sketch, batch_num_strokes, batch_stroke_len)

		# shape of pred_coords and gt_coords: (nbatch * max_num_stroke * max_stroke_len, 5)
		# shape of mask: (nbatch * max_num_stroke * max_stroke_len, 1)
		pred_coords = decode_vector_sketch(out_coords, batch_num_strokes, batch_stroke_len)
		gt_coords = decode_vector_sketch(vector_sketch, batch_num_strokes, batch_stroke_len)

		loss_mse = self.criterion_mse(pred_coords[:, :2], gt_coords[:, :2])
		loss_ce = self.criterion_ce(pred_coords[:, 2:], gt_coords[:, 2:].argmax(dim=1))

		# loss_mse = (loss_mse.mean(dim=1) * mask).sum() / mask.sum()
		# loss_ce = (loss_ce * mask).sum() / mask.sum()

		loss = loss_mse + loss_ce
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def evaluate(self, batch, batch_idx, output_dir):
		raster_sketch, image, vector_sketch, batch_num_strokes, batch_stroke_len = batch
		raster_sketch = raster_sketch.cuda()
		image = image.cuda()
		vector_sketch = vector_sketch.cuda()

		out_coords, batch_num_strokes, batch_stroke_len = self.forward(
				image, vector_sketch, batch_num_strokes, batch_stroke_len)

		draw_tensor_sketch(image, raster_sketch, out_coords, batch_num_strokes,
			batch_stroke_len, output_dir, batch_idx=batch_idx)

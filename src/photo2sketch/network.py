import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VGG_Network(nn.Module):

	def __init__(self):
		super(VGG_Network, self).__init__()
		self.backbone = torchvision.models.vgg16(pretrained=True).features
		# self.backbone = torchvision.models.resnet18(pretrained=True).features
		self.pool_method = nn.AdaptiveMaxPool2d(1)

	def forward(self, x):
		x = self.backbone(x)
		x = self.pool_method(x).view(-1, 512)
		return F.normalize(x)


class RNN_Decoder(nn.Module):
	def __init__(self, enc_dim=512, scene_hdim=512, stroke_hdim=512):
		super(RNN_Decoder, self).__init__()

		self.scene_hdim = scene_hdim
		self.stroke_hdim = stroke_hdim

		# hierarchical decoder
		self.scene_lstm = nn.LSTMCell(
			input_size=enc_dim+self.stroke_hdim, hidden_size=scene_hdim)
		self.stroke_lstm = nn.LSTM(
			input_size=self.scene_hdim+5, hidden_size=stroke_hdim, num_layers=2, batch_first=True)

		self.linear_output = nn.Linear(self.stroke_hdim, 5)

	def forward(self, enc_features, vector_sketch=None, batch_num_strokes=None, batch_stroke_len=None):
		""" 
			Input
				enc_feature: (nbatch x enc_dim) 
				vector_sketch: Tensor (nbatch, max_num_strokes, max_stroke_len, 5)
				batch_num_strokes: Tensor (nbatch, 1)
				batch_stroke_len: Tensor (nbatch, max_num_strokes)

		"""
		# input (vector_sketch[:, 0, 0, :])
		nbatch = enc_features.shape[0]

		prev_scene_token = Variable(torch.zeros((nbatch, self.stroke_hdim))).cuda()
		init_stroke_token = Variable(torch.FloatTensor([[0, 0, 1, 0, 0]]).repeat(nbatch, 1)).cuda()
		scene_h = Variable(torch.zeros((nbatch, self.scene_hdim))).cuda()
		scene_c = Variable(torch.zeros((nbatch, self.scene_hdim))).cuda()


		if batch_num_strokes is not None and batch_stroke_len is not None:
			max_num_strokes = batch_num_strokes.max()
			max_stroke_len = batch_stroke_len.max()
		else:
			max_num_strokes = 200
			max_stroke_len = 200

		output_coords = torch.zeros(nbatch, max_num_strokes, max_stroke_len, 5).cuda()

		for stroke_id in range(max_num_strokes):
			scene_ctx = torch.cat([enc_features, prev_scene_token], dim=-1)
			scene_h, scene_c = self.scene_lstm(scene_ctx, (scene_h, scene_c))
			# scene_h shape: (nbatch, scene_hdim)

			if (not self.training or torch.randn(1).item() > 0) and False:
				raise ValueError
				stroke_h = Variable(torch.zeros((1, nbatch, self.stroke_hdim))).cuda()
				stroke_c = Variable(torch.zeros((1, nbatch, self.stroke_hdim))).cuda()
				# prev_stroke_token = init_stroke_token * 1.0
				# prev_stroke_token = vector_sketch[:, stroke_id, 0, :]

				stroke_len = batch_stroke_len[:, stroke_id].max()
				for coord_id in range(stroke_len):
					# (nbatch, 1, scene_hdim+5)
					stroke_ctx =  torch.cat([scene_h, init_stroke_token], dim=-1).unsqueeze(1)
					_, (stroke_h, stroke_c) = self.stroke_lstm(stroke_ctx, (stroke_h, stroke_c))
					prev_scene_token = stroke_h[0]
					out_coords = self.linear_output(stroke_h[0])
					
					init_stroke_token[:, :2] = out_coords[:, :2].detach()
					init_stroke_token[:, 2:] = F.one_hot(
						out_coords[:, 2:].detach().argmax(dim=-1), num_classes=3).float()

					output_coords[:, stroke_id, coord_id, :] = out_coords

				# semi-teacher forcing
				init_stroke_token = vector_sketch[torch.arange(nbatch).cuda(),\
					stroke_id, batch_stroke_len[:, stroke_id]-1, :]


			# if self.training and False:
			else:
				stroke_len = batch_stroke_len[:,stroke_id]
				coords_inp = torch.cat([\
					init_stroke_token.unsqueeze(1),
					vector_sketch[:, stroke_id, :stroke_len.max(), :]], dim=1)

				# (nbatch, stroke_len, 256+5)
				stroke_inp = torch.cat([\
					scene_h.unsqueeze(1).repeat(1, stroke_len.max()+1, 1),
					coords_inp], dim=-1)

				packed = pack_padded_sequence(
					stroke_inp, 
					batch_stroke_len[:, stroke_id]+1,
					batch_first=True, enforce_sorted=False)

				# stroke_h shape: (1, nbatch, 5)
				packed_out, (stroke_h, stroke_c) = self.stroke_lstm(packed)
				prev_scene_token = stroke_h[-1].detach()

				out, _ = pad_packed_sequence(packed_out, batch_first=True)
				out_coords = self.linear_output(out)

				L =  out_coords.shape[1] - 1
				output_coords[:, stroke_id, :L, :] = out_coords[:, :L, :]

				init_stroke_token = vector_sketch[torch.arange(nbatch),\
					stroke_id, (stroke_len-1).clamp(min=0), :]

		return output_coords


if __name__ == '__main__':
    from options import opts
    from torchvision import transforms
    from dataloader import OursScene
    from utils import collate_fn, draw_tensor_sketch

    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = OursScene(opts, mode='train', transform=dataset_transforms)   

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=opts.batch_size, collate_fn=collate_fn)

    rnn_decoder = RNN_Decoder().cuda()
    enc_features = torch.randn(opts.batch_size, 512).cuda()

    for (raster_sketch, image, vector_sketch, batch_num_strokes, batch_stroke_len)  in dataloader:
        print ('shape of raster_sketch: {}, image: {}, vector_sketch: {},\
            batch_num_strokes: {}, batch_stroke_len: {}'.format(
            raster_sketch.shape, image.shape, vector_sketch.shape, batch_num_strokes.shape, batch_stroke_len.shape))

        vector_sketch = vector_sketch.cuda()
        rnn_decoder(enc_features, vector_sketch, batch_num_strokes, batch_stroke_len)

import os
import re
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import matplotlib.pyplot as plt



# Configurazioni costanti
IS_MULTISCALE = True
N_CLASS = 19
COLOR_MAP = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
			 (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
			 (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

inf_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.8]
data_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.290101, 0.328081, 0.286964],
						 [0.182954, 0.186566, 0.184475])
])



class Inference(object):
	def __init__(self, model_name, model_path):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.seg_model = self.__init_model(model_name, model_path)

	def __init_model(self, model_name, model_path):
		if model_name == 'MobileNetDenseASPP':
			from cfgs.MobileNetDenseASPP import Model_CFG
			from models.MobileNetDenseASPP import DenseASPP
		elif model_name == 'DenseASPP121':
			from cfgs.DenseASPP121 import Model_CFG
			from models.DenseASPP import DenseASPP
		elif model_name == 'DenseASPP169':
			from cfgs.DenseASPP169 import Model_CFG
			from models.DenseASPP import DenseASPP
		elif model_name == 'DenseASPP201':
			from cfgs.DenseASPP201 import Model_CFG
			from models.DenseASPP import DenseASPP
		elif model_name == 'DenseASPP161':
			from cfgs.DenseASPP161 import Model_CFG
			from models.DenseASPP import DenseASPP
		else:
			from cfgs.DenseASPP161 import Model_CFG
			from models.DenseASPP import DenseASPP

		model = DenseASPP(Model_CFG, n_class=N_CLASS, output_stride=8)
		self.load_weight(model, model_path)
		model.to(self.device)
		model.eval()
		return model


	def multiscale_inference(self, test_img):
		w, h = test_img.size
		pre_list = []
		
		for scale in inf_scales:
			new_size = (int(w * scale), int(h * scale))
			img_scaled = test_img.resize(new_size, Image.BICUBIC)
			
			# normal inference
			pre_list.append(self.single_inference(img_scaled, is_flip=False))
			
			# inference after spatial flip
			img_flip = img_scaled.transpose(Image.FLIP_LEFT_RIGHT)
			
			pre_flip = self.single_inference(img_flip, is_flip=False)
			
			# flip back the prediction to original orientation
			pre_flip_corrected = np.flip(pre_flip, axis=3).copy()
			pre_list.append(pre_flip_corrected)

		return np.mean(pre_list, axis=0)


	def single_inference(self, test_img, is_flip=False):
		input_tensor = data_transforms(test_img).unsqueeze(0).to(self.device)
		
		output = self.seg_model(input_tensor)

		orig_h, orig_w = input_tensor.shape[2], input_tensor.shape[3]
		output = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=True)

		output = F.log_softmax(output, dim=1)
		output_np = output.cpu().numpy()

		if is_flip:
			# horizontal flip back
			output_np = output_np[:, :, :, ::-1]

		return output_np
	

	def load_weight(self, model, model_path):
		print(f"Loading weights from {model_path}...")

		state_dict = torch.load(model_path, map_location=self.device)
		
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			# remove module prefix if it exists
			name = k[7:] if k.startswith('module.') else k
			
			name = re.sub(r'(norm|relu|conv)\.(\d+)', r'\1_\2', name)

			new_state_dict[name] = v
		
		# debug
		missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
		if missing:
			print(f"Missing keys: {missing}")
		if unexpected:
			print(f"Unexpected keys: {unexpected}")


	@staticmethod
	def pre_to_img(pre):
		result = pre.argmax(axis=1)[0]
		row, col = result.shape
		dst = np.zeros((row, col, 3), dtype=np.uint8)
		for i in range(N_CLASS):
			dst[result == i] = COLOR_MAP[i]
		dst = np.array(dst, dtype=np.uint8)
		dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
		return dst
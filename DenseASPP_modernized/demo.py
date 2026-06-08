#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
from PIL import Image
import cv2
from inference import Inference
import torch
import numpy as np


# constants
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DenseASPP modern inference code')
	parser.add_argument('--model_name', default='DenseASPP161', help='Model name (es. DenseASPP161)')
	parser.add_argument('--model_path', default='./weights/DenseASPP161.pth', help='File path .pth')
	parser.add_argument("--input_img", required=True, help="Path to the single input image")
	parser.add_argument("--vis_out", default="./outputs/vis_outputs", help="Directory to save colored masks")
	parser.add_argument("--raw_out", default="./outputs/raw_outputs", help="Directory to save raw class ids")
	parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
	args = parser.parse_args()
	
	# check if input file exists before loading the heavy network
	if not os.path.exists(args.input_img):
		print(f"Error: input image {args.input_img} not found")
		exit(1)

	# initialize inference engine
	infer = Inference(args.model_name, args.model_path)

	# ensure output subdirectories exist safely
	os.makedirs(args.vis_out, exist_ok=True)
	os.makedirs(args.raw_out, exist_ok=True)

	# process the single image
	img_name = os.path.basename(args.input_img)
	raw_img = Image.open(args.input_img).convert("RGB")

	with torch.no_grad():
		# run the network forward pass exactly once
		prediction_np = infer.single_inference(raw_img)

		# extract raw class ids (0-18) from the unique prediction array
		pred_ids = np.argmax(prediction_np, axis=1)[0].astype(np.uint8)

		# generate the colored mask using the same prediction array
		mask = infer.pre_to_img(prediction_np) 

	# save raw class predictions for miou evaluation
	raw_prediction_filename = os.path.join(args.raw_out, f"{os.path.splitext(img_name)[0]}.png")			
	cv2.imwrite(raw_prediction_filename, pred_ids)
	if not args.quiet:
		print(f"Saved: {raw_prediction_filename}")

	# save colored visual masks
	visualize_prediction_filename = os.path.join(args.vis_out, f"{img_name}")			
	cv2.imwrite(visualize_prediction_filename, mask)
	if not args.quiet:
		print(f"Saved: {visualize_prediction_filename}")
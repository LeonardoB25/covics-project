#!/usr/bin/env python
# -*- coding:utf-8 -*-

from inference import Inference
from PIL import Image
import argparse
import cv2
import os
import numpy as np
import random
import torch


# constants
RANDOM_SEED = 42
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DenseASPP modern inference code')
	parser.add_argument('--model_name', default='DenseASPP161', help='Model name (es. DenseASPP161)')
	parser.add_argument('--model_path', default='./weights/DenseASPP161.pth', help='File path .pth')
	parser.add_argument("--input_dir", required=True, help="Path to the input directory containing images")
	parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of images to process")
	parser.add_argument("--vis_out", default="./outputs/vis_outputs", help="Directory to save colored masks")
	parser.add_argument("--raw_out", default="./outputs/raw_outputs", help="Directory to save raw class ids")
	parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
	args = parser.parse_args()

	# check if input directory exists before loading the heavy network
	if not os.path.exists(args.input_dir):
		print(f"Error: input directory {args.input_dir} not found")
		exit(1)

	# gather and sort all valid images inside the target directory
	img_names = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(VALID_EXTENSIONS)])
	if not img_names:
		print(f"Error: no valid images found in {args.input_dir}")
		exit(1)

	# sampling logic
	if args.max_samples is not None:
		random.seed(RANDOM_SEED)
		n_to_sample = min(len(img_names), args.max_samples)
		img_names = random.sample(img_names, n_to_sample)

	# initialize inference engine
	infer = Inference(args.model_name, args.model_path)

	# ensure output subdirectories exist safely
	os.makedirs(args.vis_out, exist_ok=True)
	os.makedirs(args.raw_out, exist_ok=True)

	if not args.quiet:
		print(f"Starting batch inference for {len(img_names)} images...")

	# iterate through each image inside the target folder
	for img_name in img_names:
		input_path = os.path.join(args.input_dir, img_name)
		raw_img = Image.open(input_path).convert("RGB")

		with torch.no_grad():
			# run the network forward pass exactly once per image
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

		# save colored mask visualization
		vis_prediction_filename = os.path.join(args.vis_out, img_name)
		cv2.imwrite(vis_prediction_filename, mask)
		if not args.quiet:
			print(f"Saved: {vis_prediction_filename}")
			print(f"Processed: {img_name}")

	if not args.quiet:
		print("Inference process completed successfully.")
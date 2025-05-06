import cv2


import torch
import numpy as np


def recover_image( image_tensor, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5]):
	"""
	read a tensor and recover it to image in cv2 format
	args:
		image_tensor: [C, H, W] or [B, C, H, W]
	return:
		image_save: [B, H, W, C]
	"""
	if image_tensor.ndim == 3:
		image_tensor = image_tensor.unsqueeze(0)

	x = torch.mul(image_tensor, torch.FloatTensor(STD).view(3,1,1).to(image_tensor.device))
	x = torch.add(x, torch.FloatTensor(MEAN).view(3,1,1).to(image_tensor.device) )
	x = x.data.cpu().numpy()
	# [C, H, W] -> [H, W, C]
	image_rgb = np.transpose(x, (0, 2, 3, 1))
	# RGB -> BGR
	image_bgr = image_rgb[:, :, :, [2,1,0]]
	# float -> int
	image_save = np.clip(image_bgr*255, 0, 255).astype('uint8')

	return image_save


def draw_lm(image, landmarks, color= (0, 0, 255), radius = 20, print_idx=False):
	i = 0
	image_out = image.copy()
	for x,y in landmarks:
		# Radius of circle
		# Line thickness of 2 px
		thickness = -1
		image_out = cv2.circle(image_out, (int(x), int(y)), radius, color, thickness)
	
		if print_idx:
			image_out = cv2.putText(image_out,
				text=str(i),
				org=(int(x), int(y)),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=2.0,
				color=color,
				thickness=2,
				lineType=cv2.LINE_4)
		
		i += 1
	
	return image_out


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
	"""Draw gaze angle on given image with a given eye positions."""
	image_out = image_in.copy()
	(h, w) = image_in.shape[:2]
	length = w / 2.0
	pos = (int(h / 2.0), int(w / 2.0))
	if len(image_out.shape) == 2 or image_out.shape[2] == 1:
		image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
	dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
	dy = -length * np.sin(pitchyaw[0])
	cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
				   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
				   thickness, cv2.LINE_AA, tipLength=0.2)
	return image_out
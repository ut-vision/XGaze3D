import os
import numpy as np
import imageio
import cv2
import h5py
import math

def pitchyaw_to_vector(pitchyaws):
	r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

	Args:
		pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

	Returns:
		:obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
	"""
	n = pitchyaws.shape[0]
	sin = np.sin(pitchyaws)
	cos = np.cos(pitchyaws)
	out = np.empty((n, 3))
	out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
	out[:, 1] = sin[:, 0]
	out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
	return out
	
def vector_to_pitchyaw(vectors):
	"""Convert given gaze vectors to pitch (theta) and yaw (phi) angles."""
	n = vectors.shape[0]
	out = np.empty((n, 2))
	vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
	out[:, 0] = np.arcsin(vectors[:, 1])  # theta
	out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
	return out

def angular_error(a, b):
	"""Calculate angular error (via cosine similarity)."""
	a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
	b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

	ab = np.sum(np.multiply(a, b), axis=1)
	a_norm = np.linalg.norm(a, axis=1)
	b_norm = np.linalg.norm(b, axis=1)

	# Avoid zero-values (to avoid NaNs)
	a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
	b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

	similarity = np.divide(ab, np.multiply(a_norm, b_norm))

	return np.arccos(similarity) * 180.0 / np.pi

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
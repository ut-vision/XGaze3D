
import os
import os.path as osp
import cv2
import argparse
import math
import torch
import csv
import numpy as np
import json
import scipy.io
from glob import glob
from scipy.optimize import least_squares

from .label_transform import lm50_subset, lm68_subset, lm68_to_50


def project_3d_points(points3d, camera_matrix, camera_rotation, camera_translation, eps=1e-9):
	fx,fy,cx,cy = camera_matrix[0,0],camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]
	R = camera_rotation
	t = camera_translation

	points3d = points3d @ R.T + t.reshape(1,3)
	# landmarks = (landmarks - t.reshape(1,3))@R.T
	x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
	x_ = x / (z + eps)
	y_ = y / (z + eps)
	u = fx * x_ + cx
	v = fy * y_ + cy
	landmarks_pj = np.stack([u, v, z], axis=-1)
	return landmarks_pj



def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
	ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
	## further optimize
	if iterate:
		ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

	return rvec, tvec


def fun(x, face_model, cameras):
	hr =  np.array([ x[0], x[1], x[2] ]).reshape(3,1); ht = np.array([ x[3], x[4], x[5] ]).reshape(3,1)
	hR = cv2.Rodrigues(hr)[0]
	Points = (np.dot(hR, face_model.T) + ht).T

	proj_lm2d = []
	gt_lm2d = []
	for camera in list(cameras.values()):
		lm_gt, camera_matrix, camera_translation, camera_rotation = camera[0], camera[1], camera[2], camera[3]
		gt_lm2d.append(lm_gt)
		fx,fy,cx,cy = camera_matrix[0,0],camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]
		Points_camera =  Points @ camera_rotation.T + camera_translation.reshape(1,3)
		Points_camera = Points_camera/ Points_camera[:,[2]]
		points = Points_camera @ camera_matrix.T
		points = points[:,:2]

		proj_lm2d.append(points)
	proj_lm2d = np.array(proj_lm2d)
	gt_lm2d = np.array(gt_lm2d)
	return proj_lm2d.flatten() - gt_lm2d.flatten()


def optimize_headpose(face_model, cameras_info, good_cams, num_pts=6):
	"""
	args:
		face_model: the physical size reference face model ( shape = [n, 3], 3d landmarks in mm)
		cameras_info: {'cam00': [lm_gt, camera_matrix, camera_translation, camera_rotation, camera_distortion], 
						'cam01': [], ...}	
		good_cams: ['cam00', 'cam01', ..., 'cam17']
	return:
		rvec: rotation vector in shape (3,1)
		tvec: translation vector in shape (3,1)
	"""
	face_model = lm50_subset(face_model, num_pts)

	lm_gt0, camera_matrix0, camera_translation0, camera_rotation0, camera_distortion0 = cameras_info[good_cams[0]]
	hr0, ht0 = estimateHeadPose(lm_gt0, face_model.reshape(-1, 1, 3), camera_matrix0, camera_distortion0)
	hr0 = hr0.flatten()
	ht0 = ht0.flatten()
	x0_init = np.array([hr0[0], hr0[1], hr0[2], ht0[0], ht0[1], ht0[2]])

	solution = least_squares(fun, x0_init, args=(face_model, cameras_info))
	rvec, tvec = solution.x[:3].reshape(3,1), solution.x[3:6].reshape(3,1)
	return rvec, tvec

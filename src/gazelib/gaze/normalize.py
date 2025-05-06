# -*- coding: utf-8 -*-
"""
######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Any publications arising from the use of this software, including but
not limited to academic journal and conference publications, technical
reports and manuals, must cite at least one of the following works:

Revisiting Data Normalization for Appearance-Based Gaze Estimation
Xucong Zhang, Yusuke Sugano, Andreas Bulling
in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018
######################################################################################################################################
"""

import os
import cv2
import numpy as np
import csv
# import dlib

def resize_landmarks(lm68, focal_norm, src_size, tgt_size):
	'''
	resize normalized landmarks back to original landmarks
	'''
	cam_224 = np.array([
		[focal_norm, 0, tgt_size[0]/2],
		[0, focal_norm, tgt_size[1]/2],
		[0, 0, 1.0],
	])
	S = np.array([ # scaling matrix
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, src_size[0]/tgt_size[0]],
	])
	
	# S = np.eye(3)
	cam_448 = np.array([
		[focal_norm, 0, src_size[0]/2],
		[0, focal_norm, src_size[1]/2],
		[0, 0, 1.0],
	])
	W = cam_224 @ S @ np.linalg.inv(cam_448)
	num_point = lm68.shape[0]
	landmarks_warped = cv2.perspectiveTransform(lm68.reshape(-1,1,2).astype('float32'), W)
	landmarks_warped = landmarks_warped.reshape(num_point, 2)
	return landmarks_warped
	
	

	
def normalize_woimg(landmarks, focal_norm, distance_norm, roi_size, center, hr, ht, cam, gc=None):
	center = center.reshape(3,1)
	## universal function for data normalization
	hR = cv2.Rodrigues(hr)[0] # rotation matrix

	## ---------- normalize image ----------
	distance = np.linalg.norm(center) # actual distance between eye and original camera

	z_scale = distance_norm/distance
	cam_norm = np.array([
		[focal_norm, 0, roi_size[0]/2],
		[0, focal_norm, roi_size[1]/2],
		[0, 0, 1.0],
	])
	S = np.array([ # scaling matrix
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, z_scale],
	])

	hRx = hR[:,0]
	forward = (center/distance).reshape(3)
	down = np.cross(forward, hRx)
	down /= np.linalg.norm(down)
	right = np.cross(down, forward)
	right /= np.linalg.norm(right)
	R = np.c_[right, down, forward].T # rotation matrix R

	W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix

	## ---------- normalize rotation ----------
	hR_norm = np.dot(R, hR) # rotation matrix in normalized space
	# hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors

	## ---------- normalize gaze vector ----------
	gc_normalized = None

	num_point = landmarks.shape[0]
	landmarks_warped = cv2.perspectiveTransform(landmarks.reshape(-1,1,2).astype('float32'), W)
	landmarks_warped = landmarks_warped.reshape(num_point, 2)
	if gc is not None:
		gc_normalized = gc.reshape((3,1)) - center # gaze vector
		# For modified data normalization, scaling is not applied to gaze direction (only R applied).
		# For original data normalization, here should be:
		# "M = np.dot(S,R)
		# gc_normalized = np.dot(R, gc_normalized)"
		gc_normalized = np.dot(R, gc_normalized)
		gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)

	return [None, R, hR_norm, gc_normalized, landmarks_warped, W]

	
def normalize(img, landmarks, focal_norm, distance_norm, roi_size, center, hr, ht, cam, gc=None):
	center = center.reshape(3,1)
	## universal function for data normalization
	hR = cv2.Rodrigues(hr)[0] # rotation matrix

	## ---------- normalize image ----------
	distance = np.linalg.norm(center) # actual distance between eye and original camera

	z_scale = distance_norm/distance
	cam_norm = np.array([
		[focal_norm, 0, roi_size[0]/2],
		[0, focal_norm, roi_size[1]/2],
		[0, 0, 1.0],
	])
	S = np.array([ # scaling matrix
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, z_scale],
	])

	hRx = hR[:,0]
	forward = (center/distance).reshape(3)
	down = np.cross(forward, hRx)
	down /= np.linalg.norm(down)
	right = np.cross(down, forward)
	right /= np.linalg.norm(right)
	R = np.c_[right, down, forward].T # rotation matrix R
	W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix

	# if img is not None:
	# 	img_warped = cv2.warpPerspective(img, W, roi_size) # image normalization
	# else:
	# 	img_warped = None
	
	img_warped = cv2.warpPerspective(img, W, roi_size) # image normalization
	## ---------- normalize rotation ----------
	hR_norm = np.dot(R, hR) # rotation matrix in normalized space
	# hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors

	## ---------- normalize gaze vector ----------
	gc_normalized = None
	num_point = landmarks.shape[0]
	landmarks_warped = cv2.perspectiveTransform(landmarks.reshape(-1,1,2).astype('float32'), W)
	landmarks_warped = landmarks_warped.reshape(num_point, 2)
	if gc is not None:
		gc_normalized = gc.reshape((3,1)) - center # gaze vector
		# For modified data normalization, scaling is not applied to gaze direction (only R applied).
		# For original data normalization, here should be:
		# "M = np.dot(S,R)
		# gc_normalized = np.dot(R, gc_normalized)"
		gc_normalized = np.dot(R, gc_normalized)
		gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)

	return [img_warped, R, hR_norm, gc_normalized, landmarks_warped, W]

def normalize_face(img, face, hr, ht, cam, gc=None):
	## normalized camera parameters
	focal_norm = 960 # focal length of normalized camera
	distance_norm = 600 # normalized distance between eye and camera
	roi_size = (224, 224) # size of cropped eye image

	## compute estimated 3D positions of the landmarks
	ht = ht.reshape((3,1))
	hR = cv2.Rodrigues(hr)[0] # rotation matrix
	Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks
	# fm = np.mean(Fc, axis=1).reshape((3,1)) # center of facial landmarks
	two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
	nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
	# get the face center
	face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
	# face_center = np.mean(Fc, axis=1).reshape((3,1)) 
	return normalize(img, focal_norm, distance_norm, roi_size, face_center, hr, ht, cam, gc)

def normalize_eye(img, face, hr, ht, cam, gc=None):
	## normalized camera parameters
	focal_norm = 960 # focal length of normalized camera
	distance_norm = 600 # normalized distance between eye and camera
	roi_size = (60, 36) # size of cropped eye image

	## compute estimated 3D positions of the landmarks
	ht = ht.reshape((3,1))
	hR = cv2.Rodrigues(hr)[0] # rotation matrix
	Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks
	re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
	le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye

	## normalize each eye
	data = [
		normalize(img, focal_norm, distance_norm, roi_size, re, hr, ht, cam, gc),
		normalize(img, focal_norm, distance_norm, roi_size, le, hr, ht, cam, gc)
	]
	return data

def load_calibration(calib_path):
	## load calibration data, these paramters are expected to be obtained by camera calibration functions in OpenCV
	fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
	camera_matrix = fs.getNode('camera_matrix').mat()
	camera_distortion = fs.getNode('dist_coeffs').mat()
	return camera_matrix, camera_distortion

def load_facemodel(model_path):
	# load the generic face model, which includes 6 facial landmarks: four eye corners and two mouth corners
	fs = cv2.FileStorage(model_path, cv2.FILE_STORAGE_READ)
	face_model = fs.getNode('face_model').mat()
	return face_model

def read_image(img_path, camera_matrix, camera_distortion):
	# load input image and undistort
	img_original = cv2.imread(img_path)
	img = cv2.undistort(img_original, camera_matrix, camera_distortion)

	return img

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
	ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

	## further optimize
	if iterate:
		ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

	return rvec, tvec

def detect_landmark(img, detector_path, predictor_path):
	## obtain facial landmarks using dlib
	detector = dlib.cnn_face_detection_model_v1(detector_path)
	dets = detector(img, 0)

	if len(dets) == 0:
		return None

	predictor = dlib.shape_predictor(predictor_path)
	shape = predictor(img, dets[0].rect)

	## extract required keypoints
	landmarks = np.array([
		[shape.part(36).x, shape.part(36).y],
		[shape.part(39).x, shape.part(39).y],
		[shape.part(42).x, shape.part(42).y],
		[shape.part(45).x, shape.part(45).y],
		[shape.part(48).x, shape.part(48).y],
		[shape.part(54).x, shape.part(54).y]
	])

	return landmarks


def read_landmark(img_path):
	img_file = img_path.split(os.path.sep)[-1]
	day = img_path.split(os.path.sep)[-2]
	person = img_path.split(os.path.sep)[-3]
	person_path = os.path.split(os.path.split(img_path)[0])[0]

	person_txt = os.path.join(person_path, person+'.txt')
	index = os.path.join(day,img_file)
	print(person_txt)
	print(index)

	with open(person_txt) as f:
		data = f.readlines()
	reader = csv.reader(data)
	p = {}
	for row in reader:
		words = row[0].split()
		p[words[0]] = words[1:]
	landmarks = np.array([int(i) for i in p[index][2:14]]).reshape((6,2))
	return landmarks

# def process_image(img_path, detector_path, predictor_path, camera_matrix, camera_distortion, face_model, gc=None):
#     # read input image
#     img = read_image(img_path, camera_matrix, camera_distortion)

#     # detect facial landmarks
#     landmarks = detect_landmark(img, detector_path, predictor_path)

#     if landmarks is not None:
#         # estimate head pose
#         hr, ht = estimateHeadPose(face_model, landmarks, camera_matrix, camera_distortion)

#         # data normalization for left and right eye image
#         normalized_eyes = normalize_eye(img, face_model, hr, ht, camera_matrix, gc)

#         # data normalization for full face
#         normalized_face = normalize_face(img, face_model, hr, ht, camera_matrix, gc)

#         # return a list of [reye, leye, face]
#         return normalized_eyes + [normalized_face]

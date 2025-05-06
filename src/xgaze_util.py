
import csv
import os
import os.path as osp
import cv2
import numpy as np
from glob import glob
from rich.progress import track
from utils.preprocess import crop_img
from omegaconf import OmegaConf
import torch
import face_alignment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
except:
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)


def draw_lm(image, landmarks, color= (0, 0, 255), radius = 20):
	image_out = image.copy()
	for x,y in landmarks:
		# Radius of circle
		# Line thickness of 2 px
		thickness = -1
		image_out = cv2.circle(image_out, (int(x), int(y)), radius, color, thickness)
	return image_out


class ETHXGaze(object):
	def __init__(self, base_dir, 
					initialize_csv=True):
		self.face_model = np.loadtxt( osp.join( osp.dirname(osp.realpath(__file__)),'face_model.txt')  )

		data_dir = osp.join(base_dir, 'data')
		self.train_dir = osp.join(data_dir, 'train')
		self.annotation_dir = osp.join(data_dir, 'annotation_train')
		self.cameras = ['cam00', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05', 
						'cam06', 'cam07', 'cam08', 'cam09', 'cam10', 'cam11', 
						'cam12', 'cam13', 'cam14', 'cam15', 'cam16', 'cam17']

		self.lm2d_accurate_cams = [c for c in self.cameras if c not in ['cam05', 'cam13',] ]
		self.lm2d_accurate_cams = [c for c in self.cameras ]

		subjects_pathlist = sorted( glob(self.annotation_dir + '/subject*') )
		self.subjects_list = [osp.basename(x).split('.')[0] for x in subjects_pathlist]
		if initialize_csv:
			self.csv_all = {
				subject: read_csv_as_dict( osp.join(self.annotation_dir, subject+'.csv')) for subject in track(self.subjects_list, description='pre reading csv')
			}

		
		self.calibration_dir = osp.join(base_dir,'calibration', 'cam_calibration')
		self.original_calibration = self.read_cam_xml_to_dict(self.calibration_dir)


		self.light_meta = OmegaConf.load(osp.join(base_dir, 'light_meta.yaml'))
	
		self.updated_cam_calibration_dir = osp.join(base_dir, 'avg_cams_final')
	
		self.updated_annotation_dir = osp.join(data_dir, 'annotation_updated')
		
	def read_lm(self, subject, frame, img_name):
		sub_dict = self.csv_all[subject]
		lm_gt, gc, _, _ = read_lm_gc(sub_dict, os.path.join(frame, img_name))
		return lm_gt

	def read_cam_xml_to_dict(self, cal_dir):
		calibration = {}
		for cam in self.cameras:
			camera_path = os.path.join(cal_dir, cam +'.xml')
			camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(camera_path)
			calibration[cam] = {
				'camera_matrix': camera_matrix,
				'camera_distortion': camera_distortion,
				'camera_translation': camera_translation,
				'camera_rotation': camera_rotation
			}
		return calibration
	
	def detect_lm(self, subject, frame, img_name, img):
		lm_rough = self.read_lm(subject, frame, img_name) # print('read rough lm: ', lm_rough)

		## since the img is too large, we first crop the image (roughly), and resize it to 224x224
		## the cropping/resize should also be represented as a  transformation matrix, and then this should be applied on the landmarks to recover the original scale
		roi_box = simple_parse_roi_box_from_landmark(lm_rough.T, cam=img_name.split('.')[0])
		sx, sy, ex, ey = roi_box
		img = crop_img(img, roi_box)
		img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
		scale_x = (ex - sx) / (224 - 1)
		scale_y = (ey - sy) / (224 - 1)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		lm_crop_detected_list = fa.get_landmarks(img_rgb)
		if lm_crop_detected_list is not None and len(lm_crop_detected_list) > 0:
			lm_crop_detected = lm_crop_detected_list[0]

			crop_params_inv = np.array([[scale_x, 0, sx ],
									[0, scale_y, sy ],
									[0, 0, 1]])
			
			lm_enlarge = (np.c_[lm_crop_detected, np.ones(lm_crop_detected.shape[0])] @ crop_params_inv.T)[:, :2]
			lm_enlarge = np.round(lm_enlarge).astype(int)
		else:
			lm_enlarge = None
		return lm_enlarge, lm_rough
	
def read_image(img_path, camera_matrix, camera_distortion):
	img_original = cv2.imread(img_path)
	img = cv2.undistort(img_original, camera_matrix, camera_distortion)
	return img



def read_csv_as_dict(csv_path):
	with open(csv_path, newline='') as csvfile:
		data = csvfile.readlines()
	reader = csv.reader(data)
	sub_dict = {}
	for row in reader:
		frame = row[0]
		cam_index = row[1]
		sub_dict[frame+'/'+cam_index] = row[2:]
	return sub_dict
	
def read_lm_gc(sub_dict, index):
	"""index is e.g. frame0001/cam00.JPG"""
	gaze_point_screen = np.array([int(float(i)) for i in sub_dict[index][0:2]])
	gaze_point_cam = np.array([float(i) for i in sub_dict[index][2:5]])
	head_rotation_cam = np.array([float(i) for i in sub_dict[index][5:8]])
	head_translation_cam = np.array([float(i) for i in sub_dict[index][8:11]])
	lm_2d = np.array([int(float(i)) for i in sub_dict[index][11:]]).reshape(68,2)
	return lm_2d, gaze_point_cam, head_rotation_cam, head_translation_cam

def read_xml(xml_path):
	if not os.path.isfile(xml_path):
		print('no camera calibration file is found.')
		## instead of exit, return an exception
		raise FileNotFoundError("No camera calibration file is found.")
	try:
		fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
		camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
		camera_distortion = fs.getNode('Distortion_Coefficients').mat()
		camera_translation = fs.getNode('cam_translation').mat()
		camera_rotation = fs.getNode('cam_rotation').mat()
	except:
		print('the bad xml file is: ', xml_path)
		
	return camera_matrix, camera_distortion, camera_translation, camera_rotation


## this is the simple version simply for reducing landmark detection time
def simple_parse_roi_box_from_landmark(pts, cam=None):
	from math import sqrt
	"""calc roi box from landmark"""
	bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
	
	center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

	temp_radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
	radius = 1.4 * temp_radius
	bbox = [center[0] - radius,   # left
			center[1] - 1.5 * radius,   # up
			center[0] + radius,   # right
			center[1] + 0.5 * radius ]  # bottom

	llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
	center_x = (bbox[2] + bbox[0]) / 2
	center_y = (bbox[3] + bbox[1]) / 2

	roi_box = [0] * 4
	roi_box[0] = center_x - llength / 2
	roi_box[1] = center_y - llength / 2
	roi_box[2] = roi_box[0] + llength
	roi_box[3] = roi_box[1] + llength

	return roi_box

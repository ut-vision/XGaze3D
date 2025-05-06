import os, sys, copy
import os.path as osp
import argparse
import shutil
import time
import numpy as np
import cv2
from omegaconf import OmegaConf
from glob import glob
from datetime import datetime
from rich.progress import track
from xgaze_util import read_image, read_xml, ETHXGaze
from utils.preprocess import remove_background, crop_img, parse_roi_box_from_landmark
from gazelib.utils.color_text import print_green, print_yellow, print_magenta, print_cyan, print_red
import multiprocessing
from read_agisoft import readFileXGaze, readFileAgisoft, scaleAgisoftResult, transform_camera_from_agisoft_to_cv2, transform_camera_from_cv2_to_agisoft

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import Metashape

CAM_NUM = 18
NUM_KPTS_TO_USE=50


def get_frames_before(frames, target_frame_idx):
	idx = min(len(frames), target_frame_idx)
	return frames[:idx]
	


class XGaze3DPipeline(object):
	def __init__(self, config):
		self.config = config
		self.subjects = OmegaConf.load(config.sub_grp)['group']
		self.xgaze = ETHXGaze(base_dir=config.xgaze_basedir)
		self.STD_SIZE = int(config.STD_SIZE) if config.STD_SIZE != 'full' else 'full'


		self.save_3d_model = config.save_3d_model
		self.resize_factor_color = config.resize_factor_color
		self.resize_factor_depth = config.resize_factor_depth

		self.output_dir = osp.join(config.output_path, f'{self.STD_SIZE}_data')

		self.cal_dir = self.xgaze.calibration_dir
		self.updated_cam_calibration_dir = self.xgaze.updated_cam_calibration_dir

		self.light_metadata = self.xgaze.light_meta

	
	
	
	def write_error(self, text, out_txt_path):
		print_red('''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!''') 
		print_red(text) 
		print_red('''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!''') 
		with open(out_txt_path, 'a') as f:
			f.write( text + '\n')
	
	def print_progress(self, text):
		print_cyan(' ///////////////////////////////////////////////////////////////// ') 
		print_cyan(' //////////////// ' + text) 
		print_cyan(' ///////////////////////////////////////////////////////////////// ') 

		
	def build_depth_maps(self, chunk):
		# filter -> filter_mode
		self.print_progress('build depth maps')
		chunk.buildDepthMaps(downscale=self.config.resize_factor_depth, filter_mode=Metashape.MildFiltering)
		
	
	def build_dense_cloud(self, chunk):
		self.print_progress(' build dense cloud ')
		chunk.buildPointCloud(point_colors=True, point_confidence=True) ## name updated in Metashape 2.0.1


	def build_model(self, chunk):
		self.print_progress(' build model ')
		chunk.buildModel(surface_type=Metashape.Arbitrary, source_data=Metashape.DepthMapsData,
							interpolation=Metashape.EnabledInterpolation, face_count=Metashape.HighFaceCount,
							vertex_colors=True)
		chunk.smoothModel(strength=10)  # choose the how much you want to smooth the model
	
	
	
	def build_model_with_timeout(self, chunk, timeout):
		self.print_progress(' build model ')
		process = multiprocessing.Process(target=chunk.buildModel(surface_type=Metashape.Arbitrary, source_data=Metashape.DepthMapsData,
							interpolation=Metashape.EnabledInterpolation, face_count=Metashape.HighFaceCount,
							vertex_colors=True))
		process.start()
		process.join(timeout)

		if process.is_alive():
			process.terminate()
			process.join()
			raise TimeoutError()
		

	def build_uv(self, chunk):
		self.print_progress(' build uv ')
		# mapping -> mapping_mode
		chunk.buildUV()

	
	def build_uv_with_timeout(self, chunk, timeout):
		self.print_progress(' build uv ')
		process = multiprocessing.Process(target=chunk.buildUV())
		process.start()
		process.join(timeout)

		if process.is_alive():
			process.terminate()
			process.join()
			raise TimeoutError()
		

	def build_texture(self, chunk):
		self.print_progress(' build texture ')
		# blending -> blending_mode
		# size -> texture_size
		chunk.buildTexture()  # the default texture size is 8192 * 8192

	def save_agi_cams(self, chunk, tag):
		cam_tag_save_path = os.path.join(self.temp_cam_dir, tag)
		os.makedirs(cam_tag_save_path, exist_ok=True)
		print(' save camera parameters in agisoft format to: ', os.path.join(cam_tag_save_path, 'cam_info_agisoft.xml'))
		chunk.exportCameras(os.path.join(cam_tag_save_path, 'cam_info_agisoft.xml'), format=Metashape.CamerasFormatXML)
		### no need to save opencv format camera here

	def transform_cameras_and_object(self, chunk, inverse=False):
		camera = chunk.cameras[0]
		# rotate it to be -1 for y and z axis since the default reconstruction is like that
		# if we do not have the -1 terms, it will fully align the cam0
		R = Metashape.Matrix().Rotation(camera.transform.rotation() * Metashape.Matrix().Diag([1., -1., -1.]))
		origin = (-1) * camera.center

		origin = R.inv().mulp(origin)
		rever_mat = Metashape.Matrix().Translation(origin) * R.inv()
		# do the transform on the object
		chunk.transform.matrix = rever_mat * chunk.transform.matrix
		# do the transform on all cameras
		# for cam_id in range(0, CAM_NUM):
		# 	print('cam_id: ', cam_id)
		# 	print('chunk.cameras[cam_id].transform: ', chunk.cameras[cam_id].transform)
		for cam_id in range(0, CAM_NUM):
			if chunk.cameras[cam_id].transform is not None:
				chunk.cameras[cam_id].transform = rever_mat * chunk.cameras[cam_id].transform
		return rever_mat


	def write_xgaze_xml(self, xml_tosave:dict, xml_file_path:str):
		"""write the camera calibration file (xgaze format)
		args:
			cam_dict: dict of camera calibration
			cam_path_frame: path to the camera calibration file, e.g., .../subject0000/frame0000/cam01.xml	
		"""
		fs = cv2.FileStorage(xml_file_path, flags=1)
		for k, v in xml_tosave.items():
			fs.write(name=k, val=v)
		fs.release()

	########################################################################################################################
	### Above are basic functions 
	########################################################################################################################

			
	def run(self):
		""" loop through subject / frame"""
		for subject in self.subjects:
			frames_of_this_subject = [osp.basename(x) for x in sorted(glob(osp.join(self.xgaze.train_dir, subject, 'frame*')))]
			## we only use the full-light frames because the dark images can not be reconstructed well
			if self.light_metadata is not None:
				full_light_frame_end = self.light_metadata[subject]['full_light_frame_end_idx']
				frames_remained = get_frames_before(frames_of_this_subject, full_light_frame_end)
			else:
				frames_remained = frames_of_this_subject.copy()
			self.current_subject = subject

			print_yellow(" {} - frames_remained: ".format(subject), frames_remained)
			for frame in track(frames_remained, description="Processing {} ".format(subject)):
				print_green('the current subject - frame is: {} - {}'.format(subject, frame) ) 
				self.current_frame = frame
				start_time = time.time()
				self.forward()
				print(" the total time for this frame is: ", time.time() - start_time)


	def forward(self):
		subject = self.current_subject
		frame = self.current_frame
		save_path_frame = osp.join(self.output_dir, subject, frame)
		os.makedirs(save_path_frame, exist_ok=True)
		self.temp_cam_dir = os.path.join(save_path_frame, 'camera_temp')
		self.temp_input_dir = os.path.join(save_path_frame,'input_temp')
		os.makedirs(self.temp_cam_dir, exist_ok=True)
		os.makedirs(self.temp_input_dir, exist_ok=True)
		""" forward 1 frame
			save_path_frame: <output>/subject0000/frame0000
				/camera_temp: the intermediate camera parameters, NOTE: will be deleted after the processing of this frame
				/input_temp: the intermediate dir storing the input for agisoft NOTE: will be deleted after the processing of this frame
					/cam00.jpg: the (cropped) and background-removed image
					/cam00_lm.txt: the txt file storing the landmark points, if not crop, this is just copying from the official csv annotation
					/cam00.xml: the cropped&resized camera parameters, if not crop, this is just copying the official cam00.xml 
				/mvs3d.obj
				/mvs3d.mtl
				/mvs3d.jpg
		"""

		doc = Metashape.Document()
		chunk = doc.addChunk()

		# doc_save_name = os.path.join(save_path_frame, frame + '.psz')
		# doc.save(doc_save_name)
		# self.info_this_frame = {
		# 	'subject': subject,
		# 	'frame': frame,
		# 	'img':[],
		# 	'cal_dir': self.cal_dir,
		# 	'lm_gt': [], ## list of (N, 2)
		# 	'img_crop': [],
		# 	'cal_dir_crop': None,
		# 	'lm_crop': [],   ## list of (N, 2)
		# 	'rvec': None,   ## (3,)
		# 	'tvec': None,   ## (3,)
		# 	'Fc': None,   ## (N, 3)
		# }
		
		## (crop & resize) and remove the green background
		photo_list = self.image_preprocess() # return a list of images
		chunk.addPhotos(photo_list[:CAM_NUM])

		self.initialize_cameras(chunk)

		self.print_progress( ' Match photo ')
		chunk.matchPhotos(downscale=self.config.resize_factor_color, keypoint_limit=200000, tiepoint_limit=20000, keypoint_limit_per_mpx=2000)
		self.print_progress( ' Align photos ')
		try:
			## chunk.alignCameras( ) # no need to run this because we are already using the updated cameras
			chunk.triangulateTiePoints()
			# self.save_agi_cams(chunk, 'raw')
		except:
			self.write_error( text='{} / {} : alignCameras failed'.format(subject, frame), 
							  out_txt_path= osp.join(self.output_dir, 'error.log'))
			return

		if self.save_3d_model:
			self.print_progress( ' Build 3D model (obj)  ')

			methods_3dmodel = {
				'build_depth_maps': 'build_depth_maps error',
				'build_dense_cloud': 'build_dense_cloud error',
				'build_model': 'build_model error',
				'build_uv': 'build_uv error',
				'build_texture': 'build_texture error'
			}

			for method_name, error_message in methods_3dmodel.items():
				try:
					if method_name == 'build_model':
						try:
							self.build_model_with_timeout(chunk, timeout=500)  # 500s timeout
						except TimeoutError:
							error_text = f'{subject} / {frame}: {error_message} (TimeoutError)'
							self.write_error(text=error_text, out_txt_path=osp.join(self.output_dir, 'error.log'))
							print( print_red('go to next frame -------------------------------------------------------------------------------------'))
							return
					elif method_name == 'build_uv':
						try:
							self.build_uv_with_timeout(chunk, timeout=500)  # 500s timeout
						except TimeoutError:
							error_text = f'{subject} / {frame}: {error_message} (TimeoutError)'
							self.write_error(text=error_text, out_txt_path=osp.join(self.output_dir, 'error.log'))
							print( print_red('go to next frame -------------------------------------------------------------------------------------'))
							return  # Skip to the next method
					
					else:
						method = getattr(self, method_name)
						method(chunk=chunk)
				except Exception as e:
					error_text = f'{subject} / {frame}: {error_message} ({str(e)})'
					self.write_error(text=error_text, out_txt_path=osp.join(self.output_dir, 'error.log'))
					return

		# print('/////////////// transform cam00 to origin (all other cameras and object are tranformed the same way)  /////////////// ')
		try:
			rever_mat = self.transform_cameras_and_object(chunk=chunk)
		except Exception as e:
			error_text = f'{subject} / {frame}: transform_cameras_and_object error ({str(e)})'
			self.write_error(text=error_text, out_txt_path=osp.join(self.output_dir, 'error.log'))
			return
		
		self.save_agi_cams(chunk, 'updated')
		self.post_process(save_path_frame, chunk=chunk)

		print_green('/////////////////////////  this frame runs successfully  ////////////////////////// ') 
		print_magenta('/////////////////////////  this frame runs successfully  ////////////////////////// ') 
		try:
			shutil.rmtree(self.temp_input_dir)
			shutil.rmtree(self.temp_cam_dir)
		except OSError as e:
			print(f"Error: {e.strerror}")
		



	def image_preprocess(self):
		subject = self.current_subject
		frame = self.current_frame
		
		updated_calibrations = self.xgaze.read_cam_xml_to_dict( osp.join(self.updated_cam_calibration_dir, subject) )
		print(" read from the 1st round average calibration: ", osp.join(self.updated_cam_calibration_dir, subject))
		photo_list = list()
		for i, c in enumerate(self.xgaze.cameras):
			img_name = c + '.JPG'
			img_path = os.path.join(self.xgaze.train_dir, subject, frame, img_name)
			camera_matrix = updated_calibrations[c]['camera_matrix']
			camera_distortion = updated_calibrations[c]['camera_distortion']
			camera_translation = updated_calibrations[c]['camera_translation']
			camera_rotation = updated_calibrations[c]['camera_rotation']
			img = read_image(img_path, camera_matrix, camera_distortion)
			if c in ['cam03', 'cam06','cam13']:
				img = cv2.rotate(img, cv2.ROTATE_180)
			# lm_gt = self.xgaze.read_lm(subject,frame, img_name)
			lm_gt, lm_gt_read = self.xgaze.detect_lm(subject,frame, img_name, img)
			if lm_gt is None:
				self.write_error( text='{} / {} / {} : landmark detection error'.format(subject, frame, img_name), out_txt_path= osp.join(self.output_dir, 'no_landmarks.log'))
				lm_gt = lm_gt_read.copy()
			
			# self.info_this_frame['img'].append(img)
			# self.info_this_frame['lm_gt'].append(lm_gt)

			if self.STD_SIZE != 'full':
				roi_box = parse_roi_box_from_landmark(lm_gt.T, cam=c)
				sx, sy, ex, ey = roi_box
				img = crop_img(img, roi_box)
				img = cv2.resize(img, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
				scale_x = (ex - sx) / (self.STD_SIZE - 1)
				scale_y = (ey - sy) / (self.STD_SIZE - 1)
				crop_params = np.array([[1 / scale_x, 0, -sx / scale_x],
										[0, 1 / scale_y, -sy / scale_y],
										[0, 0, 1]])

				camera_matrix_resize = crop_params @ camera_matrix
				lm_crop = (np.c_[lm_gt, np.ones(lm_gt.shape[0])] @ crop_params.T)[:, :2]
				np.savetxt(osp.join(self.temp_input_dir, '{}_lm.txt'.format(c)), lm_crop)
				

				xml_tosave = {
					'image_Width': img.shape[1],
					'image_Height': img.shape[0],
					'Camera_Matrix': camera_matrix_resize,
					'Distortion_Coefficients': camera_distortion,
					'cam_translation': camera_translation,
					'cam_rotation': camera_rotation
				}
				self.write_xgaze_xml(xml_tosave, os.path.join(self.temp_input_dir, '{}.xml'.format(c)) )

				# self.info_this_frame['lm_crop'].append(lm_crop)
				# self.info_this_frame['img_crop'].append(img)
			
			else:
				np.savetxt(osp.join(self.temp_input_dir, '{}_lm.txt'.format(c)), lm_gt)
				print('//// write the original camera calibration to the temp dir (they are just a replicated files)')
				xml_tosave = {
					'image_Width': img.shape[1],
					'image_Height': img.shape[0],
					'Camera_Matrix': camera_matrix,
					'Distortion_Coefficients': camera_distortion,
					'cam_translation': camera_translation,
					'cam_rotation': camera_rotation
				}
				self.write_xgaze_xml(xml_tosave, os.path.join(self.temp_input_dir, '{}.xml'.format(c)) )
		

			img = remove_background(img)
			img_path = os.path.join(self.temp_input_dir, img_name)
			cv2.imwrite(img_path, img)
			if osp.exists(img_path):
				photo_list.append(img_path)
				print('add image ', img_path)
		return photo_list

	def initialize_cameras(self, chunk, ):

		self.print_progress('load cameras')
		# clear all old sensors
		# while len(chunk.sensors) > 0:
		# 	chunk.remove(chunk.sensors[0])
		original_sensor_size = len(chunk.sensors)  # there are some exist sensor
		for num_i in range(0, CAM_NUM):
			print('camera id: ', num_i)
			camera = chunk.cameras[num_i]
			if num_i < original_sensor_size:
				new_sensor = chunk.sensors[num_i]
			else:
				new_sensor = chunk.addSensor()
			new_sensor.width = camera.photo.image().width
			new_sensor.height = camera.photo.image().height
			new_sensor.label = camera.label
			calib = Metashape.Calibration()
			load_filename = "cam{0:02d}.xml".format(num_i)

			_, _, camera_translation, camera_rotation = read_xml(os.path.join(self.temp_input_dir, load_filename))
			cam_transform = transform_camera_from_cv2_to_agisoft(camera_rotation, camera_translation)
			camera.transform = Metashape.Matrix(cam_transform.tolist())

			calib.load(os.path.join(self.temp_input_dir, load_filename),
						format=Metashape.CalibrationFormat.CalibrationFormatOpenCV)
			print('load file from ', os.path.join(self.temp_input_dir, load_filename) )

			new_sensor.user_calib = calib
			print('/////////////////////////////////////////////////////////////////////////////////////')
			print('new_sensor.user_calib f : ', new_sensor.user_calib.f)
			print('new_sensor.user_calib cx : ', new_sensor.user_calib.cx)
			print('new_sensor.user_calib cy : ', new_sensor.user_calib.cy)
			print('sensor type: ', new_sensor.type)
			print('sensor height: [{}] - width: [{}] '.format( new_sensor.height ,new_sensor.width))
			print('sensor fixed_params: ', new_sensor.fixed_params)
			new_sensor.fixed = True  # if fix the intrisnic parameters
			print('sensor fixed_params: ', new_sensor.fixed_params)
			camera.sensor = new_sensor  # if we want to have each camera has different calibration parameters, we must create new sensor for each cmaera
			print('-------------------------------------------------------------------------------------------')
			




		

	def post_process(self, save_path_frame, chunk=None):
		"""
		this is the final processing that transform the agisoft output camera and obj to opencv format (aligned with xgaze)
		args:
			save_path_frame: the path where the agisoft cameras and objs are saved
		"""
		pre_agisoft_xgaze_info = osp.join(self.temp_cam_dir, 'updated', 'cam_info_agisoft.xml')
		_, xgaze_cam_extri = readFileXGaze(self.cal_dir)

		## NOTE: read the output agisoft format to be the same as cam00.xml, after this, they are totally the same and we don't need to touch agisoft file anymore
		_, new_cam_extri = readFileAgisoft(pre_agisoft_xgaze_info)
		new_cam_extri, ratio_mean = scaleAgisoftResult(xgaze_cam_extri, new_cam_extri)


		"""
			No need to run this camera-saving process because we are already using the updated cameras
		"""
		# final_xgaze_save_path = osp.join(save_path_frame, 'camera_final_xgaze')
		# os.makedirs(final_xgaze_save_path, exist_ok=True)
		# for i, c in enumerate(self.xgaze.cameras):
			
		# 	cam_rotation = new_cam_extri[i][0:3, 0:3]
		# 	cam_translation = new_cam_extri[i][0:3, 3].reshape((3, 1))

		# 	## save the final xgaze anyway
		# 	file_name = os.path.join(self.cal_dir, 'cam' + str(i).zfill(2) + '.xml')
		# 	print('read original intrinsic parameters from official xml: ', file_name)
		# 	fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
		# 	camera_matrix = fs.getNode('Camera_Matrix').mat()
		# 	camera_distortion = fs.getNode('Distortion_Coefficients').mat()
		# 	img_height = fs.getNode('image_Height').real()
		# 	img_width = fs.getNode('image_Width').real()
		# 	fs.release() 
		# 	# save all files
		# 	file_name = os.path.join(final_xgaze_save_path, 'cam' + str(i).zfill(2) + '.xml')
		# 	print('save the final xml to: ', file_name)
		# 	s = cv2.FileStorage(file_name, cv2.FileStorage_WRITE)
		# 	s.write('image_Width', int(img_width))
		# 	s.write('image_Height', int(img_height))
		# 	s.write('Camera_Matrix', np.asmatrix(camera_matrix))
		# 	s.write('Distortion_Coefficients', np.asmatrix(camera_distortion))
		# 	s.write('cam_translation', np.asmatrix(cam_translation))
		# 	s.write('cam_rotation', np.asmatrix(cam_rotation))
		# 	s.release()

		if self.save_3d_model:
			# Create a scaling transformation matrix that applies transform directly inside the chunk
			scaling_matrix = np.eye(4)
			scaling_matrix[0, 0] = scaling_matrix[1, 1] = scaling_matrix[2, 2] = ratio_mean * 1000 ## to millimeter
			# Create a 180-degree rotation matrix around the x-axis
			theta = np.pi
			c, s = np.cos(theta), np.sin(theta)
			rotation_matrix = np.eye(4)
			rotation_matrix[1, 1] = c
			rotation_matrix[1, 2] = -s
			rotation_matrix[2, 1] = s
			rotation_matrix[2, 2] = c
			# Combine the scaling and rotation matrices
			combined_matrix = scaling_matrix @ rotation_matrix
			# Apply the combined transformation matrix to the chunk
			chunk.transform.matrix = Metashape.Matrix(combined_matrix) * chunk.transform.matrix

			final_obj_save_path = osp.join(save_path_frame, 'final_obj')
			os.makedirs(final_obj_save_path, exist_ok=True)
			chunk.exportModel(path=osp.join(final_obj_save_path, 'mvs3d.obj'), binary=False, save_texture=True, save_uv=True, save_normals=True,
									save_colors=True, save_cameras=True, save_markers=False, save_comment=False,
									format=Metashape.ModelFormat.ModelFormatOBJ)
		

def get_parser(**parser_kwargs):
	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ("yes", "true", "t", "y", "1"):
			return True
		elif v.lower() in ("no", "false", "f", "n", "0"):
			return False
		else:
			raise argparse.ArgumentTypeError("Boolean value expected.")

	parser = argparse.ArgumentParser(**parser_kwargs)

	parser.add_argument(
		"--debug",
		action='store_true'
	)
	parser.add_argument(
		"--xgaze_basedir",
		type=str,
		help='the base dir of raw ETH-XGaze'
	)


	parser.add_argument(
		"--output_path",
		type=str,
		help='the output base directory'
	)
	parser.add_argument(
		"-resize",
		"--resize_input_size",
		type=str,
		default='1200',
	)
	
	parser.add_argument(
		"-grp",
		"--sub_grp",
		type=str,
	)
	
	parser.add_argument(
		"-save_3d",
		"--save_3d_model",
		type=str2bool,
		default=True,
	)


	parser.add_argument(
		"--resize_factor_color",
		type=float,
		default=1.0,
	)
	parser.add_argument(
		"--resize_factor_depth",
		type=float,
		default=1.0,
	)	
	
	return parser
if __name__=="__main__":
	
		

	parser = get_parser()
	config = parser.parse_args()

	now_day = datetime.now().strftime("%Y-%m-%d")
	now_time = datetime.now().strftime("%H-%M-%S")

	config.output_path = osp.join(config.output_path, now_day, now_time)

	config.STD_SIZE = config.resize_input_size

	metashape_reconstructor = XGaze3DPipeline(config=config)
	metashape_reconstructor.run()



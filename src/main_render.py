import os, sys
import os.path as osp
import time
import torch
import argparse
import numpy as np
import cv2
from glob import glob
from rich.progress import track
import omegaconf
from omegaconf import OmegaConf
from datetime import datetime
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	PerspectiveCameras,
	PointLights, 
	DirectionalLights, 
	RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	SoftPhongShader,
	HardPhongShader,
	TexturesUV,
	TexturesVertex
)
from pytorch3d_util import SimpleShader, mySoftPhongShader, BlendParams
from gazelib.label_transform import rotation_matrix, mean_eye_nose, get_eye_nose_landmarks
from gazelib.gaze.normalize import normalize_woimg, resize_landmarks
from gazelib.gaze.gaze_utils import vector_to_pitchyaw
from gazelib.utils.h5_utils import add, to_h5
from gazelib.utils.color_text import *
from headpose import get_optimize_3d_info, project_Fc
from xgaze_util import read_csv_as_dict, ETHXGaze




def get_frames_before(frames, target_frame_idx):
	idx = min(len(frames), target_frame_idx)
	return frames[:idx]
	


if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")


def read_resize_blur(img_path, roi_size):
	background_image = cv2.imread(img_path)
	background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)
	background_image = cv2.resize(background_image, roi_size, interpolation=cv2.INTER_AREA)
	background_image = cv2.blur(background_image, (20,20))
	background_image = np.array(background_image) / 255.0
	return background_image




def tuple_to_str(t):
	return ','.join(map(str, t))
def str_to_tuple(s):
	return tuple(map(float, s.split(',')))


XGAZE_GREEN =  np.array([0, 110, 0])



class ReconOperator(object):
	"""a class for loading multiview reconstruction obj"""
	def __init__(self):
		super().__init__()
	@classmethod
	def load_multiview_obj(cls, obj_filename):
		start_time = time.time()
		vert, face, aux = load_obj(obj_filename)
		v = vert.data.numpy()
		print(' end loading obj file, time: ', time.time() - start_time)
		verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
		faces_uvs = face.textures_idx[None, ...]  # (1, F, 3)
		tex_maps = aux.texture_images
		texture_image = list(tex_maps.values())[0]
		texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)
		texture = TexturesUV(verts_uvs=verts_uvs.to(device), faces_uvs=faces_uvs.to(device), maps=texture_image)
		return v, face, texture


class RenderXGaze(object):
	def __init__(self, config):
		self.config = config
		self.output_dir = config.output_path

		self.subjects = OmegaConf.load(config.sub_grp)['group']
		self.xgaze = ETHXGaze(base_dir=config.xgaze_basedir, initialize_csv=False)
		
		self.original_calibration = self.xgaze.original_calibration

		for c in self.original_calibration:
			## delete the key: 'camera_rotation' and 'camera_translation'
			del self.original_calibration[c]['camera_rotation']
			del self.original_calibration[c]['camera_translation']
			
		self.original_intrisic = self.original_calibration.copy()
		del self.original_calibration

		self.light_metadata = self.xgaze.light_meta

		self.output_normalized_dir = osp.join(self.output_dir, 'normalized_images'); os.makedirs(self.output_normalized_dir, exist_ok=True)
		self.output_render_dir = osp.join(self.output_dir, 'render_images'); os.makedirs(self.output_render_dir, exist_ok=True)

		


		self.mean_calibration = {}
		self.avg_cams_final = self.xgaze.updated_cam_calibration_dir
		for subject in self.subjects:
			subject_str = list(subject.keys())[0] if type(subject) == omegaconf.dictconfig.DictConfig else subject
			self.mean_calibration[subject_str] = self.xgaze.read_cam_xml_to_dict(osp.join(self.avg_cams_final, subject_str))

		self.xgaze_3d_basedir = config.xgaze_3d_basedir
		## NOTE: we directly read from the updated annotation 
		self.updated_annotation_dir = self.xgaze.updated_annotation_dir
		
		self.new_csv_all = {}
		for subject in track(self.subjects, description='pre reading new csv'):
			subject_str = list(subject.keys())[0] if type(subject) == omegaconf.dictconfig.DictConfig else subject
			self.new_csv_all[subject_str] = read_csv_as_dict( osp.join(self.updated_annotation_dir, subject+'_update.csv'))


		### NOTE: this stores the random choice of each "subjectXXXX/frameXXXX"
		self.render_help =  OmegaConf.load(osp.join( osp.dirname(osp.realpath(__file__)), 'configs/render_info.yaml'))
		self.place365_dir = config.place365_dir



		# Normalization parameters
		self.focal_norm = 960 # focal length of normalized camera
		self.distance_norm = 300 # normalized distance between eye and camera
		self.roi_size = (448,448) # size of cropped eye image

		self.final_size = (448, 448) #(224, 224)
		self.normalize_renderer, self.cameras = init_renderer(self.focal_norm, self.focal_norm, self.roi_size)


		radius = 100
		num_lights = 8
		# Calculatethe midpoint between face center and origin
		midpoint = [0,0,-200]
		self.point_light_positions = []
		for i in range(num_lights):
			angle = 2 * np.pi * i / num_lights
			x = midpoint[0] + radius * np.cos(angle)
			y = midpoint[1] + radius * np.sin(angle)
			z = midpoint[2]
			self.point_light_positions.append([ np.round(x), np.round(y), np.round(z),])
		self.set_lights(self.normalize_renderer, pos=None)


		self.reset_renderer(self.normalize_renderer)

		self.renderer_split_verts = config.renderer_split_verts

	def set_lights(self, renderer, pos= [0.0, 0.0, 280.0]):
		if pos is None:
			return
		else:
			## (+) means left, up, front
			ambi = 0.2
			diff = 0.8
			spec = 0.0
			## for the position of light, see https://pytorch3d.org/docs/cameras figure
			point_lights = PointLights(
				ambient_color=[[ambi, ambi, ambi]],
				diffuse_color=[[diff, diff, diff]],
				specular_color=[[spec, spec, spec]],
				location=[pos],  
				device=device,
			)
			renderer.shader = HardPhongShader(
				device=device, 
				cameras=self.cameras,
				lights=point_lights,#PointLights(device=device, location=[[0.0, 0.0, -3.0]]),
			)
	
	
	def set_light_bg(self, renderer, subject, frame):

		ac, dc, sc = str_to_tuple(self.render_help['{}/{}'.format(subject, frame)]['light'])
		light = DirectionalLights(
			device=device, 
			ambient_color=((ac, ac, ac), ), 
			diffuse_color=((dc, dc, dc), ), 
			specular_color=((sc, sc, sc),), 
			direction=[[0.0, 0.0,1.0]]
		)
		renderer.shader=mySoftPhongShader(
			device=device,
			cameras=self.cameras,
			lights=light, 
			blend_params=BlendParams()
		)
	
		bg_info_this_frame = self.render_help['{}/{}'.format(subject, frame)]['background']
		if bg_info_this_frame['type']=='RGB':
			color_this_frame = str_to_tuple(bg_info_this_frame['value'])
			renderer.shader.blend_params = BlendParams( background_color=color_this_frame)
		elif bg_info_this_frame['type']=='IMG':
			bg_img = read_resize_blur(osp.join( self.place365_dir, bg_info_this_frame['value']), roi_size=self.roi_size )
			renderer.shader.blend_params = BlendParams( background_color=bg_img)



	def reset_renderer(self, renderer):
		## since we want to create xgaze-like dataset, we set the default background color to be the green color of XGaze
		renderer.shader = SimpleShader(
				device=device,
				blend_params=BlendParams( background_color=XGAZE_GREEN )
			)
	
			
	def forward(self):
		""" this is the final step to write the updated label in the same format as the original XGaze
		label to update:
			- gaze_point_cam: gaze point in each camera, since the camera extrinsic is updated, we need to update the gaze point
			- rvec, tvec: save the optimized head pose in opencv format for each frame
			- landmarks_3d: save the optimized 3D landmarks
			- lm_gt: 2D landmark. some of the detected landmarks are not accurate, we want to update them using the optimized 3D landmarks
		"""

		subject = self.current_subject
		frame = self.current_frame
		new_calibration = self.mean_calibration[subject] ## this is the average calibration of all the frames in the subject

		new_sub_dict = self.new_csv_all[subject] 
		rvec_new = []
		tvec_new = [] 
		lm_gt_list = []
		gc_cam_list = []
		for i, c in enumerate(self.xgaze.cameras):
			index = frame +'/' + c + '.JPG'

			gc_cam_i = np.array([float(i) for i in new_sub_dict[index][2:5]])
			head_rotation_cam = np.array([float(i) for i in new_sub_dict[index][5:8]])
			head_translation_cam = np.array([float(i) for i in new_sub_dict[index][8:11]])
			lm_2d = np.array([float(i) for i in new_sub_dict[index][11:147]]).reshape(68,2)

			gc_cam_list.append(gc_cam_i)
			rvec_new.append(head_rotation_cam.reshape(3,1))
			tvec_new.append(head_translation_cam.reshape(3,1))
			lm_gt_list.append(lm_2d)

		### rvec_new : a list of 3x1
		### tvec_new : a list of 3x1
		### Fc_new : a numpy array of (NUM_KPTS_TO_USE, 3)
		rvec_new, tvec_new, Fc_new = get_optimize_3d_info(xgaze=self.xgaze, calibration=new_calibration,
														subject=subject, frame=frame, 
														lm_gt_list=lm_gt_list,
														good_cams=self.xgaze.lm2d_accurate_cams,
														NUM_KPTS_TO_USE=6,
													)

		new_lm_proj_list = project_Fc(Fc_new, new_calibration, self.xgaze.cameras)

		to_write_normalize = {}
		for i, c in enumerate(self.xgaze.cameras):
			img_name = c + '.JPG'
			gc_cam_i = gc_cam_list[i]
			Fc_i = Fc_new @ new_calibration[c]['camera_rotation'].T + new_calibration[c]['camera_translation'].reshape(1,3)
			face_center = mean_eye_nose( get_eye_nose_landmarks(Fc_i) if Fc_i.shape[0]!=6 else Fc_i)

			normalized_results = normalize_woimg(  landmarks=lm_gt_list[i], 
											focal_norm=self.focal_norm, distance_norm=self.distance_norm, roi_size=self.roi_size, 
											center=face_center, hr=rvec_new[i], ht=tvec_new[i], cam=self.original_intrisic[c]['camera_matrix'], gc=gc_cam_i)
			_, R, hR_norm, gc_normalized, landmarks_warped, W = normalized_results[0], normalized_results[1], normalized_results[2], normalized_results[3], normalized_results[4], normalized_results[5]
			
			pj_landmarks_warped = cv2.perspectiveTransform(new_lm_proj_list[i].reshape(-1,1,2).astype('float32'), W).reshape(-1, 2)
			
			hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
			gaze_pitchyaw = vector_to_pitchyaw(-gc_normalized.reshape((1,3))).flatten()

			add(to_write_normalize, 'frame_index', int(frame[-4:]) )
			add(to_write_normalize, 'cam_index', int(img_name[-6:-4])+1 )
			add(to_write_normalize, 'face_gaze', gaze_pitchyaw)
			add(to_write_normalize, 'face_head_pose', hr_norm.astype(np.float32))
			add(to_write_normalize, 'face_mat_norm', R.astype(np.float32))
			add(to_write_normalize, 'rotation_matrix', np.eye(3)) # relative rotation matrix to source image
			add(to_write_normalize, 'landmarks_norm',  resize_landmarks(landmarks_warped, focal_norm=self.focal_norm, src_size=self.roi_size, tgt_size=self.final_size))
			add(to_write_normalize, 'pj_landmarks_norm',  resize_landmarks(pj_landmarks_warped, focal_norm=self.focal_norm, src_size=self.roi_size, tgt_size=self.final_size) )
				
				
		"""this is based on each frame"""
		v, faces, texture = ReconOperator.load_multiview_obj(osp.join(self.xgaze_3d_basedir , subject, frame, 'final_obj/mvs3d.obj'))
		render_imgs, render_imgs_aug = self.rotate_and_render( v, faces, texture, to_write_normalize, Fc_new, new_calibration)

		to_write_mvs = to_write_normalize.copy()
		to_write_mvs_aug = to_write_normalize.copy()
		_ = [ add(to_write_mvs, 'face_patch', render_imgs[i]) for i in range(len(render_imgs)) ]
		_ = [ add(to_write_mvs_aug, 'face_patch', render_imgs_aug[i]) for i in range(len(render_imgs_aug)) ]

		os.makedirs(osp.join(self.output_render_dir, 'augment'), exist_ok=True)
		to_h5(to_write=to_write_mvs, output_path=os.path.join(self.output_render_dir, f'{subject}.h5'))
		to_h5(to_write=to_write_mvs_aug, output_path=os.path.join(self.output_render_dir, 'augment', f'{subject}.h5'))
		
	def rotate_and_render(self, v, faces, texture, to_write, Fc_new, new_calibration):
		st_time = time.time()
		subject = self.current_subject
		frame = self.current_frame
		verts_list = []
		for i, c in enumerate(self.xgaze.cameras):
			Fc_i = Fc_new @ new_calibration[c]['camera_rotation'].T + new_calibration[c]['camera_translation'].reshape(1,3)
			face_center = mean_eye_nose( get_eye_nose_landmarks(Fc_i) if Fc_i.shape[0]!=6 else Fc_i)
			v_in_cam_i = (v @ new_calibration[c]['camera_rotation'].T + new_calibration[c]['camera_translation'].reshape(1,3))
			S = np.array([ [1.0, 0.0, 0.0],
						[0.0, 1.0, 0.0], 
						[0.0, 0.0, self.distance_norm/np.linalg.norm(face_center)]]) # scaling matrix
			v_in_cam_i_norm = v_in_cam_i @ to_write['face_mat_norm'][i].T @ S.T
			verts_list.append(v_in_cam_i_norm)
		
		## NOTE: this recover the renderer to the default setting (XGAZE_GREEN background and no different lighting setting)
		self.reset_renderer(self.normalize_renderer)
		render_imgs = self.forward_render(verts_list, faces, texture)
		## this set the background and lighting based on the pre-defined random setting (reproducible)
		self.set_light_bg(self.normalize_renderer, subject, frame)
		render_imgs_aug = self.forward_render(verts_list, faces, texture)

		print(" render one frame takes {} seconds".format(time.time() - st_time))
		render_imgs = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), self.final_size, interpolation=cv2.INTER_AREA).astype(np.uint8) for img in render_imgs])
		render_imgs_aug = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), self.final_size, interpolation=cv2.INTER_AREA).astype(np.uint8) for img in render_imgs_aug])
		return render_imgs, render_imgs_aug


	
	def forward_render(self, verts_list, faces, texture):
		def split_list(my_list, num_splits):
			list_length = len(my_list)
			subset_size = list_length // num_splits
			remaining_elements = list_length % num_splits

			subsets = []
			index = 0

			for _ in range(num_splits):
				subset = my_list[index:index + subset_size]

				if remaining_elements > 0:
					subset.append(my_list[index + subset_size])
					index += 1
					remaining_elements -= 1

				subsets.append(subset)
				index += subset_size

			return subsets
		render_imgs_split = []
		splited_verts_list = split_list(verts_list, self.renderer_split_verts)
		for verts_list_split in splited_verts_list:
			verts_in = [torch.tensor(v, dtype=torch.float32,device=device) * torch.tensor([1,-1,-1],device=device) for v in verts_list_split]
			faces_in = [faces.verts_idx.to(device) for v in verts_list_split]
			textures_in = texture.extend(len(verts_list_split))
			mesh = Meshes(verts=verts_in, faces=faces_in, textures=textures_in)
			rend_img = self.normalize_renderer(mesh, zfar=1500)
			render_imgs_split.append(rend_img)
		render_imgs = torch.cat(render_imgs_split, dim=0)
		render_imgs = render_imgs.cpu().data.numpy()[:,:,:,:3]*255  # batch * H * W * 3

		return render_imgs
	


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
			self.frame_idx = 0
			print_yellow(" {} - frames_remained: ".format(subject), frames_remained)
			for frame in track(frames_remained, description="Processing {} ".format(subject)):
				print_green('the current subject - frame is: {} - {}'.format(subject, frame) ) 
				self.current_frame = frame
				start_time = time.time()
				self.forward()
				print(" the total time for this frame is: ", time.time() - start_time)
				self.frame_idx += 1
			

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
		"--xgaze_basedir",
		type=str,
		help='the base dir of raw ETH-XGaze'
	)
	parser.add_argument(
		"--xgaze_3d_basedir",
		type=str,
		help='the directory of the reconstructed 3D xgaze '
	)

	parser.add_argument(
		"--output_path",
		type=str,
		help='the output base directory'
	)

	
	parser.add_argument(
		"-grp",
		"--sub_grp",
		type=str,
	)
	
	parser.add_argument(
		"--place365_dir",
		type=str,
		default=None,
		help='the path of place365'
	)
	
	parser.add_argument(
		"--renderer_split_verts",
		type=int,
		default=9,
		help='since the GPU memeory is not enough, split the verts to N subsets and combine back'
	)
	
	
	return parser




def init_renderer(focal_x, focal_y, img_size, device=device):

	ambi = 0.3
	diff = 0.1
	spec = 0.1
	directional_lights = DirectionalLights(
		direction=[[1.0, 0.0, 0.0]],
		ambient_color=[[0.5, 0.5, 0.5]],
		diffuse_color=[[0.5, 0.5, 0.5]],
		specular_color=[[0.5, 0.5, 0.5]],
		device=device
	)

	## for the position of light, see https://pytorch3d.org/docs/cameras figure
	
	point_lights = PointLights(
		ambient_color=[[ambi, ambi, ambi]],
		diffuse_color=[[diff, diff, diff]],
		specular_color=[[spec, spec, spec]],
		# location=[[-100.0, 30.0, -100.0]],  ## left-right, up-down, front-back
		location=[[0.0, 0.0, 280.0]],  ## left-right, up-down, front-back
		device=device,
	)

	### assume the camera is always fixed
	cameras = PerspectiveCameras(
		focal_length=((focal_x, focal_y),),  # (fx_screen, fy_screen)
		principal_point=((img_size[0]/2., img_size[1]/2.),),  # (px_screen, py_screen)
		in_ndc=False,
		image_size=( (img_size[1], img_size[0]),),  # (imwidth, imheight)
		R=torch.tensor(rotation_matrix(0,np.pi,0) @ np.eye(3)).unsqueeze(0),
		T=torch.tensor([0, 0, 0]).unsqueeze(0), 
		device=device
	)
	blend_params = BlendParams(background_color=XGAZE_GREEN)
	renderer = MeshRenderer(
		rasterizer=MeshRasterizer(
			cameras=cameras, 
			raster_settings=RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1, )
		),
		shader=SoftPhongShader(
			device=device, 
			cameras=cameras,
			# lights=directional_lights,
			lights=point_lights,#PointLights(device=device, location=[[0.0, 0.0, -3.0]]),
			blend_params=blend_params,
		)
		# shader=SimpleShader(
		# 	device=device,
		# 	blend_params=blend_params
		# )
	)
	return renderer, cameras




if __name__=="__main__":

	parser = get_parser()
	config = parser.parse_args()

	now_day = datetime.now().strftime("%Y-%m-%d")
	now_time = datetime.now().strftime("%H-%M-%S")
	

	config.output_path = osp.join(config.output_path, 'render_' + now_day + '-' + now_time)

	## obj path of multiview reconstruction

	xgaze_renderer = RenderXGaze(config=config)
	xgaze_renderer.run()



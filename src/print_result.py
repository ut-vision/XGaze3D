import os
import h5py
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

sns.set_theme(font_scale=1.5)

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



def rad_to_degree(head_pose):
	return head_pose * 180/np.pi


def draw_sns(distribution, name):
	plt.figure()
	df_head = pd.DataFrame({"Yaw [degree]": distribution[:,1], "Pitch [degree]":distribution[:,0]})
	h = sns.JointGrid(x="Yaw [degree]", y="Pitch [degree]", data=df_head, xlim=(-150,150), ylim=(-150,150))  
	h.ax_joint.set_aspect('equal')         
	h.plot_joint(sns.histplot)                         
	h.ax_marg_x.set_axis_off()
	h.ax_marg_y.set_axis_off()
	h.ax_joint.set_yticks([-80, -40, 0, 40, 80])
	h.ax_joint.set_xticks([-80, -40, 0, 40, 80])
	plt.savefig(name+'.jpg',bbox_inches='tight')





if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str)
	def get_config():
		config, unparsed = parser.parse_known_args()
		return config, unparsed
	config, _ = get_config()

	data_dir = config.data_dir
	sample_dir = os.path.join(data_dir, 'samples')
	os.makedirs(sample_dir, exist_ok=True)

	person_path_list = sorted(glob.glob(data_dir + '/' + 'sub*.h5'))
	plot_distribution = False
	if plot_distribution:
		head_pose_list = np.zeros((0, 2))
		gaze_list = np.zeros((0, 2))

		for person_path in person_path_list:
			with h5py.File(person_path, 'r') as f:
				print(list(f.keys()))
				print('Number of images: ', f['face_patch'].shape[0])
				p = os.path.basename(person_path).split('.')[0]
				head_pose_list = np.append(head_pose_list, f['face_head_pose'][:], axis=0)
				gaze_list = np.append(gaze_list, f['face_gaze'][:], axis=0)

		head_path = os.path.join(data_dir, 'head_distribution.txt')
		gaze_path = os.path.join(data_dir, 'gaze_distribution.txt')
		np.savetxt(head_path, head_pose_list)
		np.savetxt(gaze_path, gaze_list)


		H_deg = rad_to_degree(head_pose_list)
		G_deg = rad_to_degree(gaze_list)
		draw_sns(H_deg, data_dir+'/H_sns')
		draw_sns(G_deg, data_dir+'/G_sns')




	for person_path in person_path_list[:]:
		with h5py.File(person_path, 'r') as f:
			for key, value in f.items():
				print('number of {}: '.format(key), value.shape[0])

			number_images = f['face_patch'].shape[0]
			p = os.path.basename(person_path).split('.')[0]

			for i in range(  np.minimum(1800, number_images) ):
				cam_index = f['cam_index'][i]
				frame_index = f['frame_index'][i]
				image = f['face_patch'][i]
				gaze = f['face_gaze'][i]
				head_pose = f['face_head_pose'][i]
				
				lm_2d = f['landmarks_norm'][i]
				pj_lm_2d = f['pj_landmarks_norm'][i]

				image = draw_lm(image,lm_2d, color=(0,0,255),  radius = 5)
				image = draw_lm(image,pj_lm_2d, color=(0,255,0),  radius = 5)
				image = draw_gaze(image,gaze, thickness=4)
				image = draw_gaze(image,head_pose, thickness=3, color=(0, 255,0))
				
				cv2.imwrite( sample_dir + f'/image_{p}_{frame_index}_{cam_index}.jpg', image)


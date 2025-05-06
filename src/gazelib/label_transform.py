

import cv2

import numpy as np


def get_eye_nose_landmarks(landmarks):
    assert landmarks.shape[0]==50 or landmarks.shape[0]==68
    if landmarks.shape[0] == 50:
        lm_6 = landmarks[[20, 23, 26, 29, 15, 19], :]  # the eye and nose landmarks
    elif landmarks.shape[0] == 68:
        lm_6 = landmarks[[36, 39, 42, 45, 31, 35], :]  # the eye and nose landmarks
    return lm_6
def get_eye_mouth_landmarks(landmarks):
    assert landmarks.shape[0]==50 or landmarks.shape[0]==68
    if landmarks.shape[0] == 50:
        lm_6 = landmarks[[20, 23, 26, 29, 32, 38], :]  # the eye and nose landmarks
    elif landmarks.shape[0] == 68:
        lm_6 = landmarks[[36,39,42,45,48,54], :]  # the eye and nose landmarks
    return lm_6

def mean_eye_nose(landmarks):
    assert landmarks.shape[0]==6
    # get the face center
    two_eye_center = np.mean(landmarks[0:4, :], axis=0).reshape(1,-1)
    nose_center = np.mean(landmarks[4:6, :], axis=0).reshape(1,-1)
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=0), axis=0).reshape(1,-1)
    return face_center
def mean_eye_mouth(landmarks):
    assert landmarks.shape[0]==6
    face_center = np.mean(landmarks, axis=0).reshape(1,-1) 
    return face_center

def get_face_center_by_nose(hR, ht, face_model_load):
    face_model = get_eye_nose_landmarks(face_model_load)  # the eye and nose landmarks
    Fc = np.dot(hR, face_model.T) + ht # 3D positions of facial landmarks
    face_center = mean_eye_nose(Fc.T).reshape((3, 1))  # get the face center
    return face_center, Fc

def get_face_center_by_mouth(hR, ht, face_model_load):
    face_model = get_eye_mouth_landmarks(face_model_load)  # the eye and nose landmarks
    Fc = np.dot(hR, face_model.T) + ht # 3D positions of facial landmarks
    face_center = mean_eye_mouth(Fc.T).reshape((3, 1))  # get the face center
    return face_center, Fc

def lm68_to_50(lm_68):
	'''
	lm_68: (68,2)
	'''
	lm_50 = np.zeros((50,2))
	lm_50[0] = lm_68[8]
	lm_50[1:44] = lm_68[17:60]
	lm_50[44:47] = lm_68[61:64]
	lm_50[47:50] = lm_68[65:68]
	return lm_50


def lm68_subset(lm_68, NUM_KPTS_TO_USE):
	'''
	lm_68: (68,2)
	'''
	if NUM_KPTS_TO_USE == 6:
		lm_68 = np.array(lm_68, dtype=np.float32)
		return lm_68[[36, 39, 42, 45, 31, 35], :]
	elif NUM_KPTS_TO_USE ==50:
		return lm68_to_50(lm_68)
	else:
		print('not supported yet')
		exit(0)

def lm50_subset(lm_50, NUM_KPTS_TO_USE):
	'''
	lm_50: (50,2)
	'''
	lm_50 = lm_50.copy()
	if NUM_KPTS_TO_USE == 6:
		lm_50 = lm_50[[20, 23, 26, 29, 15, 19], :]
		return lm_50
	elif NUM_KPTS_TO_USE ==50:
		return lm_50
	else:
		print('not supported yet')
		exit(0)

def get_face_center(landmarks_3d):
	'''
	landmarks_3d: (3, 6)
	--> 
	face_center: (3,1)
	'''
	two_eye_center = np.mean(landmarks_3d[:, 0:4], axis=1).reshape((3, 1))
	nose_center = np.mean(landmarks_3d[:, 4:6], axis=1).reshape((3, 1))
	face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
	return face_center



def compute_R(lm6, dataname):
	'''
	6 landmarks in opencv coordinate
	dataname: mpii or xgaze
	the face center are computed differently 
		for mpii: the 6 landmarks are 4 eye + 2 mouth
		for xgaze: the 6 landmarks are 4 eye + 2 nose
	'''
	if dataname=='mpii':
		left_center = np.mean(lm6[2:4,:],axis=0)
		right_center = np.mean(lm6[:2,:],axis=0)
		face_center = np.mean(lm6,axis=0)
	elif dataname=='xgaze':
		left_center = np.mean(lm6[2:4,:],axis=0)
		right_center = np.mean(lm6[:2,:],axis=0)
		nose_center = np.mean(lm6[[4,5],:],axis=0)
		face_center = ( (left_center + right_center)/2 + nose_center ) /2

	distance = np.linalg.norm(face_center)

	hRx = left_center - right_center
	hRx /= np.linalg.norm(hRx)
	forward = (face_center/distance).reshape(3)
	down = np.cross(forward, hRx)
	down /= np.linalg.norm(down)
	right = np.cross(down, forward)
	right /= np.linalg.norm(right)
	R = np.c_[right, down, forward].T
	return R
	
def rotation_matrix(x, y, z):
	'''
	x, y, z: roll, pitch, yaw, (radians)
	'''
	Rx = np.array([[1,0,0],
				[0, np.cos(x), -np.sin(x)],
				[0, np.sin(x), np.cos(x)]])

	Ry = np.array([[ np.cos(y), 0, np.sin(y)],
				[ 0,         1,         0],
				[-np.sin(y), 0, np.cos(y)]])

	Rz = np.array([[np.cos(z), -np.sin(z), 0],
				[np.sin(z),  np.cos(z), 0],
				[0,0,1]])
	return Rz@Ry@Rx
def get_rotation(from_pose, target_pose):
	
	rotation1 = rotation_matrix( -from_pose[0], from_pose[1], 0)
	rotation2 = rotation_matrix(-target_pose[0], target_pose[1], 0)
	rotation = rotation2@np.linalg.inv(rotation1)
	return rotation

def hR_2_hr(hR):
	hr = np.array([np.arcsin(hR[1, 2]),
				np.arctan2(hR[0, 2], hR[2, 2])])
	return hr

def hr_2_hR(hr):
	hR = rotation_matrix( -hr[0], hr[1], 0)
	return hR



if __name__ == '__main__':
	# hr_norm = np.array([0.15, 0.2])
	# pose = np.array([-0.1, 0.3])
	# rotation1 = rotation_matrix( -hr_norm[0], hr_norm[1], 0)
	# rot = cv2.Rodrigues( np.array([hr_norm[0], hr_norm[1], 0])  )[0] 
	def to_hR(hr_norm):
		hR_norm = rotation_matrix( -hr_norm[0], hr_norm[1], 0)
		return hR_norm

	hr1 = np.array([0.15, 0.2])
	hr2 = np.array([0.10, 0.2])

	hr_t =  np.array([-0.1, 0.3])

	hR1 = to_hR(hr1)
	hR2 = to_hR(hr2)
	print('hR1: ', hR1)
	print('hR2: ', hR2)
	
	R1t = get_rotation(hr1, hr_t)

	hR1_ = np.dot(R1t, hR1)

	print('rotated hR_: ', hR1_)
	hr1_ = np.array([np.arcsin(hR1_[1, 2]),
				np.arctan2(hR1_[0, 2], hR1_[2, 2])])
	print('rotated hr1_: ', hr1_)
	print('hR t: ', to_hR(hr_t))
	hR2_ = np.dot(R1t, hR2)
	print('rotated hR2_: ', hR2_)
	# rotation2 = rotation_matrix( -pose[0], pose[1], 0)



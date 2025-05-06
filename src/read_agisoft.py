import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import xml.dom.minidom
import xml.etree.ElementTree as ET

CAM_NUM = 18

def readFileXGaze(input_path, cam_index_start=0, cam_index_end=18):
    output_cam_intri = []
    output_cam_extri = []
    for index_cam in range(cam_index_start, cam_index_end):
        file_name = os.path.join(input_path, 'cam' + str(index_cam).zfill(2) + '.xml')
        if not os.path.exists(file_name):
            print('The file does not exist: ', file_name)
        fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode('Camera_Matrix').mat()
        output_cam_intri.append(camera_matrix)
        camera_distortion = fs.getNode('Distortion_Coefficients').mat()
        img_height = fs.getNode('image_Height').real()
        img_wdith = fs.getNode('image_Width').real()
        camera_rotation = fs.getNode('cam_rotation').mat()
        cam_translation = fs.getNode('cam_translation').mat()
        RT = np.hstack((camera_rotation, cam_translation))
        transform = np.vstack((RT, np.array([0, 0, 0, 1])))
        output_cam_extri.append(transform)
        fs.release()
    return output_cam_intri, output_cam_extri


def transform_camera_from_agisoft_to_cv2(transform):
    rot_vect = cv2.Rodrigues(transform[0:3, 0:3])[0]
    rot_vect[0] *= -1  # x and y axis are reversed in two coordiantes, and the
    rot_vect[1] *= -1  # angle should be reversed, too.
    transform[0:3, 0:3] = cv2.Rodrigues(rot_vect)[0]

    def rotate_camera_z_axis_180(M):
        R = M[:3, :3]  # extract the rotation matrix
        R_z = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                        [np.sin(np.pi), np.cos(np.pi), 0],
                        [0, 0, 1]])
        R_new = np.dot(R, R_z)  # compute the new rotation matrix
        M[:3, :3] = R_new  # update the rotation matrix in M
        return M
    transform = rotate_camera_z_axis_180(transform) ## note this is self-rotating, instead of rotating the world
    def rotation_matrix_y_axis_180(M):
        theta = np.pi
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]])
        M_new = np.dot(R, M)  
        return M_new

    transform = rotation_matrix_y_axis_180(transform)  # rotate around y-axis
    transform[0, 3] *= -1  # reverse x axis, since it is reversed in two coordinates
    transform[1, 3] *= -1  # reverse y axis, since it is reversed in two coordinates

    ## this switch the definition of rotation and translation
    """
    BEFORE: the R, t is defined from the world to the camera
    AFTER: the R, t is defined from the camera to the world
    """
    R_agi = transform[:3,:3].copy()
    t_agi = transform[:3,3].reshape(3,1).copy()

    transform[:3,:3] = R_agi.T
    transform[:3,3] =  - t_agi.reshape(1,3) @ R_agi

    return transform

def transform_camera_from_cv2_to_agisoft(camera_rotation, camera_translation):
	RT = np.hstack((camera_rotation, camera_translation))
	transform = np.vstack((RT, np.array([0, 0, 0, 1])))

	# Switch the definition of rotation and translation back
	R_cv = camera_rotation.copy()
	t_cv = camera_translation.reshape(3, 1).copy()

	transform[:3, :3] = R_cv.T
	transform[:3, 3] = -t_cv.reshape(1, 3) @ R_cv

	# Reverse x and y axes
	transform[0, 3] *= -1
	transform[1, 3] *= -1

	# Rotate around y-axis
	def rotation_matrix_y_axis_180(M):
		theta = np.pi
		c, s = np.cos(theta), np.sin(theta)
		R = np.array([[c, 0, s, 0],
					[0, 1, 0, 0],
					[-s, 0, c, 0],
					[0, 0, 0, 1]])
		M_new = np.dot(R, M)
		return M_new

	transform = rotation_matrix_y_axis_180(transform)

	# Rotate the camera around the z-axis
	def rotate_camera_z_axis_180(M):
		R = M[:3, :3]
		R_z = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
						[np.sin(np.pi), np.cos(np.pi), 0],
						[0, 0, 1]])
		R_new = np.dot(R, R_z)
		M[:3, :3] = R_new
		return M

	transform = rotate_camera_z_axis_180(transform)
	# Reverse x and y axis rotation
	rot_vect = cv2.Rodrigues(transform[0:3, 0:3])[0]
	rot_vect[0] *= -1
	rot_vect[1] *= -1
	transform[0:3, 0:3] = cv2.Rodrigues(rot_vect)[0]
	transform[0:3, 3] *= 0.001
	return transform

    

def readFileAgisoft(input_path):
    dom = xml.dom.minidom.parse(input_path)
    output_cam_intri = []
    output_cam_extri = []

    root = dom.documentElement
    chunk = root.getElementsByTagName('chunk')[0]
    sensors = chunk.getElementsByTagName('sensors')[0]

    next_id = int(sensors.getAttribute('next_id'))
    print('next_id: ', next_id)
    start = next_id - 18
    next_id=min(18, next_id)
    start = 0
    # -------- get extrisnic -------------#
    for i in range(start, next_id):

        cameras = chunk.getElementsByTagName('cameras')[0]
        camera = cameras.getElementsByTagName('camera')[i]
        try:
            trans = camera.getElementsByTagName('transform')[0]
            transform = trans.firstChild.data
            transform = transform.split()
            transform = [float(i) for i in transform]
            transform = np.array(transform)
            transform = transform.reshape(4, 4)
            transform[0:3, 3] *= 1000  # convert meter to millimeters
            transform = transform_camera_from_agisoft_to_cv2(transform)
            output_cam_extri.append(transform)
        except:
            print('camera ', i, ' has no transform')
            output_cam_extri.append(None)

    # -------- intrisnic -------------#
    for i in range(next_id):
        sensor = chunk.getElementsByTagName('sensor')[i]
        resolution = sensor.getElementsByTagName('resolution')[0]
        width, height = float(resolution.getAttribute("width")), float(resolution.getAttribute("height"))
        calibration = sensor.getElementsByTagName('calibration')[0]
        f = calibration.getElementsByTagName('f')[0]
        focal = float(f.firstChild.data)
        cx = width / 2
        cy = height / 2
        intrin = np.array([
            [focal, 0., cx],
            [0., focal, cy],
            [0., 0., 1.]])
        output_cam_intri.append(intrin)

    return output_cam_intri, output_cam_extri

def scaleAgisoftResult(xgaze_cam_extri, agisoft_cam_extri):
    # scale them
    final_cam_extri = []
    trans_cam0 = xgaze_cam_extri[0][0:3, 3].reshape(3)
    trans_cam0_agisoft = agisoft_cam_extri[0][0:3, 3].reshape(3)
    distance_xgaze = []
    distance_agisoft = []
    for index_cam in range(1, CAM_NUM):
        if agisoft_cam_extri[index_cam] is None:
            continue

        trans_cam = xgaze_cam_extri[index_cam][0:3, 3].reshape(3)
        cam0 = trans_cam0
        distance = (cam0[0] - trans_cam[0]) ** 2 + (cam0[1] - trans_cam[1]) ** 2 + (cam0[2] - trans_cam[2]) ** 2
        distance = np.sqrt(distance) / 1000
        distance_xgaze.append(distance)

        trans_cam = agisoft_cam_extri[index_cam][0:3, 3].reshape(3)
        cam0 = trans_cam0_agisoft
        distance = (cam0[0] - trans_cam[0]) ** 2 + (cam0[1] - trans_cam[1]) ** 2 + (cam0[2] - trans_cam[2]) ** 2
        distance = np.sqrt(distance) / 1000
        distance_agisoft.append(distance)
    distance_xgaze = np.asarray(distance_xgaze)
    distance_agisoft = np.asarray(distance_agisoft)
    ratio = distance_xgaze / distance_agisoft
    ratio_mean = np.mean(ratio)
    ratio_mean = np.mean(distance_xgaze)/np.mean(distance_agisoft)
    print('ratio_mean: ' , ratio_mean)
    # scale it
    for index_cam in range(0, CAM_NUM):
        if agisoft_cam_extri[index_cam] is None:
            final_cam_extri.append(xgaze_cam_extri[index_cam])
        else:
            scaled_agisoft_cam_extri = agisoft_cam_extri[index_cam]
            scaled_agisoft_cam_extri[0:3, 3] *= ratio_mean
            final_cam_extri.append(scaled_agisoft_cam_extri)
    return final_cam_extri, ratio_mean


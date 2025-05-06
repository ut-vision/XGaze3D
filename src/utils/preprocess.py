


import numpy as np
import cv2

def remove_background(image):
    # remove the green background
    h, w, c = image.shape
    empty_img = np.zeros((h, w), dtype=np.uint8)
    RED, GREEN, BLUE = (2, 1, 0)
    reds = image[:, :, RED]
    greens = image[:, :, GREEN]
    blues = image[:, :, BLUE]
    mask = (greens > (reds + 20)) & (greens > (blues + 20))  # | (greens > reds) | (greens > blues)
    empty_img[mask] = 200
    kernel = np.ones((3,3), np.uint8)
    for i in range(5):
        empty_img = cv2.erode(empty_img, kernel, iterations = 3)
        empty_img =  cv2.dilate(empty_img, kernel, iterations = 3)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(100, 100), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(100, 200), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(w - 100, 100), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(w - 100, 200), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(300, 300), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(10, h-10), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(w - 10, 100), newVal=200)
    # cv2.floodFill(empty_img, mask=None, seedPoint=(w - 10, h-10), newVal=200)
    # cv2.imwrite(os.path.join(temp_folder, img_file_name), empty_img)
    mask = (empty_img == 200)
    image[mask] = (0, 0, 0)
    return image



def parse_roi_box_from_landmark(pts, cam=None):
	from math import sqrt
	"""calc roi box from landmark"""
	bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
	
	center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
	
	temp_radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

	if cam in [ 'cam14']: ## move the center down a little
		center[1] += 0.3 * temp_radius
	
	if cam in ['cam04', 'cam05', 'cam12']: ## move the center down a little-little
		center[1] += 0.2 * temp_radius
	

	if cam in ['cam12', 'cam13', 'cam14']: ## the face in these cameras are small, so use smaller radius
		radius =  1.5 * temp_radius
	elif cam in ['cam11', 'cam15']: ## the face in these cameras are large, so use larger radius
		radius = 1.65 * temp_radius
	else:
		radius = 1.5 * temp_radius
	

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

def crop_img(img, roi_box):
	h, w = img.shape[:2]

	sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
	dh, dw = ey - sy, ex - sx
	if len(img.shape) == 3:
		res = np.zeros((dh, dw, 3), dtype=np.uint8)
	else:
		res = np.zeros((dh, dw), dtype=np.uint8)
	if sx < 0:
		sx, dsx = 0, -sx
	else:
		dsx = 0

	if ex > w:
		ex, dex = w, dw - (ex - w)
	else:
		dex = dw

	if sy < 0:
		sy, dsy = 0, -sy
	else:
		dsy = 0

	if ey > h:
		ey, dey = h, dh - (ey - h)
	else:
		dey = dh

	res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
	return res


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

import os
import argparse
import math
import numpy as np
import imageio
import scipy.io
import glob
import cv2
import h5py


import matplotlib.path as mplPath
from scipy.spatial import ConvexHull
import torch
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, Fragments
from pytorch3d.renderer import (
	PerspectiveCameras, 
	FoVPerspectiveCameras,
	PointLights, 
	DirectionalLights, 
	Materials, 
	# RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	SoftPhongShader,
	TexturesUV,
	TexturesVertex,
)
from pytorch3d.renderer.mesh.shading import phong_shading
import torch.nn as nn

from typing import NamedTuple, Sequence, Union
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

"""
In Pytorch3D, I have a SimpleShader which does not use the Phong shader but directly render the texture rgb values.

class SimpleShader(nn.Module):
	def __init__(self, device="cpu", blend_params=None):
		super().__init__()
		self.blend_params = blend_params if blend_params is not None else BlendParams()

	def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
		blend_params = kwargs.get("blend_params", self.blend_params)
		texels = meshes.sample_textures(fragments)
		images = hard_rgb_blend_with_background(texels, fragments, blend_params)
		return images  # (N, H, W, 3) RGBA image

cameras = PerspectiveCameras(
	focal_length=((focal_x, focal_y),),  # (fx_screen, fy_screen)
	principal_point=((img_size[0]/2., img_size[1]/2.),),  # (px_screen, py_screen)
	in_ndc=False,
	image_size=( (img_size[1], img_size[0]),),  # (imwidth, imheight)
	R=torch.tensor(rotation_matrix(0,np.pi,0) @ np.eye(3)).unsqueeze(0),
	T=torch.tensor([0, 0, 0]).unsqueeze(0), 
	device=device
)
blend_params = BlendParams()
renderer = MeshRenderer(
	rasterizer=MeshRasterizer(
		cameras=cameras, 
		raster_settings=RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1,)
	),
	# shader=SoftPhongShader(
	# 	device=device, 
	# 	cameras=cameras,
	# 	lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]]),
	# )
	shader=SimpleShader(
		device=device,
		blend_params=blend_params
	)
)

But how to create a lihgting source (PointLights, DirectionalLights), can you write me a renderer?
"""


class BlendParams(NamedTuple):
	sigma: float = 1e-4
	gamma: float = 1e-4
	background_color: Union[torch.Tensor, Sequence[float]] = (1.0, 1.0, 1.0)
	# add background image as a param here 
	background_image: Union[torch.Tensor, np.ndarray] = None# (H, W, 3) 

def hard_rgb_blend_with_background(colors, fragments, blend_params) -> torch.Tensor:
	N, H, W, K = fragments.pix_to_face.shape
	device = fragments.pix_to_face.device

	# Mask for the background.
	is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
	if blend_params.background_image is not None:
		if torch.is_tensor(blend_params.background_image):
			background_image = blend_params.background_image.to(device)
		else:
			background_image = colors.new_tensor(blend_params.background_image)  # (H, W, 3)
		# Format background image
		# (H, W, 3) -> (N, H, W, 3) -> select only pixels which are in the background using the mask 

		#  background_image[None, ...].expand(N, -1, -1, -1).size()) :  torch.Size([12, 448, 448, 3])
		bg_index = is_background[..., None]
		# print(torch.max(is_background))
		
		background_image_masked = background_image[None, ...].expand(N, -1, -1, -1)[is_background==True]

		# Set background color pixels to the colors from the background image
		pixel_colors = colors[..., 0, :].masked_scatter(
			is_background[..., None],
			background_image_masked
		)  # (N, H, W, 3)

		# Concat with the alpha channel.
		alpha = torch.ones((N, H, W, 1), dtype=colors.dtype, device=device)
		return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)
	else:
		background_color_ = blend_params.background_color
		if isinstance(background_color_, torch.Tensor):
			background_color = background_color_.to(device)
		else:
			background_color = colors.new_tensor(background_color_)
		# Find out how much background_color needs to be expanded to be used for masked_scatter.
		num_background_pixels = is_background.sum()

		# Set background color.
		pixel_colors = colors[..., 0, :].masked_scatter(
			is_background[..., None],
			background_color[None, :].expand(num_background_pixels, -1),
		)  # (N, H, W, 3)
		# Concat with the alpha channel.
		alpha = (~is_background).type_as(pixel_colors)[..., None]

		return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)
	


class SimpleShader(nn.Module):
	def __init__(self, device="cpu", blend_params=None):
		super().__init__()
		self.blend_params = blend_params if blend_params is not None else BlendParams()

	def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
		blend_params = kwargs.get("blend_params", self.blend_params)
		texels = meshes.sample_textures(fragments)
		images = hard_rgb_blend_with_background(texels, fragments, blend_params)
		return images  # (N, H, W, 3) RGBA image

class mySoftPhongShader(SoftPhongShader):
	def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
		cameras = kwargs.get("cameras", self.cameras)
		if cameras is None:
			msg = "Cameras must be specified either at initialization \
				or in the forward pass of SoftPhongShader"
			raise ValueError(msg)

		texels = meshes.sample_textures(fragments)
		lights = kwargs.get("lights", self.lights)
		materials = kwargs.get("materials", self.materials)
		blend_params = kwargs.get("blend_params", self.blend_params)
		colors = phong_shading(
			meshes=meshes,
			fragments=fragments,
			texels=texels,
			lights=lights,
			cameras=cameras,
			materials=materials,
		)
		znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
		zfar = kwargs.get("zfar", getattr(cameras, "zfar", 1200.0))
		images = softmax_rgb_blend(
			colors, fragments, blend_params, znear=znear, zfar=zfar
		)
		return images

def softmax_rgb_blend(
	colors: torch.Tensor,
	fragments,
	blend_params: BlendParams,
	znear: Union[float, torch.Tensor] = 1.0,
	zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
	"""
	RGB and alpha channel blending to return an RGBA image based on the method
	proposed in [1]
	  - **RGB** - blend the colors based on the 2D distance based probability map and
		relative z distances.
	  - **A** - blend based on the 2D distance based probability map.
	Args:
		colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
		fragments: namedtuple with outputs of rasterization. We use properties
			- pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
			  of the faces (in the packed representation) which
			  overlap each pixel in the image.
			- dists: FloatTensor of shape (N, H, W, K) specifying
			  the 2D euclidean distance from the center of each pixel
			  to each of the top K overlapping faces.
			- zbuf: FloatTensor of shape (N, H, W, K) specifying
			  the interpolated depth from each pixel to to each of the
			  top K overlapping faces.
		blend_params: instance of BlendParams dataclass containing properties
			- sigma: float, parameter which controls the width of the sigmoid
			  function used to calculate the 2D distance based probability.
			  Sigma controls the sharpness of the edges of the shape.
			- gamma: float, parameter which controls the scaling of the
			  exponential function used to control the opacity of the color.
			- background_color: (3) element list/tuple/torch.Tensor specifying
			  the RGB values for the background color.
		znear: float, near clipping plane in the z direction
		zfar: float, far clipping plane in the z direction
	Returns:
		RGBA pixel_colors: (N, H, W, 4)
	[0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
	Image-based 3D Reasoning'
	"""

	N, H, W, K = fragments.pix_to_face.shape
	device = fragments.pix_to_face.device
	pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)

	background_ = blend_params.background_color
	if not isinstance(background_, torch.Tensor):
		background = torch.tensor(background_, dtype=torch.float32, device=device)
	else:
		background = background_.to(device)

	# Weight for background color
	eps = 1e-10

	# Mask for padded pixels.
	mask = fragments.pix_to_face >= 0

	# Sigmoid probability map based on the distance of the pixel to the face.
	prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

	# The cumulative product ensures that alpha will be 0.0 if at least 1
	# face fully covers the pixel as for that face, prob will be 1.0.
	# This results in a multiplication by 0.0 because of the (1.0 - prob)
	# term. Therefore 1.0 - alpha will be 1.0.
	alpha = torch.prod((1.0 - prob_map), dim=-1)

	# Weights for each face. Adjust the exponential by the max z to prevent
	# overflow. zbuf shape (N, H, W, K), find max over K.
	# TODO: there may still be some instability in the exponent calculation.

	# Reshape to be compatible with (N, H, W, K) values in fragments
	if torch.is_tensor(zfar):
		# pyre-fixme[16]
		zfar = zfar[:, None, None, None]
	if torch.is_tensor(znear):
		znear = znear[:, None, None, None]

	z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
	z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
	weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

	# Also apply exp normalize trick for the background color weight.
	# Clamp to ensure delta is never 0.
	# pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
	delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

	# Normalize weights.
	# weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
	denom = weights_num.sum(dim=-1)[..., None] + delta

	# Sum: weights * textures + background color
	weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)

	## JQin: Use background image---------------------------------------------
	if blend_params.background_image is not None:
		if torch.is_tensor(blend_params.background_image):
			background = blend_params.background_image.to(device)
		else:
			background = colors.new_tensor(blend_params.background_image)  # (H, W, 3)
	## --------------------------------------------------------------------------------- 
	weighted_background = delta * background

	pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
	pixel_colors[..., 3] = 1.0 - alpha

	return pixel_colors
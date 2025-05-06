import os
import numpy as np
import imageio
import cv2
import h5py
import math


def add(to_write, key, value):  # noqa
	if key not in to_write:
		to_write[key] = [value]
	else:
		to_write[key].append(value)

def to_h5(to_write, output_path):
	for key, values in to_write.items():
		to_write[key] = np.asarray(values)
		# print('%s: ' % key, to_write[key].shape)
	
	if not os.path.isfile(output_path):
		with h5py.File(output_path, 'w') as f:
			for key, values in to_write.items():
				print("values.shape: ", values.shape)
				f.create_dataset(
					key, data=values,
					chunks=(
						tuple([1] + list(values.shape[1:]))
						if isinstance(values, np.ndarray)
						else None
					),
					compression='lzf',
					maxshape=tuple([None] + list(values.shape[1:])),
				)
				print("chunks: ", f[key].chunks)
	else:
		with h5py.File(output_path, 'a') as f:
			for key, values in to_write.items():
				if key not in list(f.keys()):
					print('write it to f {}'.format(output_path))
					f.create_dataset(
						key, data=values,
						chunks=(
							tuple([1] + list(values.shape[1:]))
							if isinstance(values, np.ndarray)
							else None
						),
						compression='lzf',
						maxshape=tuple([None] + list(values.shape[1:])),
					)
				else:
					data = f[key]
					data.resize(data.shape[0] + values.shape[0], axis=0)
					data[-values.shape[0]:] = values
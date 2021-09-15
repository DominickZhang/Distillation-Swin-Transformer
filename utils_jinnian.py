## Created by Jinnian Zhang
## Github: https://github.com/DominickZhang/mylib
## Copyright reserved

import argparse
import numpy as np
import os
import glob
import cv2

def parser_init():
	args = parser()
	eval_list = ['is_NAS', 'input_shape', 'regression', 'residual', 'amsgrad', 'is_adv_training', 'gpu']
	args = proc_args(args, eval_list)
	return args


def parser():
	parser = argparse.ArgumentParser(description="NAS for Robustness")
	parser.add_argument('--data_type', type=str, help="bravo or stealth")
	parser.add_argument('--validation_fold', type=int, help="integer number ranging from 0 to 7")
	parser.add_argument('--output_folder', type=str)

	parser.add_argument('--hyperparams_path', type=str, default='')
	parser.add_argument('--random_seed', type=int, default=0)
	parser.add_argument('--basemodel', type=str, default='BlockModel2D')
	parser.add_argument('--is_NAS', type=str, default='False')
	parser.add_argument('--model_file_name', type=str, default='2dunet.h5')
	parser.add_argument('--load_model', type=str, default='')

	##---------model parameters--------------
	parser.add_argument('--input_shape', type=str, default='(256, 256, 1)', help="must be 3D")
	parser.add_argument('--filt_num', type=int, default=16)
	parser.add_argument('--num_blocks', type=int, default=3)
	parser.add_argument('--conv_size', type=int, default=3)
	parser.add_argument('--output_chan', type=int, default=1)
	parser.add_argument('--regression', type=str, default='True')
	parser.add_argument('--residual', type=str, default='False')
	parser.add_argument('--is_adv_training', type=str, default='False')

	##----------optimizer parameters---------------
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--amsgrad', type=str, default='True')
	parser.add_argument('--learning_rate', type=float, default=1e-4)

	##----------loss---------------------
	parser.add_argument('--loss', type=str, default='MSE')

	##-------callback parameters---------------------
	parser.add_argument('--patience', type=int, default=5)

	##-------training related parameters------------------------
	parser.add_argument('--batch_size', type=int, default=6)
	parser.add_argument('--epochs', type=int, default=150)
	parser.add_argument('--epsilon', type=float, default=0.1)
	parser.add_argument('--adv_type', type=str, default='concat')
	parser.add_argument('--temperature', type=float, default=1)

	##-------nas----------------
	####-----nas search strategy-------------------------
	parser.add_argument('--nas_search_strategy', type=str, default='random')
	parser.add_argument('--max_evals', type=int, default=100)
	####-----nas training parameters---------------------
	parser.add_argument('--nas_batch_size', type=int, default=6)
	parser.add_argument('--nas_epochs', type=int, default=10)
	parser.add_argument('--nas_patience', type=int, default=1)

	## ZJN: currently not supporting multiple GPUs
	parser.add_argument('--gpu', type=str, default='0')
	args = parser.parse_args()
	return args

def read_h5_to_numpy(file_name, key):
	import h5py
	hf = h5py.File(file_name, 'r')
	data = hf[key][()]
	hf.close()
	return data


def parse_models(folder='output', file_name='2dunet.h5', keywords=[], exclude=[]):
	## Example:
	## -output
	## |-- folder0
	## |-- |-- 2dunet.h5
	## |-- |-- train.log
	## |-- folder1
	## |-- |-- 2dunet.h5
	## |-- |-- train.log
	## |-- test
	## |-- |-- test.log
	model_list = list()
	for element in glob.glob(os.path.join(folder, '*')):
		if os.path.isdir(element):
			basename = os.path.basename(element)
			if len(exclude) > 0:
				if any([keyword in basename for keyword in exclude]):
					continue
			if len(keywords) > 0:
				if any([keyword in basename for keyword in keywords]):
					file = os.path.join(element, file_name)
				else:
					continue
			else:
				file = os.path.join(element, file_name)
			if os.path.exists(file):
				model_list.append(file)
	return model_list

def proc_args(args, eval_list):
	## Usage:
	# >>> args = parser()
	# >>> eval_list = ['starting_point', 'starting_duel', 'is_kernel_fixed', 'warm_start', 'a1', 'a2', 'a3', 'is_fixed_step', 'x_range', 'x_list', 'is_creating_dual_samples', 'is_debug_mode', 'is_diff_mode', 'is_order_info', 'is_no_W']
	# >>> args.proc_args(args, eval_list)
	args_dict = vars(args)
	args_processed = dict()
	for name in args_dict.keys():
		value = args_dict[name]
		if name in eval_list:
			if value is not None:
				if type(value) is list:
					value_list = list()
					for string in value:
						value_list.append(eval(string))
					value = value_list
				else:
					value = eval(value)
		args_processed[name] = value
	return argparse.Namespace(**args_processed)

def volume2slice(volume_data):
	modality, row, column = volume_data.shape[1:4]
	return np.reshape(np.squeeze(np.transpose(volume_data, (0, 4, 2, 3, 1))), (-1, row, column, modality))

def volume2slice25D(volume_data):
	# TODO: need to fix the num_modality to be 1
	# This function should not support multi-modality case
	num_volume, num_modality, row, col, channel = volume_data.shape
	if num_modality > 1:
		return NotImplementedError()
	new_data = np.zeros((num_volume, num_modality, row, col, channel, 3))
	for i in range(num_volume):
		for j in range(num_modality):
			for k in range(channel):
				if k == 0:
					new_data[i, j, :,:, k, 1:] = volume_data[i, j, :,:, k:(k+2)]
				elif k == channel-1:
					new_data[i, j, :,:, k, 0:2] = volume_data[i, j, :,:, (k-1):]
				else:
					new_data[i, j, :,:, k, :] = volume_data[i, j, :,:, (k-1):(k+2)]
	return np.squeeze(np.transpose(new_data, (0, 1, 4, 2, 3, 5))).reshape(-1, row, col, 3)

def cal_complement_list(list1, list2):
	assert(len(set(list1)) == len(list1))
	return list(set(list1) - set(list2))

def resize_image_array(image_array, output_size=(256, 256), interpolation=None):
	## image_array: (..., row, col, channel)
	## interpolation: None, cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA
	assert(len(output_size) == 2)
	assert(len(image_array.shape) > 2)

	shape = image_array.shape
	row, col, channel = shape[-3:]
	output_shape = list(shape)
	output_shape[-3:] = [output_size[0], output_size[1], channel]
	output_shape = tuple(output_shape)

	image_array_new = image_array.reshape(-1, row, col, channel)
	num = len(image_array_new)
	output_image_array = np.zeros((num, output_size[0], output_size[1], channel))
	for i in range(num):
		image = image_array_new[i]
		if interpolation is None:
			resized_image = cv2.resize(image, output_size)
		else:
			resized_image = cv2.resize(image, output_size, interpolation=interpolation)
		if channel == 1:
			resized_image = resized_image[:,:,np.newaxis]
		output_image_array[i] = resized_image
	output_image_array = output_image_array.reshape(output_shape)
	return output_image_array


if __name__ == '__main__':
	print(parse_models(keywords=['25D']))
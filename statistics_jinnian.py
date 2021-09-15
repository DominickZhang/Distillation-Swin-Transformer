## Created by Jinnian Zhang
## Copyright reserved

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
'''
matplotlib.rcParams.update({'errorbar.capsize': 4})
font = {'size':14}
rc('text', usetex=True)
rc('font',**font)
'''

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import csv

from utils_jinnian import resize_image_array

def save_table(array, filename):
	with open(filename, 'w') as f:
		csv.writer(f).writerows(array)

def save_plot(x, array, output_name, title=None, legend=None, x_label=None, y_label=None, marker_color_list=None, transparent=True):
	## array: len(x) x N
	## marker_color_list: None or list with length N
	## color_char: 'r', 'b', 'c', etc.
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	line_plot(ax, x, array, title, legend, x_label, y_label, marker_color_list)
	plt.savefig(output_name, dpi=2000, transparent=transparent)
	plt.close()

def line_plot(ax, x, array, title=None, legend=None, x_label=None, y_label=None, marker_color_list=None):
	## array: len(x) x N
	## marker_color_list: None or list with length N
	## color_char: 'r', 'b', 'c', etc.
	if marker_color_list is None:
		marker_color_list = ['b-o', 'b-s', 'b-1', 'b-*', 'b-2', 'b-3', 'b-4']
	N = array.shape[1]
	for i in range(N):
		#plt.errorbar(np.array(epsilon)[index], mean[:, i], np.array([minvalue[:,i], maxvalue[:,i]]), #	elinewidth=4, alpha=0.2, marker=Marker_lib[count][1],mfc=color_lib[header_temp])
		ax.plot(x, array[:, i], marker_color_list[i])

	if x_label:
		ax.set_xlabel(x_label)
	if y_label:
		ax.set_ylabel(y_label)
	ax.grid()
	if title:
		ax.set_title(title)
	if legend:
		ax.legend(legend, loc="lower left")
		#plt.legend(legend, loc="upper right")
	#plt.savefig(output_name, dpi=2000, transparent=transparent)
	#plt.close()

def save_array_to_image(array, file_name):
	#print(array.shape)
	plt.imsave(file_name, array, cmap = matplotlib.cm.gray)

def scatter_plot(ax, data, colors=None, legend=True, gird_on=True, title=None, x_label=None, y_label=None, fit_line=None, fit_params=None, semilogx=False, semilogy=False, exclude_index=None):
	if colors is None:
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	if gird_on:
		ax.grid()
	#legend_list = list()
	x_all = list()
	y_all = list()
	for i, (name, values) in enumerate(data.items()):
		x = values[0]
		y = values[1]
		#legend_list.append(name)
		x_all += x
		y_all += y

		ax.scatter(x, y, color=colors[i % len(colors)], label=name)
	
	if title is not None:
		ax.set_title(title)
	if x_label is not None:
		ax.set_xlabel(x_label)
	if y_label is not None:
		ax.set_ylabel(y_label)

	def func_linear(x, a, b):
		return a*x+b
	def func_exponential(x, a, b):
		return a*np.exp(b*x)
		#return a*(x**b)
	def func_power(x, a, b):
		return a*(x**b)
	def cal_R_square(x, y):
		corr_mat = np.corrcoef(x,y)
		return corr_mat[0, 1]**2

	if fit_line is not None:
		x_all = np.array(x_all)
		y_all = np.array(y_all)
		if exclude_index is not None:
			x_all = np.delete(x_all, exclude_index)
			y_all = np.delete(y_all, exclude_index)
		x_all_sorted = np.sort(x_all)
		index = np.argsort(x_all)
		y_all_sorted = y_all[index]

		if fit_line == 'linear':
			#y_pred = func_linear(x_all_sorted, *fit_params)
			popt, pcov = curve_fit(func_linear, x_all_sorted, y_all_sorted)
			y_pred = func_linear(x_all_sorted, *popt)
			rsq_score = r2_score(y_all_sorted, y_pred)
			if popt[1] < 0:
				label = '$y=%.2gx-%.2g$, $R^2=%.2g$'%(popt[0], -popt[1], rsq_score)
			else:
				label = '$y=%.2gx+%.2g$, $R^2=%.2g$'%(popt[0], popt[1], rsq_score)
			#legend_list.append(label)
			print(popt)
		elif fit_line == 'exponential':
			y_pred = func_exponential(x_all_sorted, *fit_params)
		elif fit_line == 'power':
			#popt, pcov = curve_fit(func_exponential, x_all, y_all, bounds=(-0.2, 0.2))
			popt, pcov = curve_fit(func_power, x_all_sorted, y_all_sorted)
			#popt = fit_params
			y_pred = func_power(x_all_sorted, *popt)
			rsq_score = r2_score(y_all_sorted, y_pred)
			label = '$y=%.2gx^{%.2g}$, $R^2=%.2g$'%(popt[0], popt[1], rsq_score)
			print(popt)
		ax.plot(x_all_sorted, y_pred, label=label)
	if semilogx:
		ax.set_xscale("log")
	if semilogy:
		ax.set_yscale("log")
	if legend:
		ax.legend()

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, gird_on=True, title=None, x_ticks=None, x_label=None, x_ticks_frequency=1, x_ticks_rotation=0, x_ticks_precision=2):
	# https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
	"""Draws a bar plot with multiple bars per data point.

	Parameters
	----------
	ax : matplotlib.pyplot.axis
		The axis we want to draw our plot on.

	data: dictionary
		A dictionary containing the data we want to plot. Keys are the names of the
		data, the items is a list of the values.

		Example:
		data = {
			"x":[1,2,3],
			"y":[1,2,3],
			"z":[1,2,3],
		}

	colors : array-like, optional
		A list of colors which are used for the bars. If None, the colors
		will be the standard matplotlib color cyle. (default: None)

	total_width : float, optional, default: 0.8
		The width of a bar group. 0.8 means that 80% of the x-axis is covered
		by bars and 20% will be spaces between the bars.

	single_width: float, optional, default: 1
		The relative width of a single bar within a group. 1 means the bars
		will touch eachother within a group, values less than 1 will make
		these bars thinner.

	legend: bool, optional, default: True
		If this is set to true, a legend will be added to the axis.

	Examples:
		ax = plt.gca()
    	hist, bin_edges = np.histogram(ratio_list, bins=30)
    	data = {'baidu': hist}
    	bar_plot(ax, data, x_label='object ratio', x_ticks=bin_edges, x_ticks_frequency=3, x_ticks_precision=3)
   		plt.savefig('test.png')
	"""
	# Check if colors where provided, otherwhise use the default color cycle
	if colors is None:
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	# Number of bars per group
	n_bars = len(data)
	# The width of a single bar
	bar_width = total_width / n_bars
	# List containing handles for the drawn bars, used for the legend
	bars = []
	if gird_on:
		ax.grid()
	# Iterate over all data
	for i, (name, values) in enumerate(data.items()):
		# The offset in x direction of that bar
		x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
		# Draw a bar for every value of that type
		for x, y in enumerate(values):
			bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
		# Add a handle to the last drawn bar, which we'll need for the legend
		bars.append(bar[0])
	# Draw legend if we need
	if legend:
		ax.legend(bars, data.keys())
	if title is not None:
		ax.set_title(title)
	if x_ticks is not None:
		x_tick_loc = np.arange(len(x_ticks))[::x_ticks_frequency]
		ax.set_xticks(x_tick_loc)
		if isinstance(x_ticks, np.ndarray):
			x_ticks = x_ticks.round(decimals=x_ticks_precision)
		ax.set_xticklabels(x_ticks[::x_ticks_frequency], rotation=x_ticks_rotation)
	if x_label is not None:
		ax.set_xlabel(x_label)

def saveImage(img_array, text, text_prefix, text_label=None, n_row=1, n_col=2, path='./output/saved_images.png'):
	if (int(img_array.shape[1]) != text.shape[1]) and (int(img_array.shape[1]) != text.shape[1]+1):
		raise ValueError("The number of image pairs (%d) should be either equal to the \
			 length of PSNR (%d) or PSNR+1 (%d)."%(img_array.shape[1], text.shape[1], text.shape[1]+1))

	fontsize = 7
	#wspace = 0.01
	wspace = -0.1
	hspace = 0.01
	#plt.figure(figsize=[int(5*n_col/n_row), 5])
	#plt.figure(figsize=[5.0*n_col/n_row, 5.0])
	plt.figure(figsize=[n_col, n_row])
	## -------- resize images ----------
	img_array = resize_image_array(img_array, (256, 256))

	for i in range(n_row):
		for j in range(n_col):
			if (j == 0) and (text_label is not None):
				if len(text_label[i]) > 0:
					continue
			current = i*n_col + j
			plt.subplot(n_row, n_col, current+1)
			plt.gca().xaxis.set_major_locator(plt.NullLocator())
			plt.gca().yaxis.set_major_locator(plt.NullLocator())
			plt.axis('off')

			## ------------- plot image and the bottom texts -------------
			if (int(img_array.shape[1]) == text.shape[1]+1):
				if j > 0:
					if len(text_prefix[i]) > 0:
						if text[i, j] is not None:
							plt.text(10, 240, text_prefix[i]+"%.2f"%(text[i,j]), fontsize=fontsize, color='white')
						else:
							plt.text(10, 240, text_prefix[i], fontsize=fontsize, color='white')
			else:
				if len(text_prefix[i]) > 0:
					if text[i, j] is not None:
						plt.text(10, 240, text_prefix[i]+"%.2f"%(text[i,j]), fontsize=fontsize, color='white')
					else:
						plt.text(10, 240, text_prefix[i], fontsize=fontsize, color='white')
			plt.imshow(img_array[i,j], cmap='gray')
			#plt.imshow(img_array[current], cmap='gray',aspect='auto')

		## ---------- plot the upper left texts ----------------
		if text_label is not None:
			if len(text_label[i])>0:
				current = i*n_col
				plt.subplot(n_row, n_col, current+1)
				plt.gca().xaxis.set_major_locator(plt.NullLocator())
				plt.gca().yaxis.set_major_locator(plt.NullLocator())
				plt.axis('off')
				plt.text(10, 30, text_label[i], fontsize=fontsize, color='white')
				if int(img_array.shape[1]) == text.shape[1]:
					if len(text_prefix[i]) > 0:
						if text[i, 0] is not None:
							plt.text(10, 240, text_prefix[i]+"%.2f"%(text[i,0]), fontsize=fontsize, color='white')
						else:
							plt.text(10, 240, text_prefix[i], fontsize=fontsize, color='white')
				plt.imshow(img_array[i,0], cmap='gray')
	#plt.subplots_adjust(wspace=0, hspace=0)
	plt.subplots_adjust(wspace=wspace, hspace=hspace)
	# save figures
	# bbox_inches='tight': avoid the x-labels cut off in the figure
	# pad_inches: Amount of padding around the figure when bbox_inches is 'tight'. Default: 0.1
	plt.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0, dpi=2000)

if __name__ == '__main__':
	array = [['="0.5"', '="-"'],['="1e-6"', '="0.28"']]
	filename = 'test.csv'
	save_table(array, filename)
	print('done!')
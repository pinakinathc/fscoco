# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
import argparse
import tqdm
from bresenham import bresenham
from sklearn.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
plt.style.use('ggplot') # theme for plotting
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                     random_start=100)

""" Analysis Done on Sketches
 - Average Number of strokes v/s other object dataset

 - Average length of strokes

 - Average length of strokes across time interval

 - Percentage of sketch drawn across time interval

 """

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Perform analysis on collected sketches')
	parser.add_argument('--raw_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data/raw_data',
		help='path to directory with sketches')
	opt = parser.parse_args()
	side = 256

	avg_num_strokes = []
	avg_length_strokes = []
	avg_length_strokes_time = [[] for i in range(10)]
	percentage_stroke_time = [[] for i in range(10)]

	# get vector sketch data
	print ('Analyzing sketch, please wait...')
	for raw_file in tqdm.tqdm(glob.glob(os.path.join(opt.raw_dir, '*', '*.json'))[:10]):
		our_data = json.load(open(raw_file))

		start_time = our_data[0]['timestamp']
		end_time = our_data[-1]['timestamp']
		prevX, prevY = None, None
		count_strokes = 0
		stroke_len = 0
		percent_stroke = [[] for i in range(10)]

		count_strokes = 0
		stroke_len = 0

		for data_idx, points in enumerate(our_data):
			time = points['timestamp'] - start_time
			try:
				x, y = map(float, points['coordinates'])
			except:
				print (points)
				x, y = map(float, points['coopdinates'])

			x, y = int(x*side), int(y*side)
			pen_state = list(map(int, points['pen_state']))

			if pen_state == [1, 0, 0]:
				count_strokes += 1
				avg_length_strokes.append(stroke_len)
				time_idx = min(int(time/(end_time - start_time)*10), 9)
				avg_length_strokes_time[time_idx].append(stroke_len)
				percentage_stroke_time[time_idx].append(data_idx/len(our_data))
				stroke_len = 0
			else:
				stroke_len += 1

		if list(map(int, our_data[-1]['pen_state'])): # end case
			count_strokes += 1
		avg_num_strokes.append(count_strokes)

	print ('Average number of strokes in a sketch: ', np.mean(avg_num_strokes))
	print ('Average length of strokes in a sketch: ', np.mean(avg_length_strokes))
	print ('Median number of strokes in a sketch: ', np.median(avg_num_strokes))
	print ('Median length of strokes in a sketch: ', np.median(avg_length_strokes))
	print ('Average length of strokes across time: ',
		[np.mean(stokes) for stokes in avg_length_strokes_time])
	print ('Percentage of data drawn across time: ',
		[np.mean(strokes) for strokes in percentage_stroke_time])

	# plotting graph for stroke length vs normalised time
	xdata = np.array([i for i in range(len(avg_length_strokes_time))])
	ydata = np.array([np.mean(stokes) for stokes in avg_length_strokes_time])

	gp.fit(xdata[:, np.newaxis], ydata)
	xfit = np.linspace(0, 10, 1000)
	yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
	dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region

	plt.plot(xdata, ydata, 'or')
	plt.plot(xfit, yfit, '-', color='gray')
	plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
	                 color='gray', alpha=0.2)
	plt.show()

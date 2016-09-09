import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon
from math import sqrt, pi
import plotly as py
import plotly.graph_objs as go
from plotly import tools
from PIL import Image

def query_file(folder, query_list):
	'''
	utility function to query a directory for a file
	whose filepath contains a number of strings

	note: slow if folder contains many subdirectories,
	args:
		@a folder - directory to search
		@a query_list - list of strings that returned file should match
	'''

	for root, dirs, files in os.walk(folder):
		for file in files:
			match = all(q in (root+file) for q in query_list)

			if match:
				return os.path.join(root,file)

	return None

def get_groups_dir(vasc_dir, imgname):
	'''
	Get the groups folder for a particular image in the
	vascular data respository

	args:
		@a : vasc_dir - top level vascular data repository

		@a : imgname - name of the image, e.g. OSMSC0003
	'''
	folder = vasc_dir+imgname

	for root, dirs, files in os.walk(folder):
		if 'groups-cm' in root:
			return root

def get_group_from_file(fn):
	'''
	get a dictionary mapping group numbers to a list of 3d tuples

	@a: fn - path to the groups file
	'''
	group = {}
	f = open(fn, 'r')

	while True:
		name = f.readline()
		num = f.readline()
		meta = f.readline()

		if name in ['\n', '\r\n', ""]:
			print "end of file: ", fn
			return group

		else:
			num = int(num)
			group[num] = []

			#read empty line
			l = f.readline()
			#read first group point
			l = f.readline()
			while l not in ['\n', '\r\n']:
				l = l.split()
				l = [float(i) for i in l]

				group[num].append((l[0],l[1],l[2]))

				#read next line
				l = f.readline()

def get_vec_shift(group):
	'''
	takes a list of 3d point tuples that lie on a plane
	and returns their mean and two orthogonal vectors
	in the plane

	inputs:
	- group: list of (x,y,z) tuples
	'''
	group = [np.asarray(p) for p in group]
	group.append(group[0])
	l = len(group)
	p1 = group[0]
	p2 = group[int(np.floor(float(l)/4))]

	A = np.zeros((3,2))
	A[:,0] = p1[:]
	A[:,1] = p2[:]
	mu = np.mean(A,axis=1, keepdims=True)
	A = A - mu

	q,r = np.linalg.qr(A)

	return (q,mu)

def project_group(group, q, mu):
	'''
	Take 3d group of points and project into plane
	described by q, group is additionally centered at
	origin by subtracting mu
	'''
	group = [np.asarray(p) for p in group]
	group.append(group[0])
	group_centered = [(p-mu.T)[0] for p in group]

	twod_group = [p.dot(q) for p in group_centered]
	x = [p[0] for p in group_centered]
	y = [p[1] for p in group_centered]

	return (x,y)

def make_traces(X,Y,legend, mode="lines+markers"):
	'''
	helper function to make traces out of data for plotly plotting
	args:
		@a: X - list of X data
		@a: Y - list of Y data
		@a: legend - list of strings
		@a: title - string for the title of the plot
		@a: mode - plotly line style to use
	'''
	traces = []

	for x,y, leg in zip(X,Y,legend):

		trace = go.Scatter(
		x = x,
		y = y,
		mode = mode,
		name = leg
		)

		traces.append(trace)

	return traces

def make_image_trace(fn, bounds=[-1, 1, -1, 1]):
	'''
	make a plotly heatmap of a grayscale image

	args:
		@a: fn - filename (path) of image to plot
		@a: bounds - list with 4 elements indicating the
		x and y bounds on which to map the image
	'''
	im = Image.open(fn)

	imarray = np.array(im)

	trace=go.Heatmap(
		z=imarray,
		x=np.linspace(bounds[0],bounds[1],imarray.shape[0]),
		y=np.linspace(bounds[2],bounds[3],imarray.shape[1]),
		colorscale=[[0, 'rgb(0,0,0)'],[1, 'rgb(255,255,255)']])

	return trace

def make_heat_trace(X, bounds=[-1, 1, -1, 1]):
	'''
	make a plotly heatmap of a grayscale image

	args:
		@a: X - data to plot must be 2d array
		@a: bounds - list with 4 elements indicating the
		x and y bounds on which to map the image
	'''

	trace=go.Heatmap(
		z=X,
		x=np.linspace(bounds[0],bounds[1],X.shape[0]),
		y=np.linspace(bounds[2],bounds[3],X.shape[1]),
		colorscale=[[0, 'rgb(0,0,0)'],[1, 'rgb(255,255,255)']])

	return trace

def plot_data_plotly(X, Y, legend, title='plot', mode="lines+markers",
	fn='./plots/plot.html'):
	'''
	Utility function to plot sets of X,Y data in a plotly plot

	args:
		@a: X - list of X data
		@a: Y - list of Y data
		@a: legend - list of strings
		@a: title - string for the title of the plot
		@a: mode - plotly line style to use
		@a: fn: - filename to save the plot as
	'''

	traces = make_traces(X,Y,legend,mode)

	layout = go.Layout(
		title = title
		)

	fig = go.Figure(data=traces, layout=layout)

	py.offline.plot(fig, filename=fn)

def plot_data_subfigures(Xlist, Ylist, legends, subtitles, rows, cols,
	size='600', title='plot', fn='./plots/subfigure.html'):
	'''
	TODO: Get subplots to share same colors
	TODO: Enable background image plotting
		more generally how to separate trace creation from plotting

	utility function to plot multiple sets of data in
	different subfigures

	args:
		@a: Xlist - list of lists of X data
		@a: Ylist - list of lists of Y data
		@a: legends - list of lists of legends
		@a: rows - number of rows in subfigure grid
		@a: cols - number of columns in subfigure grid
		@a: size - dimension of plot window
		@a: title - title string
		@a: fn - path to save plot
	'''

	fig = tools.make_subplots(rows=rows, cols=cols, subplot_titles=subtitles)

	stop_ind = len(Xlist)
	ind = 0
	for i in range(1,rows+1):
		for j in range(1,cols+1):
			ind = (i-1)*cols + (j-1)

			if ind == stop_ind:
				break

			traces = make_traces(Xlist[ind], Ylist[ind], legends[ind])

			for trace in traces:

				fig.append_trace(trace,i,j)

		if ind == stop_ind:
			break

	fig['layout'].update(height=size, width=size, title=title)

	py.offline.plot(fig, filename=fn)

def scatter3d(imlist, minlist, maxlist):
	'''
	wrapper function for plotly's scatter3d

	args:
		@a imlist: list of three dimensional image data

		@a minlist: list of minimum pixel values to threshold
		for a pixel to appear in the plot

		@a maxlist: list of maximum pixel values to threshold
		for a pixel to appear in the plot
	'''
	traces = []

	for im,mi,ma in zip(imlist,minlist,maxlist):
		z,y,x = np.where((im >= mi) & (im < ma))

		trace = go.Scatter3d(
			x=x,
			y=y,
			z=z,
			mode='markers',
			marker=dict(
			    size=1,
			    opacity=0.8
			)
		)

		traces.append(trace)

	py.offline.plot(traces)

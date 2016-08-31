import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon
from math import sqrt, pi
import plotly as py
import plotly.graph_objs as go

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

def plot_data_plotly(X, Y, legend, title, mode="lines+markers",
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
	traces = []

	for x,y, leg in zip(X,Y,legend):
		print x
		print y
		print leg
		trace = go.Scatter(
		x = x,
		y = y,
		mode = mode,
		name = leg
		)

		traces.append(trace)

	layout = go.Layout(
		title = title
		)

	fig = go.Figure(data=traces, layout=layout)

	py.offline.plot(fig, filename=fn)
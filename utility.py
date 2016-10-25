import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon, Point
from math import sqrt, pi
import plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.tools import FigureFactory as FF
from PIL import Image
from skimage import measure
from vtk import vtkImageExport
from vtk.util import numpy_support
from skimage import measure
from sklearn.metrics import confusion_matrix
import vtk
from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten

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

def normalize_images(images):
	'''
	Max/min normalizes a set of images

	args:
		@a images shape = (N,W,H,C) where C is number of channels
	'''
	N, Pw, Ph, C = images.shape
	images_norm = np.zeros((N,Pw,Ph,C))

	maxs = np.amax(images, axis=(1,2,3))
	mins = np.amin(images,axis=(1,2,3))

	for i in range(0,N):
	    images_norm[i,:] = (images[i]-mins[i])/(maxs[i]-mins[i]+1e-6)
	return images_norm

def train_test_split(X,Y,split):
	"""
	Splits data into train and test subsets and returns split indices

	args:
		@a X - numpy array shape = (N,...)
		@a Y - Numpy array shape = (N,...)
		@a split - split ratio, denotes percentage of data to keep in training set
	"""
	N = X.shape[0]
	inds = np.random.permutation(N)
	split_index = int(split*N)
	train_inds = inds[:split_index]
	test_inds = inds[split_index:]

	X_train = X[train_inds]
	X_test = X[test_inds]
	Y_train = Y[train_inds]
	Y_test = Y[test_inds]

	return X_train, Y_train, X_test, Y_test, inds, train_inds, test_inds

def cum_error_dist(errors, dx):
	'''
	Computes the cumulative distribution of the error

	args:
		@a errs (list) - list of errors
		@a dx (float) - interval at which to compute distribution
	'''
	thresh_errs = []

	ts = np.arange(0,1+dx,dx)
	for t in ts:
		frac = float(np.sum(errors<=t))/len(errors)
		thresh_errs.append(frac)

	return (thresh_errs, ts)

def get_codes(directory):
	'''
	searches through files in a directory for edge_codes
	'''
	files = os.listdir(directory)

	codes = []
	for f in files:
		f_list = f.split('.')
		code = f_list[3]
		if (code not in codes) and (code != 'truth'):
			codes.append(code)

	return codes

def get_data(dataDir, reshape=True):
	'''
	Function to read 2d reslice data

	args:
		@a dataDir: (string) directory containing images.npy, segmentations.npy
		metadata.npy, contours.npy and names.txt
	'''
	imString = dataDir + 'images.npy'
	segString = dataDir + 'segmentations.npy'
	metaString = dataDir + 'metadata.npy'
	contourString = dataDir + 'contours.npy'

	images = np.load(imString)
	images = images.astype(float)
	segs = np.load(segString)
	meta = np.load(metaString)
	contours = np.load(contourString)
	f = open(dataDir+'names.txt')
	f = f.readlines()
	N, Pw, Ph = images.shape

	if reshape:
		images = images.reshape((N,Pw,Ph,1))
		segs = segs.reshape((N,Pw,Ph,1))

	return images,segs,contours,meta,f

def readVTKSP(fn):
	'''
	reads a vtk structured points object from a file
	'''
	sp_reader = vtk.vtkStructuredPointsReader()
	sp_reader.SetFileName(fn)
	sp_reader.Update()
	sp = sp_reader.GetOutput()
	return sp

def readVTKPD(fn):
	'''
	reads a vtk polydata object from a file
	'''
	pd_reader = vtk.vtkPolyDataReader()
	pd_reader.SetFileName(fn)
	pd_reader.Update()
	pd = pd_reader.GetOutput()
	return pd

def validSurface(pd):
	'''
	check whether a given surface is valid, lines!=0 and numpoints == numlines

	args:
		@a pd: vtk polydata object
	'''
	Nlines = pd.GetNumberOfLines()
	Npoints = pd.GetNumberOfPoints()

	if (Nlines == 0) or (Npoints == 0) or (Nlines != Npoints):
		return False
	else:
		return True

def VTKSPtoNumpy(vol):
	'''
	Utility function to convert a VTK structured points (SP) object to a numpy array
	the exporting is done via the vtkImageExport object which copies the data
	from the supplied SP object into an empty pointer or array

	C/C++ can interpret a python string as a pointer/array

	This function was shamelessly copied from
	http://public.kitware.com/pipermail/vtkusers/2002-September/013412.html
	args:
		@a vol: vtk.vtkStructuredPoints object
	'''
	exporter = vtkImageExport()
	exporter.SetInputData(vol)
	dims = exporter.GetDataDimensions()
	if (exporter.GetDataScalarType() == 3):
		dtype = UnsignedInt8
	if (exporter.GetDataScalarType() == 4):
		dtype = np.short
	if (exporter.GetDataScalarType() == 5):
		dtype = np.int16
	if (exporter.GetDataScalarType() == 10):
		dtype = np.float32
	a = np.zeros(reduce(np.multiply,dims),dtype)
	s = a.tostring()
	exporter.SetExportVoidPointer(s)
	exporter.Export()
	a = np.reshape(np.fromstring(s,dtype),(dims[2],dims[0],dims[1]))
	return a

def VTKSPtoNumpyFromFile(fn):
	'''
	reads a .vts file into a numpy array

	args:
		@a fn - string, filename of .sp file to read
	'''
	reader = vtk.vtkStructuredPointsReader()
	reader.SetFileName(fn)
	reader.Update()
	sp = reader.GetOutput()
	return VTKSPtoNumpy(sp)

def VTKPDPointstoNumpy(pd):
	'''
	function to convert the points data of a vtk polydata object to a numpy array

	args:
		@a pd: vtk.vtkPolyData object
	'''
	return numpy_support.vtk_to_numpy(pd.GetPoints().GetData())

def VTKPDReadAndReorder(fn):
	'''
	reads a polydata file, reorders the nodes and returns a numpy array

	args:
		@a fn: string, .vtp file to read
	'''
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(fn)
	reader.Update()
	pd = reader.GetOutput()
	if not validSurface(pd):
		return np.zeros((1,2))
	else:
		return reorder_and_convert(pd)

def reorder_and_convert(pd):
	'''
	takes a polydata file, converts it to numpy and reorders the nodes

	args:
		@a pd: vtkPolyData object

	conditions:
		validSurface(pd) == True
	'''
	C = getPDConnectivity(pd)
	O = getNodeOrdering(C)
	pd_np = VTKPDPointstoNumpy(pd)
	return pd_np[O,:]

def getPDConnectivity(pd):
	'''
	Uses the cells in a vtkPolyData object to get the point connectivity

	args:
		@a pd: vtk.vtkPolyData object
	'''
	C = {}
	N = pd.GetNumberOfLines()*3
	line_data = pd.GetLines().GetData()
	for i in range(0,N,3):
		size = line_data.GetTuple(i)[0]
		p1 = int(line_data.GetTuple(i+1)[0])
		p2 = int(line_data.GetTuple(i+2)[0])
		if not C.has_key(p1):
			C[p1] = []
		if not C.has_key(p2):
			C[p2] = []

		C[p1].append(p2)
		C[p2].append(p1)

	return C

def getNodeOrdering(C):
	'''
	Gets the node ordering of a graph by depth-first searching a connectiviy matrix

	args:
		@a C: dictionary of lists, each entry maps a node id to its neighbors
	'''
	visited = {}
	ordering = []
	n = 0

	while True:
		visited[n] = 1
		ordering.append(n)
		for k in C[n]:
			if not visited.has_key(k):
				n = k
				break
			else:
				n = 0
		if n == 0:
			break

	return ordering

def contourToSeg(contour, origin, dims, spacing):
	'''
	Converts an ordered set of points to a segmentation
	(i.e. fills the inside of the contour), uses the point in polygon method

	args:
		@a contour: numpy array, shape = (num points, 2), ordered list of points
		forming a closed contour
		@a origin: The origin of the image, corresponds to top left corner of image
		@a dims: (xdims,ydims) dimensions of the image corresponding to the segmentation
		@a spacing: the physical size of each pixel
	'''
	poly = Polygon(contour)
	seg = np.zeros((dims[0],dims[1]))

	for j in range(0,dims[0]):
	    for i in range(0,dims[1]):
	        x = origin[0] + (j+0.5)*spacing[0]
	        y = origin[1] + (i+0.5)*spacing[1]
	        p = Point(x,y)

	        if poly.contains(p):
	            seg[i,j] = 1
	return seg

def segToContour(segmentation, origin=[0.0,0.0], spacing=[1.0,1.0], isovalue=0.5):
	'''
	converts a segmentation in numpy format to a contour (ordered list of points)

	args:
		@a segmentation: numpy array, shape=(xdims,ydims), binary segmentation image
		@a origin: origin of the image
		@a spacing: physical size of each pixel

	notes:
		itk spacing is in terms of (x,y) and skimage contour gives points as
		(rows,columns) so the first column returned by skimage must be converted
		using the y spacing and the second column using the x spacing
	'''
	contours = measure.find_contours(segmentation, isovalue)
	#index = 0
	# if len(contours) > 1:
	# 	xdims,ydims = segmentation.shape
	# 	xcenter = xdims/2
	# 	ycenter = ydims/2
	# 	dist = 1000
	# 	for i in range(0,len(contours)):
	# 		center = np.mean(contours[i],axis=0)
	# 		new_dist = np.sqrt((xcenter-center[1])**2 + (ycenter-center[0])**2)
	# 		if new_dist < dist:
	# 			dist = new_dist
	# 			index = i
	returned_contours = []
	for c in contours:
		points = c
		contour = np.zeros((len(points),2))

		for i in range(0,len(points)):
			contour[i,1] = (points[i,0]+0.5)*spacing[1]+origin[1]
			contour[i,0] = (points[i,1]+0.5)*spacing[0]+origin[0]

		returned_contours.append(contour)
	return returned_contours

def segToOBG(seg, border_width=1):
	'''
	Converts a binary segmentation to a background/object/border label image as in
	"Object Boundary Guided Semantic Segmentation" by Huang et al

	args:
		@a seg: shape = (H,W,1), binary image pixels = 0 or 1
		@a border_width, width that the border labels will be expanded to
	'''
	H,W,C = seg.shape
	#empty array with 3 channels for background/object/border
	out = np.zeros((H,W,3))
	#initialize all pixels as background
	out[:,:,0] = 1
	row_mask = np.asarray([-1,-1,-1,0,0,1,1,1])
	col_mask = np.asarray([-1,0,1,-1,1,-1,0,1])
	boundary_inds = []
	for i in range(0,H):
		for j in range(0,W):
			if (i == 0) or (i == H-1) or (j == 0) or (j == W-1):
				out[i,j,0] = 1
				continue
			if seg[i,j,0] == 1:
				row_neighbors = row_mask+i
				col_neighbors = col_mask+j
				#if all neighbors are 1, then we are in the object
				if np.sum(seg[row_neighbors,col_neighbors,0])==8:
					out[i,j,1] = 1
					out[i,j,0] = 0
				#otherwise we are on the boundary
				else:
					out[i,j,2] = 1
					out[i,j,0] = 0
					boundary_inds.append((i,j))

	#thicken boundary
	for tup in boundary_inds:
		i = tup[0]
		j = tup[1]
		for k in range(-border_width,border_width+1):
			if (i+k >= 0) and (i+k < H):
				out[i+k,j,2] = 1
				out[i+k,j,0:2] = 0
			if (j+k >= 0) and (j+k < W):
				out[i,j+k,2] = 1
				out[i,j+k,0:2] = 0
	return out

def listSegToContours(segmentations, origins, spacings, isovalue=0.5):
	'''
	converts a list of segmentations to a list of lists of contours

	args:
		@a segmentations: list of segmentations shape = (N,W,H)
		@a origins: list of origins shape = (N,3)
		@a spacings: list of spacings shape = (N,3)
	'''
	contours = []
	for i in range(0,len(segmentations)):
		c = segToContour(segmentations[i], origins[i], spacings[i], isovalue)
		contours.append(c)
	return contours

def eccentricity(contour):
	'''
	calculates the ratio between minor and major axis of contour

	args:
		@a contour: list of points, shape = (N,2)
	'''
	origin = np.mean(contour,axis=0)
	xcomp = contour[:,0]-origin[0]
	ycomp = contour[:,1]-origin[1]
	dists = np.sqrt(xcomp**2 + ycomp**2)
	dmax = np.max(dists)
	dmin = np.min(dists)
	return dmin/dmax

def threshold(x,value):
	'''
	sets all values below value to 0 and above to 1

	args:
		@a x: the array to threshold
		@a value: the cutoff value
	'''
	inds = x < value
	x[x < value] = 0
	x[x >= value] = 1
	return x

def areaOverlapError(truth, edge):
	'''
	Function to calculate the area of overlap error between two contours

	args:
		@a truth: numpy array, shape=(num points,2)
		@a edge: same as truth
	'''
	truth_tups = zip(truth[:,0],truth[:,1])
	edge_tups = zip(edge[:,0],edge[:,1])

	t = Polygon(truth_tups)
	e = Polygon(edge_tups)
	if not e.is_valid:
		print "invalid geometry, error = 1.0"
		return 1.0
	Aunion = e.union(t).area
	Aintersection = e.intersection(t).area

	return 1.0-float(Aintersection)/Aunion

def listAreaOverlapError(Y_pred,Y_truth):
	'''
	computes the area overlap error for a list of contours and reference contours

	args:
		@a Y_pred, list of lists of contours (numpy arrays shape = (N,2))
		@a Y_truth, dimensions same as Y_pred
	'''
	errs = []
	for i in range(0,len(Y_pred)):

	    y_true = Y_truth[i]

	    contour_pred = Y_pred[i]

	    if len(contour_pred) == 0:
	        e = 1.0
	    else:
	        y_contour_pred = contour_pred[0]

	        if len(y_contour_pred) <= 2:
	            e = 1.0
	        else:
	            e = areaOverlapError(y_true, y_contour_pred)

	    errs.append(e)
	return errs

def confusionMatrix(ytrue,ypred, as_fraction=True):
	'''
	computes confusion matrix and (optionally) converts it to fractional form

	args:
		@a ytrue: vector of true labels
		@a ypred: vector of predicted labels
	'''
	H = confusion_matrix(ytrue,ypred)

	if not as_fraction:
		return H
	else:
		H = H.astype(float)
		totals = np.sum(H,axis=1)
		totals = totals.reshape((-1,1))
		H = H/(totals+1e-6)
		return np.around(H,2)

def makeFCN(input_shape=(64,64,1), Nfilters=32, Wfilter=3,
 	num_conv_1=3, num_conv_2=3, output_channels=1, mask=True, dense_layers=1,
	dense_size=64):
	'''
	Makes an FCN neural network
	'''
	x = Input(shape=input_shape)

	#main branch
	d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(x)
	d = BatchNormalization(mode=2)(d)

	for i in range(0,num_conv_1):
		d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
		d = BatchNormalization(mode=2)(d)

	d = Convolution2D(1,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

	#mask layer
	if mask:
		m = Flatten()(x)

		for i in range(0,dense_layers):
			m = Dense(dense_size, activation='relu')(m)

		m = Dense(input_shape[0]*input_shape[1], activation='relu')(m)

		m = Reshape(input_shape)(m)

		#merge
		d = merge([d,m], mode='mul')

	#finetune
	for i in range(0,num_conv_2):
		d = BatchNormalization(mode=2)(d)
		d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

	d = BatchNormalization(mode=2)(d)
	d = Convolution2D(output_channels,
		Wfilter,Wfilter,activation='sigmoid', border_mode='same')(d)

	FCN = Model(x,d)
	return FCN
#######################################################
# Plotly stuff
#######################################################

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

def make_heat_trace(X, bounds=[-1, 1, -1, 1], showscale=False):
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
		colorscale=[[0, 'rgb(0,0,0)'],[1, 'rgb(255,255,255)']],
		showscale=showscale)

	return trace

def heatmap(X, legend='image', title='heatmap', xaxis='x', yaxis='y',
	bounds=[-1, 1, -1, 1], fn='./plots/heatmap.html'):
	'''
	Utility function to plot a 2d heatmap using plotly

	args:
		@a: X - 2d array with image data
		@a: legend - string, legend to use
		@a: title - string, title of the plot
		@a: xaxis - string, name of the x axis
		@a: yaxis - string, name of the yaxis
		@a: bounds - list, len(list)==4, x and y bounds for the image
		denotes the range to label the x and y axis with
		@a: fn - string, filename to give the plot
	'''

	trace = make_heat_trace(X,bounds)

	layout = go.Layout(
		title = title,
		xaxis = dict(
			title = xaxis,
			autorange=True
		),
		yaxis = dict(
			title = yaxis,
			autorange=True
		)
	)

	fig = go.Figure(data=[trace], layout=layout)

	py.offline.plot(fig, filename=fn)

def plot_data_plotly(X, Y, legend, title='plot', mode="lines+markers",
	xlabel='x', ylabel='y', fn='./plots/plot.html'):
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
		title = title,
		xaxis=dict(
			title=xlabel
			),
		yaxis=dict(
			title=ylabel
			),
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

def scatter3d(imlist, minlist, maxlist, fn='./plots/scatter3d.html'):
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
		z,y,x = np.where((im >= mi) & (im <= ma))

		print "Scatter num points {}".format(len(z))

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

	py.offline.plot(traces, filename=fn)

def isoSurface3d(image, level=1, fn='./plots/isosurf3d.html'):
	'''
	wrapper function for plotly's isosurface 3d plots

	args:
		@a image: 3d array of numbers

		@a level: level at which to make an isocontour

		@a fn: string, filename to save plot as
	'''

	vertices, simplices = measure.marching_cubes(image, level)

	x,y,z = zip(*vertices)

	fig = FF.create_trisurf(
		x=x,
		y=y,
		z=z,
		simplices=simplices
	)

	py.offline.plot(fig, filename=fn)

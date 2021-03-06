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
import configparser
from keras.optimizers import Adam
import util_data
from scipy.interpolate import UnivariateSpline
import SimpleITK as sitk
from vtk.util import numpy_support
import re
#from emd import emd
import scipy
import tensorflow as tf
from skimage.filters import gaussian
from skimage.segmentation import active_contour

def mkdir(fn):
    if not os.path.exists(os.path.abspath(fn)):
        os.mkdir(os.path.abspath(fn))

def parse_config(fn):
	config = configparser.ConfigParser()
	config.read(fn)

	return config

def multi_replace(s,exprlist):

    for e in exprlist:
        s = s.replace(e,'')
    return s

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
        #print '{} {} {}'.format(name, num, meta)
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

def parsePathInfo(fn):
    """
    returns a dictionary containing, origin normal and tangent from paths in
    simvascular

    args:
        fn (string): filename of file containing path info on each line

    returns:
        res (dictionary): dict[image.path.point] = [p, t, tx]
    """

    l = open(fn).readlines()
    l = [k.split(' ') for k in l]

    for j in [4,6,8]:
        for i in range(len(l)):
            l[i][j] = l[i][j].translate(None,'()').split(',')
            l[i][j] = [float(k) for k in l[i][j]]

    res = {}
    for i in range(len(l)):
        res["{}.{}.{}".format(l[i][0],l[i][1],l[i][2])] = [l[i][4],l[i][6],l[i][8]]
    return res

def denormalizeContour(c,p,t,tx):
    """
    uses simvascular path info to transform a contour from 2d to 3d

    args:
        c (np array, (num points x 2)) - contour to transform
        p (np array 1x3) - 3d origin of contour
        t (np array 1x3) - normal vector of 3d contour
        tx (np array 1x3) - vector in 3d contour plane

    returns:
        res (np array, (num points x 3)) - 3d contour
    """
    c = np.array(c)
    if c.shape[1] == 2:
        c = np.hstack((c, np.zeros((c.shape[0],1))))
    p = np.array(p)
    t = np.array(t)
    tx = np.array(tx)

    ty = np.cross(t,tx)
    ty = ty/np.linalg.norm(ty)

    res = np.array([p + k[0]*tx + k[1]*ty for k in c])
    return res[:-1]

def normalizeContour(c,p,t,tx):
    """
    uses simvascular path info to transform contour into local 2d coordinates

    args:
        c (np array, (num points x 3)) - 3d contour to transform
        p (np array 1x3) - 3d origin of contour
        t (np array 1x3) - normal vector of 3d contour
        tx (np array 1x3) - vector in 3d contour plane

    returns:
        res (np array, (num points x 3)) - 3d contour
    """

    c = list(c)

    ty = np.cross(t,tx)
    ty = ty/np.linalg.norm(ty)

    c_p = [k-p for k in c]

    res = np.array([(k.dot(tx), k.dot(ty)) for k in c_p])
    #print '{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(p,t,tx,ty,c,c_p,res)
    return res

def groupsToPoints(folder):
    files = os.listdir(folder)
    groups = []
    points = []

    for f in files:
        if '.' not in f:
            groups.append(get_group_from_file(folder+f))

    for g in groups:
        for n in g.keys():
            for p in g[n]:
                points.append((p[0],p[1],p[2]))

    return points

def reconstructSurface(folder):
    pointSource = vtk.vtkProgrammableSource()

    def readPoints():
            output = pointSource.GetPolyDataOutput()
            points = vtk.vtkPoints()
            output.SetPoints(points)

            group_points = groupsToPoints(folder)

            for p in group_points:
                points.insertNextPoint(p[0],p[1],p[2])

    pointSource.SetExecuteMethod(readPoints)


    # Construct the surface and create isosurface.
    surf = vtk.vtkSurfaceReconstructionFilter()
    surf.SetInputConnection(pointSource.GetOutputPort())

    cf = vtk.vtkContourFilter()
    cf.SetInputConnection(surf.GetOutputPort())
    cf.SetValue(0, 0.0)

    # Sometimes the contouring algorithm can create a volume whose gradient
    # vector and ordering of polygon (using the right hand rule) are
    # inconsistent. vtkReverseSense cures this problem.
    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(cf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()

    map = vtk.vtkPolyDataMapper()
    map.SetInputConnection(reverse.GetOutputPort())
    map.ScalarVisibilityOff()

    surfaceActor = vtk.vtkActor()
    surfaceActor.SetMapper(map)
    surfaceActor.GetProperty().SetDiffuseColor(1.0000, 0.3882, 0.2784)
    surfaceActor.GetProperty().SetSpecularColor(1, 1, 1)
    surfaceActor.GetProperty().SetSpecular(.4)
    surfaceActor.GetProperty().SetSpecularPower(50)

    # Create the RenderWindow, Renderer and both Actors
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the actors to the renderer, set the background and size
    ren.AddActor(surfaceActor)
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(400, 400)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetPosition(1, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 0, 1)
    ren.ResetCamera()
    ren.GetActiveCamera().Azimuth(20)
    ren.GetActiveCamera().Elevation(30)
    ren.GetActiveCamera().Dolly(1.2)
    ren.ResetCameraClippingRange()

    iren.Initialize()
    renWin.Render()
    iren.Start()

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

def normalize_images(images, normalize='max'):
    '''
    Max/min normalizes a set of images

    args:
    	@a images shape = (N,W,H,C) where C is number of channels
    '''
    N, Pw, Ph = images.shape
    images_norm = np.zeros((N,Pw,Ph))

    if normalize=='max':
        maxs = np.amax(images, axis=(1,2))
        mins = np.amin(images,axis=(1,2))

        for i in range(0,N):
            images_norm[i,:] = (images[i]-mins[i])/(maxs[i]-mins[i]+1e-6)
        return images_norm

    if normalize == 'global_max':
        max_ = np.amax(images)
        min_ = np.amin(images)

        images_norm = (images-min_)/(max_-min_)
        return images_norm
    if normalize=='mean':
        pass

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
		if ('.vtp' in f):
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
	f = [s.replace('\n','') for s in f]
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
    if np.sum(dims) == 0:
        return np.zeros((1,64,64))
    if (exporter.GetDataScalarType() == 3):
    	dtype = UnsignedInt8
    if (exporter.GetDataScalarType() == 4):
    	dtype = np.short
    if (exporter.GetDataScalarType() == 5):
    	dtype = np.int16
    if (exporter.GetDataScalarType() == 10):
    	dtype = np.float32
    if (exporter.GetDataScalarType() == 11):
    	dtype = np.float64
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

def vtkSmoothPD(pd,iters=15,relax=0.1):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(pd)
    smoothFilter.SetNumberOfIterations(iters)
    smoothFilter.SetRelaxationFactor(relax)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()

    #Update normals on newly smoothed polydata
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputConnection(smoothFilter.GetOutputPort())
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOn()
    normalGenerator.Update()

    return normalGenerator.GetOutput()

def vtkRemapImageColor(img, ranges=[0,2000],mapRange=[0,1.0],satRange=[0.0,0.0]):
    table = vtk.vtkLookupTable()
    table.SetRange(ranges) # image intensity range
    table.SetValueRange(mapRange) # from black to white
    table.SetSaturationRange(satRange) # no color saturation
    table.SetRampToLinear()
    table.Build()

    # Map the image through the lookup table
    color = vtk.vtkImageMapToColors()
    color.SetLookupTable(table)
    color.SetInputData(img)
    color.Update()
    return color.GetOutput()
def addPdToRen(ren, pd):
    """
    adds a polydata object to a render window

    args:
        ren - vtk renderer object
        pd - vtk polydata object
    """

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    ren.AddActor(actor)

def VTKScreenshotPD(pds, elevations=[0], azimuths=[0], size=(600,600), fn='ren.png'):
    """
    this function loads a list of polydata objects into a renderwindow and screenshots them

    args:

    returns:
    """
    ren = vtk.vtkRenderer()
    renwin = vtk.vtkRenderWindow()
    renwin.AddRenderer(ren)
    renwin.SetSize(size[0], size[1])

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renwin)

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renwin)

    writer = vtk.vtkPNGWriter()

    for p in pds:
        addPdToRen(ren, p)

    iren.Initialize()

    for elevation in elevations:
        for azimuth in azimuths:
            ren.ResetCamera()
            ren.GetActiveCamera().Zoom(1.0)
            ren.GetActiveCamera().Elevation(elevation)
            ren.GetActiveCamera().Azimuth(azimuth)
            ren.SetBackground(1,1,1)

            renwin.Render()

            # screenshot code:
            w2if.Update()

            writer.SetFileName(fn[:-4]+'{}{}'.format(elevation,azimuth)+fn[-4:])
            writer.SetInputData(w2if.GetOutput())
            writer.Write()

    renwin.Finalize()
    iren.TerminateApp()
    del renwin, iren

def parsePathFile(fn):
    """
    parses a simvascular 2.0 path file
    """
    f = open(fn).readlines()

    paths={}

    expr1 = ['set ', 'gPathPoints', '(',')','{','}',',name','\n']
    expr2 = ['{','}','p ','t ', 'tx ', '(', '\\\n',' ']

    for i in range(len(f)):
        if ',name' in f[i]:
            s = f[i]
            s = multi_replace(s,expr1)

            s = s.split(' ')
            if not paths.has_key(s[0]):
                paths[s[0]] = {}
                paths[s[0]]['name'] = s[1]
            else:
                paths[s[0]]['name'] = s[1]

        if ',splinePts' in f[i]:
            j = i+1
            key = multi_replace(f[i],expr1).split(',')[0]
            if not paths.has_key(key):
                paths[key] = {}
                paths[key]['points'] = []
            else:
                paths[key]['points'] = []

            while 'tx' in f[j]:
                s = f[j]
                s = multi_replace(s,expr2).replace(')',',').split(',')[:-1]
                s = [float(x) for x in s]
                paths[key]['points'].append(s)

                j = j+1
            paths[key]['points'] = np.array(paths[key]['points'])

    return paths

def parseGroupFile(fn):
    """
    parses a simvascular groups file
    """
    f = open(fn).readlines()
    f = [i.replace('\r\n','') for i in f]
    f = [i.replace('\n','') for i in f]
    nrmExpr = 'nrm {.*} '
    posExpr = 'pos {.*} '
    xhatExpr = 'xhat {.*} '

    group = {}
    for i in range(len(f)):
        if 'xhat' in f[i]:
            group_num = int(f[i-1])

            s = f[i]

            xhat_string = re.search(xhatExpr, s).group()
            xhat_string = xhat_string.split('}')[0]
            xhat = [float(x) for x in xhat_string[6:].split(' ')]

            # pos_string = re.search(posExpr, s).group()
            # pos_string = pos_string.split('}')[0]
            # pos = [float(x) for x in pos_string[5:].split(' ')]

            nrm_string = re.search(nrmExpr, s).group()
            nrm_string = nrm_string.split('}')[0]
            nrm = [float(x) for x in nrm_string[5:].split(' ')]


            group[group_num] = {}
            group[group_num]['contour'] = []
            j = i+1
            while f[j] != '':

                tup = [float(x) for x in f[j].split(' ')]
                group[group_num]['contour'].append(tup)
                j = j+1

            group[group_num]['contour'] = np.array(group[group_num]['contour'])

            pos = np.mean(group[group_num]['contour'],axis=0)
            group[group_num]['points'] = list(pos) + nrm + xhat
            i = j

    return group

def getImageReslice(img, ext, p, n, x, asnumpy=False):
    """
    gets slice of an image in the plane defined by p, n and x

    args:
        @a img: vtk image (3 dimensional)
        @a ext: extent of the reslice plane [Xext,Yext]
        @a p ((x,y,z)): origin of the plane
        @a n ((x,y,z)): vector normal to plane
        @a x ((x,y,z)): x-axis in the plane

    returns:
        ret (itk image): image in the slice plane
    """
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(img)
    reslice.SetInterpolationModeToLinear()

    #Get y axis and make sure it satisfies the left hand rule
    tr = vtk.vtkTransform()
    tr.RotateWXYZ(-90,n)
    y = tr.TransformPoint(x)

    reslice.SetResliceAxesDirectionCosines(
        x[0],x[1],x[2],y[0],y[1],y[2],n[0],n[1],n[2])
    reslice.SetResliceAxesOrigin(p[0],p[1],p[2])

    delta_min = min(img.GetSpacing())
    px = delta_min*ext[0]
    py = delta_min*ext[1]

    reslice.SetOutputSpacing((delta_min,delta_min,delta_min))
    reslice.SetOutputOrigin(-0.5*px,-0.5*py,0.0)
    reslice.SetOutputExtent(0,ext[0],0,ext[1],0,0)

    reslice.Update()
    #print asnumpy
    if asnumpy:
         return VTKSPtoNumpy(reslice.GetOutput())
    else:
        return reslice.GetOutput()

def getAllImageSlices(img,paths,ext, asnumpy=False):
    """
    traverses points in an image and gets reslices at those points

    args:
        @a img (vtk image)
        @a paths (dictionary): paths['id']['name'] = path name
            paths['id']['points'] = array Nx9, (px,py,pz,nx,ny,nz,xx,xy,xz)
    """
    slices = []
    for k in paths.keys():
        if type(paths[k]['points'][0]) != list and type(paths[k]['points'][0]) != np.ndarray:
            p = paths[k]['points']
            i = getImageReslice(img,ext,p[:3],p[3:6],p[6:9], asnumpy)
            slices.append(i)
        else:
            for p in paths[k]['points']:
                i =getImageReslice(img,ext,p[:3],p[3:6],p[6:9], asnumpy)
                slices.append(i)
    return slices

def writeAllImageSlices(imgfn,pathfn,ext,output_dir):
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(imgfn)
    reader.Update()
    img = reader.GetOutput()

    parsed_path = parsePathFile(pathfn)
    slices = getAllImageSlices(img,parsed_path,ext)

    writer = vtk.vtkJPEGWriter()

    table = vtk.vtkLookupTable()
    scalar_range = img.GetScalarRange()
    table.SetRange(scalar_range[0], scalar_range[1]) # image intensity range
    table.SetValueRange(0.0, 1.0) # from black to white
    table.SetSaturationRange(0.0, 0.0) # no color saturation
    table.SetRampToLinear()
    table.Build()

    # Map the image through the lookup table
    color = vtk.vtkImageMapToColors()
    color.SetLookupTable(table)

    mkdir(output_dir)
    for i in range(len(slices)):
        color.SetInputData(slices[i])
        writer.SetInputConnection(color.GetOutputPort())
        writer.SetFileName(output_dir+'{}.jpg'.format(i))
        writer.Update()
        writer.Write()

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
	seg = np.zeros((int(dims[0]),int(dims[1])))

	for j in range(0,int(dims[0])):
	    for i in range(0,int(dims[1])):
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
    index = 0
    if len(contours) > 1:
    	xdims,ydims = segmentation.shape
    	xcenter = xdims/2
    	ycenter = ydims/2

    	dist = 1000
    	for i in range(0,len(contours)):
    		center = np.mean(contours[i],axis=0)
    		new_dist = np.sqrt((xcenter-center[1])**2 + (ycenter-center[0])**2)
    		if new_dist < dist:
    			dist = new_dist
    			index = i
    returned_contours = []
    for c in contours:
    	points = c
    	contour = np.zeros((len(points),2))

    	for i in range(0,len(points)):
    		contour[i,1] = (points[i,0]+0.5)*spacing[1]+origin[1]
    		contour[i,0] = (points[i,1]+0.5)*spacing[0]+origin[0]

    	returned_contours.append(contour)
    if len(returned_contours) > 0:
    	return returned_contours[index]
    else:
    	return []

def snake(img,origin=[0.0,0.0],r=0.3):
    s = np.linspace(0, 2*np.pi, 50)
    x = origin[0] + r*np.cos(s)
    y = origin[1] + r*np.sin(s)
    init = np.array([x, y]).T


    # snake = active_contour(gaussian(img, 1),
    #                        init, alpha=10.0, beta=100.0, gamma=0.01)

    snake = active_contour(gaussian(img, 1),init)

    return snake

def smoothContour(c, num_modes=10):
    if len(c) < 3:
        return np.array([[0.0,0.0],[0.0,0.0]]).T
    x = c[:,0]
    y = c[:,1]
    mu = np.mean(c,axis=0)

    x = x-mu[0]
    y = y-mu[1]

    xfft = np.fft.fft(x)
    yfft = np.fft.fft(y)

    xfft[num_modes:] = 0
    yfft[num_modes:] = 0

    sx = 2*np.fft.ifft(xfft)+mu[0]
    sy = 2*np.fft.ifft(yfft)+mu[1]

    return np.array([np.real(sx),np.real(sy)]).T

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
	y = np.copy(x)
	y[x < value] = 0
	y[x >= value] = 1
	return y

def get_extents(meta):
	extents = []
	N = len(meta[0])
	for i in range(0,N):
		spacing = meta[0,i]
		origin = meta[1,i]
		dims = meta[2,i]

		left = origin[0]
		right = origin[0]+dims[0]*spacing[0]
		bottom = origin[0]
		top = origin[0]+dims[0]*spacing[0]
		extents.append([left,right,bottom,top])
	return extents

def EMDSeg(truth, pred, dx=1.0):
    """
    computes the earht mover's distance between to segmentations

    args:
        truth,pred - numpy array (HxW)

    returns:
        earth mover distance over > 0 entries
    """
    inds_truth = np.array((truth > 0).nonzero()).T
    inds_pred = np.array((pred > 0).nonzero()).T

    return emd(inds_truth,inds_pred)*dx

def areaOverlapError(truth, edge):
    '''
    Function to calculate the area of overlap error between two contours

    args:
    	@a truth: numpy array, shape=(num points,2)
    	@a edge: same as truth
    '''
    if truth==[] or edge==[]:
        return 1.0
    if len(truth)<3 or len(edge)<3:
        return 1.0
    truth_tups = zip(truth[:,0],truth[:,1])
    edge_tups = zip(edge[:,0],edge[:,1])

    t = Polygon(truth_tups)
    e = Polygon(edge_tups)
    if not (e.is_valid and t.is_valid):
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
	        y_contour_pred = contour_pred

	        if len(y_contour_pred) <= 2:
	            e = 1.0
	        else:
	            e = areaOverlapError(y_true, y_contour_pred)

	    errs.append(e)
	return errs

def contourArea(contour):
	"""
	calculates the radius of a list of a (x,y) points

	args:
		@a contour, numpy array (num points, 2)
	"""
	tup = zip(contour[:,0],contour[:,1])

	p = Polygon(tup)

	return p.area

def contourRadius(contour):
    """
    calculates the radius of a list of a (x,y) points

    args:
    	@a contour, numpy array (num points, 2)
    """
    if len(contour) < 3:
        return 0.0
    tup = zip(contour[:,0],contour[:,1])

    p = Polygon(tup)

    return np.sqrt(p.area/np.pi)

def confusionMatrix(ytrue,ypred, as_fraction=True):
	'''
	computes confusion matrix and (optionally) converts it to fractional form

	args:
		@a ytrue: vector of true labels
		@a ypred: vector of predicted labels
	'''
	H = confusion_matrix(ytrue,ypred)

	if not as_fraction:
		print 'here'
		return H
	else:
		print 'there'
		H = H.astype(float)
		totals = np.sum(H,axis=1)
		totals = totals.reshape((-1,1))
		H = H/(totals+1e-6)
		return np.around(H,2)

def pd_to_numpy_vol(pd, spacing=[1.,1.,1.], shape=None, origin=None, foreground_value=255, backgroud_value = 0):
    if shape is None:
        bnds = np.array(pd.GetBounds())
        shape = np.ceil((bnds[1::2]-bnds[::2])/spacing).astype(int)+15
    if origin is None:
        origin = bnds[::2]+(bnds[1::2]-bnds[::2])/2

    #make image
    extent = np.zeros(6).astype(int)
    extent[1::2] = np.array(shape)-1

    imgvtk = vtk.vtkImageData()
    imgvtk.SetSpacing(spacing)
    imgvtk.SetOrigin(origin)
    imgvtk.SetExtent(extent)
    imgvtk.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.ones(shape[::-1]).ravel()*backgroud_value,  # ndarray contains the fitting result from the points. It is a 3D array
                                                deep=True, array_type=vtk.VTK_FLOAT)

    imgvtk.GetPointData().SetScalars(vtk_data_array)

    #poly2 stencil
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputWholeExtent(extent)
    pol2stenc.Update()

    #stencil to image
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(imgvtk)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOn()
    imgstenc.SetBackgroundValue(foreground_value)
    imgstenc.Update()

    ndseg = numpy_support.vtk_to_numpy(imgstenc.GetOutputDataObject(0).GetPointData().GetArray(0))
    return ndseg.reshape(shape[::-1])


def pd_to_itk_image(pd, ref_img, foreground_value=255, backgroud_value = 0):
    ndseg = pd_to_numpy_vol(pd, spacing=ref_img.GetSpacing(), shape=ref_img.GetSize(), origin=ref_img.GetOrigin() )
    segitk = sitk.GetImageFromArray(ndseg.astype(np.int16))
    segitk.CopyInformation(ref_img)
    return segitk

def jaccard3D_pd(pd1,pd2):
    """
    computes the jaccard distance error for two polydata objects

    returns (error) float
    """
    union = vtk.vtkBooleanOperationPolyDataFilter()
    union.SetOperationToUnion()
    union.SetInputData(0,pd1)
    union.SetInputData(1,pd2)
    union.Update()
    u = union.GetOutput()
    massUnion = vtk.vtkMassProperties()
    massUnion.SetInputData(u)

    intersection = vtk.vtkBooleanOperationPolyDataFilter()
    intersection.SetOperationToIntersection()
    intersection.SetInputData(0,pd1)
    intersection.SetInputData(1,pd2)
    intersection.Update()
    i = intersection.GetOutput()
    massIntersection = vtk.vtkMassProperties()
    massIntersection.SetInputData(i)

    return 1 - massIntersection.GetVolume()/massUnion.GetVolume()

def jaccard3D_pd_to_itk(pd1,pd2,ref_img):

    img1 = pd_to_itk_image(pd1,ref_img)
    img2 = pd_to_itk_image(pd2,ref_img)
    return jaccard3D_itk(img1,img2)

def jaccard3D_itk(img1,img2):
    """
    computes jaccard distance via itk images
    """
    i1 = img1>0
    i2 = img2>0

    filt = sitk.StatisticsImageFilter()

    filt.Execute(i1|i2)
    U = filt.GetSum()
    filt.Execute(i1&i2)
    I = filt.GetSum()

    return 1.0 - float(I)/(U+1e-6)

EPS = 1e-5
def balanced_cross_entropy(ytrue,ytarget):
    n1 = tf.reduce_sum(ytrue)
    n0 = tf.reduce_sum(1-ytrue)
    beta0 = tf.to_float(n1)/(n1+n0)
    beta1 = tf.to_float(n0)/(n1+n0)

    loss_mat = beta0*(ytrue-1)*tf.log(1-ytarget+EPS) - beta1*ytrue*tf.log(ytarget+EPS)

    return tf.reduce_mean(loss_mat)

def scaled_cross_entropy(ytrue,ytarget):

    beta0 = 1.0
    beta1 = 10.0

    loss_mat = beta0*(ytrue-1)*tf.log(1-ytarget+EPS) - beta1*ytrue*tf.log(ytarget+EPS)

    return tf.reduce_mean(loss_mat)

def roi_cross_entropy(ytrue,ytarget):

    beta0 = 1.0
    beta1 = 1.0
    roi = 50

    shape = ytarget.get_shape()
    midx = shape[1]/2
    midy = shape[2]/2
    mask = np.zeros(shape[1:])
    mask[midx-roi/2:midx+roi:2,midy-roi/2:midy+roi/2,:] = 1.0

    loss_mat = beta0*(ytrue-1)*tf.log(1-ytarget+EPS) - beta1*ytrue*tf.log(ytarget+EPS)
    loss_mat = loss_mat*mask
    return tf.reduce_mean(loss_mat)

def train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val, nb_batches, N=1000, downsample=False, lists=1,rotate=True,translate=20,crop=64):
    """Trains a model on 2d vascular data (optionally with boundary data)
    """
    train_loss = []
    val_loss = []
    x_train, y_train = vasc_train.get_subset(N,rotate=ROTATE,translate=translate,crop=crop)

    for lr in lrates:
    	opt = Adam(lr=lr)
        if (type(y_train)==list) or (y_train.shape[3] == 1):
            if LOSS_TYPE == 'binary':
                net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            elif LOSS_TYPE == 'balanced':
                net.compile(optimizer=opt, loss=balanced_cross_entropy, metrics=['accuracy'])
            elif LOSS_TYPE == 'scaled':
                net.compile(optimizer=opt, loss=scaled_cross_entropy, metrics=['accuracy'])
            elif LOSS_TYPE == 'roi':
                net.compile(optimizer=opt, loss=roi_cross_entropy, metrics=['accuracy'])
            else:
                raise RuntimeError('LOSS_TYPE Wrongly defined in train()')
        else:
        	train = y_train.reshape((y_train.shape[0],-1,3))
        	val = y_val.reshape((y_val.shape[0],-1,3))
        	net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        for j in range(nb_batches):
            print "Batch {}, lr {}".format(j,lr)
            x_val, y_val = vasc_val.get_subset(vasc_val.images.shape[0],rotate=False,translate=None,crop=crop)
            x_train, y_train = vasc_train.get_subset(N,rotate=rotate,translate=translate,crop=crop)
            if lists > 1:
                y_train = [y_train]*lists
                y_val = [y_val]*lists
            if downsample:
                downsampled_train = np.array([scipy.misc.imresize(y[:,:,0],(crop/2,crop/2),'nearest') for y in y_train])
                downsampled_train = downsampled_train.reshape((y_train.shape[0],crop/2,crop/2,1))
                downsampled_train /= np.max(downsampled_train)

                downsampled_val = np.array([scipy.misc.imresize(y[:,:,0],(crop/2,crop/2),'nearest') for y in y_val])
                downsampled_val = downsampled_val.reshape((y_val.shape[0],crop/2,crop/2,1))
                downsampled_val /= np.max(downsampled_val)

                y_train = [y_train,downsampled_train]
                y_val = [y_val,downsampled_val]
            if (type(y_train)==list) or (y_train.shape[3] == 1):
                #net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
                history = net.fit(x_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data=(x_val,y_val))
            else:
            	train = y_train.reshape((y_train.shape[0],-1,3))
            	val = y_val.reshape((y_val.shape[0],-1,3))
            	#net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            	history = net.fit(x_train, train,batch_size=batch_size, nb_epoch=nb_epoch,
            	validation_data=(x_val,val))
    		train_loss = train_loss + history.history['loss']
    		val_loss = val_loss + history.history['val_loss']
    return net, train_loss, val_loss
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

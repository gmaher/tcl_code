import numpy as np
import utility.utility as utility
import argparse
import time

import os
from sklearn.metrics import roc_curve
import configparser
import SimpleITK as sitk
from medpy.metric.binary import hd, assd, ravd, dc
from joblib import Parallel, delayed
import multiprocessing
import vtk
np.random.seed(0)

DX = 0.01
ts = np.arange(0,1+DX,DX)
#########################################
# Parse arguments
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('vtkfolder')
parser.add_argument('pattern')
parser.add_argument('img')
parser.add_argument('--smooth', default=True)
args = parser.parse_args()
img_fn = args.img
pattern = args.pattern
folder = args.vtkfolder
smooth = args.smooth

#Get polydatas
files = os.listdir(folder)
files = [f for f in files if pattern in f]
pds = []
for f in files:
    pds.append(utility.readVTKPD(folder+f))

#Read img
reader = vtk.vtkMetaImageReader()
reader.SetFileName(img_fn)
reader.Update()
img = reader.GetOutput()


#getreslices
origin = img.GetOrigin()
ranges = img.GetScalarRange()
spacing = img.GetSpacing()
dims = img.GetExtent()
p = (-70*spacing[0],-70*spacing[1],-50*spacing[2])
# center = [origin[0]+spacing[0]*(float(dims[1]-dims[0]))/2,
# origin[1]+spacing[1]*(float(dims[3]-dims[2])/2),
# origin[2]+spacing[2]*(float(dims[5]-dims[4]))/2]
centers = [0]*3
centers[0] = [origin[0]+spacing[0]*(float(dims[1]-dims[0]))/2,
origin[1]+spacing[1]*(float(dims[3]-dims[2])/2),
origin[2]+spacing[2]*(float(dims[5]-dims[4])/2)]

centers[1] = [origin[0]+spacing[0]*(float(dims[1]-dims[0]))/2,
origin[1]+spacing[1]*(float(dims[3]-dims[2])/2),
origin[2]+spacing[2]*(float(dims[5]-dims[4])/2)]

centers[2] = [origin[0]+spacing[0]*float(dims[1]-dims[0])/2,
origin[1]+spacing[1]*(float(dims[3]-dims[2])/2),
origin[2]+spacing[2]*(float(dims[5]-dims[4]))/2]

axial = [(0,0,1),(1,0,0)]
coronal = [(0,1,0),(1,0,0)]
sagital = [(1,0,0),(0,0,1)]
ext = [200,200]

s = [0]*3
s[0] = utility.getImageReslice(img, ext, p, axial[0], axial[1], asnumpy=False)
s[1] = utility.getImageReslice(img, ext, p, coronal[0], coronal[1], asnumpy=False)
s[2] = utility.getImageReslice(img, ext, p, sagital[0], sagital[1], asnumpy=False)

actors = []
orientations = [(0.0,0.0,0.0),(90.0,0.0,0.0),(0.0,90.0,0.0)]
for s in s:
    actors.append(vtk.vtkImageActor())
    s = utility.vtkRemapImageColor(s,ranges=ranges,mapRange=[0,1],satRange=[0.0,0.0])
    actors[-1].SetInputData(s)


actors[0].SetOrientation(orientations[0])
actors[0].SetPosition(centers[0])
actors[1].SetOrientation(orientations[1])
actors[1].SetPosition(centers[1])
actors[2].SetOrientation(orientations[2])
actors[2].SetPosition(centers[2])

renderer = vtk.vtkRenderer()
for a in actors:
    renderer.AddActor(a)

smoother = vtk.vtkSmoothPolyDataFilter()
for p in pds:
    if smooth:
        p = utility.vtkSmoothPD(p,iters=20,relax=0.5)
        utility.addPdToRen(renderer,p)
    else:
        utility.addPdToRen(renderer,p)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

# Set up the interaction
interactorStyle = vtk.vtkInteractorStyleImage()
interactor = vtk.vtkRenderWindowInteractor()
#interactor.SetInteractorStyle(interactorStyle)
window.SetInteractor(interactor)

renderer.ResetCamera()
renderer.GetActiveCamera().Zoom(1.0)
#renderer.GetActiveCamera().Elevation(elevation)
#renderer.GetActiveCamera().Azimuth(azimuth)
renderer.SetBackground(1,1,1)

window.Render()
interactor.Start()

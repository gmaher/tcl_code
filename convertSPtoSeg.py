import numpy as np
import utility
import vtk
import argparse
from shapely.geometry import Polygon, Point

parser = argparse.ArgumentParser()
parser.add_argument('groupsDir')
args = parser.parse_args()
groupsDir = args.groupsDir

mag = groupsDir + 'OSMSC0001.arch.56.truth.mag.vts'
ls = groupsDir + 'OSMSC0001.arch.56.truth.ls.vtp'

contour = utility.VTKPDReadAndReorder(ls)
poly = Polygon(contour)

sp_reader = vtk.vtkStructuredPointsReader()
sp_reader.SetFileName(mag)
sp_reader.Update()
mag_sp = sp_reader.GetOutput()

spacing = mag_sp.GetSpacing()
origin = mag_sp.GetOrigin()
dims = mag_sp.GetDimensions()

seg = np.zeros((dims[0],dims[1]))

for j in range(0,dims[0]):
    for i in range(0,dims[1]):
        x = origin[0] + j*spacing[0]
        y = origin[1] + i*spacing[1]
        p = Point(x,y)

        if poly.contains(p):
            seg[i,j] = 1

utility.heatmap(seg, fn='./plots/seg.html')
utility.heatmap(utility.VTKSPtoNumpy(mag_sp)[0], fn='./plots/mag.html')

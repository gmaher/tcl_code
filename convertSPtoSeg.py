import numpy as np
import utility
import vtk
import argparse
from shapely.geometry import Polygon, Point
import os

parser = argparse.ArgumentParser()
parser.add_argument('groupsDir')
parser.add_argument('--plot', action='store_true', default=False)
args = parser.parse_args()
groupsDir = args.groupsDir
plot = args.plot

files = os.listdir(groupsDir)
images = []
segmentations = []
for f in files:
    if "truth.ls" in f:
        mag = f.replace('truth.ls.vtp','truth.mag.vts')
        ls = f

        contour = utility.VTKPDReadAndReorder(groupsDir+ls)
        poly = Polygon(contour)

        mag_sp = utility.readVTKSP(groupsDir+mag)

        spacing = mag_sp.GetSpacing()
        origin = mag_sp.GetOrigin()
        dims = mag_sp.GetDimensions()

        seg = utility.contourToSeg(contour, origin, dims, spacing)
        mag_np = utility.VTKSPtoNumpy(mag_sp)[0]

        segmentations.append(seg)
        images.append(mag_np)

if plot:
    for i in range(0,5):
        index = np.random.randint(len(segmentations))
        utility.heatmap(segmentations[index], fn='./plots/seg{}.html'.format(i))
        utility.heatmap(images[index],
        fn='./plots/mag{}.html'.format(i))

segmentations = np.asarray(seg)
images = np.asarray(images)

np.save(groupsDir+'segmentations', segmentations)
np.save(groupsDir+'images', images)

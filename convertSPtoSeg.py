import numpy as np
import utility
import vtk
import argparse
from shapely.geometry import Polygon, Point
import os

parser = argparse.ArgumentParser()
parser.add_argument('groupsDir')
parser.add_argument('--convert', action='store_true', default=False)
parser.add_argument('--plot', action='store_true', default=False)
args = parser.parse_args()
groupsDir = args.groupsDir
plot = args.plot
convert = args.convert

files = os.listdir(groupsDir)
images = []
segmentations = []
meta_data = [[],[],[]]
if convert:
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
            meta_data[0].append(spacing)
            meta_data[1].append(origin)
            meta_data[2].append(dims)

    segmentations = np.asarray(segmentations)
    images = np.asarray(images)
    meta_data = np.asarray(meta_data)

    np.save(groupsDir+'../segmentations', segmentations)
    np.save(groupsDir+'../images', images)
    np.save(groupsDir+'../metadata', meta_data)

else:
    images = np.load(groupsDir+'../images.npy')
    segmentations = np.load(groupsDir+'../segmentations.npy')
    meta_data = np.load(groupsDir+'../metadata.npy')

if plot:
    for i in range(0,5):
        index = np.random.randint(len(segmentations))
        utility.heatmap(segmentations[index], fn='./plots/seg{}.html'.format(i))
        utility.heatmap(images[index],
        fn='./plots/mag{}.html'.format(i))

        spacing = meta_data[0][index]
        origin = meta_data[1][index]
        dims = meta_data[2][index]

        segCon = utility.segToContour(segmentations[index],
        origin,
        spacing)

        utility.plot_data_plotly([segCon[:,0]], [segCon[:,1]], ['segcon'],
        fn='./plots/segcon{}.html'.format(i))

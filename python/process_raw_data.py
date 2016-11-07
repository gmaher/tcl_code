import numpy as np
import utility.utility as utility
import vtk
import argparse
from shapely.geometry import Polygon, Point
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('groupsDir')
parser.add_argument('outputDir')
parser.add_argument('--convert', action='store_true', default=False)
parser.add_argument('--plot', action='store_true', default=False)
args = parser.parse_args()
groupsDir = args.groupsDir
outputDir = args.outputDir
plot = args.plot
convert = args.convert

files = os.listdir(groupsDir)
images = []
segmentations = []
meta_data = [[],[],[]]
contours = []
contours_ls = []
names = open(groupsDir+'../names.txt','w')

eccentricity_limit = 0.2

if convert:
    for f in tqdm(files):
        if "truth.ls" in f:
            mag = f.replace('truth.ls.vtp','truth.mag.vts')
            ls = f
            ls_image = f.replace('truth','image')

            contour = utility.VTKPDReadAndReorder(groupsDir+ls)
            contour_image = utility.VTKPDReadAndReorder(groupsDir+ls_image)
            if utility.eccentricity(contour) < eccentricity_limit:
                continue

            #convert contours to 2d
            contour = contour[:,:2]
            contour_image = contour_image[:,:2]
            poly = Polygon(contour)

            mag_sp = utility.readVTKSP(groupsDir+mag)

            spacing = mag_sp.GetSpacing()
            origin = mag_sp.GetOrigin()
            dims = mag_sp.GetDimensions()

            seg = utility.contourToSeg(contour, origin, dims, spacing)
            mag_np = utility.VTKSPtoNumpy(mag_sp)[0]

            segmentations.append(seg)
            images.append(mag_np)
            contours.append(contour)
            contours_ls.append(contour_image)

            meta_data[0].append(spacing)
            meta_data[1].append(origin)
            meta_data[2].append(dims)
            names.write(f+'\n')
    segmentations = np.asarray(segmentations)
    images = np.asarray(images)
    meta_data = np.asarray(meta_data)

    np.save(outputDir+'segmentations', segmentations)
    np.save(outputDir+'images', images)
    np.save(outputDir+'metadata', meta_data)
    np.save(outputDir+'contours', contours)
    np.save(outputDir+'ls_image', contours_ls)
    names.close()

else:
    images = np.load(outputDir+'images.npy')
    segmentations = np.load(outputDir+'segmentations.npy')
    meta_data = np.load(outputDir+'metadata.npy')
    contours = np.load(outputDir+'contours.npy')
    contours_ls = np.load(outputDir+'ls_image.npy')

if plot:
    for i in range(0,1):
        index = np.random.randint(len(segmentations))
        utility.heatmap(segmentations[index], fn='./plots/seg{}.html'.format(i))
        utility.heatmap(images[index],
        fn='./plots/mag{}.html'.format(i))

        spacing = meta_data[0][index]
        origin = meta_data[1][index]
        dims = meta_data[2][index]

        segCon = utility.segToContour(segmentations[index],
        origin,
        spacing)[0]

        utility.plot_data_plotly([segCon[:,0]], [segCon[:,1]], ['segcon'],
        fn='./plots/segcon{}.html'.format(i))

        c = contours[index]
        utility.plot_data_plotly([c[:,0]], [c[:,1]], ['truth_con'],
        fn='./plots/truth_con{}.html'.format(i))

        c_ls = contours_ls[index]
        utility.plot_data_plotly([c_ls[:,0]], [c_ls[:,1]], ['ls_con'],
        fn='./plots/ls_con{}.html'.format(i))

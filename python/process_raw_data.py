import numpy as np
import utility.utility as utility
import vtk
import argparse
from shapely.geometry import Polygon, Point
import os
from tqdm import tqdm
import configparser
np.random.seed(0)
#############################
#Parse input arguments
#############################
parser = argparse.ArgumentParser()
parser.add_argument('groupsDir')
args = parser.parse_args()
groupsDir = args.groupsDir

#############################
#Parse config params
#############################
config = configparser.ConfigParser()
config.read('options.cfg')
dirs = {}
dirs['train'] = config['learn_params']['train_dir']
dirs['val'] = config['learn_params']['val_dir']
dirs['test'] = config['learn_params']['test_dir']
split = float(config['learn_params']['split'])

################################
# Create variables to store data
################################
files = os.listdir(groupsDir)
images = []
segmentations = []
meta_data = [[],[],[]]
contours = []
contours_ls = []
contours_edge = []
names = open(groupsDir+'../names.txt','w')

eccentricity_limit = 0.2

#################################
# Make model train/val/test split
#################################
#models = [f.split('.')[0] for f in files if 'OSMSC' in f]
models = open('./data/mr_images.list').readlines()
models = [m.replace('\n','') for m in models]
models = list(set(models))
inds = np.random.permutation(len(models))
cut = int(round(split*len(models)))

split_models = {}
split_models['test'] = [models[i] for i in inds[:cut]]
#split_models['test'].append('OSMSC0002')
split_models['val'] = [models[i] for i in inds[cut:2*cut] if models[i] != "OSMSC0002"]
split_models['train'] = [models[i] for i in inds[2*cut:] if models[i] != "OSMSC0002"]

split_inds = {}
split_inds['train'] = []
split_inds['val'] = []
split_inds['test'] = []

count = 0
files = [f for f in files if 'truth.ls' in f and (not "OSMSC0159" in f) and any(s.lower() in f.lower() for s in models)]

for f in tqdm(files):
    if "truth.ls" in f:
        mag = f.replace('truth.ls.vtp','truth.mag.vts')
        ls = f
        ls_image = f.replace('truth','image')
        ls_edge = f.replace('truth','edge96')

        contour = utility.VTKPDReadAndReorder(groupsDir+ls)
        contour_image = utility.VTKPDReadAndReorder(groupsDir+ls_image)
        contour_edge = utility.VTKPDReadAndReorder(groupsDir+ls_edge)
        if utility.eccentricity(contour) < eccentricity_limit:
            continue

        #convert contours to 2d
        contour = contour[:,:2]
        contour_image = contour_image[:,:2]
        contour_edge = contour_edge[:,:2]
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
        contours_edge.append(contour_edge)

        meta_data[0].append(spacing)
        meta_data[1].append(origin)
        meta_data[2].append(dims)
        names.write(f+'\n')
        #convert file to just file name
        file_model = f.split('.')[0]
        for k in split_models.keys():
            if any(file_model in s for s in split_models[k]):
                split_inds[k].append(count)
        count+=1

segmentations = np.asarray(segmentations)
images = np.asarray(images)
meta_data = np.asarray(meta_data)
names.close()

f = open(groupsDir+'../names.txt')
s = f.readlines()
for k in dirs.keys():
    group_names = open(dirs[k]+'names.txt','w')
    group_names.writelines([s[i] for i in split_inds[k]])
    group_names.close()
    np.save(dirs[k]+'segmentations', segmentations[split_inds[k]])
    np.save(dirs[k]+'images', images[split_inds[k]])
    np.save(dirs[k]+'metadata', meta_data[:,split_inds[k],:])
    np.save(dirs[k]+'contours', [contours[i] for i in split_inds[k]])
    np.save(dirs[k]+'ls_image', [contours_ls[i] for i in split_inds[k]])
    np.save(dirs[k]+'ls_edge', [contours_edge[i] for i in split_inds[k]])

f = open('./data/train.txt','w')
for m in split_models['train']:
    f.write(m+'\n')
f.close()

f = open('./data/val.txt','w')
for m in split_models['val']:
    f.write(m+'\n')
f.close()

f = open('./data/test.txt','w')
for m in split_models['test']:
    f.write(m+'\n')
f.close()
#
# else:
#     images = np.load(outputDir+'images.npy')
#     segmentations = np.load(outputDir+'segmentations.npy')
#     meta_data = np.load(outputDir+'metadata.npy')
#     contours = np.load(outputDir+'contours.npy')
#     contours_ls = np.load(outputDir+'ls_image.npy')

# if plot:
#     for i in range(0,1):
#         index = np.random.randint(len(segmentations))
#         utility.heatmap(segmentations[index], fn='./plots/seg{}.html'.format(i))
#         utility.heatmap(images[index],
#         fn='./plots/mag{}.html'.format(i))
#
#         spacing = meta_data[0][index]
#         origin = meta_data[1][index]
#         dims = meta_data[2][index]
#
#         segCon = utility.segToContour(segmentations[index],
#         origin,
#         spacing)[0]
#
#         utility.plot_data_plotly([segCon[:,0]], [segCon[:,1]], ['segcon'],
#         fn='./plots/segcon{}.html'.format(i))
#
#         c = contours[index]
#         utility.plot_data_plotly([c[:,0]], [c[:,1]], ['truth_con'],
#         fn='./plots/truth_con{}.html'.format(i))
#
#         c_ls = contours_ls[index]
#         utility.plot_data_plotly([c_ls[:,0]], [c_ls[:,1]], ['ls_con'],
#         fn='./plots/ls_con{}.html'.format(i))

import numpy as np
import utility.utility as utility
import argparse
import time

import os
from sklearn.metrics import roc_curve
import configparser
import SimpleITK as sitk
np.random.seed(0)
import vtk
import matplotlib.pyplot as plt
from tqdm import tqdm
#############################
#Parse input arguments
#############################
parser = argparse.ArgumentParser()
parser.add_argument('output_dir')
args = parser.parse_args()
output_dir = args.output_dir

ext = [191, 191]
#############################
# Get model names
#############################
dirs={}
dirs['train'] = output_dir+'train/'
dirs['val'] = output_dir+'val/'
dirs['test'] = output_dir+'test/'

split_models = {}
split_models['train'] = open('./data/jameson_train.txt').readlines()
split_models['val'] = open('./data/jameson_val.txt').readlines()
split_models['test'] = open('./data/jameson_test.txt').readlines()
for k in split_models.keys():
    split_models[k] = [s.replace('\n','') for s in split_models[k]]
models = split_models['train'] + split_models['val'] + split_models['test']

#############################
# Start processing data
#############################
reader = vtk.vtkMetaImageReader()
reader2 = vtk.vtkMetaImageReader()

mhas = open(output_dir+'images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]

truths = open(output_dir+'truths.txt').readlines()
truths = [i.replace('\n','') for i in truths]

paths = open(output_dir+'paths.txt').readlines()
paths = [i.replace('\n','') for i in paths]

#for i in range(len(mahs)):
images = []
contours3D = []
names = []
contours = []
meta_data = [[],[],[]]
minmaxes = []
segs = []
f = open(output_dir+'pathinfo.txt','w')

for i in tqdm(range(len(mhas))):
#for i in tqdm(range(6)):
    img = mhas[i]
    img_name = img.split('/')[-2]

    reader.SetFileName(img)
    reader.Update()
    the_image = reader.GetOutput()

    model = truths[i]
    model_name = model.split('/')[-2]

    print img,model

    reader2.SetFileName(model)
    reader2.Update()
    the_model = reader2.GetOutput()

    spacing = the_image.GetSpacing()
    dims = the_image.GetDimensions()
    origin = [-ext[0]*spacing[0]/2,ext[1]*spacing[1]/2]
    minmax = the_image.GetScalarRange()

    path_dict = utility.parsePathFile(paths[i])

    for k in path_dict.keys():

        names = names + ['{}.{}.{}'.format(img_name,path_dict[k]['name'],i) for i in range(len(path_dict[k]['points']))]

    #pathinfo.txt
    for k in path_dict.keys():
        pa = path_dict[k]['points']
        for i in range(len(pa)):
            p = pa[i]
            f.write('{} {} {} p ({},{},{}) t ({},{},{}) tx ({},{},{})\n'
            .format(img_name,path_dict[k]['name'],i,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]))

    tmpimages = utility.getAllImageSlices(the_image, path_dict, ext, True)
    #tmpimages = [(k-minmax[0])/(minmax[1]-minmax[0]+1e-5) for k in tmpimages]
    images = images + tmpimages

    tmpsegs = utility.getAllImageSlices(the_model, path_dict, ext, True)
    segs = segs + tmpsegs
    for j in range(len(tmpimages)):
        meta_data[0].append(spacing)
        meta_data[1].append(origin)
        meta_data[2].append([ext[0]+1,ext[1]+1,1])
        minmaxes.append(minmax)

f.close()

segs = [segs[i] for i in range(0,len(segs),3)]
images = [images[i] for i in range(0,len(images),3)]
minmaxes = [minmaxes[i] for i in range(0,len(minmaxes),3)]
meta_data = [[meta_data[0][i], meta_data[1][i], meta_data[2][i]] for i in range(0,len(meta_data[0]),3)]
names = [names[i] for i in range(0,len(names),3)]
meta_data = np.array(meta_data)
minmaxes=np.array(minmaxes)
segs = np.array(segs)
segs = segs/np.amax(segs)
images = np.array(images)
split_inds = {}
split_inds['train'] = []
split_inds['val'] = []
split_inds['test'] = []

for i in range(len(names)):
    file_model = names[i].split('.')[0]
    for k in split_models.keys():
        if any(file_model in s for s in split_models[k]):
            split_inds[k].append(i)


for k in dirs.keys():
    utility.mkdir(dirs[k])
    group_names = open(dirs[k]+'names.txt','w')
    group_names.writelines([names[i]+'\n' for i in split_inds[k]])
    group_names.close()
    np.save(dirs[k]+'segmentations', segs[split_inds[k]])
    np.save(dirs[k]+'images', images[split_inds[k]])
    np.save(dirs[k]+'metadata', meta_data[split_inds[k]])
    np.save(dirs[k]+'minmaxes', minmaxes[split_inds[k]])


f = open(dirs['train']+'train.txt','w')
for m in split_models['train']:
    f.write(m+'\n')
f.close()

f = open(dirs['val']+'val.txt','w')
for m in split_models['val']:
    f.write(m+'\n')
f.close()

f = open(dirs['test']+'test.txt','w')
for m in split_models['test']:
    f.write(m+'\n')
f.close()

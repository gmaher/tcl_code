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

ext = [99, 99]
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

mhas = open(output_dir+'images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]

truths = open(output_dir+'truths.txt').readlines()
truths = [i.replace('\n','') for i in truths]

groups = open(output_dir+'groups.txt').readlines()
groups = [i.replace('\n','') for i in groups]

#for i in range(len(mahs)):
images = []
contours3D = []
names = []
contours = []
meta_data = [[],[],[]]
segs = []
f = open(output_dir+'pathinfo.txt','w')

for i in tqdm(range(len(mhas))):
#for i in range(2):
    img = mhas[i]
    img_name = img.split('/')[-2]

    reader.SetFileName(img)
    reader.Update()
    the_image = reader.GetOutput()

    group_folder = groups[i]
    group_files = os.listdir(group_folder)

    spacing = the_image.GetSpacing()
    dims = the_image.GetDimensions()
    origin = [-ext[0]*spacing[0]/2,ext[1]*spacing[1]/2]
    minmax = the_image.GetScalarRange()

    for grp in group_files:

        group_dict = utility.parseGroupFile(groups[i]+'/'+grp)
        contours3D = contours3D + [group_dict[k]['contour'] for k in group_dict.keys()]
        contours = contours + \
            [utility.normalizeContour(group_dict[k]['contour'],\
            group_dict[k]['points'][:3],group_dict[k]['points'][3:6],group_dict[k]['points'][6:])\
             for k in group_dict.keys()]

        names = names + ['{}.{}.{}'.format(img_name,grp,k) for k in group_dict.keys()]

        #pathinfo.txt
        for k in group_dict.keys():
            p = group_dict[k]['points']
            f.write('{} {} {} p ({},{},{}) t ({},{},{}) tx ({},{},{})\n'
            .format(img_name,grp,k,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]))

        tmpimages = utility.getAllImageSlices(the_image, group_dict, ext, True)
        tmpimages = [(k-minmax[0])/(minmax[1]-minmax[0]+1e-5) for k in tmpimages]
        images = images + tmpimages

    for j in range(len(contours)):
        meta_data[0].append(spacing)
        meta_data[1].append(origin)
        meta_data[2].append([ext[0]+1,ext[1]+1,1])

f.close()
for i in tqdm(range(len(contours))):
    segs.append(utility.contourToSeg(contours[i], meta_data[1][i],\
    meta_data[2][i], meta_data[0][i]))

for i in range(0,10):
    k = np.random.randint(0,len(contours))
    plt.figure()
    plt.imshow(segs[k],extent=[origin[0], origin[0]+ext[0]*spacing[0],\
        -origin[1],-origin[1]+ext[1]*spacing[1]])
    plt.plot(contours[k][:,0],contours[k][:,1], linewidth=3, color='g')
    plt.savefig('{}.png'.format(k))


meta_data = np.array(meta_data)
segs = np.array(segs)
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
mag_seg = []
contours_ls = []
ls_seg = []

raw_dir = './data/64cabg/raw/'
for n in names:
    pname = n.split('.')
    ls = '{}.{}.{}.image.ls.vtp'.format(pname[0],pname[1],pname[2])
    contour = utility.VTKPDReadAndReorder(raw_dir+ls)
    contour = contour[:,:2]
    contours_ls.append(contour)

    mseg = '{}.{}.{}.seg3d.mag.vts'.format(pname[0],pname[1],pname[2])
    msegnp = utility.VTKSPtoNumpyFromFile(raw_dir+mseg)[0]
    mag_seg.append(msegnp)

mag_seg=np.array(mag_seg)

for k in dirs.keys():
    utility.mkdir(dirs[k])
    group_names = open(dirs[k]+'names.txt','w')
    group_names.writelines([names[i]+'\n' for i in split_inds[k]])
    group_names.close()
    np.save(dirs[k]+'segmentations', segs[split_inds[k]])
    np.save(dirs[k]+'images', images[split_inds[k]])
    #np.save(dirs[k]+'images_seg', im_seg[split_inds[k]])
    np.save(dirs[k]+'mag_seg', mag_seg[split_inds[k]])
    np.save(dirs[k]+'metadata', meta_data[:,split_inds[k]])
    np.save(dirs[k]+'contours', [contours[i] for i in split_inds[k]])
    np.save(dirs[k]+'ls_image', [contours_ls[i] for i in split_inds[k]])
    #np.save(dirs[k]+'ls_edge', [contours_edge[i] for i in split_inds[k]])
    #np.save(dirs[k]+'ls_seg', [contours_seg[i] for i in split_inds[k]])

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

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

ext = [63, 63]
#############################
# Get model names
#############################
output_dir = '/home/marsdenlab/projects/tcl_code/python/output/9/'
data_dir = '/home/marsdenlab/projects/tcl_code/python/data/OOF_2/'
#############################
# Start processing data
#############################
reader = vtk.vtkMetaImageReader()

mhas = open(data_dir+'output_images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]

#for i in range(len(mahs)):
im_dict = {}

f = open(data_dir+'pathinfo.txt','w')
paths = utility.parsePathInfo(data_dir+'pathInfo.txt')
path_keys = []
for i in tqdm(range(len(mhas))):
#for i in range(6):
    img = mhas[i]
    img_name = img.split('/')[-1]
    img_name = img_name.replace('-cm.mha','')

    reader.SetFileName(img)
    reader.Update()
    the_image = reader.GetOutput()

    spacing = the_image.GetSpacing()
    dims = the_image.GetDimensions()
    origin = [-ext[0]*spacing[0]/2,ext[1]*spacing[1]/2]
    minmax = the_image.GetScalarRange()

    locs = [k for k in paths.keys() if img_name in k]
    for l in locs:
        tup = paths[l]
        p = tup[0]
        n = tup[1]
        x = tup[2]
        im = utility.getImageReslice(the_image, ext, p, n, x, asnumpy=True)[0]
        im = im[range(ext[0],-1,0-1),:]
        im_dict[l] = im


f.close()

names = open(data_dir+'names.txt').readlines()
names = [n.replace('.truth.ls.vtp\n','') for n in names]
images = []
for n in names:
    images.append(im_dict[n])

images = np.array(images)
# images = images.reshape((images.shape[0],images.shape[1],images.shape[2]))
images = utility.normalize_images(images)
np.save(output_dir+'predictions/OOF_2.npy',images)
for i in range(0,50):
    k = np.random.randint(0,len(images))
    plt.figure()
    plt.imshow(images[k,:,:],cmap='gray')
    plt.colorbar()
    plt.savefig(output_dir+'OOF{}_2.png'.format(k))

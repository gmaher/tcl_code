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

mhas = open(output_dir+'images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]

truths = open(output_dir+'truths.txt').readlines()
truths = [i.replace('\n','') for i in truths]

groups = open(output_dir+'groups.txt').readlines()
groups = [i.replace('\n','') for i in groups]
group = groups[0]

group_files = os.listdir(groups[0])
g1 = group_files[0]
g = open(groups[0]+'/'+g1).readlines()
g = [i.replace('\r\n','') for i in g]
for s in g:
    if 'xhat' in s:
        string = s

the_group = utility.parseGroupFile(groups[0]+'/'+g1)

paths = open(output_dir+'paths.txt').readlines()
paths = [f.replace('\n','') for f in paths]
path = paths[0]

p = (2.723052,-0.230689,-12.755765)
z = (0.305266,-0.617498,0.724920)
x = (0.000000,0.761257,0.648451)
y = np.cross(z,x)

# p = (0,0,0)
# z = (0,0,1)
# x = (1,0,0)
# y = (0,1,0)

#read mha
# for i in tqdm(range(len(mhas))):
#     imgname = mhas[i].split('/')[-2]
#     utility.writeAllImageSlices(mhas[i], paths[i], [191,191], output_dir+imgname+'/')
#     utility.writeAllImageSlices(truths[i], paths[i], [191,191], output_dir+imgname+'_truth/')

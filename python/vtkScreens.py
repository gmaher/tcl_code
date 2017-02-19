import numpy as np
import utility.utility as utility
import argparse
import time

import os
from sklearn.metrics import roc_curve
import configparser
import SimpleITK as sitk
np.random.seed(0)

DX = 0.01
ts = np.arange(0,1+DX,DX)
#########################################
# Parse arguments
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('config_file')
args = parser.parse_args()

config_file = args.config_file
config = utility.parse_config(config_file)

output_dir = config['learn_params']['output_dir']
plot_dir = config['learn_params']['plot_dir']
vtk_dir = output_dir+'vtk/'

screen_dir = output_dir+'screens/'
pd_files = os.listdir(vtk_dir)

codes = [p.split('.') for p in pd_files if '.vtp' in p]
codes = [p[0]+'.'+p[1] for p in codes]
codes = list(set(codes))

xangles = [0,50,90, 120, 180,270]
yangles = [0,90,180,270]
for code in codes:
    files = [f for f in pd_files if code in f]
    pds = [utility.readVTKPD(vtk_dir+f) for f in files]

    utility.mkdir(screen_dir)
    utility.mkdir(screen_dir+code)

    fn = screen_dir+code+'/{}.png'.format(code)
    utility.VTKScreenshotPD(pds,elevations=xangles,azimuths=yangles,fn=fn)

print "starting 3d analysis"
vtk_dir = config['learn_params']['output_dir']+'vtk/'
files = os.listdir(vtk_dir)
files = [f for f in files if 'truth' in f]
mhas = open(output_dir+'images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]
img_file = 'blah'

codes = ['RSN','ls_seg','ls']
g = open(plot_dir+'jaccard3d.txt','w')
for c in codes:

    errs = []
    for f in files:
        mod = f.replace('truth',c)

        img_name = f.split('.')[0]
        img_file = [i for i in mhas if img_name in i][0]

        print f, mod, img_file

        ref_img = sitk.ReadImage(img_file)

        p1 = utility.readVTKPD(vtk_dir+f)
        p2 = utility.readVTKPD(vtk_dir+mod)
        e = utility.jaccard3D_pd_to_itk(p1,p2,ref_img)
        errs.append(e)
    g.write('{} : {}, {}\n'.format(c,np.mean(errs),np.std(errs)))
g.close()
import numpy as np
import utility.utility as utility
import argparse
import time

import os
from sklearn.metrics import roc_curve
import configparser
import SimpleITK as sitk
from medpy.metrics.binary import hd
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

xangles = [0,90,180]
yangles = [0,90,180]
#for code in codes:
#    files = [f for f in pd_files if code in f]
#    pds = [utility.readVTKPD(vtk_dir+f) for f in files]
#
#    utility.mkdir(screen_dir)
#    utility.mkdir(screen_dir+code)
#
#    for x in xangles:
#        for y in yangles:
#
#
#            fn = screen_dir+code+'/{}{}{}.png'.format(code,x,y)
#            utility.VTKScreenshotPD(pds,elevations=[x],azimuths=[y],fn=fn)

print "starting 3d analysis"
vtk_dir = config['learn_params']['output_dir']+'vtk/'
files = os.listdir(vtk_dir)
files = [f for f in files if 'truth' in f]
mhas = open(output_dir+'images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]
img_file = 'blah'

codes = ['I2INet','ls']
g = open(plot_dir+'3derrs.txt','w')
g.write('code, jacc_mean, jacc_std, hausdorf\n')
for c in codes:

    errs = []
    dorf = []
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

        np1 = pd_to_numpy_vol(p1, spacing=ref_img.GetSpacing(), shape=ref_img.GetSize(), origin=ref_img.GetOrigin() )
        np2 = pd_to_numpy_vol(p2, spacing=ref_img.GetSpacing(), shape=ref_img.GetSize(), origin=ref_img.GetOrigin() )
        if np.sum(np1) > 0 and np.sum(np2) > 0:
            e = hd(np1,np2)
            dorf.append(e)
    g.write('{} : {}, {}, {}\n'.format(c,np.mean(errs),np.std(errs),np.mean(dorf)))
g.close()

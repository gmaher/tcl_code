import numpy as np
import utility.utility as utility
import argparse
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

import utility.util_data as util_data
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import os
from sklearn.metrics import roc_curve
import configparser

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

dataDir = config['learn_params']['data_dir']+'test/'
image_dir = config['learn_params']['output_dir']+'groups/'
utility.mkdir(image_dir)
model_dir = config['learn_params']['model_dir']
pred_dir = config['learn_params']['pred_dir']

THRESHOLD = float(config['learn_params']['threshold'])
ISOVALUE = float(config['learn_params']['isovalue'])
NUM_MODES = int(config['learn_params']['num_modes'])
#make folder name for plots and make directory
plot_dir = config['learn_params']['plot_dir']

if not os.path.exists(os.path.abspath(plot_dir)):
    os.mkdir(os.path.abspath(plot_dir))

vasc = util_data.VascData2D(dataDir)
#############################################
# Get model and path info
#############################################
pinfo = utility.parsePathInfo(dataDir+'../../pathInfo.txt')

paths = open(dataDir+'names.txt').readlines()
paths = [s.split('.') for s in paths]
models = open(dataDir+'test.txt').readlines()
models = [m.replace('\n','') for m in models]

model_dict = {}
for k in models:
    model_dict[k] = {}

for i in range(len(paths)):
    p = paths[i]
    if not model_dict[p[0]].has_key(p[1]):
        model_dict[p[0]][p[1]] = []

    model_dict[p[0]][p[1]].append((i,p[2]))

##############################################
# Load segmentations, convert to contours
##############################################
pred = ['HED','ls', 'I2INet', 'truth', 'HEDFC', 'I2INetFC', 'FCI2INet']
for pred_type in pred:
    if pred_type == 'I2INet':
        meta = np.load(dataDir+'metadata.npy')
        seg = np.load(pred_dir+'I2INet.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'HED':
        meta = np.load(dataDir+'metadata.npy')
        seg = np.load(pred_dir+'HED.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'HEDFC':
        meta = np.load(dataDir+'metadata.npy')
        seg = np.load(pred_dir+'HEDFC.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'I2INetFC':
        meta = np.load(dataDir+'metadata.npy')
        seg = np.load(pred_dir+'I2INetFC.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'I2INetFCMask':
        meta = np.load(dataDir+'metadata.npy')
        seg = np.load(pred_dir+'I2INetFCMask.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'FCI2INet':
        meta = np.load(dataDir+'metadata.npy')
        seg = np.load(pred_dir+'FCI2INet.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'I2IVanilla':
        meta = np.load('/home/marsdenlab/projects/tcl_code/python/data/64final/jameson/test/metadata.npy')
        seg = np.load('/home/marsdenlab/projects/tcl_code/python/output/11/predictions/I2INet.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'HEDVanilla':
        meta = np.load('/home/marsdenlab/projects/tcl_code/python/data/64final/jameson/test/metadata.npy')
        seg = np.load('/home/marsdenlab/projects/tcl_code/python/output/11/predictions/HED.npy')
        seg = seg.reshape((seg.shape[:3]))
        seg_thresh = utility.threshold(seg,THRESHOLD)
        contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
            meta[0,:,:], ISOVALUE)

    if pred_type == 'ls':
        contour = vasc.contours_ls

    # if pred_type == 'ls_seg':
    #     meta = np.load(dataDir+'metadata.npy')
    #     seg = vasc.mag_seg/255
    #     seg_thresh = utility.threshold(seg,THRESHOLD)
    #     contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
    #         meta[0,:,:], ISOVALUE)

    if pred_type == 'truth':
        contour = vasc.contours

    ##############################################
    # output to groups files for each model
    ##############################################
    for m in model_dict.keys():
        groups_dir = image_dir+m+'.'+pred_type+'/'
        utility.mkdir(groups_dir)

        for p in model_dict[m].keys():
            points = model_dict[m][p]
            points = sorted(points, key=lambda x: int(x[1]))

            f = open(groups_dir+p,'w')
            for t in points:
                contour_index = t[0]
                c = contour[contour_index]
                key = '{}.{}.{}'.format(m,p,t[1])
                path_info = pinfo[key]

                if c != [] and len(c) > 1:
                    if pred_type != 'ls' and pred_type != 'truth':
                        c = utility.smoothContour(c, num_modes=NUM_MODES)
                    cdenorm = utility.denormalizeContour(c,path_info[0],path_info[1],
                        path_info[2])

                    f.write('/group/{}/{}\n'.format(p,t[1]))
                    f.write(t[1]+'\n')
                    f.write("posId {}\n".format(t[1]))

                    for cp in cdenorm:
                        f.write('{} {} {}\n'.format(cp[0],cp[1],cp[2]))
                    f.write('\n')
            f.close()

        #group contents file
        f = open(groups_dir+'group_contents.tcl','w')
        f.write("""# geodesic_groups_file 2.1
    #
    # Group Stuff
    #

    proc group_autoload {} {
        global gFilenames
        set grpdir $gFilenames(groups_dir)\n""")

        for p in model_dict[m].keys():
            f.write('   group_readProfiles {'+p+'} [file join $grpdir {'+p+'}]\n')

        f.write('}')
        f.close()

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

#make folder name for plots and make directory
plot_dir = config['learn_params']['plot_dir']

if not os.path.exists(os.path.abspath(plot_dir)):
    os.mkdir(os.path.abspath(plot_dir))

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
meta = np.load(dataDir+'metadata.npy')
seg = np.load(pred_dir+'FCN.npy')
seg = seg.reshape((seg.shape[:3]))
seg_thresh = utility.threshold(seg,THRESHOLD)
contour = utility.listSegToContours(seg_thresh, meta[1,:,:],
    meta[0,:,:], ISOVALUE)

##############################################
# output to groups files for each model
##############################################
for m in model_dict.keys():
    groups_dir = image_dir+m+'/'
    utility.mkdir(groups_dir)

    for p in model_dict[m].keys():
        points = model_dict[m][p]
        points = sorted(points, key=lambda x: int(x[1]))

        f = open(groups_dir+p+'pred','w')
        for t in points:
            f.write('/group/{}/{}\n'.format(p,t[1]))
            f.write(t[1]+'\n')
            f.write("posId {}\n".format(t[1]))

            key = '{}.{}.{}'.format(m,p,t[1])
            path_info = pinfo[key]
            contour_index = t[0]
            c = contour[contour_index]

            if c != []:
                c = utility.smoothContour(c, num_modes=3)
                cdenorm = utility.denormalizeContour(c,path_info[0],path_info[1],
                    path_info[2])

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
        f.write('   group_readProfiles {'+p+'pred} [file join $grpdir {'+p+'pred}]\n')

    f.write('}')
    f.close()

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
model_dir = config['learn_params']['model_dir']
pred_dir = config['learn_params']['pred_dir']
path_types = args.paths

THRESHOLD = float(config['learn_params']['threshold'])
ISOVALUE = float(config['learn_params']['isovalue'])

#make folder name for plots and make directory
plot_dir = config['learn_params']['plot_dir']
plot_dir = plot_dir+"{}/".format("_".join(path_types))

if not os.path.exists(os.path.abspath(plot_dir)):
    os.mkdir(os.path.abspath(plot_dir))

#############################################
# Get model and path info
#############################################
paths = open(dataDir+'names.txt').readlines()
models = open(dataDir+'test.txt').readlines()

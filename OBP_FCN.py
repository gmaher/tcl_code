import numpy as np
import utility
import argparse

from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

import util_data

np.random.seed(0)

###########################
# Parse arguments
###########################
parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
args = parser.parse_args()

dataDir = args.dataDir

##########################
# Load data
#########################
vascdata = util_data.VascData2D(dataDir)

#creates OBG data with pixel boundary 1
vascdata.createOBG()

N,Pw,Ph,C = vascdata.images_norm.shape

#train test split
X_train, Y_train, X_test, Y_test, inds, train_inds, test_inds = \
    utility.train_test_split(vascdata.images_norm, vascdata.obg,0.75)

print 'X_train.shape={},Y_train.shape={},X_test.shape={},Y_test.shape={}'.format(
    X_train.shape,Y_train.shape,X_test.shape,Y_test.shape
)

###############################
# Make and train Neural Network
###############################
FCN = utility.makeFCN((Ph,Pw,1), Nfilters, Wfilter, num_conv_1=3, num_conv_2=2,
    output_channels=3, mask=True, dense_layers=1, dense_size=Pw)

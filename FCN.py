import numpy as np
import utility
import argparse
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

import util_data

np.random.seed(0)
##########################
# Parse args
##########################
parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
args = parser.parse_args()

dataDir = args.dataDir

###########################
# Load data and preprocess
###########################
vascdata = util_data.VascData2D(dataDir)

images = vascdata.images_tf
segs = vascdata.segs_tf
meta = vascdata.meta
contours = vascdata.contours

N, Pw, Ph, C = images.shape

Pmax = np.amax(images)
Pmin = np.amin(images)

print "Images stats\n N={}, Width={}, Height={}, Max={}, Min={}".format(
    N,Pw,Ph,Pmax,Pmin)

images_norm = vascdata.images_norm

#train test split
X_train, Y_train, X_test, Y_test, inds, train_inds, test_inds = \
    utility.train_test_split(images_norm, segs,0.75)

##############################
# Neural network construction
##############################
Nfilters = 64
Wfilter = 3
lr = 1e-3
threshold = 0.3
output_channels = 1
dense_size = 100
opt = Adam(lr=lr)

x = Input(shape=(Ph,Pw,1))

#main branch
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(x)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(output_channels,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

#mask
m = Flatten()(x)
m = Dense(dense_size, activation='relu')(m)
m = Dense(Ph*Pw, activation='relu')(m)
m = Reshape((Ph,Pw,output_channels))(m)

#merge
d = merge([d,m], mode='mul')

#finetune
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(output_channels,
	Wfilter,Wfilter,activation='sigmoid', border_mode='same')(d)

FCN = Model(x,d)

FCN.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#FCN.compile(optimizer=opt, loss='hinge', metrics=['accuracy'])

###############################
# Training
###############################
FCN.fit(X_train, Y_train, batch_size=32, nb_epoch=10,
validation_data=(X_test,Y_test))

FCN.save('./models/FCN.h5')

###############################
# confusion matrix
###############################
Y_pred = FCN.predict(X_test)
Y_pred_flat = np.ravel(Y_pred)
Y_pred_flat = utility.threshold(Y_pred_flat,0.3)

Y_true_flat = np.ravel(Y_test)
Conf_mat = utility.confusionMatrix(Y_true_flat, Y_pred_flat)
print "Confusion matrix:\n {}".format(Conf_mat)

##############################
# Visualize
##############################
Ntest = Y_pred.shape[0]
for i in range(0,5):
    index = np.random.randint(Ntest)
    img = X_test[index]
    img = img.reshape((Pw,Ph))
    seg_pred = Y_pred[index]
    seg_pred = seg_pred.reshape((Pw,Ph))
    seg_true = Y_test[index]
    seg_true = seg_true.reshape((Pw,Ph))

    utility.heatmap(img, fn='./plots/segimg{}.html'.format(i))
    utility.heatmap(seg_pred, fn='./plots/segpred{}.html'.format(i))
    utility.heatmap(seg_true, fn='./plots/segtrue{}.html'.format(i))
    time.sleep(1)

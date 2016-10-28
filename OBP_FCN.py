import numpy as np
import utility
import argparse

from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

import util_data
import util_plot

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

Y_train = Y_train.reshape((Y_train.shape[0],Pw*Ph,3))
Y_test = Y_test.reshape((Y_test.shape[0],Pw*Ph,3))

print 'X_train.shape={},Y_train.shape={},X_test.shape={},Y_test.shape={}'.format(
    X_train.shape,Y_train.shape,X_test.shape,Y_test.shape
)

##############################
# Neural network construction
##############################
Nfilters = 64
Wfilter = 3
lr = 1e-3
threshold = 0.3
output_channels = 3
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
m = Dense(Ph*Pw*output_channels, activation='relu')(m)
m = Reshape((Ph,Pw,output_channels))(m)

#merge
d = merge([d,m], mode='mul')

#finetune
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(output_channels,
	Wfilter,Wfilter,activation='sigmoid', border_mode='same')(d)

d_out = Reshape((Ph*Pw,3))(d)

OBP_FCN = Model(x,d_out)
OBP_FCN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

OBP_FCN_output = Model(x,d)
OBP_FCN_output.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#OBP_FCN.compile(optimizer=opt, loss='hinge', metrics=['accuracy'])

###############################
# Training
###############################
OBP_FCN.fit(X_train, Y_train, batch_size=32, nb_epoch=30,
validation_data=(X_test,Y_test))

OBP_FCN.save('./models/OBP_FCN.h5')
OBP_FCN_output.save('./models/OBP_FCN_output.h5')

###############################
# confusion matrix
###############################
Y_pred = OBP_FCN.predict(X_test)
Y_pred = np.argmax(Y_pred,axis=2)
Y_pred_flat = np.ravel(Y_pred)

Y_true = np.argmax(Y_test,axis=2)
Y_true_flat = np.ravel(Y_true)
Conf_mat = utility.confusionMatrix(Y_true_flat, Y_pred_flat)
print "Confusion matrix:\n {}".format(Conf_mat)

##############################
# Visualize
##############################
Y_pred = OBP_FCN_output.predict(X_test)
Ntest = Y_pred.shape[0]
for i in range(0,1):
    index = np.random.randint(Ntest)
    img = X_test[index]
    img = img.reshape((Pw,Ph))
    seg_pred = Y_pred[index,:,:,1]
    boundary_pred = Y_pred[index,:,:,2]
    seg_test = Y_test[index]
    seg_test = seg_test.reshape((Pw,Ph,output_channels))
    seg_true = seg_test[:,:,1]
    boundary_true = seg_test[:,:,2]
    utility.heatmap(img, fn='./plots/segimg{}.html'.format(i))
    utility.heatmap(seg_pred, fn='./plots/segpred{}.html'.format(i))
    utility.heatmap(seg_true, fn='./plots/segtrue{}.html'.format(i))
    utility.heatmap(boundary_pred, fn='./plots/segpred{}.html'.format(i))
    utility.heatmap(boundary_true, fn='./plots/segtrue{}.html'.format(i))

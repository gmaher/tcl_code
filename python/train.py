import numpy as np
import utility.utility as utility
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

from utility import util_data
from utility import util_model
import argparse
np.random.seed(0)
##########################
# Parse args
##########################
#args
parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()
model_to_train = args.model

#config
config = utility.parse_config('options.cfg')
dataDir = config['learn_params']['train_dir']
valDir = config['learn_params']['val_dir']
testDir = config['learn_params']['test_dir']

vasc_train = util_data.VascData2D(dataDir)
vasc_train.createOBG(border_width=1)
vasc_val = util_data.VascData2D(valDir)
vasc_val.createOBG(border_width=1)
vasc_test = util_data.VascData2D(testDir)
vasc_test.createOBG(border_width=1)

##############################
# Neural network construction
##############################
Nfilters = 64
Wfilter = 3
lr = 1e-3
threshold = 0.3
output_channels = 1
dense_size = 64
dense_layers = 1
nb_epoch=10
batch_size=32
Pw=Ph=64
opt = Adam(lr=lr)
opt2 = Adam(lr=lr/10)
###############################
# Training
###############################
if model_to_train == 'FCN':
    net = util_model.FCN(Nfilters=Nfilters,Wfilter=Wfilter,
    dense_layers=dense_layers,dense_size=dense_size)
    net.name ='FCN'
    net = utility.train(net, lr, batch_size, nb_epoch, vasc_train, vasc_val)
    net = utility.train(net, lr/10, batch_size, nb_epoch, vasc_train, vasc_val)
    net.save('./models/FCN.h5')

if model_to_train == 'OBP_FCN':
    net,net_categorical = util_model.FCN(Nfilters=Nfilters,Wfilter=Wfilter, output_channels=3,
    dense_layers=dense_layers,dense_size=dense_size, obg=True)
    #have to manually compile net because it isn't directly trained
    net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    net.name = 'OBP_FCN'
    net_categorical = utility.train(net_categorical, lr, batch_size, nb_epoch, vasc_train, vasc_val,obg=True)
    net_categorical = utility.train(net_categorical, lr/10, batch_size, nb_epoch, vasc_train, vasc_val,obg=True)
    net.save('./models/OBP_FCN.h5')
    net_categorical.save('./models/OBP_FCN_categorical.h5')

if model_to_train == 'OBG_FCN':
    fcn = load_model('./models/FCN.h5')
    obp = load_model('./models/OBP_FCN.h5')

    net = util_model.OBG_FCN(fcn,obp,Nfilters=Nfilters,Wfilter=Wfilter)

    net = utility.train(net, lr, batch_size, nb_epoch, vasc_train, vasc_val)
    net = utility.train(net, lr/10, batch_size, nb_epoch, vasc_train, vasc_val)
    net.save('./models/OBG_FCN.h5')

if model_to_train == 'HED':
    net = load_model('./models/hed_bsds_vasc.h5')
    #high learning rate
    net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    net.fit(vasc_train.images_norm, [vasc_train.segs_tf]*6,
    batch_size=batch_size, nb_epoch=nb_epoch,
     validation_data=(vasc_val.images_norm,[vasc_val.segs_tf]*6))

    #lower learning rate
    net.compile(optimizer=opt2, loss='binary_crossentropy', metrics=['accuracy'])
    net.fit(vasc_train.images_norm, [vasc_train.segs_tf]*6,
    batch_size=batch_size, nb_epoch=nb_epoch,
     validation_data=(vasc_val.images_norm,[vasc_val.segs_tf]*6))
    net.save('./models/HED.h5')

if model_to_train == 'HED_dense':
    hed = load_model('./models/hed_bsds_vasc.h5')
    hed.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    net = util_model.hed_dense(hed, dense_size=dense_size)
    #high learning rate
    net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    net.fit(vasc_train.images_norm, [vasc_train.segs_tf],batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(vasc_val.images_norm,[vasc_val.segs_tf]))

    #lower learning rate
    net.compile(optimizer=opt2, loss='binary_crossentropy', metrics=['accuracy'])
    net.fit(vasc_train.images_norm, [vasc_train.segs_tf],batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(vasc_val.images_norm,[vasc_val.segs_tf]))
    net.save('./models/HED_dense.h5')

prediction = net.predict(vasc_test.images_norm)
np.save('./predictions/{}'.format(model_to_train), prediction)
###############################
# confusion matrix
###############################
X_test = vasc_val.images_norm
Y_pred = net.predict(X_test)
if 'HED' in model_to_train:
    Y_pred = Y_pred[0]
if Y_pred.shape[3] == 1:
    Y_test = vasc_val.segs_tf
else:
    Y_test = vasc_val.obg

Y_pred_flat = np.ravel(Y_pred)
Y_pred_flat = utility.threshold(Y_pred_flat,0.3)

Y_true_flat = np.ravel(Y_test)
Conf_mat = utility.confusionMatrix(Y_true_flat, Y_pred_flat)
print "Confusion matrix:\n {}".format(Conf_mat)

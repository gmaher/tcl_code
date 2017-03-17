import numpy as np
import utility.utility as utility
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from utility import util_data
from utility import util_model
import scipy
import argparse
np.random.seed(0)
##########################
# Parse args
##########################
#args
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config_file')
args = parser.parse_args()
model_to_train = args.model
config_file = args.config_file

#config
config = utility.parse_config(config_file)
dataDir = config['learn_params']['data_dir']+'train/'
valDir = config['learn_params']['data_dir']+'val/'
testDir = config['learn_params']['data_dir']+'test/'
output_dir = config['learn_params']['output_dir']
model_dir = config['learn_params']['model_dir']
pred_dir = config['learn_params']['pred_dir']
plot_dir = config['learn_params']['plot_dir']
mask = config['learn_params']['mask']=='True'
utility.mkdir(output_dir)
utility.mkdir(model_dir)
utility.mkdir(pred_dir)
utility.mkdir(plot_dir)

vasc_train = util_data.VascData2D(dataDir)
#vasc_train.createOBG(border_width=1)
vasc_val = util_data.VascData2D(valDir)
#vasc_val.createOBG(border_width=1)
vasc_test = util_data.VascData2D(testDir, rotate_data=False)
#vasc_test.createOBG(border_width=1)

##############################
# Neural network construction
##############################
Nfilters = 32
Wfilter = 3
num_conv=6
lr = 1e-3
threshold = 0.3
output_channels = 1
dense_size = 200
dense_layers = 1
nb_epoch=1
batch_size=64
Pw=Ph=int(config['learn_params']['image_dims'])
input_shape = (Pw,Ph,1)
opt = Adam(lr=lr)
lrates = [lr,lr/10,lr/100,lr/1000]
#lrates=[lr/10]
l2_reg=0.1
r_finetune = 0.5
N = 2500
nb_batches = 10
translate = 5
#lrates = [lr]
###############################
# Training
###############################
#train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val, nb_batches, downsample=False, lists=1,rotate=True,translate=20,crop=64)
if model_to_train == 'FCN':
    net, net_f = util_model.FCN(input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter,
    num_conv_1=num_conv,num_conv_2=num_conv,dense_layers=dense_layers,
    dense_size=dense_size, l2_reg=l2_reg, mask=mask)
    net.name ='FCN'
    net_f.name='FCN'
    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val, nb_batches=nb_batches, N=N, crop=Pw, translate=translate)
    net.save(model_dir+'FCN.h5')
    net_f.save(model_dir+'FCN_f.h5')

if model_to_train == 'FCN_multi':
    net_f = load_model(output_dir+'../4/models/FCN_f.h5')
    net = util_model.FCN_multi(net_f,input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter,
    l2_reg=l2_reg)

    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val, nb_batches=nb_batches, N=N, crop=Pw, translate=translate)
    net.save(model_dir+'FCN_multi.h5')

if model_to_train == 'FCN_finetune':
    net,net_f = util_model.FCN(input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter,
    num_conv_1=num_conv,num_conv_2=num_conv,dense_layers=dense_layers,
    dense_size=dense_size, l2_reg=l2_reg, mask=mask)
    net.name ='FCN_finetune'

    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train.images_norm, vasc_train.segs_tf,
         vasc_val.images_norm,vasc_val.segs_tf)

    N = vasc_train.images_norm.shape[0]
    rads = [utility.contourRadius(x) for x in vasc_train.contours]
    inds = [i for i in range(len(rads)) if rads[i] <= r_finetune]
    X_small = vasc_train.images_norm[inds]
    Y_small = vasc_train.segs_tf[inds]

    net,train_loss,val_loss =\
        utility.train(net, lrates[1:], batch_size, nb_epoch, X_small, Y_small,
         vasc_val.images_norm,vasc_val.segs_tf)

    net.save(model_dir+'FCN_finetune.h5')

if model_to_train == 'FC_branch':
    net = util_model.FC_branch(input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter,
    num_conv_1=num_conv,num_conv_2=num_conv,dense_layers=dense_layers,
    dense_size=dense_size, l2_reg=l2_reg, mask=mask)
    net.name ='FC_branch'

    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train.images_norm, vasc_train.segs_tf,
         vasc_val.images_norm,vasc_val.segs_tf)
    net.save(model_dir+'FC_branch.h5')

if model_to_train == 'ConvFC':
    net = util_model.ConvFC(input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter,
    num_conv_1=2*num_conv,dense_layers=dense_layers,
    dense_size=dense_size, l2_reg=l2_reg)
    net.name ='ConvFC'

    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train.images_norm, [vasc_train.segs_tf]*2,
         vasc_val.images_norm,[vasc_val.segs_tf]*2)
    net.save(model_dir+'ConvFC.h5')

if model_to_train == 'OBP_FCN':
    net,net_categorical = util_model.FCN(input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter, output_channels=3,
    dense_layers=dense_layers,dense_size=dense_size, obg=True,l2_reg=l2_reg, mask=mask)
    #have to manually compile net because it isn't directly trained
    net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    net.name = 'OBP_FCN'

    net_categorical,train_loss,val_loss = utility.train(net_categorical, lrates, batch_size, nb_epoch, vasc_train.images_norm,
     vasc_train.obg, vasc_val.images_norm,vasc_val.obg)

    net.save(model_dir+'OBP_FCN.h5')
    net_categorical.save(model_dir+'OBP_FCN_categorical.h5')

if model_to_train == 'OBG_FCN':
    fcn = load_model(model_dir+'FCN.h5')
    obp = load_model(model_dir+'OBP_FCN.h5')

    net = util_model.OBG_FCN(fcn,obp,input_shape=input_shape,Nfilters=Nfilters,Wfilter=Wfilter,l2_reg=l2_reg)

    net,train_loss,val_loss = utility.train(net, lrates, batch_size, nb_epoch, vasc_train.images_norm, vasc_train.segs_tf,
     vasc_val.images_norm,vasc_val.segs_tf)

    net.save(model_dir+'OBG_FCN.h5')

if model_to_train == 'HED':
    #net = load_model('./models/hed_bsds_vasc.h5')
    net = util_model.hed_keras(input_shape=input_shape, Wfilter=Wfilter, mask=mask, Nfilters=Nfilters,
    dense_layers=dense_layers, dense_size=dense_size, num_conv=num_conv, l2_reg=0.0)
    #high learning rate
    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val, nb_batches=nb_batches, N=N, crop=Pw, translate=translate, lists=4)
    net.save(model_dir+'HED.h5')

if model_to_train == 'HEDFC':
    #net = load_model('./models/hed_bsds_vasc.h5')
    net = util_model.hed_dense(input_shape=input_shape, Wfilter=Wfilter, mask=mask, Nfilters=Nfilters,
    dense_layers=dense_layers, dense_size=dense_size, num_conv=num_conv, l2_reg=0.0)
    #high learning rate
    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val, nb_batches=nb_batches, N=N, crop=Pw, translate=translate, lists=4)
    net.save(model_dir+'HEDFC.h5')

if model_to_train == 'I2INet':

    net = util_model.I2INet(input_shape=input_shape, Wfilter=Wfilter, mask=mask, Nfilters=Nfilters,
    dense_layers=dense_layers, dense_size=dense_size, num_conv=num_conv, l2_reg=0.0, batchnorm=True)
    #high learning rate

    net,train_loss,val_loss =\
        utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val,
            nb_batches=nb_batches, N=N, crop=Pw, translate=translate, downsample=True, lists=1)
    net.save(model_dir+'I2INet.h5')

if model_to_train == 'I2INetFC':

    net = util_model.I2INet_dense(input_shape=input_shape, Wfilter=Wfilter, mask=mask, Nfilters=Nfilters,
    dense_layers=dense_layers, dense_size=dense_size, num_conv=num_conv, l2_reg=0.0, batchnorm=True)
    #high learning rate

    net,train_loss,val_loss = utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val,
        nb_batches=nb_batches, N=N, crop=Pw, translate=translate, downsample=True, lists=1)
    net.save(model_dir+'I2INetFC.h5')

if model_to_train == 'I2INetFCMask':

    net = util_model.I2INet_dense_mask(input_shape=input_shape, Wfilter=Wfilter, mask=mask, Nfilters=Nfilters,
    dense_layers=dense_layers, dense_size=dense_size, num_conv=num_conv, l2_reg=0.0, batchnorm=True)
    #high learning rate

    net,train_loss,val_loss = utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val,
        nb_batches=nb_batches, N=N, crop=Pw, translate=translate, downsample=True, lists=1)
    net.save(model_dir+'I2INetFCMask.h5')

if model_to_train == 'FCI2INet':

    net = util_model.FCI2INet(input_shape=input_shape, Wfilter=Wfilter, mask=mask, Nfilters=Nfilters,
    dense_layers=dense_layers, dense_size=dense_size, num_conv=num_conv, l2_reg=0.0, batchnorm=True)
    #high learning rate

    net,train_loss,val_loss = utility.train(net, lrates, batch_size, nb_epoch, vasc_train, vasc_val,
        nb_batches=nb_batches, N=N, crop=Pw, translate=translate, downsample=True, lists=1)
    net.save(model_dir+'FCI2INet.h5')

images_norm = vasc_test.get_all(crop=Pw)[0]
prediction = net.predict(images_norm)
if "I2I" in model_to_train or model_to_train == 'HED' or model_to_train == "HEDFC" or model_to_train == 'I2INet' or model_to_train == 'ConvFC':
    prediction = prediction[0]

np.save(pred_dir+'{}'.format(model_to_train), prediction)

plt.figure()
plt.plot(range(0,len(train_loss)),train_loss, color='red', label='train loss', linewidth=2)
plt.plot(range(0,len(val_loss)),val_loss, color='green', label='validation loss', linewidth=2)
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig(plot_dir+'{}_loss.png'.format(model_to_train))
# ###############################
# # confusion matrix
# ###############################
# X_test = vasc_val.images_norm
# Y_pred = net.predict(X_test)
# if 'I2INet' in model_to_train or "HED" in model_to_train or model_to_train == 'HED' or model_to_train == 'I2INet' or model_to_train == 'ConvFC' or model_to_train == 'I2INetFC' or model_to_train == 'HEDFC':
#     Y_pred = Y_pred[0]
# if Y_pred.shape[3] == 1:
#     Y_test = vasc_val.segs_tf
# else:
#     Y_test = vasc_val.obg
#
# Y_pred_flat = np.ravel(Y_pred)
# Y_pred_flat = utility.threshold(Y_pred_flat,0.3)
#
# Y_true_flat = np.ravel(Y_test)
# Conf_mat = utility.confusionMatrix(Y_true_flat, Y_pred_flat.astype(int))
# print "Confusion matrix:\n {}".format(Conf_mat)

import numpy as np
import utility.utility as utility

from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

from utility import util_data
from utility import util_plot
from utility import util_model
np.random.seed(0)

##########################
# Parse args
##########################
config = utility.parse_config('options.cfg')
dataDir = config['learn_params']['train_dir']
valDir = config['learn_params']['val_dir']


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
Pw = Ph = 64
nb_epoch = 10
batch_size = 32

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

###############################
# Training
###############################
OBP_FCN,_ = utility.train(OBP_FCN, lr, batch_size, nb_epoch, dataDir, valDir, True,1)
OBP_FCN, Data = utility.train(OBP_FCN, lr/10, batch_size, nb_epoch, dataDir, valDir, True,1)

OBP_FCN.save('./models/OBP_FCN.h5')
OBP_FCN_output.save('./models/OBP_FCN_output.h5')

###############################
# confusion matrix
###############################
X_test = Data[2]
Y_test = Data[3]
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
    utility.heatmap(boundary_pred, fn='./plots/obgpred{}.html'.format(i))
    utility.heatmap(boundary_true, fn='./plots/obgtrue{}.html'.format(i))

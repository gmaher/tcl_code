import numpy as np
import utility.utility as utility
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm

from utility import util_data
from utility import util_model
np.random.seed(0)
##########################
# Parse args
##########################
config = utility.parse_config('options.cfg')
dataDir = config['learn_params']['train_dir']
valDir = config['learn_params']['val_dir']

vasc_train = util_data.VascData2D(dataDir)
vasc_train.createOBG(border_width=1)
vasc_val = util_data.VascData2D(valDir)
vasc_val.createOBG(border_width=1)

##############################
# Neural network construction
##############################
Nfilters = 64
Wfilter = 3
lr = 1e-3
threshold = 0.3
output_channels = 1
dense_size = 64
nb_epoch=10
batch_size=32
Pw=Ph=64

###############################
# Training
###############################
net = util_model.FCN()
net = utility.train(net, lr, batch_size, nb_epoch, vasc_train, vasc_val)
net = utility.train(net, lr/10, batch_size, nb_epoch, vasc_train, vasc_val)
net.save('./models/FCN.h5')

###############################
# confusion matrix
###############################
X_test = vasc_val.images_norm
Y_test = vasc_val.segs_tf
Y_pred = net.predict(X_test)
Y_pred_flat = np.ravel(Y_pred)
Y_pred_flat = utility.threshold(Y_pred_flat,0.3)

Y_true_flat = np.ravel(Y_test)
Conf_mat = utility.confusionMatrix(Y_true_flat, Y_pred_flat)
print "Confusion matrix:\n {}".format(Conf_mat)

##############################
# Visualize
##############################
Ntest = Y_pred.shape[0]
for i in range(0,1):
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

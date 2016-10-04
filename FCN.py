import numpy as np
import utility
import argparse
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization
from keras.optimizers import Adam
##########################
# Parse args
##########################
parser = argparse.ArgumentParser()
parser.add_argument('images')
parser.add_argument('segmentations')
args = parser.parse_args()

imString = args.images
segString = args.segmentations

###########################
# Load data and preprocess
###########################
images = np.load(imString)
images = images.astype(float)
segs = np.load(segString)

N, Pw, Ph = images.shape

images = images.reshape((N,Pw,Ph,1))
segs = segs.reshape((N,Pw,Ph,1))
Pmax = np.amax(images)
Pmin = np.amin(images)

print "Images stats\n N={}, Width={}, Height={}, Max={}, Min={}".format(
    N,Pw,Ph,Pmax,Pmin)

maxs = np.amax(images, axis=(1,2))
maxs = maxs[:,0]
mins = np.amin(images,axis=(1,2))
mins = mins[:,0]

images_norm = np.zeros((N,Pw,Ph,1))
for i in range(0,N):
    images_norm[i,:] = (images[i]-mins[i])/(maxs[i]-mins[i]+1e-6)

X_train, X_test, Y_train, Y_test = train_test_split(
    images_norm, segs, test_size=0.25, random_state=0
)

##############################
# Neural network construction
##############################
Nfilters = 32
Wfilter = 3
lr = 1e-3
threshold = 0.03

opt = Adam(lr=lr)

x = Input(shape=(Pw,Ph,1))
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(x)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(1,Wfilter,Wfilter,activation='sigmoid', border_mode='same')(d)

FCN = Model(x,d)
FCN.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#FCN.compile(optimizer=opt, loss='hinge', metrics=['accuracy'])

###############################
# Training
###############################
class_weight = {0:1.0,1:10}
FCN.fit(X_train, Y_train, batch_size=32, nb_epoch=10,
validation_data=(X_test,Y_test))

###############################
# confusion matrix
###############################
Y_pred = FCN.predict(X_test)
Y_pred_flat = np.ravel(Y_pred)
#Y_pred_flat = np.rint(Y_pred_flat)
Y_pred_flat[Y_pred_flat < threshold] = 0
Y_pred_flat[Y_pred_flat >= threshold] = 1

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

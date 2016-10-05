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

np.random.seed(0)
##########################
# Parse args
##########################
parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
args = parser.parse_args()

dataDir = args.dataDir
imString = dataDir + 'images.npy'
segString = dataDir + 'segmentations.npy'
metaString = dataDir + 'metadata.npy'
contourString = dataDir + 'contours.npy'

###########################
# Load data and preprocess
###########################
images = np.load(imString)
images = images.astype(float)
segs = np.load(segString)
meta = np.load(metaString)
contours = np.load(contourString)

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

#train test split
inds = np.random.permutation(N)
split_index = int(0.75*N)
train_inds = inds[:split_index]
test_inds = inds[split_index:]

X_train = images_norm[train_inds]
X_test = images_norm[test_inds]
Y_train = segs[train_inds]
Y_test = segs[test_inds]

##############################
# Neural network construction
##############################
Nfilters = 64
Wfilter = 3
lr = 1e-3
threshold = 0.03

opt = Adam(lr=lr)

x = Input(shape=(Pw,Ph,1))

#main branch
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(x)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,Wfilter,Wfilter,activation='relu', border_mode='same')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(1,Wfilter,Wfilter,activation='relu', border_mode='same')(d)

#mask layer
m = Flatten()(x)
m = Dense(Pw, activation='relu')(m)
m = Dense(Pw*Ph, activation='relu')(m)
m = Reshape((Pw,Ph,1))(m)

#merge
d = merge([d,m], mode='mul')

#finetune
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
Y_pred_flat = np.rint(Y_pred_flat)
#Y_pred_flat[Y_pred_flat < threshold] = 0
#Y_pred_flat[Y_pred_flat >= threshold] = 1

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

##################################
# Calculate area of overlap error
##################################
meta_test = meta[:,test_inds,:]
contours_test = contours[test_inds]

Y_pred = Y_pred.reshape((-1,64,64))
err_list = []
ts_list = []
isos = []
for iso in np.arange(0.1,1,0.1):
    isos.append(str(iso))
    errs = []
    for i in tqdm(range(0,len(Y_pred))):
        y = Y_pred[i]
        spacing, origin, dims = meta_test[:,i,:]
        y_true = contours_test[i]

        y = (y-np.amin(y))/(np.amax(y)-np.amin(y))

        y_contour_pred = utility.segToContour(y, origin, spacing,iso)

        e = utility.areaOverlapError(y_true, y_contour_pred)
        errs.append(e)

    cum_err, ts = utility.cum_error_dist(errs,0.025)
    err_list.append(cum_err)
    ts_list.append(ts)

utility.plot_data_plotly(ts_list, err_list, isos)

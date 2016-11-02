import numpy as np
import utility
import argparse
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm
from models.FCN import FCN

import util_data
import matplotlib.pyplot as plt

np.random.seed(0)
THRESHOLD = 0.4
ISOVALUE = 0.5
DX = 0.01

def get_level_sets(f, dataDir, inds, code):
    '''
    returns the level set contours
    '''
    f = [dataDir+s for s in f]
    f = [s.replace('truth',code) for s in f]
    f = [f[i] for i in inds]
    ret = [utility.VTKPDReadAndReorder(s) for s in f]
    return ret

def plot_segs(images, segs, n_ex=4, dim=(4,4), figsize=(10,10)):
    plot_count = 1
    plt.figure(figsize=figsize)

    for i in range(0,n_ex):
        plt.subplot(dim[0],dim[1],plot_count)
        img = images[i,:,:,0]
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plot_count += 1

        for seg in segs:
            plt.subplot(dim[0],dim[1],plot_count)
            plt.imshow(seg[i], cmap='gray')
            plt.axis('off')
            plot_count += 1

    plt.tight_layout()

def plot_contours(images, contours, extents, n_ex=4, dim=(4,4), figsize=(10,10)):
    plt.figure(figsize=figsize)
    plot_count = 1
    for i in range(0,n_ex):
        img = images[i,:,:,0]

        for contour in contours:
            plt.subplot(dim[0],dim[1],plot_count)
            plt.imshow(img, extent=extents[i], cmap='gray')
            if len(FCN_contour[i]) > 0:
                plt.scatter(contour[i][0][:,0],contour[i][0][:,1])
            else:
                plt.scatter([0],[0])
            plt.axis('off')
            plot_count+=1

    plt.tight_layout()

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
images,segs,contours,meta,f = utility.get_data(dataDir)

N, Pw, Ph, C = images.shape

Pmax = np.amax(images)
Pmin = np.amin(images)
print "Images stats\n N={}, Width={}, Height={}, Max={}, Min={}".format(
    N,Pw,Ph,Pmax,Pmin)

images_norm = utility.normalize_images(images)

#train test split
X_train, Y_train, X_test, Y_test, inds, train_inds, test_inds = \
    utility.train_test_split(images_norm, segs,0.75)

Ntest = X_test.shape[0]

meta_test = meta[:,test_inds,:]
contours_test = contours[test_inds]
extents_test = utility.get_extents(meta_test)
############################
# Load Model 1 (FCN)
############################
FCN = load_model('./models/FCN.h5')
FCN_seg = FCN.predict(X_test)
FCN_seg = FCN_seg.reshape((Ntest,Pw,Ph))
FCN_contour = utility.listSegToContours(FCN_seg, meta_test[1,:,:],
    meta_test[0,:,:], ISOVALUE)
FCN_errs = utility.listAreaOverlapError(FCN_contour, contours_test)
FCN_thresh,ts = utility.cum_error_dist(FCN_errs,DX)

###########################
# Load Model 2 (OBP_FCN)
###########################
OBP_FCN = load_model('./models/OBP_FCN_output.h5')
OBP_FCN_seg = OBP_FCN.predict(X_test)[:,:,:,1]
OBP_FCN_seg = OBP_FCN_seg.reshape((Ntest,Pw,Ph))
OBP_FCN_contour = utility.listSegToContours(OBP_FCN_seg, meta_test[1,:,:],
    meta_test[0,:,:], ISOVALUE)
OBP_FCN_errs = utility.listAreaOverlapError(OBP_FCN_contour, contours_test)
OBP_FCN_thresh,ts = utility.cum_error_dist(OBP_FCN_errs,DX)

#############################
# Load Model 3 (OBP_FCN_full)
#############################
OBP_full = load_model('./models/OBP_full.h5')
OBP_full_seg = OBP_full.predict(X_test)
OBP_full_seg = OBP_full_seg.reshape((Ntest,Pw,Ph))
OBP_full_contour = utility.listSegToContours(OBP_full_seg, meta_test[1,:,:],
    meta_test[0,:,:], ISOVALUE)
OBP_full_errs = utility.listAreaOverlapError(OBP_full_contour, contours_test)
OBP_full_thresh,ts = utility.cum_error_dist(OBP_full_errs,DX)
#############################
# Get Level set
#############################
# ls_contours = get_level_sets(f,'/home/marsdenlab/projects/ls_full/raw/',test_inds,
#     'edge96_LS')
# ls_errs = utility.listAreaOverlapError(ls_contours,contours_test)

#############################
# Visualize results
#############################
plt.figure()
plot_segs(X_test, [Y_test[:,:,:,0],FCN_seg, OBP_FCN_seg,
    OBP_full_seg], n_ex=10, dim=(10,5),figsize=(10,10))
plt.savefig('./plots/segs.png')

plt.figure()
plot_contours(X_test, [FCN_contour, OBP_FCN_contour, OBP_full_contour], extents_test,
    n_ex=4, dim=(4,3))
plt.savefig('./plots/contours.png')

plt.figure()
plt.plot(ts,FCN_thresh, color='red', label='FCN', linewidth=2)
plt.plot(ts,OBP_FCN_thresh, color='blue', label='OBP_FCN', linewidth=2)
plt.plot(ts,OBP_full_thresh, color='green', label='OBG_FCN', linewidth=2)
plt.xlabel('IOU error')
plt.ylabel('Fraction of contours below error')
plt.legend()
plt.savefig('./plots/IOU.png')

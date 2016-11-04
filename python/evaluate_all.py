import numpy as np
import utility.utility as utility
import argparse
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm
from models.FCN import FCN

import utility.util_data as util_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

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

def plot_contours(images, contours, extents, colors, labels,
    n_ex=4, dim=(4,4), figsize=(10,10)):
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
ROC = {}
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
ROC['FCN'] = roc_curve(np.ravel(Y_test),np.ravel(FCN_seg), pos_label=1)

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
ROC['OBP_FCN'] = roc_curve(np.ravel(Y_test),np.ravel(OBP_FCN_seg), pos_label=1)

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
ROC['OBG_FCN'] = roc_curve(np.ravel(Y_test), np.ravel(OBP_full_seg),pos_label=1)

#############################
# Get Level set
#############################
# ls_contours = get_level_sets(f,'/home/marsdenlab/projects/ls_full/raw/',test_inds,
#     'edge96_LS')
# ls_errs = utility.listAreaOverlapError(ls_contours,contours_test)

#############################
# Visualize results
#############################
#Figure 1 segmentations
plt.figure()
Nplots = 4
fig_seg, ax_seg = plt.subplots(Nplots,5)
for i in range(0,Nplots):
    ax_seg[i,0].imshow(X_test[i,:,:,0],cmap='gray')
    ax_seg[i,0].set_title('Image')
    ax_seg[i,1].imshow(Y_test[i,:,:,0],cmap='gray')
    ax_seg[i,1].set_title('User segmentation')
    ax_seg[i,2].imshow(FCN_seg[i,:,:],cmap='gray')
    ax_seg[i,2].set_title('FCN')
    ax_seg[i,3].imshow(OBP_FCN_seg[i,:,:],cmap='gray')
    ax_seg[i,3].set_title('OBP_FCN')
    ax_seg[i,4].imshow(OBP_full_seg[i,:,:],cmap='gray')
    ax_seg[i,4].set_title('OBG_FCN')
    [a.set_axis_off() for a in ax_seg[i, :]]
    [a.set_axis_off() for a in ax_seg[:, i]]
plt.axis('off')
fig_seg.tight_layout()
plt.savefig('./plots/segs.png')

#Figure 2 contours
plt.figure()
Nplots = 4
fig_contour, ax_contour = plt.subplots(Nplots,3)
plot_count = 0
for i in range(0,Nplots):
    if (FCN_contour[i] == [] or OBP_FCN_contour[i] == [] or OBP_full_contour[i] == []):
        continue
    fcn_contour = FCN_contour[i][0]
    obp_fcn_contour = OBP_FCN_contour[i][0]
    obg_fcn_contour = OBP_full_contour[i][0]

    ax_contour[plot_count,0].imshow(X_test[i,:,:,0], extent=extents_test[i],cmap='gray')
    ax_contour[plot_count,0].scatter(fcn_contour[:,0],fcn_contour[:,1], color='green',label='FCN')
    ax_contour[plot_count,1].imshow(X_test[i,:,:,0], extent=extents_test[i],cmap='gray')
    ax_contour[plot_count,1].scatter(obp_fcn_contour[:,0],obp_fcn_contour[:,1], color='red',label='OBP_FCN')
    ax_contour[plot_count,2].imshow(X_test[i,:,:,0], extent=extents_test[i],cmap='gray')
    ax_contour[plot_count,2].scatter(obg_fcn_contour[:,0],obg_fcn_contour[:,1], color='blue',label='OBG_FCN')
    [a.set_axis_off() for a in ax_contour[plot_count, :]]
    [a.set_axis_off() for a in ax_contour[:, plot_count]]
    plot_count += 1
plt.axis('off')
fig_contour.tight_layout()
plt.savefig('./plots/contours.png')

#Figure 3, IOU
plt.figure()
plt.plot(ts,FCN_thresh, color='red', label='FCN', linewidth=2)
plt.plot(ts,OBP_FCN_thresh, color='blue', label='OBP_FCN', linewidth=2)
plt.plot(ts,OBP_full_thresh, color='green', label='OBG_FCN', linewidth=2)
plt.xlabel('IOU error')
plt.ylabel('Fraction of contours below error')
plt.legend()
plt.savefig('./plots/IOU.png')

#Figure 4 ROC
plt.figure()
plt.plot(ROC['FCN'][0],ROC['FCN'][1], color='red', label='FCN', linewidth=2)
plt.plot(ROC['OBP_FCN'][0],ROC['OBP_FCN'][1], color='blue', label='OBP_FCN', linewidth=2)
plt.plot(ROC['OBG_FCN'][0],ROC['OBG_FCN'][1], color='green', label='OBG_FCN', linewidth=2)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig('./plots/roc.png')

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
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
import configparser

np.random.seed(0)
THRESHOLD = 0.4
ISOVALUE = 0.5
DX = 0.01
#########################################
#function definitions
#########################################
def contour_plot(x_test, C, extent, labels, colors, filename):
    '''
    function to plot contours on top of images

    args:
        @a index - (int) index of the contour to plot
        @a x_test - (shape=(W,H,1)) image to plot
        @a C - list, list containing predicted contour for each model
        @a extent - numpy array containing left,right,top,bottom bounds
        @a labels - list of labels to use
        @a filename - filename to save plot as
    '''
    N = len(labels)
    plt.figure()
    fig_contour, ax_contour = plt.subplots(1,N)
    plot_count = 0
    handles = []
    for j in range(0,len(C)):
        contour = C[j]
        label = labels[j]
        color = colors[j]
        ax_contour[j].imshow(x_test[:,:,0], extent=extent,cmap='gray')
        h, = ax_contour[j].plot(contour[:,0],contour[:,1], color=color,label=label)
        handles.append(h)

    [a.set_axis_off() for a in ax_contour]

    plt.legend(handles=handles,
        labels=labels, loc='lower right', bbox_to_anchor=(1.0,1.0))
    plt.axis('off')
    fig_contour.tight_layout()
    plt.savefig(filename)
##########################
# Parse args
##########################
config = configparser.ConfigParser()
config.read('options.cfg')

dataDir = config['learn_params']['test_dir']

###########################
# Load data and preprocess
###########################
vasc2d = util_data.VascData2D(dataDir)

N, Pw, Ph, C = vasc2d.images_tf.shape

print "Images stats\n N={}, Width={}, Height={}".format(
    N,Pw,Ph)

X_test = vasc2d.images_norm
Y_test = vasc2d.segs_tf
Ntest = X_test.shape[0]

meta_test = vasc2d.meta
contours_test = vasc2d.contours


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
# Level set
#############################
ls_test = vasc2d.contours_ls
ls_errs = utility.listAreaOverlapError(ls_test, contours_test)
ls_thresh,ts = utility.cum_error_dist(ls_errs,DX)
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
contours_to_plot = [contours_test[0], ls_test[0], FCN_contour[0],
OBP_FCN_contour[0], OBP_full_contour[0]]
labels = ['user','level set','FCN','OBP_FCN','OBG_FCN']
colors = ['green','red','blue','yellow','magenta']
contour_plot(X_test[0],contours_to_plot,extents_test[0],labels,colors,'./plots/contours.png')

#Figure 3, IOU
plt.figure()
plt.plot(ts,ls_thresh, color='red', label='level set', linewidth=2)
plt.plot(ts,FCN_thresh, color='blue', label='FCN', linewidth=2)
plt.plot(ts,OBP_FCN_thresh, color='yellow', label='OBP_FCN', linewidth=2)
plt.plot(ts,OBP_full_thresh, color='magenta', label='OBG_FCN', linewidth=2)
plt.xlabel('IOU error')
plt.ylabel('Fraction of contours below error')
plt.legend(loc='lower right')
plt.savefig('./plots/IOU.png')

#Figure 4 ROC
plt.figure()
plt.plot(ROC['FCN'][0],ROC['FCN'][1], color='red', label='FCN', linewidth=2)
plt.plot(ROC['OBP_FCN'][0],ROC['OBP_FCN'][1], color='blue', label='OBP_FCN', linewidth=2)
plt.plot(ROC['OBG_FCN'][0],ROC['OBG_FCN'][1], color='green', label='OBG_FCN', linewidth=2)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.savefig('./plots/roc.png')

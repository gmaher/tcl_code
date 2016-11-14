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

import utility.util_data as util_data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab

from sklearn.metrics import roc_curve
import configparser

np.random.seed(0)
THRESHOLD = 0.4
ISOVALUE = 0.5
DX = 0.01
#########################################
#function definitions
#########################################
def get_outputs(seg):
    contour = utility.listSegToContours(seg, meta_test[1,:,:],
        meta_test[0,:,:], ISOVALUE)
    errs = utility.listAreaOverlapError(contour, contours_test)
    thresh,ts = utility.cum_error_dist(errs,DX)
    roc = roc_curve(np.ravel(Y_test),np.ravel(seg), pos_label=1)
    return (contour,errs,thresh,roc)

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
    fig_contour, ax_contour = plt.subplots(1,N,figsize=(20,8))
    plot_count = 0
    handles = []
    for j in range(0,len(C)):
        contour = C[j]
        label = labels[j]
        color = colors[j]
        ax_contour[j].imshow(x_test[:,:,0], extent=extent,cmap='gray')
        if contour == []:
            print "failed contour"
            h, = ax_contour[j].plot([0],[0], color=color,label=label,linewidth=4)
        else:
            h, = ax_contour[j].plot(contour[:,0],contour[:,1], color=color,label=label,linewidth=4)
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
config = utility.parse_config('options.cfg')

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
FCN_seg = np.load('./predictions/FCN.npy').reshape((Ntest,Pw,Ph))
FCN_contour,FCN_errs,FCN_thresh,FCN_roc =\
get_outputs(FCN_seg)
ROC['FCN'] = FCN_roc

###########################
# Load Model 2 (OBP_FCN)
###########################
OBP_FCN_out = np.load('./predictions/OBP_FCN.npy')
OBP_FCN_seg = OBP_FCN_out[:,:,:,1].reshape((Ntest,Pw,Ph))
OBP_FCN_contour, OBP_FCN_errs, OBP_FCN_thresh, OBP_roc =\
get_outputs(OBP_FCN_seg)
ROC['OBP_FCN'] = OBP_roc

#############################
# Load Model 3 (OBP_FCN_full)
#############################
OBP_full_seg = np.load('./predictions/OBG_FCN.npy').reshape((Ntest,Pw,Ph))
OBP_full_contour, OBP_full_errs, OBP_full_thresh, OBP_roc =\
get_outputs(OBP_full_seg)
ROC['OBG_FCN'] = OBP_roc

#############################
# Load HED
#############################
HED_seg = np.load('./predictions/HED.npy')[0].reshape((Ntest,Pw,Ph))
HED_contour, HED_errs, HED_thresh, HED_roc =\
get_outputs(HED_seg)
ROC['HED'] = HED_roc

#############################
# Load HED
#############################
HED_dense_seg = np.load('./predictions/HED_dense.npy').reshape((Ntest,Pw,Ph))
HED_dense_contour, HED_dense_errs, HED_dense_thresh, HED_dense_roc =\
get_outputs(HED_dense_seg)
ROC['HED_dense'] = HED_dense_roc

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
fig_seg, ax_seg = plt.subplots(Nplots,7,figsize=(35,25))
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
    ax_seg[i,5].imshow(HED_seg[i,:,:],cmap='gray')
    ax_seg[i,5].set_title('HED')
    ax_seg[i,6].imshow(HED_dense_seg[i,:,:],cmap='gray')
    ax_seg[i,6].set_title('HED_dense')
    [a.set_axis_off() for a in ax_seg[i, :]]
    [a.set_axis_off() for a in ax_seg[:, i]]
plt.axis('off')
fig_seg.tight_layout()
plt.savefig('./plots/segs.png')

#Figure 2 contours
contours_to_plot = [contours_test[0], ls_test[0], FCN_contour[0],
OBP_FCN_contour[0], OBP_full_contour[0], HED_contour[0], HED_dense_contour[0]]
labels = ['user','level set','FCN','OBP_FCN','OBG_FCN','HED','HED_dense']
colors = ['green','red','blue','yellow','magenta','teal','orange']
contour_plot(X_test[0],contours_to_plot,extents_test[0],labels,colors,'./plots/contours.png')

#Figure 3, IOU
plt.figure()
plt.plot(ts,ls_thresh, color='red', label='level set', linewidth=2)
plt.plot(ts,FCN_thresh, color='blue', label='FCN', linewidth=2)
plt.plot(ts,OBP_FCN_thresh, color='yellow', label='OBP_FCN', linewidth=2)
plt.plot(ts,OBP_full_thresh, color='magenta', label='OBG_FCN', linewidth=2)
plt.plot(ts,HED_thresh, color='black', label='HED', linewidth=2)
plt.plot(ts,HED_dense_thresh, color='orange', label='HED_dense', linewidth=2)
plt.xlabel('IOU error')
plt.ylabel('Fraction of contours below error')
plt.legend(loc='lower right')
plt.savefig('./plots/IOU.png')

#Figure 4 ROC
plt.figure()
plt.plot(ROC['FCN'][0],ROC['FCN'][1], color='red', label='FCN', linewidth=2)
plt.plot(ROC['OBP_FCN'][0],ROC['OBP_FCN'][1], color='blue', label='OBP_FCN', linewidth=2)
plt.plot(ROC['OBG_FCN'][0],ROC['OBG_FCN'][1], color='green', label='OBG_FCN', linewidth=2)
plt.plot(ROC['HED'][0],ROC['HED'][1], color='black', label='HED', linewidth=2)
plt.plot(ROC['HED_dense'][0],ROC['HED_dense'][1], color='orange', label='HED_dense', linewidth=2)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.savefig('./plots/roc.png')

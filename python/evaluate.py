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
import os
from sklearn.metrics import roc_curve
import configparser

np.random.seed(0)
THRESHOLD = 0.4
ISOVALUE = 0.5
DX = 0.01
ts = np.arange(0,1+DX,DX)
#########################################
# Parse arguments
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir',default='./predictions/')
parser.add_argument('-p','--paths',nargs='+',default=['all'])
args = parser.parse_args()

pred_dir = args.pred_dir
path_types = args.paths

#make folder name for plots and make directory
plot_dir = "./plots/{}/".format("_".join(path_types))
if not os.path.exists(os.path.abspath(plot_dir)):
    os.mkdir(os.path.abspath(plot_dir))

config = utility.parse_config('options.cfg')

dataDir = config['learn_params']['test_dir']
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

def add_pred_to_dict(pred_dict,pred_string,pred_code,pred_color,inds,shape):
    out = np.load(pred_string)
    if len(out.shape) == 5:
        out = out[0]
        out = out[inds].reshape(shape)
    elif out.shape[3] == 3:
        out = out[inds,:,:,1].reshape(shape)
    else:
        out = out[inds].reshape(shape)

    contour,errs,thresh,roc = get_outputs(out)

    pred_dict['seg'][pred_code] = out
    pred_dict['contour'][pred_code] = contour
    pred_dict['error'][pred_code] = errs
    pred_dict['thresh'][pred_code] = thresh
    pred_dict['ROC'][pred_code] = roc
    pred_dict['color'][pred_code] = pred_color

def add_contour_to_dict(pred_dict,pred_code,pred_color,contours,contours_test,inds,DX):
    ls_test = [contours[i] for i in inds]
    errs = utility.listAreaOverlapError(ls_test, contours_test)
    thresh,ts = utility.cum_error_dist(errs,DX)

    pred_dict['thresh'][pred_code] = thresh
    pred_dict['error'][pred_code] = errs
    pred_dict['contour'][pred_code] = ls_test
    pred_dict['color'][pred_code] = pred_color

def contour_plot(X_test, Clist, extent, labels, colors, start_index, Nplots, filename,size=(20,20)):
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
    fig_contour, ax_contour = plt.subplots(Nplots,N,figsize=size)
    handles = []
    for i in range(Nplots):
        x_test = X_test[start_index+i]
        for j in range(0,len(Clist)):
            contour = Clist[j][start_index+i]
            label = labels[j]
            color = colors[j]
            ax_contour[i,j].imshow(x_test[:,:,0], extent=extent[start_index+i],cmap='gray')
            if contour == []:
                print "failed contour"
                h, = ax_contour[i,j].plot([0],[0], color=color,label=label,linewidth=4)
            else:
                h, = ax_contour[i,j].plot(contour[:,0],contour[:,1], color=color,label=label,linewidth=4)
            handles.append(h)

            [a.set_axis_off() for a in ax_contour[i, :]]
            [a.set_axis_off() for a in ax_contour[:, j]]

    plt.legend(handles=handles,
        labels=labels, loc='lower right', bbox_to_anchor=(1.0,1.0))
    plt.axis('off')
    fig_contour.tight_layout()
    plt.savefig(filename)

def image_grid_plot(imlist, labels, Nplots, fn, size=(20,20)):
    plt.figure()
    fig_seg, ax_seg = plt.subplots(Nplots,len(imlist),figsize=size)
    for i in range(0,Nplots):
        for j in range(len(imlist)):
            x = imlist[j]
            if len(x.shape) == 4:
                ax_seg[i,j].imshow(x[i,:,:,0],cmap='gray')
            else:
                ax_seg[i,j].imshow(x[i,:,:],cmap='gray')
            ax_seg[i,j].set_title(labels[j],fontsize=100)
        [a.set_axis_off() for a in ax_seg[i, :]]
        [a.set_axis_off() for a in ax_seg[:, j]]
    plt.axis('off')
    fig_seg.tight_layout()
    plt.savefig(fn)


###########################
# Load data and preprocess
###########################
vasc2d = util_data.VascData2D(dataDir)

N, Pw, Ph, C = vasc2d.images_tf.shape

print "Images stats\n N={}, Width={}, Height={}".format(
    N,Pw,Ph)

if path_types != ['all']:
    f = open(dataDir+'names.txt').readlines()
    inds = [any(s in p.lower() for s in path_types) for p in f]
else:
    inds = range(0,N)

X_test = vasc2d.images_norm[inds]
Y_test = vasc2d.segs_tf[inds]
Ntest = X_test.shape[0]

meta_test = vasc2d.meta[:,inds,:]
extents_test = utility.get_extents(meta_test)
contours_test = [vasc2d.contours[i] for i in inds]

PREDS = {}
PREDS['seg'] = {}
PREDS['contour'] = {}
PREDS['thresh'] = {}
PREDS['ROC'] = {}
PREDS['color'] = {}
PREDS['error'] = {}

shape = (Ntest,Pw,Ph)

#Load all predictions
add_pred_to_dict(PREDS,pred_dir+'FCN.npy','FCN','red',inds,shape)
add_pred_to_dict(PREDS,pred_dir+'OBP_FCN.npy','OBP_FCN','green',inds,shape)
add_pred_to_dict(PREDS,pred_dir+'OBG_FCN.npy','OBG_FCN','blue',inds,shape)
add_pred_to_dict(PREDS,pred_dir+'HED.npy','HED','black',inds,shape)
add_pred_to_dict(PREDS,pred_dir+'HED_dense.npy','HED_dense','orange',inds,shape)

#load contours
add_contour_to_dict(PREDS,'level set','yellow',vasc2d.contours_ls,contours_test,inds,DX)
add_contour_to_dict(PREDS,'level set edge map','purple',vasc2d.contours_edge,contours_test,inds,DX)

#load OBP by itself for vizualization purposes
OBP_FCN_out = np.load(pred_dir+'OBP_FCN.npy')[inds]

#############################
# Visualize results
#############################
#Figure 0 OBP plot
vasc2d.createOBG(border_width=1)
image_grid_plot([X_test,vasc2d.obg[:,:,:,1],vasc2d.obg[:,:,:,2],
OBP_FCN_out[:,:,:,1],OBP_FCN_out[:,:,:,2]],
['image','user segmentation','boundary','OBP_FCN segmentation','OBG_FCN boundary'],
10,plot_dir+'/OBP.png',(80,80))

#Figure 1 segmentations
keys_seg = ['FCN','OBP_FCN','OBG_FCN','HED','HED_dense']
segs = [X_test,Y_test]+[PREDS['seg'][k] for k in keys_seg]
labels = ['image', 'user segmentation']+keys_seg
image_grid_plot(segs,labels,10,plot_dir+'/segs.png',(80,80))

#Figure 2 contours
keys = ['level set', 'level set edge map', 'FCN', 'OBP_FCN',
'OBG_FCN', 'HED', 'HED_dense']
contours_to_plot = [contours_test]+[PREDS['contour'][k] for k in keys]
labels = ['user']+keys
colors = ['yellow']+[PREDS['color'][k] for k in keys]
contour_plot(X_test,contours_to_plot,extents_test,labels,colors,0,5,plot_dir+'contours1.png',(20,20))
contour_plot(X_test,contours_to_plot,extents_test,labels,colors,100,5,plot_dir+'contours2.png',(20,20))

# #Figure 3, IOU
plt.figure()
for k in keys:
    plt.plot(ts,PREDS['thresh'][k], color=PREDS['color'][k],
     label=k, linewidth=2)
plt.xlabel('IOU error')
plt.ylabel('Fraction of contours below error')
plt.legend(loc='upper left')
plt.savefig(plot_dir+'IOU.png')

# #Figure 3, IOU
plt.figure()
for k in keys_seg:
    plt.plot(PREDS['ROC'][k][0],PREDS['ROC'][k][1], color=PREDS['color'][k],
     label=k, linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig(plot_dir+'roc.png')

# #Summary statistics

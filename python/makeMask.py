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
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import os
from sklearn.metrics import roc_curve
import configparser
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm
np.random.seed(0)

DX = 0.01
ts = np.arange(0,1+DX,DX)
#########################################
# Parse arguments
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('config_file')
#parser.add_argument('--pred_dir',default='./predictions/')
parser.add_argument('-p','--paths',nargs='+',default=['all'])
args = parser.parse_args()

config_file = args.config_file
config = utility.parse_config(config_file)

dataDir = config['learn_params']['data_dir']+'test/'
model_dir = config['learn_params']['model_dir']
pred_dir = config['learn_params']['pred_dir']
path_types = args.paths

THRESHOLD = float(config['learn_params']['threshold'])
ISOVALUE = float(config['learn_params']['isovalue'])
image_dims = int(config['learn_params']['image_dims'])
#make folder name for plots and make directory
plot_dir = config['learn_params']['plot_dir']
plot_dir = plot_dir+"{}/".format("_".join(path_types))

if not os.path.exists(os.path.abspath(plot_dir)):
    os.mkdir(os.path.abspath(plot_dir))


#########################################
#function definitions
#########################################
def calc_accuracy(seg, seg_truth):
    ypred = utility.threshold(np.ravel(seg),THRESHOLD)
    ytruth = utility.threshold(np.ravel(seg_truth),THRESHOLD)
    H = utility.confusionMatrix(ytruth,ypred)
    N1 = np.sum(ytruth[:]==1)
    N2 = np.sum(ytruth[:]==0)
    acc = (N2*H[0,0] + N1*H[1,1])/(N1+N2)
    mean_acc = 0.5*(H[0,0]+H[1,1])
    print "N1 {} N2 {} H {} acc {} mean_acc {}".format(
    N1,N2,H, acc, mean_acc
    )
    return (acc, mean_acc)

def get_outputs(seg, seg_truth):
    print seg.shape, seg_truth.shape
    seg_thresh = utility.threshold(seg,THRESHOLD)
    print seg_thresh.shape
    contour = utility.listSegToContours(seg_thresh, meta_test[1,:],
        meta_test[0,:], ISOVALUE)
    errs = utility.listAreaOverlapError(contour, contours_test)
    thresh,ts = utility.cum_error_dist(errs,DX)
    roc = roc_curve(np.ravel(Y_test),np.ravel(seg), pos_label=1)

    dorf = []
    emd = []
    asdl = []
    for i in range(len(seg_truth)):
        if np.sum(seg_thresh[i,:,:]) > 0.1 and np.sum(seg_truth[i,:,:,0]) > 0.1:
            e= hd(seg_thresh[i,:,:],seg_truth[i,:,:,0],meta_test[0,i][0])
            dorf.append(e)

            e_asd= assd(seg_thresh[i,:,:],seg_truth[i,:,:,0],meta_test[0,i][0])
            asdl.append(e_asd)
            # if np.sum(seg_thresh[i,:,:]) < 600 and np.sum(seg_truth[i,:,:,0]) < 600:
            #     print i,np.sum(seg_thresh[i,:,:]),np.sum(seg_truth[i,:,:,0])
            #     e_emd = utility.EMDSeg(seg_truth[i,:,:,0],seg_thresh[i,:,:], meta_test[0,i][0])
            #     emd.append(e_emd)
    acc,mean_acc = calc_accuracy(seg, seg_truth)
    return (contour,errs,thresh,roc,acc,mean_acc,dorf,asdl)

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
                h, = ax_contour[i,j].plot([0],[0], color=color,label=label,linewidth=6)
            else:
                h, = ax_contour[i,j].plot(contour[:,0],-contour[:,1], color=color,label=label,linewidth=6)
            handles.append(h)

            [a.set_axis_off() for a in ax_contour[i, :]]
            [a.set_axis_off() for a in ax_contour[:, j]]

    #plt.legend(handles=handles,
    #    labels=labels, loc='lower right', bbox_to_anchor=(1.2,1.2))
    plt.axis('off')
    fig_contour.tight_layout(pad=0.0)
    fig_contour.subplots_adjust(wspace=0, hspace=0)
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
            #if i == 0:
                #ax_seg[i,j].set_title(labels[j],fontsize=int(1.5*size[0]), fontweight='bold')
        [a.set_axis_off() for a in ax_seg[i, :]]
        [a.set_axis_off() for a in ax_seg[:, j]]
    plt.axis('off')
    fig_seg.tight_layout(pad=0.0)
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
    inds = [i for i in range(N) if any(s.lower() in f[i].lower() for s in path_types)]
    paths = [f[i] for i in inds]
    inds = [x for (x,y) in sorted(zip(inds,paths), key=lambda pair:int(pair[1].split('.')[2]))]
else:
    inds = range(0,N)

X_test = vasc2d.images_norm[inds]
Y_test = vasc2d.segs_tf[inds]

vasc2d.images_norm = vasc2d.images_norm[inds]
vasc2d.segs_tf = vasc2d.segs_tf[inds]
vasc2d.segs = vasc2d.segs[inds]
#vasc2d.mag_seg = vasc2d.mag_seg[inds]

Ntest = X_test.shape[0]

meta_test = vasc2d.meta[:,inds]
extents_test = utility.get_extents(meta_test)
contours_test = [vasc2d.contours[i] for i in inds]

#HED Plot
from keras import backend as K
def get_activation(x,name,model):
   f = K.function([model.layers[0].input],
       [model.get_layer(name).output])

   return f([x])[0]

#hed = load_model(model_dir+'HED.h5')
#Y_hed = hed.predict(X_test)
#mask = get_activation(X_test,"mask",hed)
#inside_output = get_activation(X_test,"new-score-weighting",hed)
#merged = get_activation(X_test,'merged',hed)
#
# image_grid_plot([X_test]+Y_hed+[mask,inside_output,merged],
# ['image','hed1','hed2','hed3','hed4','mask','inside_output','merged'],
# 15,plot_dir+'/hed.png',(40,40))
#
#I2INet
i2i = load_model(model_dir+'I2INetFC.h5')
Y_i2i = i2i.predict(X_test, batch_size=1)
mask = get_activation(X_test[0:10],"mask",i2i)
inside_output = get_activation(X_test[0:10],"conv5_4",i2i)
#merged = get_activation(X_test[0:10],'merged',i2i)
#
plt.figure()
plt.imshow(X_test[2,:,:,0])
plt.colorbar()
plt.savefig(plot_dir+'/i2iimage.png')

plt.figure()
plt.imshow(mask[2,:,:,0])
plt.colorbar()
plt.savefig(plot_dir+'/i2imask.png')

plt.figure()
plt.imshow(inside_output[2,:,:,0])
plt.colorbar()
plt.savefig(plot_dir+'/i2iinsideoutput.png')

# plt.figure()
# plt.imshow(merged[2,:,:,0])
# plt.colorbar()
# plt.savefig(plot_dir+'/i2imerged.png')

image_grid_plot([X_test]+Y_i2i+[mask, inside_output],
['image','i2i1','i2i2','mask','inside','merged'],
10,plot_dir+'/i2i.png',(40,40))

# image_grid_plot([X_test]+Y_fc,
# ['image','i2i1','i2i2'],
# 15,plot_dir+'/fc.png',(40,40))

#PR curve
i2i2_c = np.load('./output/9/predictions/I2INet.npy')
pr_i2ic = precision_recall_curve(np.ravel(Y_test),np.ravel(i2i2_c), pos_label=1)
hed_c = np.load('./output/9/predictions/HED.npy')
pr_hedc = precision_recall_curve(np.ravel(Y_test),np.ravel(hed_c), pos_label=1)
i2i2 = np.load('./output/11/predictions/I2INet.npy')
pr_i2i = precision_recall_curve(np.ravel(Y_test),np.ravel(i2i2), pos_label=1)
hed = np.load('./output/11/predictions/HED.npy')
pr_hed = precision_recall_curve(np.ravel(Y_test),np.ravel(hed), pos_label=1)
convfc = np.load('./output/11/predictions/ConvFC.npy')
pr_conv = precision_recall_curve(np.ravel(Y_test),np.ravel(convfc), pos_label=1)

plt.figure()

plt.plot(pr_i2ic[0],pr_i2ic[1], color='b', marker='o',markevery=100000,label='I2I-2D (ours)', linewidth=2)
plt.plot(pr_hedc[0],pr_hedc[1], color='r', marker='o',markevery=100000,label='HED (ours)', linewidth=2)
plt.plot(pr_i2i[0],pr_i2i[1], color='b', marker='^',markevery=100000,label='I2I-2D', linewidth=2)
plt.plot(pr_hed[0],pr_hed[1], color='r', marker='^',markevery=100000,label='HED', linewidth=2)
plt.plot(pr_conv[0],pr_conv[1], color='g', marker='s',markevery=100000,label='ConvFC', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(b='on')
plt.savefig(plot_dir+'PR.png')

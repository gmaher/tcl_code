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
matplotlib.rcParams.update({'font.size': 22})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import os
from sklearn.metrics import roc_curve, precision_recall_curve
import configparser
from medpy.metric.binary import hd, assd,dc,precision
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
    roc = roc_curve(np.ravel(seg_truth),np.ravel(seg), pos_label=1)
    pr = precision_recall_curve(np.ravel(seg_truth),np.ravel(seg), pos_label=1)
    dorf = []
    emd = []
    asdl = []
    dice = []
    prec = []
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

        edc = dc(seg_thresh[i,:,:],seg_truth[i,:,:,0])
        dice.append(edc)
        prec.append(precision(seg_thresh[i,:,:],seg_truth[i,:,:,0]))
    acc,mean_acc = calc_accuracy(seg, seg_truth)
    return (contour,errs,thresh,roc,pr,acc,mean_acc,dorf,dice, prec,asdl)

def add_pred_to_dict(pred_dict,pred_string,pred_code,pred_color,inds,shape,seg_truth):
    out = np.load(pred_string)
    if len(out) < max(inds):
        out = out.reshape(shape)
    elif len(out.shape) == 5 or len(out.shape) == 2:
        out = out[0]
        out = out[inds].reshape(shape)
    elif len(out.shape) == 3:
        out = out[inds]
    elif out.shape[3] == 3:
        out = out[inds,:,:,1].reshape(shape)
    else:
        out = out[inds].reshape(shape)

    contour,errs,thresh,roc,pr,acc,mean_acc,dorf,dice,prec,asdl = get_outputs(out,seg_truth)

    pred_dict['seg'][pred_code] = out
    pred_dict['contour'][pred_code] = contour
    pred_dict['error'][pred_code] = errs
    pred_dict['thresh'][pred_code] = thresh
    pred_dict['ROC'][pred_code] = roc
    pred_dict['PR'][pred_code] = pr
    pred_dict['color'][pred_code] = pred_color
    pred_dict['acc'][pred_code] = acc
    pred_dict['mean_acc'][pred_code] = mean_acc
    pred_dict['dorf'][pred_code] = dorf
    pred_dict['asd'][pred_code] = asdl
    pred_dict['dc'][pred_code] = dice
    pred_dict['precision'][pred_code] = prec

def add_contour_to_dict(pred_dict,pred_code,pred_color,contours,contours_test,inds,DX):
    if len(contours) >= max(inds):
        ls_test = [contours[i] for i in inds]
    else:
        ls_test = contours
    errs = utility.listAreaOverlapError(ls_test, contours_test)
    thresh,ts = utility.cum_error_dist(errs,DX)

    dorf = []
    asdl = []
    dice = []
    prec = []
    # for i in tqdm(range(len(ls_test))):
    #     dims = meta_test[2,i]
    #     spacing = meta_test[0,i]
    #     origin = meta_test[1,i]
    #     if ls_test[i] != [] and len(ls_test[i]) > 2:
    #         seg_truth = utility.contourToSeg(contours_test[i], origin, dims, spacing)
    #         seg_thresh = utility.contourToSeg(ls_test[i], origin, dims, spacing)
    #         if np.sum(seg_thresh) > 0.1 and np.sum(seg_truth) > 0.1:
    #             e= hd(seg_thresh,seg_truth,meta_test[0,i][0])
    #             dorf.append(e)
    #
    #             edc = dc(seg_thresh,seg_truth)
    #             dice.append(edc)
    #             prec.append(precision(seg_thresh,seg_truth))
    #             #e_asd= assd(seg_thresh,seg_truth,meta_test[0,i][0])
    #             #asdl.append(e_asd)
    #     else:
    #         e = np.sqrt(2)*32*meta_test[0,i][0]
    #         dorf.append(e)
    #
    #         e_asd = np.sqrt(2)*32*meta_test[0,i][0]
    #         asdl.append(e_asd)
    #
    #         dice.append(1.0)
    #         prec.append(0.0)



    pred_dict['thresh'][pred_code] = thresh
    pred_dict['error'][pred_code] = errs
    pred_dict['contour'][pred_code] = ls_test
    pred_dict['color'][pred_code] = pred_color
    pred_dict['dorf'][pred_code] = dorf
    pred_dict['asd'][pred_code] = asdl
    pred_dict['dc'][pred_code] = dice
    pred_dict['precision'][pred_code] = prec

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
                h, = ax_contour[i,j].plot(contour[:,0],-contour[:,1], color=color,label=label,linewidth=4)
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
vasc2d = util_data.VascData2D(dataDir, normalize='minmax',rotate_data=False)
X_test = vasc2d.get_all(crop=image_dims)[0]
Y_test = vasc2d.get_all(crop=image_dims)[1]

N, Pw, Ph, C = X_test.shape

print "Images stats\n N={}, Width={}, Height={}".format(
    N,Pw,Ph)

if path_types != ['all']:
    f = open(dataDir+'names.txt').readlines()
    inds = [i for i in range(N) if any(s.lower() in f[i].lower() for s in path_types)]
    paths = [f[i] for i in inds]
    inds = [x for (x,y) in sorted(zip(inds,paths), key=lambda pair:int(pair[1].split('.')[2]))]
else:
    inds = range(0,N)

X_test = X_test[inds]
Y_test = Y_test[inds]

# vasc2d.images_norm = vasc2d.images_norm[inds]
# vasc2d.segs_tf = vasc2d.segs_tf[inds]
# vasc2d.segs = vasc2d.segs[inds]
if len(vasc2d.meta) == 3:
    meta_test = vasc2d.meta[:,inds]
else:
    meta_test = vasc2d.meta[inds,:].T
extents_test = utility.get_extents(meta_test)

if hasattr(vasc2d, 'mag_seg'):
    vasc2d.mag_seg = vasc2d.mag_seg[inds]
else:
    vasc2d.mag_seg = Y_test

if not hasattr(vasc2d, 'contours'):
    vasc2d.contours = utility.listSegToContours(Y_test[:,:,:,0], meta_test[1,:],
        meta_test[0,:], ISOVALUE)

    vasc2d.contours_ls = vasc2d.contours

Ntest = X_test.shape[0]


contours_test = vasc2d.contours

PREDS = {}
PREDS['seg'] = {}
PREDS['contour'] = {}
PREDS['thresh'] = {}
PREDS['ROC'] = {}
PREDS['color'] = {}
PREDS['error'] = {}
PREDS['acc'] = {}
PREDS['mean_acc'] = {}
PREDS['dorf'] = {}
PREDS['asd'] = {}
PREDS['PR'] = {}
PREDS['dc'] = {}
PREDS['precision'] = {}
shape = (Ntest,Pw,Ph)

#Load all predictions
# add_pred_to_dict(PREDS,pred_dir+'FCN.npy','RSN','red',inds,shape,Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'OBP_FCN.npy','OBP_SN','green',inds,shape,Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'OBG_FCN.npy','OBG_RSN','blue',inds,shape,Y_test)
# add_pred_to_dict(PREDS,pred_dir+'HED.npy','HED','black',inds,shape,Y_test)
# add_pred_to_dict(PREDS,pred_dir+'I2INet.npy','I2I-2D','orange',inds,shape, Y_test)
# add_pred_to_dict(PREDS,pred_dir+'HEDFC.npy','HEDFC','g',inds,shape,Y_test)
# add_pred_to_dict(PREDS,pred_dir+'I2INetFC.npy','I2I-2DFC','m',inds,shape, Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'I2INetFCMask.npy','FCI2I','teal',inds,shape, Y_test)
# add_pred_to_dict(PREDS,pred_dir+'FCI2INet.npy','FCI2I','teal',inds,shape, Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'FC_branch.npy','FC_branch','teal',inds,shape, Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'ConvFC.npy','ConvFC','gray',inds,shape, Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'FCN_finetune.npy','RSN_finetune','m',inds,shape, Y_test)
# #add_pred_to_dict(PREDS,pred_dir+'FCN_multi.npy','RSN_multi','c',inds,shape, Y_test)
#
# # np.save(pred_dir+'mag_seg',vasc2d.mag_seg/255)
# # add_pred_to_dict(PREDS,pred_dir+'mag_seg.npy','3D segmentation','yellow',inds,shape, Y_test)
#
# #load contours
# add_contour_to_dict(PREDS,'level set','pink',vasc2d.contours_ls,contours_test,inds,DX)
# #add_contour_to_dict(PREDS,'level set edge map','purple',vasc2d.contours_edge,contours_test,inds,DX)
# #add_contour_to_dict(PREDS,'3D segmentation','yellow',vasc2d.contours_seg,contours_test,inds,DX)
#
# #load OBP by itself for vizualization purposes
# #OBP_FCN_out = np.load(pred_dir+'OBP_FCN.npy')[inds]
#
# #############################
# # Visualize results
# #############################
# #plot all contour plots
# names = open(dataDir+'names.txt').readlines()
# names = [n.replace('\n','') for n in names]
# names = [n.replace('ls','') for n in names]
# utility.mkdir(plot_dir+'boundaries')
# for i in range(Ntest):
#     plt.figure()
#     plt.imshow(X_test[i,:,:,0],extent=extents_test[i], cmap='gray')
#     plt.savefig(plot_dir+'boundaries/'+names[i]+'_image.png')
#     plt.close()
#
#     plt.figure()
#     plt.imshow(Y_test[i,:,:,0],extent=extents_test[i], cmap='gray')
#     plt.savefig(plot_dir+'boundaries/'+names[i]+'_seg.png')
#     plt.close()
#
#     plt.figure()
#     plt.imshow(PREDS['seg']['I2I-2DFC'][i,:,:],extent=extents_test[i], cmap='gray')
#     plt.savefig(plot_dir+'boundaries/'+names[i]+'_segI2INetFC.png')
#     plt.close()
#
#     plt.figure()
#     plt.imshow(X_test[i,:,:,0],extent=extents_test[i], cmap='gray')
#     plt.plot(contours_test[i][:,0],-contours_test[i][:,1], color='red',linewidth=3)
#     plt.savefig(plot_dir+'boundaries/'+names[i]+'.png')
#     plt.close()
#
#     plt.figure()
#     plt.imshow(X_test[i,:,:,0],extent=extents_test[i], cmap='gray')
#     plt.plot(vasc2d.contours_ls[i][:,0],-vasc2d.contours_ls[i][:,1], color='b',linewidth=3)
#     plt.savefig(plot_dir+'boundaries/'+names[i].replace('truth','ls')+'.png')
#     plt.close()
#
#     if len(PREDS['contour']['I2I-2DFC'][i]) > 2:
#         plt.figure()
#         plt.imshow(X_test[i,:,:,0],extent=extents_test[i], cmap='gray')
#         plt.plot(PREDS['contour']['I2I-2DFC'][i][:,0],-PREDS['contour']['I2I-2DFC'][i][:,1], color='g',linewidth=3)
#         plt.savefig(plot_dir+'boundaries/'+names[i].replace('truth','I2INetFC')+'.png')
#     plt.close('all')
#
# #Plot lots of images
# plot_inds = np.random.randint(Ntest,size=(5,5))
# #plot_inds[4,4] = 1380
# image_grid_plot([Y_test[plot_inds[0,:]],Y_test[plot_inds[1,:]],Y_test[plot_inds[2,:]],
# Y_test[plot_inds[3,:]],Y_test[plot_inds[4,:]]],
# ['segmentation','segmentation','segmentation','segmentation','segmentation'],
# 5,plot_dir+'/segmentations.png',(40,40))
#
# #plot lots of segmentations
# image_grid_plot([X_test[plot_inds[0,:]],X_test[plot_inds[1,:]],X_test[plot_inds[2,:]],
# X_test[plot_inds[3,:]],X_test[plot_inds[4,:]]],
# ['image','image','image','image','image'],
# 5,plot_dir+'/images.png',(40,40))
#
# # image_grid_plot([vasc2d.mag_seg[plot_inds[0,:]],vasc2d.mag_seg[plot_inds[1,:]],vasc2d.mag_seg[plot_inds[2,:]],
# # vasc2d.mag_seg[plot_inds[3,:]],vasc2d.mag_seg[plot_inds[4,:]]],
# # ['image','image','image','image','image'],
# # 5,plot_dir+'/mag_seg.png',(40,40))
#
# #plot lots of predicted segmentations
# seg = PREDS['seg']['I2I-2D']
# image_grid_plot([seg[plot_inds[0,:]],seg[plot_inds[1,:]],seg[plot_inds[2,:]],
# seg[plot_inds[3,:]],seg[plot_inds[4,:]]],
# ['image','image','image','image','image'],
# 5,plot_dir+'/predicted_segmentations.png',(40,40))
#
# # #Figure 0 OBP plot
# # vasc2d.createOBG(border_width=1)
# # image_grid_plot([X_test,Y_test,vasc2d.obg[:,:,:,1],vasc2d.obg[:,:,:,2],
# # OBP_FCN_out[:,:,:,1],OBP_FCN_out[:,:,:,2]],
# # ['image','user segmentation','object','boundary','OBP_SN segmentation','OBG_SN boundary'],
# # 15,plot_dir+'/OBP.png',(40,40))
#
# #Figure 1 segmentations
# keys_seg = ['RSN','HED','I2I-2D', 'HEDFC', 'I2I-2DFC', 'FCI2I']
# #keys_seg = ['HED','I2I-2D', '3D segmentation']
#
# segs = [X_test,Y_test]+[PREDS['seg'][k] for k in keys_seg]
# labels = ['image', 'user segmentation']+keys_seg
# image_grid_plot(segs,labels,5,plot_dir+'/segs.png',(40,40))
#
# #Figure segmentations for high error vessels
# inds_high = [i for i in range(X_test.shape[0]) if PREDS['error']['I2I-2DFC'][i] > 0.8]
#
# inds_big_rad = [i for i in range(X_test.shape[0]) if utility.contourRadius(PREDS['contour']['I2I-2DFC'][i]) > 0.5]
# #keys_seg = ['RSN','HED','I2I-2D','HEDFC', 'I2I-2DFC', 'FCI2I']
# keys = ['level set', 'RSN', 'HED','I2I-2D']
#
# labels = ['image', 'user segmentation']+keys_seg
# l = 10
# if len(inds_high) < l:
#     l = len(inds_high)-1
#
# if l > 0:
#     segs = [X_test[inds_high],Y_test[inds_high]]+[PREDS['seg'][k][inds_high] for k in keys_seg]
#     image_grid_plot(segs,labels,l,plot_dir+'/segs_higherr.png',(40,40))
#
# labels = ['image', 'user segmentation']+keys_seg
# l = 10
# if len(inds_big_rad) < l:
#     l = len(inds_big_rad)-1
#
# if l > 0:
#     segs = [X_test[inds_big_rad],Y_test[inds_big_rad]]+[PREDS['seg'][k][inds_big_rad] for k in keys_seg]
#     image_grid_plot(segs,labels,l,plot_dir+'/segs_highrad.png',(40,40))
#
# #Figure 2 contours
# keys = ['level set', 'RSN', 'HED','I2I-2D','HEDFC', 'I2I-2DFC', 'FCI2I']
# #keys = ['level set', 'HED','I2I-2D', '3D segmentation']
#
# PREDS['contour']['I2I-2DFC'] = [utility.smoothContour(c,num_modes=6) for c in PREDS['contour']['I2I-2DFC']]
# #keys = ['level set', '3D segmentation', 'RSN','OBG_RSN', 'HED','I2I-2D', 'ConvFC', 'RSN_finetune', 'RSN_multi']
# contours_to_plot = [contours_test]+[PREDS['contour'][k] for k in keys]
# labels = ['user']+keys
# colors = ['yellow']+[PREDS['color'][k] for k in keys]
# n1 = 0
# n2 = int(Ntest/2)
# n3 = (Ntest-10)
# n4 = plot_inds[4,4]
#
# contour_plot(X_test,contours_to_plot,extents_test,labels,colors,n1,5,plot_dir+'contours1.png',(14,20))
# contour_plot(X_test,contours_to_plot,extents_test,labels,colors,n2,5,plot_dir+'contours2.png',(20,20))
# contour_plot(X_test,contours_to_plot,extents_test,labels,colors,n3,4,plot_dir+'contours3.png',(20,14))
# contour_plot(X_test,contours_to_plot,extents_test,labels,colors,n4,4,plot_dir+'contours4.png',(20,14))
#
# #High error contours
# contours_to_plot = [[contours_test[i] for i in inds_high]]+[[PREDS['contour'][k][i] for i in inds_high] for k in keys]
# contour_plot(X_test[inds_high],contours_to_plot,extents_test,labels,colors,0,10,plot_dir+'contours_higherr.png',(20,20))
#
# contours_to_plot = [[contours_test[i] for i in inds_big_rad]]+[[PREDS['contour'][k][i] for i in inds_big_rad] for k in keys]
# contour_plot(X_test[inds_big_rad],contours_to_plot,extents_test,labels,colors,0,l,plot_dir+'contours_highrad.png',(20,20))
#
# # #Figure 3, IOU
# plt.figure()
# for k in keys:
#     plt.plot(ts,PREDS['thresh'][k], color=PREDS['color'][k],
#      label=k, linewidth=3)
# plt.xlabel('Jaccard distance')
# plt.ylabel('Cumulative error distribution, F(x)')
# lg = plt.legend(loc='center right', bbox_to_anchor=(1.5,0.5))
# plt.grid(b='on')
# plt.savefig(plot_dir+'IOU.png',bbox_extra_artists=(lg,), bbox_inches='tight')
#
# # #Figure 4, ROC
# plt.figure()
# for k in keys_seg:
#     plt.plot(PREDS['ROC'][k][0],PREDS['ROC'][k][1], color=PREDS['color'][k],
#      label=k, linewidth=2)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.grid(b='on')
# plt.savefig(plot_dir+'roc.png')
#
# # #Figure 5, PR
# plt.figure()
# for k in keys_seg:
#     plt.plot(PREDS['PR'][k][1],PREDS['PR'][k][0], color=PREDS['color'][k],
#      label=k, linewidth=2)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='lower left')
# plt.grid(b='on')
# plt.savefig(plot_dir+'PR.png')
#
# # #Summary statistics
# f = open(plot_dir+'seg_acc.csv','w')
# f.write(','+','.join(PREDS['acc'].keys()))
# f.write('\n')
# f.write('accuracy,'+','.join([str(PREDS['acc'][k]) for k in PREDS['acc']]))
# f.write('\n')
# f.write('mean accuracy,'+','.join([str(PREDS['mean_acc'][k]) for k in PREDS['mean_acc']]))
# f.close()

# f = open(plot_dir+'contour_acc.csv','w')
# f.write(','+','.join(keys))
# f.write('\n')
# f.write('jaccard distance,'+','.join([str(np.mean(PREDS['error'][k])) for k in keys]))
# f.write('\n')
# f.write('IOU,'+','.join([str(1-np.mean(PREDS['error'][k])) for k in keys]))
# f.write('\n')
# f.write('Hausdorf,'+','.join([str(np.mean(PREDS['dorf'][k])) for k in keys]))
# f.write('\n')
# f.write('DICE,'+','.join([str(np.mean(PREDS['dc'][k])) for k in keys]))
# f.write('\n')
# f.write('PREC,'+','.join([str(np.mean(PREDS['precision'][k])) for k in keys]))
# f.write('\n')
# f.write('ASSD,'+','.join([str(np.mean(PREDS['asd'][k])) for k in keys]))
# f.close()

#HED Plot
# from keras import backend as K
# def get_activation(x,name,model):
#    f = K.function([model.layers[0].input],
#        [model.get_layer(name).output])
#
#    return f([x])[0]

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
#I2I-2D
# i2i = load_model(model_dir+'I2I-2D.h5')
# Y_i2i = i2i.predict(X_test, batch_size=10)
# mask = get_activation(X_test,"mask",i2i)
# inside_output = get_activation(X_test,"conv5_4",i2i)
# merged = get_activation(X_test,'merged',i2i)
#
# plt.figure()
# plt.imshow(X_test[2,:,:,0])
# plt.colorbar()
# plt.savefig(plot_dir+'/i2iimage.png')
#
# plt.figure()
# plt.imshow(mask[2,:,:,0])
# plt.colorbar()
# plt.savefig(plot_dir+'/i2imask.png')
#
# plt.figure()
# plt.imshow(inside_output[2,:,:,0])
# plt.colorbar()
# plt.savefig(plot_dir+'/i2iinsideoutput.png')

#plt.figure()
#plt.imshow(merged[2,:,:,0])
#plt.colorbar()
#plt.savefig(plot_dir+'/hedmerged.png')
#
# image_grid_plot([X_test]+Y_i2i+[mask, inside_output, merged],
# ['image','i2i1','i2i2','mask','inside','merged'],
# 15,plot_dir+'/i2i.png',(40,40))

# image_grid_plot([X_test]+Y_fc,
# ['image','i2i1','i2i2'],
# 15,plot_dir+'/fc.png',(40,40))

#Radius scatter plot
# radius_vector = [utility.contourRadius(x) for x in contours_test]
# plt.figure()
# plt.scatter(radius_vector, PREDS['error']['RSN'])
# plt.xlabel('radius (cm)')
# plt.ylabel('error')
# plt.xlim(-0.2,2.0)
# plt.ylim(-0.2,1.2)
# plt.savefig(plot_dir+'sn_scatter.png')
#
# plt.figure()
# plt.scatter(radius_vector, PREDS['error']['level set'])
# plt.xlabel('radius (cm)')
# plt.ylabel('error')
# plt.xlim(-0.2,2.0)
# plt.ylim(-0.2,1.2)
# plt.savefig(plot_dir+'ls_scatter.png')
#
# #Radius scatter plot
# pixel_vector = np.array(radius_vector)/np.array([k[0] for k in meta_test[0,:]])
# plt.figure()
# plt.scatter(pixel_vector, PREDS['error']['RSN'])
# plt.xlabel('radius (pixels)')
# plt.ylabel('error')
# plt.xlim(-0.2,30.0)
# plt.ylim(-0.2,1.2)
# plt.savefig(plot_dir+'sn_scatter_pixel.png')
#
# plt.figure()
# plt.scatter(pixel_vector, PREDS['error']['level set'])
# plt.xlabel('radius (pixels)')
# plt.ylabel('error')
# plt.xlim(-0.2,30.0)
# plt.ylim(-0.2,1.2)
# plt.savefig(plot_dir+'ls_scatter_pixel.png')
#
# #Radius error binning plot
# def radiusErrorCount(errors,radiuses,err_thresh, rad_range):
#     """
#     computes number of errors smaller than err_thresh for all vessels with radius
#     in rad_range
#
#     args:
#         @a errors: list of errors
#         @a radiuses: list of radiuses
#         @a err_thresh: error threshold below which to count
#         @a rad_range: radius range in which to consider errors
#
#     returns:
#         @a err_count: number of errors below threshold
#     """
#     r = [1 for i in range(len(radiuses)) if radiuses[i] >= rad_range[0] and
#     radiuses[i] < rad_range[1]]
#     e = [1 for i in range(len(radiuses)) if radiuses[i] >= rad_range[0] and
#     radiuses[i] < rad_range[1] and errors[i] <= err_thresh]
#
#     err_count = float(np.sum(e))/np.sum(r)
#     return err_count
#
# r =[[0,0.3],[0.3,1.0],[1.0,2.5]]
# rad_err_thresh = 0.25
#
# sn = [radiusErrorCount(PREDS['error']['RSN'],radius_vector,rad_err_thresh,rad) for rad in
# r]
# # obg = [radiusErrorCount(PREDS['error']['OBG_RSN'],radius_vector,rad_err_thresh,rad) for rad in
# # r]
# hed_rad = [radiusErrorCount(PREDS['error']['HEDFC'],radius_vector,rad_err_thresh,rad) for rad in
# r]
# i2i_rad = [radiusErrorCount(PREDS['error']['I2I-2DFC'],radius_vector,rad_err_thresh,rad) for rad in
# r]
# ls = [radiusErrorCount(PREDS['error']['level set'],radius_vector,rad_err_thresh,rad) for rad in
# r]
#ls_seg = [radiusErrorCount(PREDS['error']['3D segmentation'],radius_vector,rad_err_thresh,rad) for rad in
#r]
#sn_ft = [radiusErrorCount(PREDS['error']['RSN_finetune'],radius_vector,rad_err_thresh,rad) for rad in
#r]
#sn_multi = [radiusErrorCount(PREDS['error']['RSN_multi'],radius_vector,rad_err_thresh,rad) for rad in
#r]
# ind = np.array([0,3,6])
# width=0.3
#
# plt.figure()
# plt.bar(ind,sn,width,color='r', label='RSN')
# plt.bar(ind+width,obg,width,color='b', label='OBG_RSN')
# plt.bar(ind+2*width,hed_rad,width,color='black', label='HED')
# plt.bar(ind+3*width,i2i_rad,width,color='orange', label='I2I-2D')
# plt.bar(ind+4*width,sn_ft,width,color='m', label='RSN_finetune')
# plt.bar(ind+5*width,sn_multi,width,color='c', label='RSN_multi')
# plt.bar(ind+6*width,ls,width,color='pink', label='level set')
# plt.bar(ind+7*width,ls_seg,width,color='green', label='3D segmentation')

# ind = np.array([0,2,4])
# width=0.2
#
# plt.figure()
# plt.bar(ind,sn,width,color='r', label='RSN')
# # plt.bar(ind+width,obg,width,color='b', label='OBG_RSN')
# plt.bar(ind+1*width,hed_rad,width,color='black', label='HEDFC')
# plt.bar(ind+2*width,i2i_rad,width,color='orange', label='I2I-FC')
# plt.bar(ind+3*width,ls,width,color='pink', label='level set')
#
# plt.ylim(0,1.0)
# plt.ylabel('Fraction of vessels with error below threshold')
# plt.xlabel('radius')
# plt.xticks(ind+width, ['0-0.3cm','0.3-1.0cm','1.0-2.5cm'])
# lg = plt.legend(loc='center right', bbox_to_anchor=(1.5,0.5))
#
# plt.savefig(plot_dir+'radiusBar.png',bbox_extra_artists=(lg,), bbox_inches='tight')

############################
# pixel error plot
############################
# rpix =[[0,5.0],[5.0,10.0],[15.0,30.0]]
# rad_err_thresh = 0.25
#
# sn_pix = [radiusErrorCount(PREDS['error']['RSN'],pixel_vector,rad_err_thresh,rad) for rad in
# rpix]
# # obg_pix = [radiusErrorCount(PREDS['error']['OBG_RSN'],pixel_vector,rad_err_thresh,rad) for rad in
# # rpix]
# hed_rad_pix = [radiusErrorCount(PREDS['error']['HED'],pixel_vector,rad_err_thresh,rad) for rad in
# rpix]
# i2i_rad_pix = [radiusErrorCount(PREDS['error']['I2I-2D'],pixel_vector,rad_err_thresh,rad) for rad in
# rpix]
# ls_pix = [radiusErrorCount(PREDS['error']['level set'],pixel_vector,rad_err_thresh,rad) for rad in
# rpix]
#
# ind = np.array([0,2,4])
# width=0.3
#
# plt.figure()
# plt.bar(ind,sn_pix,width,color='r', label='RSN')
# # plt.bar(ind+width,obg_pix,width,color='b', label='OBG_RSN')
# plt.bar(ind+1*width,hed_rad_pix,width,color='black', label='HED')
# plt.bar(ind+2*width,i2i_rad_pix,width,color='orange', label='I2I-2D')
# plt.bar(ind+3*width,ls_pix,width,color='pink', label='level set')
#
# plt.ylim(0,1.0)
# plt.ylabel('Fraction of vessels with error below threshold')
# plt.xlabel('radius (pixels)')
# plt.xticks(ind+width, ['0-5 pixels','5-10 pixels','10-30 pixels'])
# lg = plt.legend(loc='upper right', bbox_to_anchor=(1.6,0.5))
#
# plt.savefig(plot_dir+'radiusBar_pixels.png',bbox_extra_artists=(lg,), bbox_inches='tight')
#
# #Central pixel probability vs error
# cprob = PREDS['seg']['RSN'][:,image_dims/2,image_dims/2]
# s = np.mean(PREDS['seg']['RSN'], axis=(1,2))
# m = np.max(PREDS['seg']['RSN'], axis=(1,2))
# sd = np.std(PREDS['seg']['RSN'], axis=(1,2))
#
# plt.figure()
# plt.scatter(cprob, PREDS['error']['RSN'])
# plt.xlabel('center pixel probability')
# plt.ylabel('error')
# plt.savefig(plot_dir+'prob_center.png')
#
# plt.figure()
# plt.scatter(s, PREDS['error']['RSN'])
# plt.xlabel('mean pixel probability')
# plt.ylabel('error')
# plt.savefig(plot_dir+'prob_mean.png')
#
# plt.figure()
# plt.scatter(m, PREDS['error']['RSN'])
# plt.xlabel('max pixel probability')
# plt.ylabel('error')
# plt.savefig(plot_dir+'prob_max.png')
#
# plt.figure()
# plt.scatter(sd, PREDS['error']['RSN'])
# plt.xlabel('pixel probability std')
# plt.ylabel('error')
# plt.savefig(plot_dir+'prob_std.png')

#Compute radius change statistics

# names = open(dataDir+'names.txt').readlines()
# names = [names[i] for i in inds]
# names = [n.split('.') for n in names]
# names = [names[i] for i in range(len(names)) if PREDS['contour']['RSN'][i] != []]
# radius_vector = [utility.contourRadius(x) for x in PREDS['contour']['RSN'] if x != []]
# radius_vector_true = [utility.contourRadius(contours_test[i]) for i in range(Ntest) if
#  PREDS['contour']['RSN'][i] != []]
# vs = sorted(zip(radius_vector,names), key=lambda x:(x[1][0]+x[1][1],int(x[1][2])))
# radius = [vs[i][0] for i in range(len(vs))]
# paths = [vs[i][1][0]+vs[i][1][1] for i in range(len(vs))]
# vs = sorted(zip(radius_vector_true,names), key=lambda x:(x[1][0]+x[1][1],int(x[1][2])))
# radius_true = [vs[i][0] for i in range(len(vs))]
#
# ratios = []
# ratios_true = []
# for i in range(len(radius)-1):
#     if paths[i] == paths[i+1]:
#         ratios.append(float(radius[i])/radius[i+1])
#         ratios_true.append(float(radius_true[i])/radius_true[i+1])
#
# plt.figure()
# plt.hist(ratios, bins=40, alpha=0.5, label='RSN')
# plt.hist(ratios_true, bins=40, alpha=0.5, label='Ground truth')
# plt.xlabel('radius ratio')
# plt.legend()
# plt.savefig(plot_dir+'radiusratio.png')

####################################
# 3D errors
####################################
# print "starting 3d analysis"
# vtk_dir = config['learn_params']['output_dir']+'vtk/'
# files = os.listdir(vtk_dir)
# files = [f for f in files if 'truth' in f]
#
# codes = ['RSN','ls_seg','ls']
# g = open(plot_dir+'jaccard3d.txt','w')
# # for c in codes:
# #     errs = []
# #     for f in files:
# #         mod = f.replace('truth',c)
# #         print f, mod
# #         p1 = utility.readVTKPD(vtk_dir+f)
# #         p2 = utility.readVTKPD(vtk_dir+mod)
# #         e = utility.jaccard3D(p1,p2)
# #         errs.append(e)
# #     g.write('{} : {}, {}\n'.format(c,np.mean(errs),np.std(errs)))
# g.close()

############################
# Path perturbation analysis
############################

translations = [1,3,5,7,10]
net = load_model(model_dir+'I2INetFC.h5',
    custom_objects={'scaled_cross_entropy':utility.scaled_cross_entropy, 'balanced_cross_entropy':utility.balanced_cross_entropy})
prs = []
pr_snakes = []
f = open(plot_dir+'translations.csv','w')
f.write('translation, hausdorf, dice, assd, precision\n')
for t in translations:
    print t
    X,Y = vasc2d.get_subset(N=Ntest,translate=t,crop=image_dims,random=False)
    ypred = net.predict(X)[0]
    contour,errs,thresh,roc,pr,acc,mean_acc,dorf,dice, prec,asdl = get_outputs(ypred[:,:,:,0], Y)
    f.write('{},{}, {}, {}, {}, {}\n'.format('I2INetFC',t,np.mean(dorf),np.mean(dice),np.mean(asdl),np.mean(prec)))
    prs.append(pr)

    #snakes
    print 'snake'
    snakes = [utility.snake(X[i,:,:,0]*255,[Pw/2,Pw/2],10) for i in range(len(X))]
    snake_seg = [utility.contourToSeg(snakes[i],[0,0],[Pw,Pw],[1.0,1.0]) for i in range(len(snakes))]
    snake_seg = np.array(snake_seg)
    contour,errs,thresh,roc,pr,acc,mean_acc,dorf,dice, prec,asdl = get_outputs(snake_seg, Y)
    f.write('{},{}, {}, {}, {}, {}\n'.format('level set',t,np.mean(dorf),np.mean(dice),np.mean(asdl),np.mean(prec)))
    pr_snakes.append(pr)
f.close()

colors = ['b','r','g','k','y']
labels = ['2%','7%','10%','15%','22%']
plt.figure()
for i in range(len(translations)):
    plt.plot(prs[i][1],prs[i][0], label=labels[i],linewidth=2, color=colors[i])
plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.savefig(plot_dir+'pr_translations.png',dpi=500)

plt.figure()
for i in range(len(translations)):
    plt.plot(pr_snakes[i][1],pr_snakes[i][0], label=labels[i], linewidth=2, color=colors[i])
plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.savefig(plot_dir+'pr_snakes_translations.png',dpi=500)

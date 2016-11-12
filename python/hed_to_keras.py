#trains hed and generates predictions on test set
#see http://stackoverflow.com/questions/32379878/cheat-sheet-for-caffe-pycaffe
import sys
import os

caffe_root = "/home/marsdenlab/projects/hed/"
sys.path.insert(0,os.path.abspath(caffe_root+'python'))
import caffe
from utility import util_data
import configparser

from tqdm import tqdm
from utility import util_model
import numpy as np

def caffe_conv_to_keras(layer):
    """converts a caffe convolution layer to a keras convolution weight and bias
    """
    weights = layer[0].data
    bias = layer[1]
    filters,C,H,W = layer[0].data.shape

    ret_mat = np.zeros((H,W,C,filters))

    for i in range(filters):
        for j in range(C):
            ret_mat[:,:,j,i] = np.rot90(weights[i,j],2)
    ret_weight=[]
    ret_weight.append(ret_mat)
    ret_weight.append(layer[1].data)
    return ret_weight

m = util_model.hed_keras((320,480,3))

weights = './models/hed_pretrained_bsds.caffemodel'
netfile = './models/deploy.prototxt'
caffe.set_mode_cpu()

net = caffe.Net(os.path.abspath(netfile),
os.path.abspath(weights),caffe.TEST)

for k in net.params.keys():
    w = net.params[k]

    for i in range(len(w)):
        print "layer {} weight {} has shape {}".format(k,i,w[i].data.T.shape)

for i in range(0,len(m.layers)):
    k_name = m.layers[i].name
    for k in net.params.keys():
        if k == k_name:
            print "name match found for {}".format(k)
            w = caffe_conv_to_keras(net.params[k])
            m.layers[i].set_weights(w)

m.save('./models/hed_bsds_keras.h5')

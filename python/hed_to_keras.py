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
from PIL import Image

#parse config
config = configparser.ConfigParser()
config.read('options.cfg')
vasc_dims = int(config['learn_params']['image_dims'])

#set test dims
dim_x = 320
dim_y = 480
channels = 3
def caffe_conv_to_keras(layer):
    """converts a caffe convolution layer to a keras convolution weight and bias
    """
    weights = layer[0].data
    bias = layer[1]
    filters,C,H,W = layer[0].data.shape

    ret_mat = np.zeros((H,W,C,filters))

    for i in range(filters):
        for j in range(C):
            ret_mat[:,:,j,i] = np.rot90(weights[i,j],0)
    #ret_mat = weights.T[:,:,::-1,::-1]
    ret_weight=[]
    ret_weight.append(ret_mat)
    ret_weight.append(layer[1].data)
    return ret_weight

def transfer_caffe_weights(m,net):
    for i in range(0,len(m.layers)):
        k_name = m.layers[i].name
        for k in net.params.keys():
            if k == k_name:
                print "name match found for {}".format(k)
                w = caffe_conv_to_keras(net.params[k])
                C = m.layers[i].get_weights()[0].shape[2]
                if C == 1:
                    w[0] = w[0][:,:,0,:]
                    s = w[0].shape
                    w[0] = w[0].reshape((s[0],s[1],1,s[2]))
                m.layers[i].set_weights(w)

m = util_model.hed_keras((dim_x,dim_y,channels))
m_vasc = util_model.hed_keras((vasc_dims, vasc_dims,1))

weights = './models/hed_pretrained_bsds.caffemodel'
netfile = './models/deploy.prototxt'
caffe.set_mode_cpu()

net = caffe.Net(os.path.abspath(netfile),
os.path.abspath(weights),caffe.TEST)

for k in net.params.keys():
    w = net.params[k]

    for i in range(len(w)):
        print "layer {} weight {} has shape {}".format(k,i,w[i].data.T.shape)

transfer_caffe_weights(m,net)
transfer_caffe_weights(m_vasc,net)

m.save('./models/hed_bsds_keras.h5')
m_vasc.save('./models/hed_bsds_vasc.h5')
######################################
# Generate input and compare networks
######################################

from keras import backend as K
def get_activation(x,name,model):
    f = K.function([model.layers[0].input],
        [model.get_layer(name).output])

    return f([x])[0]

im = Image.open('./test/100039.jpg')
im = im.crop((0,0,480,320))
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_caffe = in_.transpose((2,0,1))

#compute caffe
net.blobs['data'].reshape(1,*in_caffe.shape)
net.blobs['data'].data[...] = in_caffe
net.forward()
out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
y_caffe = [fuse,out1,out2,out3,out4,out5]
in_ = in_.reshape((1,in_.shape[0],in_.shape[1],in_.shape[2]))
y_keras = m.predict(in_)

for i in range(len(y_caffe)):
    err = np.mean(abs(y_caffe[i]-y_keras[i][0,:,:,0]))
    print "error out {} = {}".format(i,err)

for i in range(len(m.layers)):
    name = m.layers[i].name
    k_act = get_activation(in_,name,m)
    if name in net.blobs.keys():
        c_act = net.blobs[name].data
        print "k shape ={}, c shape = {}".format(k_act.shape,c_act.shape)

        if "conv" in name:
            err = np.mean(abs(k_act - np.transpose(c_act,(0,2,3,1))))
            print "{} err = {}".format(name,err)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm

def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    N = len(scale_lst)
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,N,i+1)
        plt.imshow(1-scale_lst[i][0][:,:,0], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()

plot_single_scale(y_keras,20)
plt.show()

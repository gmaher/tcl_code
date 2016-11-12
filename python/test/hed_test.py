import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from utility import util_model
from keras.models import Model, load_model
m = util_model.hed_keras((224,224,1))

x = np.ones((1,224,224,1))

y = m.predict(x)

for i in range(len(y)):
    print "output {} has shape {}".format(i,y[i].shape)

w = m.get_weights()

for i in range(len(w)):
    print "weight {} has shape {}".format(i, w[i].shape)

##########################
# Test pretrained model
##########################
#Note some code taken from https://github.com/s9xie/hed
from PIL import Image
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

im = Image.open('100039.jpg')
im = im.crop((0,0,480,320))
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.reshape((1,in_.shape[0],in_.shape[1],in_.shape[2]))
m_pretrained = load_model('../models/hed_bsds_keras.h5')
y = m_pretrained.predict(in_)

plot_single_scale(y,20)
plt.show()

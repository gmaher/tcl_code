#trains hed and generates predictions on test set
#see http://stackoverflow.com/questions/32379878/cheat-sheet-for-caffe-pycaffe
import sys
import os
sys.path.append(os.path.abspath('../../hed/python'))
import caffe
from utility import util_data
import configparser

############################
# Parse config
############################
config = configparser.ConfigParser()
config.read('options.cfg')

dataDir = config['learn_params']['train_dir']

############################
# Get data
############################
vasc2d = util_data.VascData2D(dataDir)

########################
# Train
########################
weights = './models/hed_pretrained_bsds.caffemodel'

caffe.set_mode_gpu()

solver = caffe.SGDSolver('./models/solver.prototxt')

test_net = solver.test_nets[0]

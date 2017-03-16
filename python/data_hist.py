import numpy as np
import utility.utility as utility
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, BatchNormalization, Dense, merge, Reshape, Flatten
from keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from utility import util_data
from utility import util_model
import scipy
import argparse
np.random.seed(0)
##########################
# Parse args
##########################
#args
parser = argparse.ArgumentParser()
parser.add_argument('config_file')
args = parser.parse_args()
config_file = args.config_file

#config
config = utility.parse_config(config_file)
dataDir = config['learn_params']['data_dir']+'train/'
valDir = config['learn_params']['data_dir']+'val/'
testDir = config['learn_params']['data_dir']+'test/'
output_dir = config['learn_params']['output_dir']
model_dir = config['learn_params']['model_dir']
pred_dir = config['learn_params']['pred_dir']
plot_dir = config['learn_params']['plot_dir']
mask = config['learn_params']['mask']=='True'

vascMR = util_data.VascData2D('./data/64final/mr/train/')
vascCT = util_data.VascData2D('./data/64final/ct/train/')
vascALL = util_data.VascData2D('./data/64final/jameson/train/')

vascMR_glob = util_data.VascData2D('./data/64final/mr/train/', 'global_max')
vascCT_glob = util_data.VascData2D('./data/64final/ct/train/', 'global_max')
vascALL_glob = util_data.VascData2D('./data/64final/jameson/train/', 'global_max')


plt.figure()
plt.hist(np.ravel(vascALL.images), bins=50)
plt.savefig('./plots/hist_all.png')

plt.figure()
plt.hist(np.ravel(vascMR.images), bins=50)
plt.savefig('./plots/hist_mr.png')

plt.figure()
plt.hist(np.ravel(vascCT.images), bins=50)
plt.savefig('./plots/hist_ct.png')

plt.figure()
plt.hist(np.ravel(vascALL.images_norm), bins=50)
plt.savefig('./plots/hist_all_norm.png')

plt.figure()
plt.hist(np.ravel(vascMR.images_norm), bins=50)
plt.savefig('./plots/hist_mr_norm.png')

plt.figure()
plt.hist(np.ravel(vascCT.images_norm), bins=50)
plt.savefig('./plots/hist_ct_norm.png')

plt.figure()
plt.hist(np.ravel(vascALL_glob.images_norm), bins=50)
plt.savefig('./plots/hist_all_glob.png')

plt.figure()
plt.hist(np.ravel(vascMR_glob.images_norm), bins=50)
plt.savefig('./plots/hist_mr_glob.png')

plt.figure()
plt.hist(np.ravel(vascCT_glob.images_norm), bins=50)
plt.savefig('./plots/hist_ct_glob.png')

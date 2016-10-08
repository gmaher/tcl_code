from keras.models import Model, load_model
import numpy as np
import utility
import util_plot
import argparse
from tqdm import tqdm

np.random.seed(0)
##########################
# Parse args
##########################
parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
args = parser.parse_args()

dataDir = args.dataDir
imString = dataDir + 'images.npy'
segString = dataDir + 'segmentations.npy'
metaString = dataDir + 'metadata.npy'
contourString = dataDir + 'contours.npy'

###########################
# Load data and preprocess
###########################
images = np.load(imString)
images = images.astype(float)
segs = np.load(segString)
meta = np.load(metaString)
contours = np.load(contourString)

N, Pw, Ph = images.shape

images = images.reshape((N,Pw,Ph,1))
segs = segs.reshape((N,Pw,Ph,1))
Pmax = np.amax(images)
Pmin = np.amin(images)
print "Images stats\n N={}, Width={}, Height={}, Max={}, Min={}".format(
    N,Pw,Ph,Pmax,Pmin)

images_norm = utility.normalize_images(images)

#train test split
X_train, Y_train, X_test, Y_test, inds, train_inds, test_inds = \
    utility.train_test_split(images_norm, segs,0.75)

############################
# Load model
############################
FCN = load_model('./models/FCN.h5')

##############################
# Calculate errors and analyze
##############################
Y_pred = FCN.predict(X_test)

meta_test = meta[:,test_inds,:]
contours_test = contours[test_inds]

Y_pred = Y_pred.reshape((-1,Pw,Ph))
err_list = []
ts_list = []
iso = 0.55
errs = []
bad_error_inds = []

predicted_contours = \
    utility.listSegToContours(Y_pred,meta_test[1,:,:], meta_test[0,:,:], iso)

errs = utility.listAreaOverlapError(predicted_contours, contours_test)

for i in range(0,len(errs)):
    if errs[i] > 0.5:
        bad_error_inds.append(i)

cum_err, ts = utility.cum_error_dist(errs,0.025)
err_list.append(cum_err)
ts_list.append(ts)

utility.plot_data_plotly(ts_list, err_list, [str(iso)])

#investigate bad_errors
Nplots = 20
Y_test = Y_test.reshape((-1,Pw,Ph))

subtitles = []
f = open(dataDir+'names.txt')
f = f.readlines()
for j in range(1,Nplots+1):
    subtitles = subtitles + [f[test_inds[j]]]*4

fig = util_plot.Figure(Nplots,4, subtitles=subtitles, height=250*Nplots, width=2000)

for j in range(1,Nplots+1):
    i = bad_error_inds[j]
    img = images[test_inds[i]].reshape((Pw,Ph))
    img_norm = X_test[i].reshape((Pw,Ph))
    seg_truth = Y_test[i]
    seg_pred = Y_pred[i]
    c_pred = predicted_contours[i]
    c_truth = contours_test[i]

    #get bounds
    spacing, origin, dims = meta[:,test_inds[j],:]
    dims = dims-1
    bounds = [origin[0], origin[0]+dims[0]*spacing[0],
        origin[1], origin[1]+dims[1]*spacing[1]]

    #plot images
    fig.add_heatmap(img, bounds, row=j, col=1)
    fig.add_heatmap(img_norm, bounds, row=j, col=2)
    fig.add_heatmap(seg_truth, bounds, row=j, col=3)
    fig.add_heatmap(seg_pred, bounds, row=j, col=4)

    #plot truth contours
    fig.add_scatter2d(c_truth[:,0], c_truth[:,1], 'truth', row=j, col=3)
    fig.add_scatter2d(c_truth[:,0], c_truth[:,1], 'truth', row=j, col=4)

    #plot predicted contours
    for k in range(0,len(c_pred)):
        c = c_pred[k]
        fig.add_scatter2d(c[:,0], c[:,1], 'predicted{}'.format(k), row=j, col=3)
        fig.add_scatter2d(c[:,0], c[:,1], 'predicted{}'.format(k), row=j, col=4)

fig.plot()

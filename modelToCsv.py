from keras.models import Model, load_model
import numpy as np
import utility
import util_plot
import argparse
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
parser.add_argument('--models',nargs='+')
args = parser.parse_args()

dataDir = args.dataDir
models = args.models

images,segs,contours,meta,f = utility.get_data(dataDir)

images_norm = utility.normalize_images(images)
data = []
THRESHOLD = 0.4
for model in models:
    print model
    mod = load_model(model)

    Y_pred = mod.predict(images_norm)
    N,W,H,C = Y_pred.shape
    Y_pred = Y_pred.reshape((N,W,H))
    Y_pred = utility.threshold(Y_pred, THRESHOLD)

    contours_pred =\
        utility.listSegToContours(Y_pred,meta[1,:,:], meta[0,:,:],0.5)

    errs = utility.listAreaOverlapError(contours_pred,contours)
    print np.mean(errs)

    #Get model name as code
    s_model = model.split('/')[-1]
    s_model = s_model.replace('.h5','')

    for i in range(0,len(f)):
        dentry = {}
        s = f[i].split('.')


        dentry['image'] = s[0]
        dentry['path'] = s[1]
        dentry['member'] = s[2]
        dentry['code'] = s_model
        dentry['overlap_error'] = errs[i]
        data.append(dentry)

df = pd.DataFrame(data)
df.to_csv('model_groups_errors.csv')

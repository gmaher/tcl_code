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

for model in models:
    for i in range(0,len(f)):

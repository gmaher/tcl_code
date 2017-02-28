import numpy as np
import utility.utility as utility
import argparse
import time

import os
from sklearn.metrics import roc_curve
import configparser
import SimpleITK as sitk
from medpy.metric.binary import hd, assd, ravd, dc
from joblib import Parallel, delayed
import multiprocessing
import vtk
np.random.seed(0)

DX = 0.01
ts = np.arange(0,1+DX,DX)
#########################################
# Parse arguments
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('vtkfolder')

args = parser.parse_args()

folder = args.vtkfolder


#Get polydatas
files = os.listdir(folder)
#files = [f for f in files if pattern in f]
pds = []
for f in files:
    pds.append(utility.readVTKPD(folder+f))

writer = vtk.vtkPolyDataWriter()

smoother = vtk.vtkSmoothPolyDataFilter()
for i in range(len(pds)):

    p = utility.vtkSmoothPD(pds[i],iters=20,relax=0.5)
    writer.SetInputData(p)
    writer.SetFileName(folder+files[i])
    writer.Update()
    writer.Write()

import argparse
import numpy as np
import pandas as pd
import os
import vtk
from utility import areaOverlapError, VTKPDPointstoNumpy

argparser = argparse.ArgumentParser()
argparser.add_argument('directory')
args = argparser.parse_args()

codes = ['edge96']

dataDir = args.directory

files = os.listdir(dataDir)

data = []

reader = vtk.vtkPolyDataReader()

def read_pd(reader, fpath):
    reader.SetFileName(fpath)
    reader.Update()
    return reader.GetOutput()

for file in files:
    #print file
    if 'truth.ls' in file:
        fp = os.path.join(dataDir, file)
        attr = file.split('.')

        for code in codes:
            dentry = {}
            dentry['image'] = attr[0]
            dentry['path'] = attr[1]
            dentry['member'] = attr[2]
            dentry['code'] = code

            fp_code = fp.replace('truth',code)

            print fp
            print fp_code

            if os.path.isfile(fp_code):
                pd_truth = read_pd(reader,fp)
                pd_truth_np = VTKPDPointstoNumpy(pd_truth)
                pd_edge = read_pd(reader,fp_code)
                pd_edge_np = VTKPDPointstoNumpy(pd_edge)

                dentry['overlap_error'] =\
                    areaOverlapError(pd_truth_np, pd_edge_np)
            else:
                dentry['overlap_error'] = 1.0

            data.append(dentry)

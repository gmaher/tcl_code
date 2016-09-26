import argparse
import numpy as np
import pandas as pd
import os
import vtk
from utility import *

argparser = argparse.ArgumentParser()
argparser.add_argument('directory')
args = argparser.parse_args()

dataDir = args.directory

codes = get_codes(dataDir)

files = os.listdir(dataDir)

data = []

reader = vtk.vtkPolyDataReader()

def read_pd(reader, fpath):
    reader.SetFileName(fpath)
    reader.Update()
    return reader.GetOutput()

def reorder_and_convert(pd):
    C = getPDConnectivity(pd)
    O = getNodeOrdering(C)
    pd_np = VTKPDPointstoNumpy(pd)
    return pd_np[O,:]

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
                pd_truth_np = reorder_and_convert(pd_truth)

                pd_edge = read_pd(reader,fp_code)
                if validSurface(pd_edge):
                    pd_edge_np = reorder_and_convert(pd_edge)
                    dentry['overlap_error'] =\
                        areaOverlapError(pd_truth_np, pd_edge_np)
                else:
                    dentry['overlap_error'] = 1.0
            else:
                dentry['overlap_error'] = 1.0

            data.append(dentry)
df = pd.DataFrame(data)
df.to_csv('vtk_groups_errors.csv')

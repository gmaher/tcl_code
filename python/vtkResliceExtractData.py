import numpy as np
import utility.utility as utility
import argparse
import time

import os
from sklearn.metrics import roc_curve
import configparser
import SimpleITK as sitk
np.random.seed(0)
import vtk
import matplotlib.pyplot as plt
#########################################
# Parse arguments
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('config_file')
args = parser.parse_args()

config_file = args.config_file
config = utility.parse_config(config_file)

output_dir = config['learn_params']['output_dir']

mhas = open(output_dir+'images.txt').readlines()
mhas = [i.replace('\n','') for i in mhas]

p = (2.723052,-0.230689,-12.755765)
z = (0.305266,-0.617498,0.724920)
x = (0.000000,0.761257,0.648451)
y = np.cross(z,x)

# p = (0,0,0)
# z = (0,0,1)
# x = (1,0,0)
# y = (0,1,0)

#read mha
reader = vtk.vtkMetaImageReader()
reader.SetFileName(mhas[0])
reader.Update()
print mhas[0]
img = reader.GetOutput()
ext = [100,100]
#img.SetOrigin(0.0,0.0,0.0)
print img.GetOrigin()

#vtk reslice
c = utility.getImageReslice(img,ext,p,z,x)
plt.imshow(c[0])
plt.colorbar()
plt.show()

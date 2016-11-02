import SimpleITK as sitk
import numpy as np
import plotly as py
import plotly.graph_objs as go
import utility
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('impath')
args = parser.parse_args()
impath = args.impath

im = sitk.ReadImage(impath)

imarray = sitk.GetArrayFromImage(im)

print imarray.shape
print np.max(imarray)
print np.min(imarray)
print np.std(imarray)

utility.scatter3d([imarray],[1],[255])

h = utility.make_heat_trace(imarray[100,:,:])
py.offline.plot([h], filename='heat.html')

utility.isoSurface3d(imarray, None)

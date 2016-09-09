import SimpleITK as sitk
import numpy as np
import plotly as py
import plotly.graph_objs as go
import utility

impath = "/media/gabriel/Data/marsden_data/vascular_data/OSMSC0001/\
OSMSC0001-cm.mha"

im = sitk.ReadImage(impath)

imarray = sitk.GetArrayFromImage(im)

print imarray.shape
print np.max(imarray)
print np.std(imarray)

utility.scatter3d([imarray],[1000],[1070])

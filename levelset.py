import SimpleITK as sitk
from utility import *
import plotly as py
###########################
# Set some input parameters
###########################
sliceid = 50
impath = '/home/marsdenlab/Dropbox/vascular_data/OSMSC0006/OSMSC0006-cm.mha'

xstart = 100
ystart = 10
dim = 64

sigma = 0.1

seedx = dim/2
seedy = dim/2
############################
# Load image and get patch
############################
reader = sitk.ImageFileReader()
reader.SetFileName(impath)
img = reader.Execute()
print img.GetSize()

patch = img[xstart:xstart+dim, ystart:ystart+dim,sliceid]
print patch
print type(patch)

np_patch = sitk.GetArrayFromImage(patch)
heatmap(np_patch, fn='./plots/patch.html')

##########################
# Compute feature image
##########################
gradMagFilter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
gradMagFilter.SetSigma(sigma)
filt_patch = gradMagFilter.Execute(patch)

np_patch = sitk.GetArrayFromImage(filt_patch)
heatmap(np_patch, fn='./plots/blur.html')

###############################
# Create initialization image
###############################
seed_img = sitk.Image(dim,dim,sitk.sitkUInt8)
seed_img.SetSpacing(patch.GetSpacing())
seed_img.SetOrigin(patch.GetOrigin())
seed_img.SetDirection(patch.GetDirection())
seed_img[seedx,seedy] = 1

distance = sitk.SignedMaurerDistanceMapImageFilter()
distance.InsideIsPositiveOff()
distance.UseImageSpacingOn()

dis_img = distance.Execute(seed_img)
np_patch = sitk.GetArrayFromImage(dis_img)
heatmap(np_patch, fn='./plots/distance.html')

init_img = sitk.BinaryThreshold(dis_img, -1000, 10)
init_img = sitk.Cast(init_img, filt_patch.GetPixelIDValue())*-1+0.5
np_patch = sitk.GetArrayFromImage(init_img)
heatmap(np_patch, fn='./plots/init.html')

#####################################
# Run GeodesicActiveContour level set
#####################################
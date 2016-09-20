import vtk
from utility import *

###############################
# Read structured points
###############################
imgPath = '/home/marsdenlab/projects/I2INet/OSMSC0006_runs\
/OSMSC0006.superior_mesentaric.8.pot_truth.vts'

reader = vtk.vtkStructuredPointsReader()
reader.SetFileName(imgPath)
reader.Update()
sp = reader.GetOutput()

################################
# Convert to numpy
################################
imarray = VTKSPtoNumpy(sp)


################################
# Read polydata
################################
pdPath = '/home/marsdenlab/projects/I2INet/OSMSC0006_runs\
/OSMSC0006.superior_mesentaric.8.ls.truth.vtp'
pdReader = vtk.vtkPolyDataReader()
pdReader.SetFileName(pdPath)
pdReader.Update()
pd = pdReader.GetOutput()

################################
#   Convert To Numpy
################################
parray = VTKPDPointstoNumpy(pd)
print parray

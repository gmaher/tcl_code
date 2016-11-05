import vtk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

fn = args.filename

#create reader
imr = vtk.vtkMetaImageReader()
imr.SetFileName(fn)
imr.Update()
print imr

#create polydataalgorithm
#this operates on an image and
#returns a polydata object
mc = vtk.vtkMarchingCubes()
mc.AddInputData(imr.GetOutput())
mc.ComputeNormalsOn()
mc.ComputeGradientsOn()
mc.SetValue(0,1.0)

#create polydata mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(mc.GetOutput())
mapper.ScalarVisibilityOff()

#create actor to store the polydata object
actor = vtk.vtkActor()
actor.GetProperty().SetColor(1,1,1)
actor.SetMapper(mapper)

#create renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

#create render window
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(renderer)

#create render window interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renwin)
iren.Initialize()
iren.Start()

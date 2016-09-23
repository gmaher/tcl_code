import vtk
from utility import *
import plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.tools import FigureFactory as FF

BOUNDS = [-2.5, 2.5, -2.5, 2.5]
##########################################
# Get FileNames
##########################################
groups_dir = '/home/marsdenlab/projects/I2INet/'
code = 'edge96'
pot_truth = groups_dir + 'OSMSC0006.aorta.100.truth.pot.vts'
mag_truth = groups_dir + 'OSMSC0006.aorta.100.truth.mag.vts'
ls_truth = groups_dir + 'OSMSC0006.aorta.100.truth.ls.vtp'
pot_edge = groups_dir + 'OSMSC0006.aorta.100.{}.pot.vts'.format(code)
mag_edge = groups_dir + 'OSMSC0006.aorta.100.{}.mag.vts'.format(code)
ls_edge = groups_dir + 'OSMSC0006.aorta.100.{}.ls.vtp'.format(code)
ls_image = groups_dir + 'OSMSC0006.aorta.100.{}.ls.vtp'.format('image')

###########################################
# Convert To Numpy
###########################################
np_ls_truth = VTKPDReadAndReorder(ls_truth)
np_ls_edge = VTKPDReadAndReorder(ls_edge)
np_ls_image = VTKPDReadAndReorder(ls_image)

np_pot_truth = VTKSPtoNumpyFromFile(pot_truth)
np_mag_truth = VTKSPtoNumpyFromFile(mag_truth)
np_pot_edge = VTKSPtoNumpyFromFile(pot_edge)
np_mag_edge = VTKSPtoNumpyFromFile(mag_edge)

###########################################
# Start figure construction
###########################################
pot_truth_trace = make_heat_trace(np_pot_truth[0], BOUNDS)
mag_truth_trace = make_heat_trace(np_mag_truth[0], BOUNDS)
pot_edge_trace = make_heat_trace(np_pot_edge[0], BOUNDS)
mag_edge_trace = make_heat_trace(np_mag_edge[0], BOUNDS)

ls_trace = make_traces(
    [np_ls_truth[:,0], np_ls_image[:,0], np_ls_edge[:,0]],
    [np_ls_truth[:,1], np_ls_image[:,1], np_ls_edge[:,1]],
    ['user','image','edge'])

fig = tools.make_subplots(rows=2,cols=3)
fig.append_trace(pot_truth_trace,1,1)
fig.append_trace(mag_truth_trace,1,2)
fig.append_trace(pot_truth_trace,1,3)

fig.append_trace(pot_edge_trace,2,1)
fig.append_trace(mag_edge_trace,2,2)
fig.append_trace(pot_edge_trace,2,3)

for t in ls_trace:
    fig.append_trace(t,1,3)
    fig.append_trace(t,2,3)

fig['layout'].update(height=1000, width=1200, title='plot')
py.offline.plot(fig,filename='./plots/edgecompare.html')

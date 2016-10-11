import utility
import numpy as np
import util_plot
import argparse
from skimage import segmentation

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()
f = args.file

pd = utility.readVTKPD('/home/marsdenlab/projects/OBG_full/raw/{}.truth.ls.vtp'.format(f))
sp = utility.readVTKSP('/home/marsdenlab/projects/OBG_full/raw/{}.truth.mag.vts'.format(f))
contour = utility.VTKPDReadAndReorder('/home/marsdenlab/projects/OBG_full/raw/{}.truth.ls.vtp'.format(f))
mag = utility.VTKSPtoNumpyFromFile('/home/marsdenlab/projects/OBG_full/raw/{}.truth.mag.vts'.format(f))
spacing = sp.GetSpacing()
origin = sp.GetOrigin()
bounds = [origin[0], origin[0]+ 64*spacing[0],
    origin[1], origin[1]+64*spacing[1]]

seg = utility.contourToSeg(contour,origin,[64,64],spacing)
contour_pred = utility.segToContour(seg, origin, spacing, 0.55)[0]

ps = np.linspace(0,2*np.pi-0.1,50)
init_snake = 0.3*np.asarray([np.sin(ps),np.cos(ps)])
init_snake = init_snake.T
snake_contour = segmentation.active_contour(seg, init_snake, w_edge=100)

fig = util_plot.Figure(1,1, height=500,width=500)
fig.add_heatmap(seg, bounds)
fig.add_scatter2d(contour[:,0],contour[:,1], legend='contour')
fig.add_scatter2d(contour_pred[:,0],contour_pred[:,1], legend='contour_pred')
fig.add_scatter2d(init_snake[:,0],init_snake[:,1], legend='snake init')
fig.add_scatter2d(snake_contour[:,0],snake_contour[:,1], legend='contour_snake')
fig.plot('plot1.html')

fig2 = util_plot.Figure(1,1, height=500,width=500)
fig2.add_heatmap(mag[0], bounds)
fig2.add_scatter2d(contour[:,1],contour[:,0], legend='contour')
fig2.plot('plot2.html')
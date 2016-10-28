import utility
import numpy as np
import util_plot
import argparse
from skimage import segmentation
from keras.models import Model, load_model

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='?', default='OSMSC0001.arch.23')
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
H,W = seg.shape
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
fig.plot('./plots/plot1.html')

fig2 = util_plot.Figure(1,1, height=500,width=500)
fig2.add_heatmap(mag[0], bounds)
fig2.add_scatter2d(contour[:,1],contour[:,0], legend='contour')
fig2.plot('./plots/plot2.html')

#test OBG stuff
obg = utility.segToOBG(seg.reshape((H,W,1)),1)
fig2 = util_plot.Figure(1,3, height=1000,width=1000)
fig2.add_heatmap(obg[:,:,0], row=1, col=1)
fig2.add_heatmap(obg[:,:,1], row=1, col=2)
fig2.add_heatmap(obg[:,:,2], row=1, col=3)
fig2.plot('./plots/plotobg.html')

OBP_FCN = load_model('./models/OBP_FCN_output.h5')
mag = mag.reshape((64,64,1))
img_norm = utility.normalize_images(mag.reshape(1,
    mag.shape[0],mag.shape[1],mag.shape[2]))
y_obg = OBP_FCN.predict(img_norm)

fig2 = util_plot.Figure(1,3, height=1000,width=1000)
fig2.add_heatmap(y_obg[0,:,:,0], row=1, col=1)
fig2.add_heatmap(y_obg[0,:,:,1], row=1, col=2)
fig2.add_heatmap(y_obg[0,:,:,2], row=1, col=3)
fig2.plot('./plots/plotobg_pred.html')

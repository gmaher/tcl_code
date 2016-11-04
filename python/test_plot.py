import numpy as np
import utility.util_plot as util_plot

x1 = np.random.randn(100)
x2 = np.random.randn(100)

fig = util_plot.Figure(2,2,'blah')

fig.add_scatter2d(x1,x1,'normal1', 'markers')
fig.add_scatter2d(x2,x2,'normal2', 'markers')
fig.add_scatter2d(x1,x2,'normal3', 'markers',2,1)
fig.add_scatter2d(x2,x1,'normal4', 'markers',1,2)
fig.add_scatter2d(x2,x1,'normal5', 'markers',1,2)
fig.add_scatter3d(x1,x2,x1,'normal6', 'markers',2,2)
fig.plot()

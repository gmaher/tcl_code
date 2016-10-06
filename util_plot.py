import plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.tools import FigureFactory as FF

#TODO: how to incorporate layout/plot property setting e.g. line thickness?
#TODO: legend groups how do they work?
#TODO: 3D subplots https://plot.ly/python/3d-subplots/

class Figure:
    """
    Utility class to act as a wrapper around plotting libraries.
    """
    def __init__(self, rows, cols, title="", height=1000, width=1000):
        """
        Initialization.
        """
        self.rows = rows
        self.cols = cols

        self.fig = tools.make_subplots(rows=rows,cols=cols)
        self.fig['layout'].update(height=height,width=width, title=title)

    def plot(self,filename='plot.html'):
        """
        plots the graphs that have been added so far
        """
        py.offline.plot(self.fig, filename=filename)

    def add_scatter2d(self, x, y, legend, mode="lines+markers", row=1,col=1):
        """
        Add a 2d scatter plot to the figure.

        arguments:
        x -- x data to plot
        y -- y data to plot
        legend -- (string) legend to use
        mode -- (string) mode to use
        row -- (int) row of subplot grid to add to
        column -- (int) column of subplot grid to add to
        """
        trace = go.Scatter(
            x = x,
            y = y,
            mode = mode,
            name = legend
        )
        self.fig.append_trace(trace,row,col)

    def add_scatter3d(self, x, y, z, legend, mode='lines+markers', row=1, col=1):
        """
        Add a 3d scatter plot to the figure

        arguments:
        z -- z data to plot
        for other arguments see add_scatter2d
        """
        trace = go.Scatter3d(
			x=x,
			y=y,
			z=z,
			mode=mode,
			marker=dict(
			    size=1,
			    opacity=0.8
			)
		)
        self.fig.append_trace(trace,row,col)

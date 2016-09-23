import plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.tools import FigureFactory as FF

#TODO: how to incorporate layout/plot property setting e.g. line thickness?

class Figure:
    """
    Utility class to act as a wrapper around plotting libraries.
    """
    def __init__(self):
        """
        Initialization.
        """
    def add_scatter2d(self,x,y):
        """
        Add a 2d scatter plot to the figure.

        arguments:
        x -- x data to plot
        y -- y data to plot
        """

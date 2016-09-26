import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import utility

df = pd.read_csv('vtk_groups_errors.csv')
codes = df['code'].unique()
threshes = []
dx = 0.05
DX = []

for code in codes:
    df_e = df.loc[df['code'] == code, 'overlap_error']
    thresh, ts = utility.cum_error_dist(df_e, dx)
    threshes.append(thresh)
    DX.append(ts)

utility.plot_data_plotly(DX, threshes, codes,
    title='cumulative error distribution OSMSC0006', fn='./plots/cum_error_osmsc006.html')

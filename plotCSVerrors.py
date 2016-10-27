import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import utility
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ls_csv')
parser.add_argument('model_csv')
args = parser.parse_args()

ls_csv = args.ls_csv
model_csv = args.model_csv

df = pd.read_csv(ls_csv)
df_models = pd.read_csv(model_csv)
df = pd.concat([df,df_models])

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
    title='cumulative error distribution OSMSC0006',
    xlabel='fraction of groups below error',
    ylabel='normalized area of overlap error',
    fn='./plots/cum_error_osmsc0006.html')

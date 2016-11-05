import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import utility.utility as utility
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


def make_plot(pathtype=None):
    threshes = []
    dx = 0.05
    DX = []
    for code in codes:
        if pathtype == None:
            df_e = df.loc[df['code'] == code, 'overlap_error']
        else:
            df_e = df.loc[(df['code'] == code) & (df['path']==pathtype)\
             ,'overlap_error']
        thresh, ts = utility.cum_error_dist(df_e, dx)
        threshes.append(thresh)
        DX.append(ts)

    if pathtype == None:
        pathtype = 'Full Data'

    utility.plot_data_plotly(DX, threshes, codes,
        title='cumulative error distribution {}'.format(pathtype),
        xlabel='fraction of groups below error',
        ylabel='normalized area of overlap error',
        fn='./plots/cum_error_osmsc0006{}.html'.format(pathtype))

make_plot()
make_plot('aorta')
#make_plot('LCA')
make_plot('RCA')
make_plot('RPA')
make_plot('LPA')

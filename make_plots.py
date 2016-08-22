import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go

apps = ['_edge48',
'_image',
'_edge96']

colors = {'_edge48':'red',
'_image':'blue',
'_edge96':'green'}

markers = {'_edge48':'o',
'_image':'v',
'_edge96':'x'}


df = pd.read_csv('all_errors.csv')

df_rca = df.loc[(df['path'] == 'RCA') | (df['path']=='LCA')]

AORTAind = [(("aorta" in s) | ("Aorta" in s) | ("AORTA" in s) ) for s in df['path']]
df_aorta = df.loc[AORTAind]

PAind = ["PA" in s for s in df['path']]
df_pa = df.loc[PAind]
 
def make_thresh(df, val):
	thresh_errs = {}
	for app in apps:
		thresh_errs[app] = []

	ts = np.arange(0,1+val,val)
	for t in ts:
		for app in apps:
			frac = float(np.sum(df['error'+app]<=t))/len(df['error'+app])
			thresh_errs[app].append(frac)

	return (thresh_errs, ts)

thresh_aorta, ts_aorta = make_thresh(df_aorta,0.025)
thresh_rca, ts_rca = make_thresh(df_rca, 0.025)
thresh_pa, ts_pa = make_thresh(df_pa, 0.025)

graphs = []

for app in apps:
	#plt.plot(ts_aorta,thresh_aorta[app], color = colors[app], marker=markers[app], markersize=8, label=app, linewidth=2)
	trace = go.Scatter(
		x = ts_aorta,
		y = thresh_aorta[app],
		mode = 'lines+markers',
		name = 'aorta_'+app
		)
	graphs.append(trace)

for app in apps:
	#plt.plot(ts_rca,thresh_rca[app], color = colors[app], marker=markers[app], markersize=8, label=app, linewidth=2)
	trace = go.Scatter(
		x = ts_rca,
		y = thresh_rca[app],
		mode = 'lines+markers',
		name = 'coronary_'+app
		)

	graphs.append(trace)

for app in apps:
	#plt.plot(ts_rca,thresh_rca[app], color = colors[app], marker=markers[app], markersize=8, label=app, linewidth=2)
	trace = go.Scatter(
		x = ts_pa,
		y = thresh_pa[app],
		mode = 'lines+markers',
		name = 'pulmonary_'+app
		)

	graphs.append(trace)

py.offline.plot(graphs, filename="plot.html")

